# Importa o interpretador TFLite específico para o Pi
import tflite_runtime.interpreter as tflite 
import numpy as np
import cv2 # OpenCV
import requests
from PIL import Image
from io import BytesIO
from collections import Counter

# Adicione esta função perto do topo, após as importações
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

# --- CONFIGURAÇÕES ---
MODEL_PATH = 'yolov4-tiny_416_quant.tflite'
LABELS_PATH = 'coco-labels-2014_2017.txt'
INPUT_SIZE = (416, 416)
CONFIDENCE_THRESHOLD = 0.5 # Confiança mínima para considerar uma detecção
NMS_THRESHOLD = 0.4 # Limiar para o Non-Max Suppression (evita caixas duplicadas)

def load_labels():
    """Carrega os nomes das classes do arquivo coco.names."""
    with open(LABELS_PATH, 'r') as f:
        return [line.strip() for line in f.readlines()]

def get_image_from_url(url):
    """Baixa uma imagem de uma URL e a converte para o formato OpenCV."""
    try:
        response = requests.get(url)
        # Abre com PIL para garantir a conversão correta de formatos
        image_pil = Image.open(BytesIO(response.content)).convert('RGB')
        # Converte de PIL para o formato de array do OpenCV (BGR)
        image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        return image_cv2
    except Exception as e:
        print(f"Erro ao baixar ou abrir a imagem: {e}")
        return None

def preprocess_image(image):
    """Prepara a imagem para o formato que o modelo YOLO (quantizado) espera."""
    # Obtém a forma original e a forma de destino
    original_shape = image.shape
    target_h, target_w = INPUT_SIZE
    
    # Redimensiona a imagem para 416x416
    image_resized = cv2.resize(image, (target_w, target_h))
    
    # Converte a imagem para o formato INT8 (range -128 a 127)
    # 1. Converte para float32 (temporariamente) para fazer a subtração
    image_data = image_resized.astype(np.float32)
    # 2. Muda o range de 0..255 para -128..127
    image_data = image_data - 128.0
    # 3. Converte para o tipo final INT8 que o modelo espera
    image_data = image_data.astype(np.int8) 
    
    # Adiciona a dimensão do lote (batch)
    image_data = np.expand_dims(image_data, axis=0)
    
    return image_data, original_shape

def postprocess_results(detections, output_details, labels, original_shape):
    """
    Processa a saída bruta do modelo (quantizado, com feature maps) para obter caixas, classes e confianças.
    """
    all_boxes = []
    all_scores = []
    all_class_ids = []
    
    orig_h, orig_w, _ = original_shape
    input_h, input_w = INPUT_SIZE # (416, 416)

    # Âncoras do YOLOv4-tiny (precisamos delas para decodificar)
    anchors = np.array([
        [10, 14], [23, 27], [37, 58], 
        [81, 82], [135, 169], [344, 319]
    ], dtype=np.float32)
    # Máscaras de quais anchors usar para cada saída
    anchor_masks = [[3, 4, 5], [1, 2, 3]] # O segundo (maior) é 13x13, o primeiro (menor) é 26x26
    
    # O modelo tem 2 saídas (detections[0] e detections[1])
    # Precisamos garantir que estamos aplicando a máscara de anchor correta para cada
    if detections[0].shape[1] == 13: # Saída 13x13
         output_indices = [0, 1] # Ordem: [13x13, 26x26]
         masks_to_use = [anchor_masks[0], anchor_masks[1]]
    else: # Saída 26x26
         output_indices = [1, 0] # Ordem: [13x13, 26x26]
         masks_to_use = [anchor_masks[1], anchor_masks[0]]
    
    for i, det_index in enumerate(output_indices):
        # 1. Obter tensor (ele já é FLOAT32, sem de-quantizar)
        output_tensor = detections[det_index]
        
        # 2. Obter forma e anchors
        grid_h, grid_w = output_tensor.shape[1:3] # 13x13 ou 26x26
        num_anchors = 3
        num_classes = 80 # O modelo COCO tem 80 classes
        
        # 3. Reformata para (1, H, W, 3, 85)
        output_tensor = output_tensor.reshape((1, grid_h, grid_w, num_anchors, 5 + num_classes))
        
        # 4. Decodificar
        # Cria a grid de células (cx, cy)
        grid_y, grid_x = np.meshgrid(np.arange(grid_h), np.arange(grid_w))
        grid = np.stack((grid_x, grid_y), axis=-1).reshape((1, grid_h, grid_w, 1, 2)).astype(np.float32)
        
        # Decodifica as coordenadas xy (centro) e wh (tamanho)
        box_xy = (sigmoid(output_tensor[..., 0:2]) + grid) / (grid_w, grid_h) # Normalizado 0-1
        box_wh = (np.exp(output_tensor[..., 2:4]) * anchors[masks_to_use[i]]) / (input_w, input_h) # Normalizado 0-1
        
        # Converte xy (centro) e wh para xmin, ymin, xmax, ymax
        box_xymin = box_xy - (box_wh / 2.0)
        box_xymax = box_xy + (box_wh / 2.0)
        
        boxes = np.concatenate((box_xymin, box_xymax), axis=-1)
        
        # 5. Obter confianças
        object_confidence = sigmoid(output_tensor[..., 4:5])
        class_confidences = sigmoid(output_tensor[..., 5:])
        
        box_scores = object_confidence * class_confidences
        
        # 6. Achar a melhor classe para cada box
        box_class_ids = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        
        # 7. Filtrar por confiança
        filtering_mask = box_class_scores > CONFIDENCE_THRESHOLD
        
        boxes = boxes[filtering_mask]
        scores = box_class_scores[filtering_mask]
        class_ids = box_class_ids[filtering_mask]
        
        # 8. Ajustar caixas para o tamanho da imagem original
        boxes[..., [0, 2]] = boxes[..., [0, 2]] * orig_w # xmin, xmax
        boxes[..., [1, 3]] = boxes[..., [1, 3]] * orig_h # ymin, ymax
        
        # 9. Converter para formato (x, y, w, h) para o NMS do OpenCV
        boxes_for_nms = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            boxes_for_nms.append([int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)])
        
        all_boxes.extend(boxes_for_nms)
        all_scores.extend(scores)
        all_class_ids.extend(class_ids)

    # 10. Aplicar Non-Max Suppression (NMS) final
    indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    
    final_detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            class_name = labels[all_class_ids[i]]
            final_detections.append(class_name)
            
    return final_detections

def generate_description(detected_objects):
    """Gera uma frase simples baseada nos objetos detectados."""
    if not detected_objects:
        return "Nenhum objeto reconhecido na imagem."
        
    # Conta a ocorrência de cada objeto
    counts = Counter(detected_objects)
    
    parts = []
    for obj, count in counts.items():
        if count == 1:
            parts.append(f"um {obj}")
        else:
            # Lida com plural simples (adicionando 's')
            parts.append(f"{count} {obj}s")
            
    return "A imagem contém " + ", ".join(parts) + "."

def main():
    """Função principal que orquestra todo o processo."""
    # URL da imagem com os dois leões
    image_url = "https://s4.static.brasilescola.uol.com.br/be/2021/04/casal-de-leoes.jpg"
    # URL de teste com mais objetos:
    # image_url = "https://static.todamateria.com.br/upload/pe/ss/pessoas-na-praia-cke.jpg"

    print("Carregando o modelo YOLOv4-tiny TFLite...")
    # Usa o 'tflite.Interpreter' que importamos
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Modelo carregado.")

    labels = load_labels()
    
    print(f"\nBaixando imagem de: {image_url}")
    image = get_image_from_url(image_url)
    
    if image is not None:
        # Prepara a imagem e obtém a sua forma original
        processed_image, original_shape = preprocess_image(image)
        
        # Define o tensor de entrada e executa a inferência
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        print("Analisando a imagem com YOLO...")
        interpreter.invoke()
        
        # Obtém os resultados brutos da detecção
        detections = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        
        # Linha corrigida
        detected_objects = postprocess_results(detections, output_details, labels, original_shape)
        
        # Gera e imprime a descrição final
        description = generate_description(detected_objects)
        
        print("\n--- Resultado da Detecção ---")
        print(description)
        print("-----------------------------")

if __name__ == "__main__":
    main()