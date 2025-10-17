import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Modelo usado
MODEL_URL = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/classification/2"
# Resolução esperada
IMAGE_SHAPE = (224, 224)
# URL do arquivo com os nomes das 1000 categorias que o modelo consegue identificar
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"

def load_model():
    # Carrega o modelo de classificação de imagem do TensorFlow Hub. 
    print("Carregando o modelo...")
    # Cria a camada e retorna 
    classifier_layer = hub.KerasLayer(MODEL_URL, input_shape=IMAGE_SHAPE + (3,))
    print("Modelo carregado com sucesso!")
    return classifier_layer


def load_labels():
    # Carrega os rótulos (nomes das categorias) da internet. 
    print("Baixando os rótulos (labels)...")
    response = requests.get(LABELS_URL)
    labels_text = response.text
    # Separa o texto em uma lista de rótulos, um por linha
    labels_list = labels_text.splitlines()
    # Remove o primeiro item que é background
    return labels_list[1:] 

def get_image_from_url(url):
    # Baixa uma imagem de uma URL e a abre. 
    try:
        response = requests.get(url)
        # BytesIO para tratar os dados da imagem em memória, sem salvar no disco
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except Exception as e:
        print(f"Erro ao baixar ou abrir a imagem: {e}")
        return None

def preprocess_image(image):
    # Preparar a imagem para o formato que o modelo espera
    # Redimensiona a imagem para o tamanho esperado pelo modelo (224x224)
    image = image.resize(IMAGE_SHAPE)
    
    # Converte a imagem para um array NumPy
    image_array = np.array(image)
    
    # Normaliza os pixels da imagem para o intervalo [0, 1]
    image_array = image_array / 255.0
    
    # Adiciona uma dimensão extra no início (batch dimension)
    # O modelo espera um "lote" de imagens, mesmo que seja uma só.
    # O formato muda de (224, 224, 3) para (1, 224, 224, 3).
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def main():
    # URL da imagem
    image_url = "https://s4.static.brasilescola.uol.com.br/be/2021/04/casal-de-leoes.jpg"

    # Carrega o modelo e os rótulos
    classifier_model = load_model()
    labels = load_labels()
    
    print(f"\nBaixando imagem de: {image_url}")
    image = get_image_from_url(image_url)
    
    if image:
        processed_image = preprocess_image(image)
        
        print("Analisando a imagem...")
        result = classifier_model(processed_image)
        
        # O resultado é um array de probabilidades, pega o índice do maior valor
        predicted_class_index = np.argmax(result)
        
        # Usar o índice para encontrar o nome da classe na lista de rótulos
        predicted_class_name = labels[predicted_class_index].strip()
        
        # Pega a "confiança" da previsão (o valor da %)
        confidence = result[0][predicted_class_index]
        
        # Imprime o resultado final.
        print("\n--- Resultado ---")
        print(f"Descrição: {predicted_class_name}")
        print(f"Confiança: {confidence:.2%}")
        print("-----------------")

# Executa a função principal quando o script é rodado.
if __name__ == "__main__":
    main()