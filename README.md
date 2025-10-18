# TCC_Image_Captioning


# Brach "yolo" - YOLOv4-tiny

Este projeto usa um modelo YOLOv4-tiny TFLite para detectar objetos em uma imagem.

## Instalação

1.  Clone o repositório:
    ```bash
    git clone [https://github.com/seu-usuario/seu-repo.git](https://github.com/seu-usuario/seu-repo.git)
    cd seu-repo
    ```

2.  Crie um ambiente virtual e instale as dependências:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Baixe o modelo:**
    O modelo não está incluído no repositório. Baixe os arquivos necessários:
    ```bash
    # Baixar o modelo
    wget [https://huggingface.co/gbahlnxp/yolov4tiny/download/main/yolov4-tiny_416_quant.tflite](https://huggingface.co/gbahlnxp/yolov4tiny/download/main/yolov4-tiny_416_quant.tflite)

    # Baixar os rótulos
    wget [https://huggingface.co/gbahlnxp/yolov4tiny/download/main/coco-labels-2014_2017.txt](https://huggingface.co/gbahlnxp/yolov4tiny/download/main/coco-labels-2014_2017.txt)
    ```

## Execução

```bash
python3 img_description.py