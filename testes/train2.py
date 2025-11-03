from ultralytics import YOLO
import torch

def main():
    """
    Função aprimorada para treinar o modelo YOLOv8 com de data augmentation
    """
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")

    # --- 2. Carregamento do Modelo ---
    model = YOLO("yolov8n.pt")

    # --- 3. Treinamento com Data Augmentation ---
    print("Iniciando o treinamento com data augmentation...")
    results = model.train(
        # --- Parâmetros Essenciais ---
        data="mammography_dataset.yaml",
        epochs=100,
        imgsz=640,
        device=device,
        patience=40, # Parada antecipada se a métrica de validação não melhorar
        batch=-1,    # Deixe a YOLO encontrar o maior batch size que cabe na sua VRAM (autobatch)
                     # Se der erro, defina um valor fixo, ex: 8, 16, 32.

        # --- Organização dos Experimentos ---
        project='yolo_training_mammography', # Pasta principal para salvar todos os experimentos
        name='yolov8n_augmented_run1',       # Nome específico para este treinamento

        # --- TÉCNICAS DE DATA AUGMENTATION ---

        # Augmentations de Cor (Fotométricas)
        hsv_h=0.015,   # (hue) variação de matiz da cor
        hsv_s=0.7,     # (saturation) variação de saturação da cor
        hsv_v=0.4,     # (value) variação de brilho/valor da cor

        # Augmentations Geométricas
        degrees=15.0,    # rotação aleatória da imagem em +/- 15 graus
        translate=0.1, # translação aleatória (mover a imagem) em +/- 10%
        scale=0.5,     # escala aleatória (zoom) em +/- 50%
        shear=5.0,     # cisalhamento (inclinação) da imagem em +/- 5 graus
        perspective=0.0, # transformação de perspectiva (geralmente 0 para a maioria dos casos)
        flipud=0.5,    # (flip up-down) virar a imagem de cabeça para baixo com 50% de chance
        fliplr=0.5,    # (flip left-right) virar a imagem horizontalmente com 50% de chance

        # Augmentations Avançadas (muito eficazes)
        mosaic=1.0,    # combina 4 imagens de treino em uma só (desativado nas últimas 10 épocas por padrão)
        mixup=0.1,     # mistura duas imagens e seus rótulos (probabilidade de 10%)
        copy_paste=0.1 # copia objetos de uma imagem e cola em outra (probabilidade de 10%)

        # --- Otimização do Desempenho ---
        # cache=True,  # Descomente se o seu dataset for pequeno o suficiente para caber na RAM.
                       # Isso acelera o carregamento dos dados após a primeira época.
    )
    print("Treinamento concluído.")
    print(f"Resultados, pesos e logs salvos em: {results.save_dir}")

if __name__ == '__main__':
    main()
    # data augmentation
    # aritgo yolo 8v n  
    # como ele faz a previsao:
    # 70 15 15 / 80 10 10
    # analise de yolo
    # pelo menos 150 img 
    # 110 20 20 