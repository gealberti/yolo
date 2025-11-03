from ultralytics import YOLO
import torch

def main():
    """
    Função para avaliar, validar e prever com um modelo YOLOv8 treinado.
    """
    # --- 1. Configuração do Dispositivo ---
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")

    # --- 2. Carregar o Modelo Treinado ---
    model_path = 'yolo_training_mammography/yolov8n_radical_augmentation_run/weights/best.pt'
    model = YOLO(model_path)
    print(f"Modelo carregado de: {model_path}")

    # --- 3. AVALIAR O MODELO (no conjunto de validação) ---
    print("\nIniciando a avaliação no conjunto de validação...")
    metrics = model.val(
        data="mammography_dataset.yaml",
        split='val', 
        imgsz=640,
        batch=6,
        project='yolo_evaluation_mammography',
        name='validation_on_best_model'
    )
    print(f"mAP50-95 da validação: {metrics.box.map}")
    print(f"Resultados da validação salvos em: yolo_evaluation_mammography/validation_on_best_model")


    # --- 4. VALIDAR / TESTAR (no conjunto de teste) ---
    print("\nIniciando a avaliação final no conjunto de teste...")
    test_metrics = model.val(
        data="mammography_dataset.yaml",
        split='test', 
        imgsz=640,
        project='yolo_evaluation_mammography',
        name='test_on_best_model'
    )
    print(f"mAP50-95 do teste: {test_metrics.box.map}")
    print(f"Resultados do teste salvos em: yolo_evaluation_mammography/test_on_best_model")


    # --- 5. PREVER (em novas imagens) ---
    print("\nIniciando a previsão em imagens do conjunto de teste...")
    results = model.predict(
        source='dataset/images/test',
        save=True, 
        imgsz=640,
        conf=0.5, # Agora só mostra confiança mais alta
        project='yolo_prediction_mammography',
        name='predictions'
    )
    print(f"Previsões salvas em: yolo_prediction_mammography/predictions")
    

    # for r in results:
    #     boxes = r.boxes  # caixas delimitadoras
    #     masks = r.masks  # máscaras de segmentação
    #     keypoints = r.keypoints # keypoints
    #     probs = r.probs  # probabilidades para classificação

if __name__ == '__main__':
    main()