# from ultralytics import YOLO
# import torch

# def main():
#     """
#     Função aprimorada para treinar o modelo YOLOv8 com um dataset um pouco maior.
#     """
#     # --- 1. Configuração do Dispositivo ---
#     device = 0 if torch.cuda.is_available() else 'cpu'
#     print(f"Usando dispositivo: {device}")

#     # --- 2. Seleção do Modelo ---
#     model = YOLO("yolov8n.pt") # Modelo Nano - Mais rápido
#     # model = YOLO("yolov8s.pt")   # Modelo Small 

#     # --- 3. Treinamento Otimizado ---
#     print("Iniciando o treinamento otimizado...")
#     results = model.train(
#         # --- Parâmetros de Dataset e Hardware ---
#         data="mammography_dataset.yaml",
#         device=device,
#         cache=True,  

#         # --- Parâmetros de Treinamento e Otimização ---
#         epochs=150,    
#         patience=55,   
#         batch=-1,     
#         imgsz=640,
        
#         optimizer='auto',    # Deixa a Ultralytics escolher o melhor otimizador
#         lr0=0.01,           
#         lrf=0.01,         
#         warmup_epochs=3.0,  
        
#         # --- Organização dos Experimentos ---
#         project='yolo_training_mammography',
#         name='yolov8n_tuned_run1',

#         # --- TÉCNICAS DE DATA AUGMENTATION ---
#         hsv_v=0.4,
#         degrees=15.0,
#         translate=0.1,
#         scale=0.4,   
#         shear=5.0,
#         flipud=0.5,
#         fliplr=0.5,
#         mosaic=1.0,
#         mixup=0.1,
#         copy_paste=0.1,
#     )
#     print("Treinamento concluído.")
#     print(f"Resultados, pesos e logs salvos em: {results.save_dir}")

# if __name__ == '__main__':
#     main()
    
#     # avaliar o modelo e validar
#     # prever 
#     # prever pra teste
     
#      #    # treinar novamente com CC


from ultralytics import YOLO
import torch

def main():
    """
    Função para treinar o YOLOv8 com o dataset pré-aumentado.
    """
    # --- 1. Configuração do Dispositivo ---
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")

    # --- 2. Seleção do Modelo ---
    model = YOLO("yolo11n.pt") # Modelo Nano - Mais rápido

    print("Iniciando o treinamento com o dataset aumentado...")
    results = model.train(
        # --- Parâmetros de Dataset e Hardware ---
        data="mammography_dataset.yaml",
        device=device,
        cache=True,

        # --- Parâmetros de Treinamento ---
        epochs=250,
        patience=55,
        batch=100,  # -1 para auto-batch
        imgsz=640,
        optimizer='auto',
        lr0=0.01,
        lrf=0.01,

        # --- Organização dos Experimentos ---
        project='yolo_training_mammography',
        name='yolov8n_radical_augmentation_run',

        # --- DATA AUGMENTATION (SIMPLIFICADO) ---
        #mosaic=0,   
        fliplr=0.5  
    )
    print("Treinamento concluído.")
    print(f"Resultados, pesos e logs salvos em: {results.save_dir}")

if __name__ == '__main__':
    main()
