# from ultralytics import YOLO
# import torch

# def main():
#     """
#     Função principal para treinar o modelo YOLOv8.
#     """
#     device = 0 if torch.cuda.is_available() else 'cpu'
#     print(f"Usando dispositivo: {device}")

#     # YOLOv8n
#     model = YOLO("yolov8n.pt")

#     print("Iniciando o treinamento...")
#     results = model.train(
#         data="mammography_dataset.yaml",
#         epochs=100,
#         imgsz=640,
#         device=device,
#         patience=30 # Parada antecipada se não houver melhora após 30 épocas
#     )
#     print("Treinamento concluído.")

# if __name__ == '__main__':
#     main()
