import os
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

base_dir = r"C:\inetpub\wwwroot\img-TCIA\manifest-ZkhPvrLo5216730872708713142\CBIS-DDSM"

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.lower().endswith(".dcm"):
            caminho_dicom = os.path.join(root, file)
            
            print(f"Processando arquivo de exemplo: {caminho_dicom}")

            try:
                # 1. Ler o arquivo DICOM
                ds = pydicom.dcmread(caminho_dicom)
                pixel_array = ds.pixel_array

                # --- INÍCIO DA LÓGICA ---

                # 2. Pré-processamento da Imagem
                # Normaliza a imagem para 8-bit.
                pixel_array_normalized = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                #desfoque Gaussiano
                pixel_array_blurred = cv2.GaussianBlur(pixel_array_normalized, (5, 5), 0)

                #Threshold
                _, thresh = cv2.threshold(pixel_array_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # 4. Encontrar Contornos
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 5. Filtrar Contornos e Obter ROIs
                rois_detectadas = []
                min_area = 500   # chutei um min
                max_area = 10000 # chutei um max
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if min_area < area < max_area:
                        x, y, w, h = cv2.boundingRect(cnt)
                        rois_detectadas.append((x, y, w, h))

                # --- FIM DA LÓGICA DE IA ---

                fig, ax = plt.subplots(1, figsize=(12, 12))
                ax.imshow(pixel_array, cmap=plt.cm.gray)
                
                if rois_detectadas:
                    print(f"Encontradas {len(rois_detectadas)} ROIs candidatas.")
                    for (x, y, w, h) in rois_detectadas:
                        rect = patches.Rectangle(
                            (x, y), w, h,
                            linewidth=2, 
                            edgecolor='cyan', # Cor ciano pq sim
                            facecolor='none'
                        )
                        ax.add_patch(rect)
                    ax.set_title(f"Detecção Automática de ROIs\nArquivo: {os.path.basename(caminho_dicom)}")
                else:
                    print("Nenhuma ROI candidata encontrada com os critérios atuais.")
                    ax.set_title(f"Nenhuma ROI encontrada\nArquivo: {os.path.basename(caminho_dicom)}")
                
                plt.show()

            except Exception as e:
                print(f"Erro ao processar {caminho_dicom}: {e}")

            exit() 

print("Nenhum arquivo .dcm encontrado no diretório.")
