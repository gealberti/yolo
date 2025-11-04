import os
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

base_dir = r"C:\inetpub\wwwroot\img-TCIA\manifest-ZkhPvrLo5216730872708713142\CBIS-DDSM"

# Percorre as subpastas para encontrar um arquivo DICOM
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.lower().endswith(".dcm"):
            caminho_dicom = os.path.join(root, file)
            
            print(f"Processando arquivo de exemplo: {caminho_dicom}")

            try:
                # 1. Ler o arquivo DICOM
                ds = pydicom.dcmread(caminho_dicom)
                pixel_array = ds.pixel_array

                # --- INÍCIO: Pré-processamento e Segmentação da Mama ---
                pixel_array_normalized = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                img_for_segmentation = pixel_array_normalized.copy()

                blurred_for_mama = cv2.GaussianBlur(img_for_segmentation, (21, 21), 0)

                # Separar a mama do resto da img
                _, thresh_mama = cv2.threshold(blurred_for_mama, 20, 255, cv2.THRESH_BINARY)
                
                # Encontra contornos na imagem limiarizada da mama
                contours_mama, _ = cv2.findContours(thresh_mama, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                mama_contour = None
                if contours_mama:
                    mama_contour = max(contours_mama, key=cv2.contourArea)
                    
                    mama_mask = np.zeros_like(pixel_array_normalized)
                    cv2.drawContours(mama_mask, [mama_contour], -1, 255, cv2.FILLED)
                    
                    mama_only = cv2.bitwise_and(pixel_array_normalized, pixel_array_normalized, mask=mama_mask)
                else:
                    print("Não foi possível isolar. Prosseguindo com a imagem completa.")
                    mama_only = pixel_array_normalized.copy() # Se a mama não for encontrada, usa a imagem inteira

                # --- FIM: Pré-processamento e Segmentação da Mama ---


                # --- INÍCIO DA LÓGICA DE DETECÇÃO DE ROIs DENTRO DA MAMA ---
                
                # Aplica um desfoque Gaussiano 
                pixel_array_blurred = cv2.GaussianBlur(mama_only, (5, 5), 0)

                #Threshold
                _, thresh = cv2.threshold(pixel_array_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                kernel = np.ones((3,3),np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

                # Encontrar Contornos
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Filtrar Contornos e Obter ROIs
                rois_detectadas = []
                min_area = 100   # Área mínima chutada
                max_area = 5000  # Área máxima
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    # Verifica se a área está dentro do range e se o contorno está dentro da mama
                    if min_area < area < max_area:
                        x, y, w, h = cv2.boundingRect(cnt)
                        if mama_contour is not None and cv2.pointPolygonTest(mama_contour, (x + w/2, y + h/2), False) >= 0:
                             rois_detectadas.append((x, y, w, h))

                # --- FIM DA LÓGICA DE DETECÇÃO DE ROIs ---

                # 7. Visualizar a imagem original com as ROIs detectadas
                fig, ax = plt.subplots(1, figsize=(12, 12))
                ax.imshow(pixel_array, cmap=plt.cm.gray) # Mostra a imagem DICOM original
                
                if rois_detectadas:
                    print(f"Encontradas {len(rois_detectadas)} ROIs candidatas DENTRO DA MAMA!!")
                    for (x, y, w, h) in rois_detectadas:
                        rect = patches.Rectangle(
                            (x, y), w, h,
                            linewidth=2, 
                            edgecolor='cyan', # ciano de nov
                            facecolor='none'
                        )
                        ax.add_patch(rect)
                    ax.set_title(f"Detecção Automática de ROIs (Dentro da Mama)\nArquivo: {os.path.basename(caminho_dicom)}")
                else:
                    print("Nenhuma ROI candidata encontrada com os critérios atuais dentro da mama.")
                    ax.set_title(f"Nenhuma ROI encontrada na mama\nArquivo: {os.path.basename(caminho_dicom)}")
                
                # Opcional: Desenhar o contorno da mama para visualização
                if mama_contour is not None:
                    # Desenha o contorno da mama em outra cor para diferenciar
                    x_mama, y_mama, w_mama, h_mama = cv2.boundingRect(mama_contour)
                    rect_mama = patches.Rectangle(
                        (x_mama, y_mama), w_mama, h_mama,
                        linewidth=1, 
                        linestyle='--',
                        edgecolor='red', 
                        facecolor='none'
                    )
                    ax.add_patch(rect_mama)
                plt.show()

            except Exception as e:
                print(f"Erro ao processar {caminho_dicom}: {e}")

            exit() 

print("Nenhum arquivo .dcm encontrado no diretório.")


# contorno da mama
# baixar imagens de massa teste e treino ok
# testar com as img de massa
# rede neural de deteccao de objetos yolo, faster r-cnn, ssd
# https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/ - ensina o que o youtube ensina e já tem cod pronto

# identificar as massas 
# converter img de dicom para png
#anotar num software de anotação de img como labelimg
# cvat
# treinar com as imagens de treino de teste

