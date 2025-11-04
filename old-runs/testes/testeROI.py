import os
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Pasta base
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
                
                # Pega a matriz de pixels da imagem
                pixel_array = ds.pixel_array

                # 2. Definir a Região de Interesse (ROI)
                roi_x = 500
                roi_y = 1200
                roi_largura = 400
                roi_altura = 100

                # 3. Visualizar a imagem completa com a ROI desenhada
                fig, ax = plt.subplots(1, figsize=(10, 10))
                
                # Mostra a imagem em escala de cinza
                ax.imshow(pixel_array, cmap=plt.cm.gray)
                
                # Cria um retângulo para representar a ROI
                rect = patches.Rectangle(
                    (roi_x, roi_y), 
                    roi_largura, 
                    roi_altura, 
                    linewidth=2, 
                    edgecolor='lime',  # Cor verde-limão
                    facecolor='none' # Sem preenchimento
                )
                
                # Adiciona o retângulo à imagem
                ax.add_patch(rect)
                
                # 4. Definir e desenhar a "linha de interesse" (perfil central)
                linha_y = roi_y + (roi_altura // 2)
                ax.axhline(y=linha_y, color='red', linestyle='--', linewidth=1, xmin=(roi_x/pixel_array.shape[1]), xmax=((roi_x+roi_largura)/pixel_array.shape[1]))

                ax.set_title(f"Imagem DICOM com ROI e Linha de Interesse\nArquivo: {os.path.basename(caminho_dicom)}")
                plt.show()

                # 5. Extrair os dados da linha de interesse e plotar o perfil
                perfil_intensidade = pixel_array[linha_y, roi_x : roi_x + roi_largura]

                plt.figure(figsize=(10, 5))
                plt.plot(perfil_intensidade)
                plt.title("Perfil de Intensidade da Linha de Interesse")
                plt.xlabel("Posição do Pixel na Linha (dentro da ROI)")
                plt.ylabel("Intensidade do Pixel")
                plt.grid(True)
                plt.show()

            except Exception as e:
                print(f"Erro ao processar {caminho_dicom}: {e}")

            # Para o loop após processar o primeiro arquivo para não abrir dezenas de janelas
            exit() 

print("Nenhum arquivo .dcm encontrado no diretório.")