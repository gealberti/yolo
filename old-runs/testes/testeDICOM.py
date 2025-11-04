import pydicom
import matplotlib.pyplot as plt
import numpy as np

aqvDicom = "C:/inetpub/wwwroot/img-TCIA/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM/Calc-Test_P_00038_LEFT_CC/08-29-2017-DDSM-NA-96009/1.000000-full mammogram images-63992/1-1.dcm"

try:
    # Carrega o arquivo DICOM
    ds = pydicom.dcmread(aqvDicom)

    patient_id = ds.PatientID
    partes = patient_id.split("_")

    # Concaternando partes do ID do paciente para extrair informações
    lateralidade = partes[-2]   # LEFT ou RIGHT
    posicao = partes[-1]        # CC ou MLO

    print("--- Informações da Imagem ---")
    print(f"ID do Paciente: {patient_id}")
    print(f"Lateralidade da Imagem (L/R): {lateralidade}")
    print(f"Posição da Visualização: {posicao}")

    imagem = ds.pixel_array

    if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
        imagem = imagem.astype(np.float64) * slope + intercept

    plt.figure(figsize=(10, 10))
    plt.imshow(imagem, cmap=plt.cm.gray)
    plt.title(f"Mamografia - {lateralidade} - {posicao}")
    plt.xlabel('Colunas de Pixels')
    plt.ylabel('Linhas de Pixels')
    plt.colorbar(label='Intensidade do Pixel')
    plt.show()

except FileNotFoundError:
    print(f"O arquivo não foi encontrado em: '{aqvDicom}'")
except Exception as e:
    print(f"Ocorreu um erro: {e}")
