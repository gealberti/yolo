import os
import pydicom
import numpy as np
import re
from PIL import Image

def convert_dicoms_to_png(source_dir, output_dir):
    """
    Varre um diretório de origem, encontra arquivos DICOM em subpastas,
    converte-os para PNG e os salva em um único diretório de destino
    com um nome de arquivo padronizado.

    Args:
        source_dir (str): O caminho para a pasta principal contendo os arquivos DICOM.
        output_dir (str): O caminho para a pasta onde as imagens PNG serão salvas.
    """
    # 1. Garante que o diretório de origem realmente existe antes de começar.
    if not os.path.isdir(source_dir):
        print(f"--- ERRO CRÍTICO ---")
        print(f"O diretório de origem não foi encontrado: '{source_dir}'")
        print("Por favor, verifique se o caminho no script está 100% correto.")
        print("Script encerrado.")
        return

    # 2. Cria a pasta de destino se ela não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Pasta de destino criada em: {output_dir}")

    patient_id_pattern = re.compile(r"P_(\d{5})")
    converted_count = 0
    
    print(f"Iniciando a varredura em '{source_dir}'...")
    # 3. Percorre todas as pastas e subpastas no diretório de origem
    for root, _, files in os.walk(source_dir):
        # --- LÓGICA MELHORADA ---
        # Primeiro, verifica se o NOME DO DIRETÓRIO ATUAL contém o que procuramos.
        is_original_folder = "full mammogram images" in root
        is_roi_folder = "ROI mask images" in root

        # Se não for uma pasta de interesse, pula para a próxima iteração
        if not is_original_folder and not is_roi_folder:
            continue

        # Se for uma pasta de interesse, tenta processar CADA arquivo dentro dela
        for filename in files:
            file_path = os.path.join(root, filename)
            
            try:
                # Tenta ler o arquivo como DICOM. Se falhar, não é um DICOM e será ignorado.
                dicom_data = pydicom.dcmread(file_path)

                # Extrai o ID do paciente do caminho da pasta
                match = patient_id_pattern.search(root)
                if not match:
                    continue
                
                patient_id = int(match.group(1))

                # Determina o tipo de imagem (original ou roi)
                image_type = "original" if is_original_folder else "roi"

                # Converte os dados de pixel para um formato de imagem
                pixel_array = dicom_data.pixel_array
                
                pixel_range = np.max(pixel_array) - np.min(pixel_array)
                if pixel_range == 0:
                    pixel_array_normalized = pixel_array
                else:
                    pixel_array_normalized = (pixel_array - np.min(pixel_array)) / pixel_range * 255.0
                
                pixel_array_uint8 = pixel_array_normalized.astype(np.uint8)
                image = Image.fromarray(pixel_array_uint8)

                # Remove a extensão original do arquivo para criar o novo nome
                filename_without_ext = os.path.splitext(filename)[0]
                new_filename = f"{patient_id}-{image_type}-{filename_without_ext}.png"
                output_path = os.path.join(output_dir, new_filename)
                
                image.save(output_path)
                print(f"SUCESSO: Convertido '{file_path}' -> '{output_path}'")
                converted_count += 1

            except pydicom.errors.InvalidDicomError:
                # Ignora silenciosamente arquivos que não são DICOM (ex: Thumbs.db)
                continue
            except Exception as e:
                print(f"ERRO: Não foi possível processar o arquivo '{file_path}'. Motivo: {e}")
                    
    print(f"\nConversão concluída! Total de {converted_count} arquivos convertidos.")


if __name__ == '__main__':
    # --- CONFIGURAÇÃO ---
    # O caminho foi corrigido para usar hífen, com base no seu exemplo.
    # Se o nome real da sua pasta usar espaço, apenas altere aqui.
    # source_folder = r"C:\Users\gealb\Downloads\testes-originais"
    source_folder = r"C:\inetpub\wwwroot\img-TCIA-completo\manifest-1758565166288\CBIS-DDSM"

    # Nome da pasta onde todas as imagens PNG serão salvas
    output_folder = "imagens-convertidas-teste"
    
    # Chama a função principal
    convert_dicoms_to_png(source_folder, output_folder)

