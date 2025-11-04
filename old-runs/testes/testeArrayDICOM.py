import os
import pydicom

# Pasta base
base_dir = r"C:\inetpub\wwwroot\img-TCIA\manifest-ZkhPvrLo5216730872708713142\CBIS-DDSM"

# Lista para armazenar resultados
dados = []

# Percorre todas as subpastas
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.lower().endswith(".dcm"):
            caminho = os.path.join(root, file)

            try:
                ds = pydicom.dcmread(caminho, stop_before_pixels=True)

                patient_id = ds.PatientID
                partes = patient_id.split("_")

                lateralidade = partes[-2]   # LEFT / RIGHT
                posicao = partes[-1]        # CC / MLO (às vezes vem "_1", "_2")

                # Se houver sufixo numérico (ex: MLO_2), separa
                if posicao in ["CC", "MLO"]:
                    repeticao = None
                else:
                    # Exemplo: "MLO_2" → posicao="MLO", repeticao="2"
                    if "_" in posicao:
                        posicao, repeticao = posicao.split("_", 1)
                    else:
                        repeticao = None

                dados.append({
                    "arquivo": caminho,
                    "patient_id": patient_id,
                    "lateralidade": lateralidade,
                    "posicao": posicao,
                    "repeticao": repeticao
                })

            except Exception as e:
                print(f"Erro ao ler {caminho}: {e}")

# Mostra resumo
for d in dados[:10]:  # mostra só os 10 primeiros
    print(d)
