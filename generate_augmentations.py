import cv2
import albumentations as A
import os
from tqdm import tqdm

# --- CONFIGURAÇÕES ---
INPUT_DIR = 'dataset'
OUTPUT_DIR = 'augmented_dataset'
NUM_AUGMENTATIONS_PER_IMAGE = 10 

# --- FIM DAS CONFIGURAÇÕES ---

# Caminhos de entrada (agora apontando para as subpastas 'train')
images_path = os.path.join(INPUT_DIR, 'images', 'train') # <--- MUDANÇA AQUI
labels_path = os.path.join(INPUT_DIR, 'labels', 'train') # <--- MUDANÇA AQUI

# Caminhos de saída (cria as pastas se não existirem)
aug_images_path = os.path.join(OUTPUT_DIR, 'images')
aug_labels_path = os.path.join(OUTPUT_DIR, 'labels')
os.makedirs(aug_images_path, exist_ok=True)
os.makedirs(aug_labels_path, exist_ok=True)


# --- PIPELINE DE AUMENTAÇÃO "RADICAL" (VERSÃO ATUALIZADA) ---
transform = A.Compose([
    # 1. Alterações de Cor, Brilho e Contraste
    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.8),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.7),
    A.ToGray(p=0.1),
    # A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),

    # 2. Transformações Geométricas (usando a função Affine, mais moderna)
    A.Affine(
        scale=(0.9, 1.1), 
        translate_percent=(-0.1, 0.1), 
        rotate=(-15, 15), 
        p=0.9,
        mode=cv2.BORDER_CONSTANT,
        cval=0 
    ),
    A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.GridDistortion(p=0.3),
    A.OpticalDistortion(distort_limit=1.5, p=0.3),

    # 3. Flips
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),

    # 4. Adição de Ruído e Efeitos
    A.GaussNoise(var_limit=(20.0, 100.0), mean=0, p=0.4),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.MedianBlur(blur_limit=(3, 7), p=0.5),
    ], p=0.3),

], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def augment_and_save():
    """
    Função principal que lê os dados, aplica as augmentations e salva os resultados.
    """
    # Adicionei mais extensões de arquivo por segurança
    valid_extensions = ('.jpg', '.jpeg', '.png', '.PNG', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(images_path) if f.endswith(valid_extensions)]
    
    print(f"Lendo de: {images_path}")
    print(f"Encontradas {len(image_files)} imagens. Gerando {NUM_AUGMENTATIONS_PER_IMAGE} augmentations para cada uma...")

    for filename in tqdm(image_files):
        # Monta os caminhos completos
        image_path = os.path.join(images_path, filename)
        label_name = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(labels_path, label_name)

        # Carrega a imagem e as anotações
        image = cv2.imread(image_path)
        if image is None:
            print(f"Aviso: Não foi possível ler a imagem {image_path}. Pulando.")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Lê as anotações do arquivo YOLO
        bboxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    bboxes.append(coords)
                    class_labels.append(class_id)
        
        # Gera N novas imagens a partir da original
        for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_image = augmented['image']
                aug_bboxes = augmented['bboxes']

                # Define o novo nome para os arquivos de saída
                base_name = os.path.splitext(filename)[0]
                new_filename = f"{base_name}_aug_{i}.png" # Salvar como PNG para evitar compressão
                new_labelname = f"{base_name}_aug_{i}.txt"

                # Salva a imagem aumentada
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(aug_images_path, new_filename), aug_image_bgr)
                
                # Salva as novas anotações no formato YOLO
                with open(os.path.join(aug_labels_path, new_labelname), 'w') as f:
                    for bbox, class_id in zip(aug_bboxes, augmented['class_labels']):
                        f.write(f"{class_id} {' '.join(map(str, bbox))}\n")
            except Exception as e:
                print(f"Não foi possível aumentar {filename} na iteração {i}: {e}")

    print("\nProcesso de Augmentation concluído!")
    print(f"Novos arquivos salvos em: {OUTPUT_DIR}")


if __name__ == '__main__':
    augment_and_save()
    
    
    # pegar user c3 mandar   
    # aprender a mexer em env
    # aprender a mexer em server linux
    # mini conda