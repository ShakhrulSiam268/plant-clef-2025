import os
import pandas as pd
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

class BioclipFeatureExtractor:
    def __init__(self):
        self.model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')
        self.preprocess = preprocess_train
        self.device=torch.device('cuda')
        self.model.to(self.device)
        print('Bioclip-2 Model Loaded...')

    def extract_features(self, image_path: str) -> torch.Tensor or None:
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                return image_features.squeeze(0).cpu() # Returns a (D,) tensor

        except FileNotFoundError:
            print(f"Warning: Image file not found at {image_path}. Skipping.")
            return None
        except Exception as e:
            print(f"Error processing image {image_path}: {e}. Skipping.")
            return None


def main():
    train_image_path = '/local/scratch1/siam/dataset/plant_clef/train/images_max_side_800'
    train_label_path = '/local/scratch1/siam/dataset/plant_clef/train/PlantCLEF2024singleplanttrainingdata.csv'
    test_image_path = '/local/scratch1/siam/dataset/plant_clef/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package'
    train_data = pd.read_csv(train_label_path, sep=';', dtype={'partner': str})

    e = BioclipFeatureExtractor()
    results = []

    for index, row in tqdm(train_data.iterrows(), total=train_data.shape[0], desc="Extracting Features"):
        image_name = row['image_name']
        species_id = str(row['species_id'])  # Ensure species_id is treated as a string for pathing

        image_path = os.path.join(train_image_path, species_id, image_name)

        embedding = e.extract_features(image_path)

        if embedding is not None:
            results.append({
                'image_name': image_name,
                'species_id': species_id,
                'embedding': embedding
            })

    output_file = "/local/scratch1/siam/dataset/plant_clef/train/image_embeddings_bioclip2.pkl"
    embeddings_df = pd.DataFrame(results)
    embeddings_df.to_pickle(output_file)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()


