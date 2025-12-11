import torch
import torch.nn as nn
from PIL import Image
import open_clip
import torch.serialization


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


class BioclipFeatureExtractor:
    def __init__(self):
        self.model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')
        self.preprocess = preprocess_train
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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