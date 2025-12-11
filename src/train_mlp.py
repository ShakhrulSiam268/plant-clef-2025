import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os
from model_utils import SimpleMLP

# --- Configuration ---
EMBEDDINGS_FILE = "/local/scratch1/siam/dataset/plant_clef/train/image_embeddings_bioclip2.pkl"
MODEL_SAVE_PATH = "/local/scratch1/siam/saved_models/plant_clef/bioclip2_mlp_classifier.pth"
INPUT_SIZE = 768  # Dimensionality of BiCLIP-2 embeddings
HIDDEN_SIZE = 512
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 30

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Load Data and Prepare Labels ---
print("Loading embeddings and preparing data...")
try:
    embeddings_df = pd.read_pickle(EMBEDDINGS_FILE)
except FileNotFoundError:
    print(f"Error: File not found at {EMBEDDINGS_FILE}")
    exit()

# Extract raw labels (species_id)
y_raw = embeddings_df['species_id'].values

# --- NEW STEP: Filter out singleton classes ---
# 1a. Count occurrences of each species
class_counts = embeddings_df['species_id'].value_counts()
# 1b. Identify species with only 1 sample
singleton_classes = class_counts[class_counts == 1].index
# 1c. Filter the DataFrame
filtered_df = embeddings_df[~embeddings_df['species_id'].isin(singleton_classes)].copy()

print(f"Total initial samples: {len(embeddings_df)}")
print(f"Number of singleton classes removed: {len(singleton_classes)}")
print(f"Total samples remaining after filter: {len(filtered_df)}")

# Re-extract features and labels from the filtered data
X_raw = np.stack(filtered_df['embedding'].values) # Features
y_raw = filtered_df['species_id'].values         # Raw labels (species_id)

# 1d. Encode species_id to contiguous integers (0 to C-1)
# Must re-fit the encoder on the filtered data
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
NUM_CLASSES = len(label_encoder.classes_)
print(f"Total number of species classes after filter: {NUM_CLASSES}")

# 1e. Split data into training and validation sets (THIS WILL NOW WORK)
X_train, X_val, y_train, y_val = train_test_split(
    X_raw, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
)

# --- 2. PyTorch Dataset and DataLoader ---

class BiCLIPDataset(Dataset):
    """Custom Dataset for BiCLIP Embeddings"""
    def __init__(self, embeddings, labels):
        # Convert numpy arrays to PyTorch tensors
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) # Labels must be long type

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Create Dataset and DataLoader instances
train_dataset = BiCLIPDataset(X_train, y_train)
val_dataset = BiCLIPDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. Define the MLP Model ---

model = SimpleMLP(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)

# --- 4. Training Setup ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. Training Loop ---
print("\nStarting model training...")

best_val_accuracy = 0.0

for epoch in tqdm(range(NUM_EPOCHS)):
    # Training phase
    model.train()
    running_loss = 0.0
    for i, (embeddings, labels) in enumerate(train_loader):
        embeddings, labels = embeddings.to(device), labels.to(device)

        # Forward pass
        outputs = model(embeddings)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * embeddings.size(0)

    epoch_loss = running_loss / len(train_dataset)

    # Validation phase
    model.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for embeddings, labels in val_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    # Save the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save({
            'model_state_dict': model.state_dict(),
            'label_encoder': label_encoder,
            'num_classes': NUM_CLASSES,
            'input_size': INPUT_SIZE,
            'hidden_size': HIDDEN_SIZE
        }, MODEL_SAVE_PATH)
        print(f"Model saved with improved validation accuracy: {best_val_accuracy:.2f}%")

print("\nTraining complete.")
print(f"Best model saved to {MODEL_SAVE_PATH}")