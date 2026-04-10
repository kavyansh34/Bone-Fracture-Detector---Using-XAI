import os
import pandas as pd
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from captum.attr import LayerGradCam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
CSV_PATH = os.path.join('FracAtlas', 'dataset.csv')
IMAGE_DIR = os.path.join('FracAtlas', 'images')  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. FIXED CUSTOM DATASET (Corrected Pathing)
# ==========================================
class FractureDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['image_id']
        is_fractured = row['fractured']
        
        # Determine subdirectory based on fractured label
        sub_dir = 'Fractured' if is_fractured == 1 else 'Non_fractured'
        img_path = os.path.join(IMAGE_DIR, sub_dir, img_id)
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Fallback if the path is still wrong (e.g. check root IMAGE_DIR)
        if image is None:
            img_path = os.path.join(IMAGE_DIR, img_id)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Fallback for common extensions if the CSV ID didn't have one
            base_path = os.path.join(IMAGE_DIR, sub_dir, img_id.split('.')[0])
            for ext in ['.jpg', '.png', '.jpeg']:
                if os.path.exists(base_path + ext):
                    image = cv2.imread(base_path + ext, cv2.IMREAD_GRAYSCALE)
                    break

        if image is None:
            # Print which file failed so you can verify manually
            print(f"Warning: Could not load {img_id} from {IMAGE_DIR}")
            return torch.zeros((3, 224, 224)), torch.tensor(is_fractured, dtype=torch.long), img_id

        # Apply CLAHE Preprocessing
        image = self.clahe.apply(image)
        # Convert to RGB (3-channel) for DenseNet
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(is_fractured, dtype=torch.long)
        return image, label, img_id

# ==========================================
# 3. DATA PREPARATION (SPLITTING & LOADERS)
# ==========================================
df = pd.read_csv(CSV_PATH)

# Split: 80% Train, 20% Test (which we further split into Val and Test)
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['fractured'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['fractured'], random_state=42)

# Training Augmentation
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Validation/Test (No Augmentation)
val_test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_loader = DataLoader(FractureDataset(train_df, train_transforms), batch_size=32, shuffle=True)
val_loader = DataLoader(FractureDataset(val_df, val_test_transforms), batch_size=32, shuffle=False)
test_loader = DataLoader(FractureDataset(test_df, val_test_transforms), batch_size=32, shuffle=False)

# Class Weights (Normal: 3366, Fractured: 717)
weights = torch.tensor([0.60, 2.84], dtype=torch.float).to(DEVICE)

# ==========================================
# 4. MODEL, CRITERION, OPTIMIZER
# ==========================================
model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ==========================================
# 5. XAI & UTILITY FUNCTIONS
# ==========================================
def get_gradcam(model, input_tensor, target_class):
    gc = LayerGradCam(model, model.features.norm5)
    attr = gc.attribute(input_tensor.to(DEVICE), target=target_class)
    return attr

def train_one_epoch():
    model.train()
    running_loss = 0.0
    for images, labels, _ in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Avg Training Loss: {running_loss/len(train_loader):.4f}")

def evaluate_model(model, dataloader, title="Validation"):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print(f"\n--- {title} Report ---")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Fracture']))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fracture'], yticklabels=['Normal', 'Fracture'])
    plt.title(f'Confusion Matrix - {title}')
    plt.savefig(f'confusion_matrix_{title.replace(" ", "_")}.png')
    plt.close()

    return f1_score(all_labels, all_preds)

def train_model(epochs=15):
    best_f1 = 0.0
    for epoch in range(epochs):
        print(f"\n🚀 Epoch {epoch+1}/{epochs}")
        train_one_epoch()
        
        # Use val_loader to monitor progress and save the best model
        current_f1 = evaluate_model(model, val_loader, title=f"Val Epoch {epoch+1}")
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(model.state_dict(), 'best_fracture_model.pth')
            print("⭐ Best model updated and saved!")

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Start Training
    train_model(epochs=2)
    
    # Final Test on the unseen test_loader
    print("\n🏁 FINAL TEST PERFORMANCE:")
    model.load_state_dict(torch.load('best_fracture_model.pth'))
    evaluate_model(model, test_loader, title="Final Test Set")