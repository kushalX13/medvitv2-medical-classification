import os
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from collections import Counter

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# BloodMNIST class order (official)
bloodmnist_classes = [
    "neutrophil",         # 0
    "eosinophil",         # 1
    "basophil",           # 2
    "lymphocyte",         # 3
    "monocyte",           # 4
    "immature_granulocyte",  # 5
    "erythroblast",       # 6
    "platelet"            # 7 (not in Czech set)
]

# Correct Czech folder names mapped to BloodMNIST indices
czech_to_bloodmnist_label_map = {
    "Neutrophile Segment": 0,
    "Neutrophile Band": 0,
    "Eosinophile": 1,
    "Basophile": 2,
    "Lymphocyte": 3,
    "Monocyte": 4,
    "Myeloblast": 5,
    "Normoblast": 6
    # "Lymphoblast" is dropped
}

# Load Czech dataset
root_dir = '/scratch/kgangara/dl_final_project/blood_dataset_external/czech_9_classes'
filtered_images = []

print("\n🧼 Filtering Czech dataset...")
for folder in os.listdir(root_dir):
    folder_key = folder.strip()
    if folder_key not in czech_to_bloodmnist_label_map:
        print(f"⛔ Skipping folder: {folder}")
        continue

    label_index = czech_to_bloodmnist_label_map[folder_key]
    folder_path = os.path.join(root_dir, folder)
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            filtered_images.append((os.path.join(folder_path, file), label_index))

print(f"\n✅ Loaded {len(filtered_images)} valid images.")

# Dataset class
class CustomCzechDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformations
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# DataLoader
dataset = CustomCzechDataset(filtered_images, transform=test_transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

# Load model
from MedViT import MedViT_tiny

# Build model for 7-class Czech WBC dataset
model = MedViT_tiny(num_classes=7)

# Load checkpoint trained on 8-class BloodMNIST
checkpoint_path = '/scratch/kgangara/dl_final_project/MedViTV2/my_checkpoints/MedViT_tiny_bloodmnist_custom.pth'
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Remove the 8-class classification head weights
filtered_state_dict = {k: v for k, v in checkpoint['model'].items() if not k.startswith('proj_head')}

# Load only matching layers
model.load_state_dict(filtered_state_dict, strict=False)

# Done — last layer is randomly initialized to output 7 classes
model.to(device)
model.eval()

# Evaluation
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in tqdm(loader, desc="🔍 Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Results
print("\n📊 Classification Report (Czech Dataset vs BloodMNIST):")
print(classification_report(
    all_labels,
    all_preds,
    labels=[0, 1, 2, 3, 4, 5, 6],  # Only include Czech classes
    target_names=bloodmnist_classes[:7]
))



import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3, 4, 5, 6])

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=bloodmnist_classes[:7], yticklabels=bloodmnist_classes[:7])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Czech WBC vs BloodMNIST Model")
plt.tight_layout()
plt.show()
