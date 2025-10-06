import os
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Step 1: Set device
# -------------------------------
device = torch.device('cuda')

# -------------------------------
# Step 2: Define transforms
# -------------------------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# -------------------------------
# Step 3: Load dataset
# -------------------------------
data_dir = '/scratch/kgangara/dl_final_project/blood_dataset_external/iran_2_classes/Test-B'
iran_dataset = ImageFolder(root=data_dir, transform=test_transform)
iran_loader = DataLoader(iran_dataset, batch_size=32, shuffle=False, num_workers=4)

# -------------------------------
# Step 4: Add safe globals
# -------------------------------
from torch.serialization import add_safe_globals
add_safe_globals({np.dtype, np.core.multiarray.scalar})

# -------------------------------
# Step 5: Load checkpoint
# -------------------------------
checkpoint_path = '/scratch/kgangara/dl_final_project/MedViTV2/my_checkpoints/MedViT_tiny_bloodmnist_custom.pth'
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# -------------------------------
# Step 6: Load model
# -------------------------------
from MedViT import MedViT_tiny
model = MedViT_tiny(num_classes=2)

# Remove classification head weights from checkpoint
filtered_state_dict = {k: v for k, v in checkpoint['model'].items() if not k.startswith("proj_head.0")}
model.load_state_dict(filtered_state_dict, strict=False)
model.eval()
model.to(device)

# -------------------------------
# Step 7: Evaluate
# -------------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(iran_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# -------------------------------
# Step 8: Metrics
# -------------------------------
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=iran_dataset.classes))

conf_mat = confusion_matrix(all_labels, all_preds)
acc = accuracy_score(all_labels, all_preds)

print("Confusion Matrix:")
print(conf_mat)
print(f"\nAccuracy: {acc * 100:.2f}%")

# -------------------------------
# Step 9: Save visual outputs
# -------------------------------
base_dir = "external_data_results/Iran_dataset"
os.makedirs(f"{base_dir}/accuracy_chart", exist_ok=True)
os.makedirs(f"{base_dir}/confusion_matrix", exist_ok=True)

# Accuracy bar chart
plt.figure(figsize=(6, 4))
plt.bar(["Accuracy"], [acc * 100], color='skyblue')
plt.ylabel("Percentage (%)")
plt.ylim(0, 100)
plt.title("Model Accuracy on Iran Dataset")
plt.savefig(f"{base_dir}/accuracy_chart/accuracy_barplot.png")
plt.close()

# Confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap="Blues",
            xticklabels=iran_dataset.classes,
            yticklabels=iran_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig(f"{base_dir}/confusion_matrix/confusion_matrix.png")
plt.close()
