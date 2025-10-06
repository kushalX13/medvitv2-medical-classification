#######################################


########################################
#           train_hybrid_70_30.py      #
#     Hybrid training Czech + Blood    #
########################################

import os
import shutil
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import ImageFolder

from MedViT import MedViT_tiny

# --------------------
# Config & Class Maps
# --------------------

CLASS_MAPPING = {
    'Basophile': 'basophil',
    'Eosinophile': 'eosinophil',
    'Monocyte': 'monocyte',
    'Lymphocyte': 'lymphocyte',
    'Neutrophile Band': 'neutrophil',
    'Neutrophile Segment': 'neutrophil',
    'Normoblast': 'erythroblast',
    'Myeloblast': 'ig',
    'Basophil': 'basophil',
    'Eosinophil': 'eosinophil',
    'Monocyte': 'monocyte',
    'Lymphocyte': 'lymphocyte',
    'Neutrophil': 'neutrophil',
    'Erythroblast': 'erythroblast',
    'IG': 'ig'
}
EXCLUDE_CLASSES = ['Lymphoblast', 'Platelet']

# --------------------------
# Normalize Czech directory
# --------------------------
def normalize_and_filter_dataset(src_path, prefix):
    norm_dir = f"/tmp/{prefix}_normalized"
    if os.path.exists(norm_dir):
        shutil.rmtree(norm_dir)
    os.makedirs(norm_dir, exist_ok=True)

    for cls in os.listdir(src_path):
        if cls in EXCLUDE_CLASSES:
            continue
        unified = CLASS_MAPPING.get(cls)
        if unified is None:
            continue
        src_cls_dir = os.path.join(src_path, cls)
        dst_cls_dir = os.path.join(norm_dir, unified)
        os.makedirs(dst_cls_dir, exist_ok=True)
        for img in os.listdir(src_cls_dir):
            src_img_path = os.path.join(src_cls_dir, img)
            unique_name = f"{cls}_{img}".replace(" ", "_")
            dst_img_path = os.path.join(dst_cls_dir, unique_name)
            if not os.path.exists(dst_img_path):
                os.symlink(src_img_path, dst_img_path)
    return norm_dir

# -------------------------
# Load BloodMNIST from NPZ
# -------------------------
def load_bloodmnist_npz(npz_path):
    data = np.load(npz_path)
    # 1) combine train/val/test
    imgs = []
    labs = []
    for split in ('train', 'val', 'test'):
        imgs.append(data[f'{split}_images'])            # (N, 224, 224, 3)
        labs.append(data[f'{split}_labels'].squeeze())  # (N,)
    images = np.concatenate(imgs, axis=0)
    labels = np.concatenate(labs, axis=0)

    # 2) filter out platelets (label==7)
    valid_idx = labels != 7
    images = images[valid_idx]
    labels = labels[valid_idx]

    # 3) wrap in a Dataset
    class BloodDataset(torch.utils.data.Dataset):
        def __init__(self, images, labels, transform):
            self.images = images
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img = self.images[idx].astype(np.uint8)          # H×W×C
            img = torch.tensor(img / 255., dtype=torch.float32)
            img = img.permute(2, 0, 1)                       # → C×H×W
            if self.transform:
                img = self.transform(img)
            label = int(self.labels[idx])
            return img, label

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    return BloodDataset(images, labels, transform)

# -------------------------
# Build Hybrid Dataset
# -------------------------
def build_hybrid_dataset(czech_path, bloodmnist_npz_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    norm_czech = normalize_and_filter_dataset(czech_path, "czech")
    czech_dataset = ImageFolder(norm_czech, transform=transform)
    blood_dataset = load_bloodmnist_npz(bloodmnist_npz_path)

    czech_len = int(0.7 * len(czech_dataset))
    blood_len = int(0.3 * len(blood_dataset))

    hybrid_dataset = data.ConcatDataset([
        torch.utils.data.Subset(czech_dataset, range(czech_len)),
        torch.utils.data.Subset(blood_dataset, range(blood_len))
    ])

    y = [hybrid_dataset[i][1] for i in range(len(hybrid_dataset))]
    train_idx, val_idx = train_test_split(list(range(len(hybrid_dataset))), test_size=0.2, stratify=y)
    train_dataset = torch.utils.data.Subset(hybrid_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(hybrid_dataset, val_idx)

    return train_dataset, val_dataset, len(czech_dataset.classes), czech_dataset, blood_dataset

# --------------------
# Training
# --------------------
def train_model(model, train_loader, val_loader, device, save_path, plot_dir, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader), eta_min=5e-6
    )

    best_acc = 0.0
    train_accs, val_accs = [], []
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # --------- TRAINING ---------
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ------- VALIDATION -------
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = running_loss / total
        val_acc  = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # ----- EPOCH SUMMARY -----
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}"
        )

        # ----- SAVE BEST MODEL -----
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    # ----- PLOT & SAVE CURVES -----
    os.makedirs(plot_dir, exist_ok=True)

    # Accuracy curve
    plt.figure()
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs,   label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve (70:30)")
    plt.savefig(os.path.join(plot_dir, "accuracy_curve_70_30.png"))
    plt.close()

    # Loss curve
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve (70:30)")
    plt.savefig(os.path.join(plot_dir, "loss_curve_70_30.png"))
    plt.close()

########## evaulating seperatley ####################


def evaluate(model, dataset, name, plot_dir, device):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix - {name} (70:30)")
    plt.savefig(os.path.join(plot_dir, f"confusion_matrix_{name}_70_30.png"))
    plt.close()

    # Metrics
    acc       = np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall    = recall_score(all_labels, all_preds, average='weighted')
    f1        = f1_score(all_labels, all_preds, average='weighted')

    with open(os.path.join(plot_dir, f"metrics_{name}_70_30.txt"), "w") as f:
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
# -------------
# Main script
# -------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--czech_path', type=str, required=True)
    parser.add_argument('--bloodmnist_npz', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--plot_dir', type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, num_classes, full_czech, full_blood = build_hybrid_dataset(args.czech_path, args.bloodmnist_npz)
    train_loader = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = MedViT_tiny(num_classes=num_classes).to(device)

    train_model(model, train_loader, val_loader, device, args.save_path, args.plot_dir, args.epochs, args.lr)

    print("\n\nEvaluating separately...")
    model.load_state_dict(torch.load(args.save_path))
    evaluate(model, full_czech, "czech", args.plot_dir, device)
    evaluate(model, full_blood, "bloodmnist", args.plot_dir, device)

if __name__ == '__main__':
    main()
