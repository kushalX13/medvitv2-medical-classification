import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import argparse
from MedViT import MedViT_tiny
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def build_czech_dataset(data_path, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    full_dataset = ImageFolder(root=data_path, transform=transform)
    class_names = full_dataset.classes

    train_idx, val_idx = train_test_split(list(range(len(full_dataset))), test_size=0.2, stratify=full_dataset.targets)
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

    return train_dataset, val_dataset, len(class_names)

def train_model(model, train_loader, val_loader, device, save_path, plot_dir, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=5e-6)

    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_train_loss = running_loss / len(train_loader)
        train_accuracies.append(train_acc)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        val_loss_total = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss_total += loss.item()

                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        avg_val_loss = val_loss_total / len(val_loader)
        val_acc = val_correct / val_total
        val_accuracies.append(val_acc)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("\n✅ Saved best model so far!")

    print("\nTraining complete.")

    # Create plot directory
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss - Czech Dataset')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'czech_loss_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy - Czech Dataset')
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig(os.path.join(plot_dir, 'czech_accuracy_curve.png'))
    plt.close()

    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Czech Dataset')
    plt.savefig(os.path.join(plot_dir, 'czech_confusion_matrix.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to Czech dataset (root folder)')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, default='./medvit_czech_scratch.pth')
    parser.add_argument('--plot_dir', type=str, default='/scratch/kgangara/dl_final_project/results_czech')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"\n🚀 Using device: {device}\n")

    train_dataset, val_dataset, num_classes = build_czech_dataset(args.data_path)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = MedViT_tiny(num_classes=num_classes).to(device)

    train_model(model, train_loader, val_loader, device, args.save_path, args.plot_dir, args.epochs, args.lr)

if __name__ == '__main__':
    main()
