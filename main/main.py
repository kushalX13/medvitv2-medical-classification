import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import argparse
import requests
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from tqdm import tqdm
from datasets import build_dataset
from distutils.util import strtobool
import medmnist
from medmnist import INFO, Evaluator
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize

from MedViT import MedViT_tiny, MedViT_small, MedViT_base, MedViT_large

# ─── MODEL REGISTRY ─────────────────────────────────────────────────────────────
model_classes = {
    'MedViT_tiny':  MedViT_tiny,
    'MedViT_small': MedViT_small,
    'MedViT_base':  MedViT_base,
    'MedViT_large': MedViT_large
}

model_urls = {
    "MedViT_tiny":  "https://dl.dropbox.com/scl/fi/496jbihqp360jacpji554/MedViT_tiny.pth?dl=1",
    "MedViT_small": "https://dl.dropbox.com/scl/fi/6nnec8hxcn5da6vov7h2a/MedViT_small.pth?dl=1",
    "MedViT_base":  "https://dl.dropbox.com/scl/fi/q5c0u515dd4oc8j55bhi9/MedViT_base.pth?dl=1",
    "MedViT_large": "https://dl.dropbox.com/scl/fi/owujijpsl6vwd481hiydd/MedViT_large.pth?dl=1"
}

def download_checkpoint(url, path):
    print(f"Downloading checkpoint from {url}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(path, 'wb') as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    print(f"Saved checkpoint to {path}")

# ─── METRIC HELPERS ───────────────────────────────────────────────────────────────
def specificity_per_class(cm):
    spec = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i,:].sum() + cm[:,i].sum() - cm[i,i])
        fp = cm[:,i].sum() - cm[i,i]
        spec.append(tn / (tn + fp))
    return spec

def overall_accuracy(cm):
    return cm.trace() / cm.sum()

# ─── TRAINING FOR MEDMNIST ───────────────────────────────────────────────────────
def train_mnist(epochs, net, train_loader, test_loader,
                optimizer, scheduler, loss_fn, device,
                save_path, data_flag, task):
    os.makedirs(f'losses/{data_flag}', exist_ok=True)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_acc = 0.0

    for epoch in range(1, epochs+1):
        # — train —
        net.train()
        total_loss = 0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", file=sys.stdout):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = net(imgs)
            if task.startswith('multi-label'):
                lbls = lbls.float()
                loss = loss_fn(out, lbls)
            else:
                lbls = lbls.squeeze().long()
                loss = loss_fn(out.squeeze(0), lbls)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # — validation forward & collect outputs for AUC —
        net.eval()
        all_scores = []
        with torch.no_grad():
            for imgs, _ in test_loader:
                out = net(imgs.to(device))
                scores = out.softmax(dim=1).cpu().numpy()
                all_scores.append(scores)
        all_scores = np.vstack(all_scores)

        # — compute val loss —
        total_vloss = 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = net(imgs)
                if task.startswith('multi-label'):
                    lbls = lbls.float()
                    vloss = loss_fn(out, lbls)
                else:
                    lbls = lbls.squeeze().long()
                    vloss = loss_fn(out.squeeze(0), lbls)
                total_vloss += vloss.item()
        avg_val_loss = total_vloss / len(test_loader)

        # — compute accuracies —
        correct_t, total_t = 0, 0
        with torch.no_grad():
            for imgs, lbls in train_loader:
                preds = net(imgs.to(device)).argmax(dim=1).cpu()
                correct_t += (preds == lbls).sum().item()
                total_t   += lbls.size(0)
        train_acc = correct_t / total_t

        correct_v, total_v = 0, 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                preds = net(imgs.to(device)).argmax(dim=1).cpu()
                correct_v += (preds == lbls).sum().item()
                total_v   += lbls.size(0)
        val_acc = correct_v / total_v

        # — MedMNIST evaluator (AUC + accuracy) —
        auc, macc = Evaluator(data_flag, 'test', size=224, root='./data').evaluate(all_scores)

        print(
            f"[{epoch}/{epochs}] "
            f"Train loss: {avg_train_loss:.3f}  Val loss: {avg_val_loss:.3f}  "
            f"Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}  "
            f"AUC: {auc:.4f}  MedMNIST Acc: {macc:.4f}  "
            f"LR: {scheduler.get_last_lr()[-1]:.1e}"
        )

        # save best MedMNIST accuracy
        if macc > best_acc:
            best_acc = macc
            torch.save({
                'model': net.state_dict(),
                'opt': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'epoch': epoch,
                'acc': best_acc
            }, save_path)
            print("→ checkpoint saved")

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    # ─── PLOTS ───────────────────────────────────────────────────────────────────

    # Loss curve
    plt.figure()
    plt.plot(train_losses, label='Train Loss',  linewidth=2)
    plt.plot(val_losses,   label='Val Loss',    linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve — {data_flag}')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'losses/{data_flag}/loss_curve.png')
    plt.show()

    # Accuracy curve (percent)
    train_pct = [a * 100 for a in train_accs]
    val_pct   = [a * 100 for a in val_accs]
    plt.figure()
    plt.plot(train_pct, label='Train Acc (%)', linewidth=2, marker='o')
    plt.plot(val_pct,   label='Val   Acc (%)', linewidth=2, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.title(f'Accuracy Curve — {data_flag}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'losses/{data_flag}/accuracy_curve_pct.png')
    plt.show()

    # Confusion matrix
    all_preds, all_lbls = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            preds = net(imgs.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_lbls.extend(lbls.numpy())
    cm = confusion_matrix(all_lbls, all_preds)
    disp = ConfusionMatrixDisplay(cm)
    os.makedirs(f'confusion_matrices/{data_flag}', exist_ok=True)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'CM — {data_flag}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrices/{data_flag}/confusion_matrix.png')
    plt.show()

# ─── TRAINING FOR OTHER DATASETS ────────────────────────────────────────────────
def train_other(epochs, net, train_loader, test_loader,
                optimizer, scheduler, loss_fn, device, save_path):
    best_acc = 0.0
    for epoch in range(1, epochs+1):
        net.train()
        total_loss = 0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", file=sys.stdout):
            optimizer.zero_grad()
            out = net(imgs.to(device))
            loss = loss_fn(out, lbls.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # validation
        net.eval()
        all_preds, all_lbls, all_probs = [], [], []
        correct = 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                out = net(imgs.to(device))
                probs = out.softmax(dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_lbls.extend(lbls.numpy())
                correct += (preds == lbls.numpy()).sum()
        val_acc = correct / len(test_loader.dataset)

        # metrics
        prec   = precision_score(all_lbls, all_preds, average='weighted')
        rec    = recall_score(all_lbls, all_preds, average='weighted')
        f1     = f1_score(all_lbls, all_preds, average='weighted')
        cm     = confusion_matrix(all_lbls, all_preds)
        spec   = sum(specificity_per_class(cm)) / cm.shape[0]
        overall= overall_accuracy(cm)
        # multi‑class AUC
        ncls    = cm.shape[0]
        onehot  = label_binarize(all_lbls, classes=list(range(ncls)))
        try:
            auc = roc_auc_score(onehot, all_probs, multi_class='ovr')
        except:
            auc = float('nan')

        print(
            f"[{epoch}/{epochs}] Loss: {avg_loss:.3f}  Val Acc: {val_acc:.4f}  "
            f"P: {prec:.4f}  R: {rec:.4f}  F1: {f1:.4f}  Spec: {spec:.4f}  "
            f"AUC: {auc:.4f}  Overall Acc: {overall:.4f}  "
            f"LR: {scheduler.get_last_lr()[-1]:.1e}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model': net.state_dict(),
                'opt': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'epoch': epoch,
                'acc': best_acc
            }, save_path)
            print("→ checkpoint saved")

# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',     type=str,   default='MedViT_tiny')
    parser.add_argument('--dataset',        type=str,   default='bloodmnist')
    parser.add_argument('--batch_size',     type=int,   default=32)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--epochs',         type=int,   default=20)
    parser.add_argument('--pretrained',     type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--checkpoint_path',type=str,   default='./checkpoint/MedViT_tiny.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # pick loss
    if args.dataset.endswith('mnist'):
        task = INFO[args.dataset]['task']
        loss_fn = nn.BCEWithLogitsLoss() if task.startswith('multi-label') else nn.CrossEntropyLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    # build data
    train_ds, test_ds, ncls = build_dataset(args=args)
    train_loader = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = data.DataLoader(test_ds,  batch_size=args.batch_size*2, shuffle=False)

    # model
    if args.model_name in model_classes:
        net = model_classes[args.model_name](num_classes=ncls).to(device)
        if args.pretrained:
            ckpt = args.checkpoint_path
            if not os.path.exists(ckpt):
                download_checkpoint(model_urls[args.model_name], ckpt)
            st = torch.load(ckpt, map_location=device)
            # drop incompatible keys
            for k in ['proj_head.0.weight','proj_head.0.bias']:
                if k in st and st[k].shape != net.state_dict()[k].shape:
                    del st[k]
            net.load_state_dict(st, strict=False)
    else:
        net = timm.create_model(args.model_name, pretrained=args.pretrained, num_classes=ncls).to(device)

    # optimizer + scheduler
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
    total_steps = args.epochs * len(train_ds) // args.batch_size
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=5e-6)

    os.makedirs('./my_checkpoints', exist_ok=True)
    save_path = f'./my_checkpoints/{args.model_name}_{args.dataset}.pth'

    if args.dataset.endswith('mnist'):
        train_mnist(args.epochs, net, train_loader, test_loader,
                    optimizer, scheduler, loss_fn, device,
                    save_path, args.dataset, task)
    else:
        train_other(args.epochs, net, train_loader, test_loader,
                    optimizer, scheduler, loss_fn, device, save_path)

if __name__ == '__main__':
    main()
