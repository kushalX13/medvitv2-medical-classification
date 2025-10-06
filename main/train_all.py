import os

datasets = [
    "PAD",
    "ISIC2018",
    "Fetal",
    "CPN",
    "Kvasir",
    "chestmnist",
    "pathmnist",
    "dermamnist",
    "octmnist",
    "pneumoniamnist",
    "retinamnist",
    "breastmnist",
    "bloodmnist",
    "tissuemnist",
    "organamnist",
    "organcmnist",
    "organsmnist"
]

for dataset in datasets:
    print(f"\n\n========================================================================\n")
    print(f"\n\n========================================================================\n")
    print(f"\n\n========================================================================\n")
    print(f"\n\n================== TRAINING ON {dataset.upper()} ==================\n")
    print(f"\n\n========================================================================\n")
    print(f"\n\n========================================================================\n")
    print(f"\n\n========================================================================\n")
    cmd = (
        f"python MedViTV2/main.py "
        f"--model_name MedViT_tiny "
        f"--dataset {dataset} "
        f"--batch_size 64 "
        f"--lr 0.0001 "
        f"--epochs 20 "
        f"--pretrained False"
    )
    os.system(cmd)
