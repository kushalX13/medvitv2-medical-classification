# MedViTV2: Medical Image Classification with KAN-Integrated Transformers

[![Paper](https://img.shields.io/badge/arXiv-2502.13693-red.svg)](https://arxiv.org/abs/2502.13693)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<div align="center">
  <h1>MedViTV2</h1>
  <h3>Medical Image Classification with KAN-Integrated Transformers and Dilated Neighborhood Attention</h3>
  <p><em>A hybrid deep learning architecture for robust medical image classification</em></p>
</div>

---

## Abstract

MedViTV2 presents a novel hybrid architecture that integrates Kolmogorov-Arnold Networks (KAN) with Vision Transformers for medical image classification. Our approach combines the expressive power of KANs with the global context modeling capabilities of Transformers, achieving state-of-the-art performance on multiple medical imaging benchmarks while maintaining computational efficiency.

**Key Contributions:**
- First integration of FasterKAN with Vision Transformers for medical imaging
- Novel Local Feature Perception (LFP) and Global Feature Perception (GFP) modules
- Cross-domain generalization evaluation on BloodMNIST and Czech WBC datasets
- Comprehensive ablation studies and performance analysis

## Experimental Results

### Cross-Domain Blood Cell Classification Performance

| Dataset | Domain Type | Accuracy | Precision | Recall | F1-Score |
|---------|-------------|----------|-----------|--------|----------|
| BloodMNIST | Source | **98.0%** | 97.69% | 97.68% | 97.68% |
| Czech WBC | Target | **97.0%** | 98.92% | 98.86% | 98.87% |
| Overall (Mixed) | Cross-domain | **97.5%** | - | - | - |

### Detailed Training Configurations

| Dataset | Training Strategy | Accuracy | Precision | Recall | F1-Score |
|---------|------------------|----------|-----------|--------|----------|
| BloodMNIST | Hybrid (83:17) | **97.68%** | 97.69% | 97.68% | 97.68% |
| Czech Dataset | Hybrid (83:17) | **98.86%** | 98.92% | 98.86% | 98.87% |
| BloodMNIST | Hybrid (70:30) | **97.9%** | - | - | - |
| Czech Dataset | Hybrid (70:30) | **95.8%** | - | - | - |

## Datasets and Evaluation

### Primary Evaluation Datasets

- **BloodMNIST**: 8-class blood cell microscopy dataset from MedMNIST v2
- **Czech WBC Dataset**: Real-world white-blood-cell images with visual domain differences

Both datasets were resized to 224 × 224 pixels and combined to test cross-domain generalization capabilities.

### Extended Dataset Support

**MedMNIST Collection:**
- BloodMNIST, ChestMNIST, PathMNIST, DermaMNIST, OCTMNIST
- PneumoniaMNIST, RetinaMNIST, BreastMNIST, TissueMNIST
- OrganAMNIST, OrganCMNIST, OrganSMNIST

**External Medical Datasets:**
- Iran Dataset, ISIC2018, Kvasir, CPN X-ray, Fetal Planes, PAD-UFES-20

## Model Architecture

MedViTV2 incorporates a hierarchical hybrid strategy for balanced local-global learning:

### Core Architectural Components

1. **Local Feature Perception (LFP) Blocks**: Capture localized spatial patterns through convolutional attention mechanisms
2. **Global Feature Perception (GFP) Blocks**: Extract broader contextual relationships using self-attention and KAN integration
3. **KAN Integration**: FasterKAN layers for enhanced non-linear feature learning and expressiveness
4. **Hybrid Attention Mechanisms**: 
   - Multi-Head Convolutional Attention (MHCA)
   - Efficient Multi-Head Self Attention (E_MHSA)
5. **Stem Network**: Multi-layer convolutional stem for initial feature extraction
6. **Classification Head**: Linear projection layer for final predictions

### Model Variants

| Model | Dimensions | Depths | Parameters | Use Case |
|-------|------------|--------|------------|----------|
| `MedViT_tiny` | [64, 128, 192, 384] | [2, 2, 6, 1] | ~5M | Lightweight deployment |
| `MedViT_small` | [64, 128, 256, 512] | [2, 2, 6, 2] | ~12M | Balanced performance |
| `MedViT_base` | [96, 192, 384, 768] | [2, 2, 6, 2] | ~25M | High accuracy |
| `MedViT_large` | [96, 256, 512, 1024] | [2, 2, 6, 2] | ~50M | Maximum performance |

## Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Framework | PyTorch (MPS-accelerated) | Deep learning framework |
| Optimizer | AdamW | Learning rate: 1e-4, Weight decay: 0.05 |
| Batch Size | 32 | Training batch size |
| Epochs | 20-50 | Training epochs |
| Scheduler | Cosine Annealing | Learning rate scheduling |
| Metrics | Accuracy, Loss, Confusion Matrix | Evaluation metrics |
| Visualization | Matplotlib | Training curves and plots |

## Installation and Setup

### System Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA-capable GPU (recommended) or MPS (Apple Silicon)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/MedViTV2.git
cd MedViTV2
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥1.9.0 | Deep learning framework |
| torchvision | ≥0.10.0 | Computer vision utilities |
| timm | ≥0.6.0 | Model architectures |
| medmnist | 3.0.2 | Medical imaging datasets |
| einops | latest | Tensor operations |
| scikit-learn | latest | Machine learning utilities |
| matplotlib | latest | Visualization |
| tqdm | latest | Progress bars |

## Usage Examples

### 1. Standard MedMNIST Training

Train MedViT on BloodMNIST dataset:

```bash
python main.py \
    --model_name MedViT_tiny \
    --dataset bloodmnist \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-4
```

### 2. Hybrid Training (Czech + BloodMNIST)

Train with mixed datasets for cross-domain generalization:

```bash
python train_hybrid_70_30.py \
    --czech_path /path/to/czech/dataset \
    --bloodmnist_npz /path/to/bloodmnist.npz \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-4 \
    --save_path ./checkpoints/medvit_hybrid_70_30.pth \
    --plot_dir ./results/hybrid_70_30
```

### 3. Fine-tuning with Different Ratios

**83:17 ratio:**
```bash
python train_hybrid_83_17.py \
    --czech_path /path/to/czech \
    --bloodmnist_npz /path/to/bloodmnist.npz
```

**90:10 ratio:**
```bash
python train_hybrid_90_10.py \
    --czech_path /path/to/czech \
    --bloodmnist_npz /path/to/bloodmnist.npz
```

## Project Structure

```
MedViTV2/
├── src/                           # Core implementation
│   ├── MedViT.py                  # MedViT architecture
│   ├── main.py                    # Main training script
│   ├── datasets.py                # Dataset utilities
│   ├── fasterkan.py               # FasterKAN implementation
│   ├── train_hybrid_*.py          # Hybrid training scripts
│   ├── testing_data*.py           # Evaluation scripts
│   ├── checkpoints/               # Model checkpoints
│   └── Tutorials/                 # Jupyter notebooks
├── fine_tuned_results/             # Fine-tuning results
│   ├── 20 epoch (70:30)/          # 70:30 ratio results
│   ├── 20 epoch (83:17)/          # 83:17 ratio results
│   └── 20 epoch (90:10)/          # 90:10 ratio results
├── baseline_results/               # Pre-fine-tuning results
├── data/                          # Dataset storage
│   ├── bloodmnist/               # BloodMNIST data
│   └── czech_wbc/                # Czech WBC data
├── results/                       # Training outputs
│   ├── confusion_matrices/        # Confusion matrix plots
│   └── loss_curves/              # Training/validation curves
├── requirements.txt               # Dependencies
├── LICENSE                        # MIT License
├── .gitignore                    # Git ignore rules
└── README.md                     # Documentation
```

## Key Features

### 1. Hybrid Training Strategy
- Combines multiple datasets for robust feature learning
- Configurable dataset ratios (70:30, 83:17, 90:10)
- Cross-dataset evaluation capabilities

### 2. Advanced Data Augmentation
- RandomResizedCrop with AugMix
- RandomHorizontalFlip
- Normalized preprocessing

### 3. Comprehensive Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix visualization
- ROC-AUC for multi-class problems
- Cross-dataset generalization testing

### 4. Model Optimization
- AdamW optimizer with cosine annealing
- Gradient checkpointing support
- Batch normalization merging for inference

## Evaluation and Visualization

Confusion matrices and training curves are stored under `results/` for both datasets. The training process generates comprehensive visualizations:

- Training/Validation loss curves
- Training/Validation accuracy curves  
- Confusion matrices for each dataset
- Detailed metrics files

### Model Checkpoints
Pre-trained models are available for:
- BloodMNIST classification
- Hybrid models (Czech + BloodMNIST)
- Various training configurations

## Evaluation Scripts

### External Dataset Testing
```bash
python testing_data_czech.py    # Czech dataset evaluation
python testing_data_iran.py     # Iran dataset evaluation
```

### Grad-CAM Visualization
```bash
python GradCam_MedViT_large.py  # Generate attention maps
```

## Tutorials and Documentation

- **Evaluation.ipynb**: Complete evaluation pipeline
- **Visualization.ipynb**: Grad-CAM and attention visualization

## Contributing

We welcome contributions to MedViTV2! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We acknowledge the following projects and datasets:

- [FasterKAN](https://github.com/AthanasiosDelis/faster-kan) for KAN implementation
- [NATTEN](https://github.com/SHI-Labs/NATTEN) for neighborhood attention
- [MedViTV1](https://github.com/Omid-Nejati/MedViT) for the original architecture
- MedMNIST team for comprehensive medical imaging datasets
- Czech WBC dataset contributors for domain adaptation evaluation

## Citation

If you use MedViTV2 in your research, please cite our paper:

```bibtex
@article{manzari2025medical,
  title={Medical Image Classification with KAN-Integrated Transformers and Dilated Neighborhood Attention},
  author={Manzari, Omid Nejati and Asgariandehkordi, Hojat and Koleilat, Taha and Xiao, Yiming and Rivaz, Hassan},
  journal={arXiv preprint arXiv:2502.13693},
  year={2025}
}
```

## Reference

Nejati O. M., Asgariandehkordi H., Koleilat T., Xiao Y., Rivaz H. (2025).  
**Medical Image Classification with KAN-Integrated Transformers and Dilated Neighborhood Attention.**  
arXiv:2502.13693

---

<div align="center">
  <p><strong>If you find this project useful, please consider giving it a star! ⭐</strong></p>
  <p><em>Note: This implementation includes modifications for MPS (Metal Performance Shaders) compatibility, with neighborhood attention temporarily disabled for broader device support.</em></p>
</div>
