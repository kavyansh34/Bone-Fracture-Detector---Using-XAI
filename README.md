# 🦴 Bone Fracture Detector — XAI Dashboard

An AI-powered X-ray fracture detection system built on a fine-tuned **ViT-B/16 Vision Transformer**, with **Explainable AI (XAI)** via Attention Rollout to visualize model focus. The model is trained on the **FracAtlas** dataset.

> ⚠️ **Disclaimer:** This tool is for research and educational purposes only. It is **not a substitute** for professional medical diagnosis. Always consult a qualified radiologist or physician for clinical decisions.

---

## 📸 Demo

| Upload X-ray | Prediction | Attention Heatmap |
|:---:|:---:|:---:|
| Upload any X-ray image | Fracture / Normal + confidence | Warm regions = model focus |

---

## 📁 Project Structure

```
bone-fracture-detector-xai/
├── app.py                          ← Streamlit dashboard (main app)
├── main.py                         ← Training script (DenseNet-121 + GradCAM)
├── detector.ipynb                  ← Jupyter notebook for experimentation
├── best_fracture_model.pth         ← Trained ViT-B/16 model weights (not in repo — see below)
├── confusion_matrix_Final_Test_Set.png
├── requirements.txt
└── README.md
```

> ⚠️ The model weights file `best_fracture_model.pth` (~327 MB) is **not included** in this repository due to size constraints. See [Model Weights](#-model-weights) below for download instructions.

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Architecture | Vision Transformer (ViT-B/16) |
| Input size | 224 × 224 px |
| Preprocessing | CLAHE → RGB → Normalize (ImageNet stats) |
| Output classes | Normal (0), Fracture (1) |
| Training script model | DenseNet-121 (with GradCAM via Captum) |
| Trained on | [FracAtlas](https://figshare.com/articles/dataset/The_dataset/22363012) dataset |

### Dataset — FracAtlas

FracAtlas is a publicly available musculoskeletal X-ray dataset introduced in the paper:

> *"FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal X-ray Images"* — Abedeen et al.

The dataset used in this project contains:

| Class | Count |
|---|---|
| Fracture | 2,000 |
| Normal | 127 |

The severe class imbalance is addressed during training via **weighted cross-entropy loss** (`weights = [0.60, 2.84]` for Normal and Fracture respectively).

**Data split (stratified):**
- 80% Training
- 10% Validation
- 10% Test

---

## 🩻 How It Works

### Inference Pipeline

1. Upload any X-ray image (PNG, JPG, JPEG, BMP, TIFF)
2. Image is preprocessed with **CLAHE** (Contrast Limited Adaptive Histogram Equalization) — the same preprocessing used during training
3. The ViT-B/16 model outputs a **Fracture / Normal** prediction with confidence scores
4. **Attention Rollout XAI** generates a heatmap showing which regions the model focused on

### XAI — Attention Rollout

Standard GradCAM does not work well on Vision Transformers because ViT operates on token sequences rather than spatial CNN feature maps. This project instead implements **Attention Rollout**, which:

- Monkey-patches each of the 12 ViT encoder blocks to capture attention weights during the forward pass
- Rolls out attention matrices across all 12 layers using matrix multiplication
- Extracts the `[CLS]` token's attention over patch tokens and reshapes it into a 14×14 spatial grid
- Upsamples to 224×224 and blends as a heatmap onto the original image

**🔴 Warm regions** = high model attention · **🔵 Cool regions** = low attention

### Training Pipeline (`main.py`)

- Model: **DenseNet-121** (pretrained on ImageNet, classifier head replaced)
- XAI during training: **GradCAM** via [Captum](https://captum.ai/) on `model.features.norm5`
- Optimizer: Adam, lr = 0.0001
- Loss: Weighted CrossEntropyLoss
- Best model checkpointed by **F1 score** on validation set

---

## 🚀 Setup & Run

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU recommended (CPU also works)

### 1. Clone the repository

```bash
git clone https://github.com/kavyansh34/bone-fracture-detector-xai.git
cd bone-fracture-detector-xai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download model weights

Download `best_fracture_model.pth` and place it in the root directory alongside `app.py`.

> *(Add your Google Drive / Hugging Face / release link here)*

### 4. Run the dashboard

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📦 Requirements

```
streamlit>=1.32.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python-headless>=4.7.0
numpy>=1.24.0
Pillow>=9.5.0
```

For training (`main.py`) you will also need:

```
pandas
scikit-learn
captum
matplotlib
seaborn
```

---

## 🔬 Citation

If you use the FracAtlas dataset, please cite the original paper:

```bibtex
@article{fracatlas2023,
  title   = {FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal X-ray Images},
  author  = {Abedeen, Iftekharul and others},
  journal = {Scientific Data},
  year    = {2023}
}
```

---

## 📄 License

This project is released for **research and educational use only**.

---

## 🙏 Acknowledgements

- [FracAtlas](https://figshare.com/articles/dataset/The_dataset/22363012) — dataset
- [PyTorch](https://pytorch.org/) & [torchvision](https://pytorch.org/vision/) — model backbone
- [Captum](https://captum.ai/) — GradCAM XAI (training script)
- [Streamlit](https://streamlit.io/) — dashboard framework
- 
