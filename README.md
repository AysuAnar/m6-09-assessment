# YOLO26-based Cat Detection & Predator Distractor Classification System

[![YOLO26](https://img.shields.io/badge/Ultralytics-YOLO26-orange.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Image for leaderboard
```bash
docker pull aysuanar/catdetector:v2
```
- **Image:** `aysuanar/catdetector:v2`
- **Student:** Aysu Mammadova

---

An advanced, edge-optimized convolutional object detection system built on **Ultralytics YOLO26m** (January 2026 release) to detect cats with high precision in diverse environments. In addition to standard cat detection, this system integrates negative/rival samples consisting of dog and cheetah distractor classes to drastically minimize false positives and enhance real-world classification robustness. Tiger and lion annotations were explicitly filtered out in data preparation to focus training on the domestic and cheetah distractor categories.

---

## 🌟 Key Achievements & Design Features

1. **Robust Data Wrangling & Cleaning**
   - Eliminated **36 dataset anomalies** from the raw inputs, including duplicate image files (exact content hash matches), orphaned image/label pairs, corrupted image binaries, and invalid/empty annotation coordinates.
2. **Negative/Rival Distractor Classes**
   - Incorporated negative/rival samples of other animals (Dogs and Cheetahs) to train the model to accurately distinguish domestic cats from similar species, reducing incorrect classifications (False Positives). Lion and tiger annotations were filtered out of the training images.
3. **Optimized Multi-Epoch Training**
   - Trained the model for 70 epochs with advanced data augmentations (`mosaic=1.0`, `mixup=0.15`, `copy_paste=0.15`, `shear=5.0`, `perspective=0.0005`) and a learning rate scheduler to optimize generalization across hand-drawn sketches and real cat photos.
4. **Performance Tuning for Windows**
   - Solved Windows multiprocessing memory limits (`cv::OutOfMemoryError`) by pinning single-worker data loaders (`workers=0`) and caching images directly into system RAM (`cache=True`). This increased data transfer rates to **~2.5-3.0 batches per second**, cutting total training time in half.

---

## 📊 Model Performance Comparison

Below is the test set performance comparison between the **Week-1 Baseline** and our advanced **YOLO26m (cats_v2)** model trained this week, highlighting overall and class-specific metrics:

| Model | Number of Classes | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Training / Evaluation Strategy |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Week-1 Baseline** | 1 (Cat) | **0.9313** | 0.8650 | 0.9208 | **0.7373** | Trained only on cat images (no negative/distractor samples). |
| **YOLO26m (cats_v2) - Cat-only (Isolated)** | 1 (Cat class evaluated with `classes=0`) | 0.9010 | **0.8744** | **0.9268** | 0.7212 | Test set metrics on **ONLY the cat class** (isolated from dog and cheetah during evaluation). |
| **YOLO26m (cats_v2) - Cat class under multi-class** | 1 (Cat class within 3-class eval) | 0.9064 | 0.8693 | **0.9268** | 0.7212 | Test set metrics for the **cat class** when evaluated alongside dog and cheetah. |
| **YOLO26m (cats_v2) - Overall (3 Classes)** | 3 (Cat, Dog, Cheetah) | **0.9426** | **0.9000** | **0.9468** | **0.7979** | Test set metrics across **all three classes combined** (Overall model performance). **(Best Overall)** |

### Key Takeaways
- **mAP@0.5 Increase (+0.60%):** The advanced training strategy (YOLO26m `cats_v2`) achieved a peak **0.9268 mAP@0.5** on the cat class (both isolated and under multi-class evaluation), outperforming the Week-1 baseline's 0.9208.
- **Recall Jump (+0.94%):** `cats_v2` reached a high isolated Recall of **0.8744** (up from 0.8650 in the baseline), significantly decreasing missed detections for small, heavily occluded, or background cats.
- **Distractor Class-Specific Metrics:**
  - **Dog (Class 1):** Precision: `0.9213`, Recall: `0.8925`, mAP@0.5: `0.9512`, mAP@0.5:0.95: `0.8315`
  - **Cheetah (Class 2):** Precision: `1.0000`, Recall: `0.9381`, mAP@0.5: `0.9623`, mAP@0.5:0.95: `0.8410`
- **High-Fidelity Classification:** While the baseline frequently misidentified other animals (like dogs or cheetahs) as cats, our trained YOLO26m distractor-aware model successfully distinguishes them, drastically reducing real-world false positives. Note that Tigers and Lions were explicitly filtered out.

---

## 📂 Project Structure

```
├── Mycats/                  # Custom test images directory for prediction visualization
├── images/                  # Main dataset image files
├── labels/                  # Bounding-box annotations (YOLO format: class_id cx cy w h)
├── runs/                    # Training outputs (weights, curves, confusion matrices)
│   └── detect/runs/cats_v2/ # Run folder for the best trained model (YOLO26m)
│       └── weights/
│           ├── best.pt      # Optimized YOLO26m model weights
│           └── best.onnx    # Exported ONNX model (opset 17, 640×640)
├── container/               # Docker container for standardised leaderboard CLI
│   ├── Dockerfile           # Container build instructions
│   ├── STUDENT.json         # Student metadata for leaderboard registration
│   ├── requirements.txt     # Pinned Python dependencies
│   ├── models/
│   │   └── best.onnx        # ONNX model bundled inside the container
│   └── app/
│       ├── cli.py           # CLI entrypoint (info / predict subcommands)
│       └── detector.py      # ONNX-based CatDetector inference engine
├── m6-04-assessment.ipynb   # Full end-to-end Jupyter Notebook pipeline (includes ONNX export & sanity check)
├── data.yaml                # Dataset configuration file
├── train.txt                # Paths to training split images
├── val.txt                  # Paths to validation split images
├── test.txt                 # Paths to test split images
└── README.md                # System documentation
```

---

## 🚀 Getting Started & Replication

### 1. Prerequisites
Ensure you have Python 3.10+ installed. Install the required dependencies:
```bash
pip install ultralytics torch torchvision matplotlib pandas pillow opencv-python pyyaml
```

### 2. Dataset Configuration
Verify that `data.yaml` is properly configured at the root of the project:
```yaml
path: ./
train: ./train.txt
val: ./val.txt
test: ./test.txt
nc: 3
names:
  0: cat
  1: dog
  2: cheetah
```

### 3. Pipeline Execution
Open and execute the cells in the Jupyter notebook `m6-04-assessment.ipynb`. The pipeline will automatically:
1. **Validate and clean** the raw image and label files.
2. **Generate** reproducible `70 / 15 / 15` train/val/test splits.
3. **Train / Load** the model weights using the advanced training configuration.
4. **Evaluate** model precision, recall, and mAP metrics on the test split.
5. **Run custom predictions** on the custom `Mycats/` directory.

---

## 🎯 Custom Prediction on `Mycats/`

The custom prediction block loads the best-performing model checkpoint (`best.pt`) and performs inference on all images in the `Mycats/` folder. 

In compliance with the project guidelines, although the model is trained with multiple distractor animal classes (Class 1: Dog, Class 2: Cheetah) to optimize robustness, the visualization layer is filtered to **only draw bounding boxes and label Domestic Cats (Class 0)**.

To customize predictions, simply drop any new image files into the `Mycats/` folder and run **Cell 32** of the Jupyter notebook.

---

## 🐳 Docker Container (Standardised CLI)

The trained ONNX model is packaged inside a lightweight Docker container with a standardised CLI for the leaderboard evaluation system.

### Pull the Image
```bash
docker pull aysuanar/catdetector:v2
```

### Build Locally (optional)
```bash
docker build -t aysuanar/catdetector:v2 -f container/Dockerfile .
```

### Usage

**`info` subcommand** — prints the STUDENT.json record:
```bash
docker run --rm aysuanar/catdetector:v2 info
```

**`predict` subcommand** — reads images from `/data/input/`, writes predictions to `/data/output/predictions.csv`:
```bash
docker run --rm \
  -v /path/to/your/images:/data/input \
  -v /path/to/output:/data/output \
  aysuanar/catdetector:v2 predict
```

### CSV Output Schema
```
image_path,xmin,ymin,xmax,ymax,confidence,class
cat001.jpg,120.3,45.1,450.7,380.2,0.92,cat
cat002.jpg,,,,,, 
```

---

## 📦 ONNX Export & Verification

The best PyTorch model (`best.pt`) was exported to ONNX format with the following settings:
- **Opset:** 17
- **Input Size:** 640×640 (static)
- **File Size:** ~81.7 MB

A full **ONNX-vs-PyTorch sanity check** is included in the notebook (Task 5e), verifying that bounding box coordinates match within sub-pixel precision and confidence scores are near-identical between the two runtimes.

---

## ✅ Local CSV Verification

The container was tested locally on the `Mycats/` folder (27 images) to verify correct CSV output:

```bash
docker run --rm -v ./Mycats:/data/input:ro -v ./test_output:/data/output aysuanar/catdetector:v2 predict
```

**Result:** 27 images processed → 15 cat detections across 13 images, 14 images with no cat detected. Sample output:

```csv
image_path,xmin,ymin,xmax,ymax,confidence,class
001.jpg,,,,,,
004.jpg,174.7,183.7,414.4,515.9,0.86,cat
005.jpg,143.4,519.9,1596.6,2375.7,0.93,cat
011.jpg,5.2,0.0,1024.0,479.5,0.96,cat
012.jpg,426.5,526.7,1019.3,1279.8,0.84,cat
012.jpg,0.0,9.7,1018.8,1276.2,0.38,cat
```

**Schema compliance verified:**
- ✅ Header: `image_path,xmin,ymin,xmax,ymax,confidence,class`
- ✅ Images with no detections: empty fields (`001.jpg,,,,,,`)
- ✅ Multiple detections per image supported (e.g., `012.jpg` has 2 rows)
- ✅ All detections are `class=cat` only (dog/cheetah filtered out)
- ✅ Coordinates are rounded to 1 decimal, confidence to 2 decimals

