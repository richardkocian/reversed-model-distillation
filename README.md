# Reversed Model Distillation

This project investigates **reversed model distillation**, where a larger student model learns from a smaller teacher model. The codebase includes training and evaluation scripts, support for multiple seeds, and robustness evaluation using FGSM adversarial attacks.

The experiments demonstrated improved accuracy and adversarial robustness on the CIFAR-10, Fashion-MNIST, and California Housing datasets.  
*See the full details in the [thesis document](Richard_Kocian_Bachelor_Thesis.pdf).*

---

## 📦 Installation

#### Recommended environment:
- Python version: 3.12.3 
- Operating system: Linux (tested on Ubuntu 22.04)

First, create a virtual environment and install the required dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Running the Scripts

Before running any script, navigate to the `scripts/` directory and set the `PYTHONPATH`:

```bash
cd scripts
export PYTHONPATH=.
```
Another important step is to set up script parameters inside `scripts/configs/config.py`.
## 🧠 Training a Teacher Model
```bash
python3 train_model.py -h
```

### 💡 Example usage:
```bash
python3 train_model.py \
  --output ../outputs/medium_teacher_cifar \
  --batch-size 64 \
  --num-workers 6 \
  --seeds-file configs/seeds.txt \
  --model TeacherModelMedium \
  --dataset cifar10 \
  --datasets-path ../datasets
```
- This command trains medium-sized teacher models on the CIFAR-10 dataset using all seeds defined in configs/seeds.txt.

 - For each seed, the following files will be saved:
   - `training_loss.txt` — training loss over epochs 
   - `accuracy.txt` — final test accuracy 
   - `teacher_model.pth` — saved model weights

## 🎓 Training a Student Model with Distillation
```bash
python3 train_student_distil.py -h
```

It is necessary to run this script on the same device (CPU/GPU) as the teacher model, which is specified via `--teacher-path`, was trained on. The device can be selected in `scripts/configs/config.py`.

### 💡 Example usage:
```bash
python3 train_student_distil.py \
  --output ../outputs/experiment1 \
  --batch-size 64 \
  --num-workers 6 \
  --seeds-file configs/seeds.txt \
  --teacher-path ../teacher_models/teacher_model_small_best_cifar_10106.pth \
  --dataset cifar10 \
  --datasets-path ../datasets \
  --alpha 0.6
```

- This command trains student models on CIFAR-10 using reversed model distillation from the specified teacher model with alpha parameter set to 0.6 (60\% hard loss, 40\% soft loss).

- For each seed in configs/seeds.txt and for each switch epoch from 1 to epochs + 1 (as defined in configs/config.py), one student model is trained. This simulates distillation from the teacher for varying durations — from only the first epoch to the entire training. 
- For each configuration, the following will be saved to the output directory:
  - training_loss.txt 
  - accuracy.txt 
  - fgsm_results.csv — accuracy under FGSM adversarial attacks with different epsilon values

## 📁 Directory Structure
```bash
xkocia19_bachelor_thesis/
├── graphs/       # Plots and graphs generated from experiment outputs
├── outputs      # Output files and logs from experiments runs
├── scripts/      # Python scripts related to training and evaluation of reversed model distillation
│   ├── configs/  # Configuration files for experiments
│   ├── data/     # Dataset definitions and data loading utilities
│   ├── models/   # PyTorch model definitions
│   ├── utils/    # Utility functions used across scripts
│   ├── fgsm_attack.py  # Script for performing FGSM adversarial attacks
│   ├── train_model.py  # Script for standard model training (without distillation)
│   └── train_student_distil.py # Script for training models using reversed model distillation
├── teacher_models/ # Best-performing teacher models (based on accuracy, trained on GPU with CUDA 12.8)
├── visualize_outputs/   # Jupyter notebooks for visualizing experiment results
├── README.md     # Project overview and setup instructions
├── Richard_Kocian_Bachelor_Thesis.pdf  # Final thesis document
└── requirements.txt  # Python dependencies for running the project
```
