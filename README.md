# WildPPG Signal Analysis — Learning Notebook

A step-by-step learning journey through PPG signal analysis using the
[WildPPG dataset](https://siplab.org/projects/WildPPG) (NeurIPS 2024, ETH Zürich).

---

## What is this?

This notebook walks through the full pipeline of analysing real-world PPG
(photoplethysmography) signals — from loading raw data to training a neural
network that estimates heart rate.

Everything is done in a single Jupyter notebook (`wildppg_learning.ipynb`),
one cell at a time, with explanations at each step.

---

## What is PPG?

PPG is how smartwatches measure heart rate. A green LED shines into your
skin and measures how much light bounces back. When your heart beats, blood
rushes into your wrist and absorbs more light. The result is a wave that
goes up and down with each heartbeat.

The problem: this works great when you are sitting still. The moment you
walk or hike, wrist movement corrupts the signal. WildPPG is a dataset
specifically built around this challenge.

---

## The Dataset

| Property | Value |
|---|---|
| Source | ETH Zürich (NeurIPS 2024) |
| Participants | 16 healthy adults |
| Activity | Full day hike — Zürich to Jungfraujoch (3,571m) |
| Total windows | ~94,800 (8 seconds each) |
| Sampling rate | 25 Hz (preprocessed) |
| Window size | 200 samples = 8 seconds |
| Ground truth | ECG-derived HR (Pan-Tompkins) |
| Body sites | Wrist, Ankle, Chest, Head |
| Sensors | PPG (Green), IMU, Temperature, Altitude |

---

## What the Notebook Covers

### 1. Loading and Understanding the Data
- Load `WildPPG.mat` with scipy
- Understand the structure: 16 subjects × ~6000 windows × 200 samples
- Clean NaN and Inf values
- Explore HR distribution across subjects

### 2. Visualising the PPG Signal
- Plot raw 8-second PPG windows
- Compare a clean window vs a motion-corrupted window
- Understand what a heartbeat looks like in the signal

### 3. Frequency Domain Analysis
- Apply Welch PSD to convert signal to frequency domain
- See why the true HR frequency is often buried under motion artifact
- Understand why simple FFT peak-picking fails (MAE ~59 bpm)

### 4. Bandpass Filtering
- Apply a 0.5–4 Hz Butterworth bandpass filter
- See filter ringing on severely corrupted windows
- Understand the limits of classical signal processing

### 5. Building a CNN from Scratch
- Understand the input shape: `(batch, 1, 200)`
- Build a 3-layer 1D CNN with `nn.Conv1d`, `MaxPool1d`, `Linear`
- Trace the shape of data through every layer
- Understand what each layer is detecting:
  - Layer 1 → local pulse shapes
  - Layer 2 → one full heartbeat cycle
  - Layer 3 → rhythm across multiple beats

### 6. Training the Model
- Split data by subject (never mix subjects between train and test)
- Train subjects 0–12, test on subjects 13–15
- Train with Huber loss (robust to artifact outliers)
- Track MAE across 130 epochs

### 7. Evaluating Results
- Predicted vs True HR scatter plot
- Per-subject breakdown revealing why some subjects are harder
- Understand regression-to-the-mean failure mode

---

## Results Summary

| Method | MAE (bpm) | Notes |
|---|---|---|
| FFT peak picking | ~59 | Completely fooled by motion artifacts |
| Simple CNN (ours) | ~10 | 18K parameters, 130 epochs |
| Best subject (16) | 3.71 | Resting-dominant, similar to training data |
| Worst subject (14) | 18.62 | High activity, out-of-distribution |
| Paper ResNet | ~4–5 | Full 3-channel 128Hz data |

---

## Key Lessons Learned

**Why classical methods fail:**
42.8% of windows are too corrupted for peak detection or FFT to work.
Motion artifacts create spectral peaks larger than the cardiac signal.

**Why subject split matters:**
A model trained on 13 subjects and tested on 3 unseen subjects shows
MAE ranging from 3.71 to 18.62 bpm. Performance depends heavily on
how similar the test subject is to the training distribution.

**Why the model underestimates high HR:**
The training data is skewed toward 70–100 bpm (resting/walking).
The model rarely sees 150+ bpm windows so it plays safe and predicts
near the mean — called regression to the mean.

**How to improve:**
- Use all 3 PPG channels (Red, Green, IR) at full 128 Hz
- Use a deeper ResNet instead of a simple CNN
- Use leave-one-subject-out cross-validation for fair evaluation
- Apply data augmentation to balance the HR distribution

---

## Setup

### Requirements
```
numpy
scipy
matplotlib
torch
jupyter
huggingface_hub
```

### Download the dataset
```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="eth-siplab/WildPPG",
    filename="WildPPG.mat",
    repo_type="dataset",
    local_dir="data/processed"
)
```

### Run
```bash
jupyter notebook wildppg_learning.ipynb
```

---

## File Structure

```
Wildppg-Analysis/
├── wildppg_learning.ipynb   ← the entire learning journey
├── data/
│   └── processed/
│       └── WildPPG.mat      ← downloaded dataset (632 MB)
└── README.md                ← this file
```

---

## Reference

```bibtex
@inproceedings{meier2024wildppg,
  title     = {WildPPG: A Real-World PPG Dataset of Long Continuous Recordings},
  author    = {Meier, Manuel and Demirel, Berken Utku and Holz, Christian},
  booktitle = {NeurIPS Datasets and Benchmarks Track},
  year      = {2024},
}
```
