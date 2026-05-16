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
- Compare all 4 body sites (wrist, chest, head, ankle) for the same window
- Understand what a heartbeat looks like in the signal

### 3. Frequency Domain Analysis
- Apply Welch PSD to convert signal to frequency domain
- See why the true HR frequency is often buried under motion artifact
- Understand why simple FFT peak-picking fails (MAE ~59 bpm)

### 4. Bandpass Filtering
- Apply a 0.5–4 Hz Butterworth bandpass filter
- See filter ringing on severely corrupted windows
- Understand the limits of classical signal processing

### 5. Building a 1-Channel CNN
- Understand the input shape: `(batch, 1, 200)`
- Build a 3-layer 1D CNN with `nn.Conv1d`, `MaxPool1d`, `Linear`
- Trace the shape of data through every layer:
  - Layer 1 → local pulse shapes (kernel sees 0.28 seconds)
  - Layer 2 → one full heartbeat cycle
  - Layer 3 → rhythm across multiple beats
- Train on subjects 0–12, test on subjects 13–15

### 6. Multi-Site Fusion (4-Channel CNN)
- Discover that the dataset has 4 body sites: wrist, chest, head, ankle
- Compare all 4 sites visually — chest is cleanest, ankle is noisiest
- Stack all 4 into a `(batch, 4, 200)` input
- Same CNN architecture, `in_channels=1` → `in_channels=4`
- Dramatic MAE improvement from multi-site fusion

### 7. ResNet with Residual Connections
- Understand the skip connection: `output = F(input) + input`
- Build `ResBlock` and stack into a deeper `ResNet1D`
- Learn why saving the best checkpoint matters (overfitting)
- Achieve near-paper-level performance

### 8. Per-Subject Analysis
- Evaluate each test subject independently
- Understand why some subjects are harder than others
- See how each improvement helps the hardest subjects most

---

## Final Results

### Overall MAE

| Model | MAE (bpm) | Parameters |
|---|---|---|
| FFT peak picking | ~59 | — |
| 1ch CNN (wrist only) | 10.35 | 18,209 |
| 4ch CNN (all sites) | 6.01 | 18,545 |
| ResNet + best checkpoint | **3.02** | 624,545 |
| Paper ResNet (128Hz, 3ch) | ~4–5 | — |

### Per-Subject Breakdown (Test Subjects)

| Subject | 1ch CNN | 4ch CNN | ResNet | Difficulty |
|---|---|---|---|---|
| Subject 14 | 19.98 bpm | 9.37 bpm | 4.34 bpm | Hard (active) |
| Subject 15 | 8.83 bpm  | 5.56 bpm | 2.63 bpm | Medium |
| Subject 16 | 3.97 bpm  | 3.37 bpm | 2.10 bpm | Easy (resting) |

---

## Key Lessons Learned

**Why classical methods fail:**
Motion artifacts create spectral peaks larger than the cardiac signal.
Simple FFT peak-picking gives ~59 bpm MAE — essentially random.

**Why multi-site fusion helps:**
When the wrist is corrupted by motion, the chest or head signal is often
clean. The model learns to rely on whichever site is most informative at
each moment. Biggest improvement on the hardest subjects (Subject 14:
19.98 → 9.37 bpm).

**Why residual connections help:**
Deeper networks can learn more complex patterns but are harder to train.
Skip connections let gradients flow freely through many layers, enabling
the ResNet to learn far more precise HR estimates.

**Why saving the best checkpoint matters:**
The ResNet eventually overfits — training loss keeps dropping while test
MAE gets worse. Always save the weights at the best validation performance.

**Why subject split matters:**
MAE ranges from 2.10 to 4.34 bpm across the three test subjects.
Performance depends on how similar the test subject is to the training
distribution. Easy subjects were already solved by the simple 1ch CNN.

**Why the model underestimates high HR:**
Training data is skewed toward 70–100 bpm. The model plays safe and
predicts near the mean for rare high-HR windows — regression to the mean.

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