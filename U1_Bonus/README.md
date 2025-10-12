# Binary Classification Noise Study

## Overview

This project investigates how noise levels, loss functions, and evaluation metrics affect a PyTorch logistic regression model trained on synthetic datasets generated with scikit-learn. It was developed as a bonus challenge for MLOps Unit 1.

## Repository Structure

```
your_project_name/
├── data/               # Optional: serialized datasets (not tracked by default)
├── notebooks/          # Jupyter notebooks with experiments
├── src/                # Source modules (Architecture class and helpers)
├── results/            # Exported figures and metric tables
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

## Quick Start

1. Create a fresh virtual environment (Python 3.10+ recommended).
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
3. Launch Jupyter Lab/Notebook from the project root and open `notebooks/noise_experiments.ipynb`.
4. Run all cells. Figures and the consolidated metrics CSV will be saved automatically under `results/`.

### Reproducing Metrics from the Command Line

If you prefer to execute the notebook non-interactively, use `jupyter nbconvert`:

```bash
jupyter nbconvert --execute --to notebook --inplace notebooks/noise_experiments.ipynb
```

## Results Summary

Once the notebook has been executed, the following artifacts will be available:

- `results/metrics_summary.csv`: aggregated accuracy, precision, recall, F1, and final losses for every noise/loss configuration.
- `results/decision_boundary_noise_*.png`: decision boundary heatmaps that visualize how separability degrades with noise.
- `results/confusion_noise_*.png`: confusion matrices highlighting the evolution of false positives/negatives.
- `results/loss_curve_noise_*.png`: training vs. validation loss trajectories for stability analysis.

### Observations to Capture

- **Noise vs. separability:** Expect accuracy and recall to decay as flip probability increases; document how boundary lines blur and overlap.
- **BCE vs. BCEWithLogits:** `BCEWithLogitsLoss` keeps logits in their native space, preventing saturation and yielding smoother loss curves, especially for near-separable (low-noise) scenarios.
- **Metric trade-offs:** Track whether precision or recall deteriorates faster—this often reveals if mislabeled samples skew toward one class.

> Update this section with concrete numbers, screenshots, and insights after running the experiments.

## Summary Checklist

| Item | Description | Status |
| --- | --- | --- |
| Dataset | Generated with scikit-learn (`make_classification`) under multiple noise levels | ☐ |
| Model | Logistic Regression implemented in PyTorch | ☐ |
| Architecture | Training handled by provided `Architecture` class | ☐ |
| Loss Functions | Comparison between `BCELoss` and `BCEWithLogitsLoss` | ☐ |
| Metrics | Accuracy, Precision, Recall, F1, Confusion Matrix logged | ☐ |
| Plots | Decision boundary and confusion matrix visuals exported | ☐ |
| Repository | Clean structure with notebook + results | ☐ |
| Video | ≤10 min overview hosted on Loom/YouTube | ☐ |

## Presentation Video

- **Link:** _Coming soon_ (record a ≤10 min video explaining setup, experiments, plots, and conclusions.)

## Interpretation Tips

- Comment on how weight magnitudes react to noise—large oscillations can indicate instability with `BCELoss`.
- Use confusion matrices to explain whether errors skew to false positives or negatives as noise increases.
- Highlight any notable divergence between training and validation losses to justify the chosen number of epochs or learning rate.

## Troubleshooting

- If CUDA is unavailable, the `Architecture` class will automatically fall back to CPU training.
- Re-run the notebook to regenerate figures after changing hyperparameters or introducing new datasets (e.g., `make_circles`, `make_moons`).
