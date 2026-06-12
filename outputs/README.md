# Outputs

This directory is reserved for generated experiment artifacts.

Recommended layout:

```text
outputs/
├── checkpoints/  # Model weights and checkpoint files
├── logs/         # Training, preprocessing, and evaluation logs
├── figures/      # Generated plots and confusion matrix images
└── metrics/      # Saved metrics, confusion matrices, and reports
```

Files generated under `outputs/` are ignored by git. Keep only this README tracked.
