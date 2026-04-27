# results/

This directory stores all generated figures and tables for the final report.
Files here are produced by `eval_pipeline.py` and `train.py` — do not edit manually.

## Structure

```
results/
├── figures/
│   ├── qualitative/     ← per-image XAI comparison panels (Normal, Benign, Malignant examples)
│   ├── quantitative/    ← metric bar charts, heatmap comparisons, radar plots
│   └── training/        ← training/validation loss and accuracy curves from Phase 0
└── tables/              ← CSV exports of all metric tables (ready for LaTeX/Word import)
```

## Naming conventions

Qualitative figures:   `<class>_<image_stem>_xai_panel.png`
Quantitative figures:  `metric_comparison_<metric_name>.png`
Training figures:      `training_curves_resnet50.png`
Tables:                `metrics_table_<split>.csv`, `classification_report.csv`

## Do not commit large figure sets to git.
Add `results/figures/qualitative/*.png` to .gitignore if the folder grows large.
Commit only the final curated figures used in the report.
