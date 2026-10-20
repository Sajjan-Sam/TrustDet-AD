# Reproducing the pipeline

Run all commands from the repository root after installing dependencies and preparing
both datasets as described in [DATASETS.md](DATASETS.md). A CUDA-capable device is required
by the current inference scripts. Set its index in `configs/paths.yaml`.

The scripts write to `data/processed/`, `results/`, `logs/` and `figures/` relative to
the repository. `output_root`, `batch_size` and `num_workers` are retained configuration
fields, but do not provide uniform output routing or execution control across scripts.
The scripts evaluate a fixed detector/condition grid declared in source; the YAML
`detectors` and `datasets` lists do not dynamically select a subset.

## 1. Detector inference and evaluation

```bash
python src/run_yolov8.py
python src/run_yolov10.py
python src/run_rtdetr.py
python src/run_fasterrcnn.py
python src/evaluate_detection.py
```

The models use COCO-pretrained weights. Raw predictions are saved to
`results/raw_detections/`; detection and latency metrics are written to `results/metrics/`.
The baseline score threshold is 0.001, input image size is 640, and the NMS setting is 0.7.
Precision, recall and F1 use the separate confidence operating point of 0.25.

The class-name mappings in `src/utils_coco.py` map COCO classes to each dataset taxonomy.
BDD100K rider and traffic-sign classes have no mapped detector output. Inspect the
per-class metrics alongside aggregate metrics when interpreting these results.

## 2. Correctness labels and calibration diagnostics

```bash
python src/create_detection_level_dataset.py
python src/calibration_metrics.py
```

Outputs are `results/reliability/detection_level_labels.csv`, `calibration_bins.csv`
and `detection_ece.csv`. Confidence calibration uses ten equal-width bins. Validation
and test diagnostics are computed separately; the reliability features use validation
bins only. Empty validation bins receive calibration error 1.0.

Correctness labels use IoU >= 0.5. The labeling function first chooses the highest-IoU
box across all classes, then checks class and whether the box was already matched.
This differs from the class-aware unmatched-box search in the precision/recall evaluator.
The current labeling and consistency thresholds are fixed at 0.5 in source; changing
`iou_threshold` in YAML alone does not change them everywhere.

## 3. Reliability features and condition calibration

```bash
python src/run_trustdet_scoring.py
```

The scorer keeps detections with confidence >= 0.25 and samples at most 150 images per
dataset/condition/detector, using seed 42, across validation and test combined. These
settings are constants at the top of the scoring script, independent of the preparation
seed in YAML. Sampling happens before splitting scored rows back into validation and test.
Reliability evaluation therefore covers a sampled detection population, distinct from the
full baseline detection evaluation.

Four inference perturbations are used: brightness adjustment, Gaussian blur, 0.85 resize
and horizontal flip. Consistency measures same-class IoU agreement. Uncertainty uses
confidence dispersion over matched predictions. EigenCAM stability compares saliency
crops. A missing image or saliency computation is logged and receives fallback values.

The condition calibrator uses logistic regression on validation rows, combining the
numeric features with one-hot condition IDs. Runtime overhead is included as a feature,
so hardware and timing variation can affect fitted scores. Test rows are scored using
the frozen fitted model. The same validation population is used for calibration fitting
and threshold tuning; there is no separate calibration holdout in this implementation.

Outputs include `reliability_scores_val.csv`, `reliability_scores_test.csv`, per-combination
checkpoints in `_combo_scores/`, and `trustdet_calibrator.pkl` under `results/reliability/`.

## 4. Gating, ablations and confidence intervals

```bash
python src/tune_thresholds.py
python src/evaluate_gating.py
python src/run_ablation.py
python src/bootstrap_confidence_intervals.py
```

Validation threshold selection maximizes trusted precision subject to coverage >= 0.50.
The rejected threshold maximizes rejected-set purity subject to rejection rate >= 0.10.
The test evaluation applies the saved thresholds unchanged. Outputs are written to
`results/ablations/`.

`evaluate_gating.py` defines false-trusted rate as incorrect trusted detections divided
by all detections. The bootstrap script's FTR instead means 1 minus trusted precision,
with trusted detections as the denominator. These columns should not be compared as the
same statistic. Bootstrap intervals use 1,000 detection-level resamples, seed 42, and
percentile 95% intervals; they do not model clustering by image or condition.

BDD100K conditions may overlap, and the custom validation/test assignments are made per
condition. They do not guarantee disjoint underlying images across conditions. Consider
that dependence when interpreting pooled calibration, gating and uncertainty estimates.

## 5. Optional plots and qualitative examples

```bash
python src/make_plots.py
python src/select_qualitative_examples.py
python src/make_qualitative_panels.py
```

These commands create local files in `figures/`. Qualitative panels require the original
dataset images. No figures or sample images are distributed in this repository.

## Reruns and environment records

Several scripts reuse existing detector JSONs, completed-combination markers, or scored
combination CSVs. Latency and some evaluation outputs are appended. Before changing data,
model versions, thresholds or feature settings, archive existing outputs and rerun in a
fresh checkout. Keep generated output sets separate between experiments.

Record the Python, PyTorch, torchvision, Ultralytics, NumPy and scikit-learn versions,
checkpoint identities, CUDA version, GPU model, config files and source commit with each
run. The source dependency list is unpinned, and no original environment lockfile is
provided. This release does not assert that installing current package versions reproduces
previous numerical results exactly.
