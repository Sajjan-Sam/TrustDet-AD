# TrustDet-AD

**Condition-Calibrated Reliability Scoring and Risk-Aware Gating for Object Detection in Adverse Visual Conditions**

*Research implementation | PyTorch | Object detection | Reliability assessment | Adverse conditions*

[Overview](#overview) · [Method](#method-overview) · [Datasets](#datasets-and-detectors) · [Installation](#installation) · [Reproduction](#reproducing-the-experiments) · [Evaluation](#evaluation-protocol) · [Limitations](#implementation-notes-and-limitations)

## Overview

Modern object detectors may produce high-confidence errors when images are captured under low illumination or challenging weather. Detection confidence alone does not describe whether a particular prediction should be trusted.

**TrustDet-AD** is a post-hoc framework for studying *detection-level reliability* under adverse visual conditions. It combines prediction confidence, validation-derived calibration error, consistency under image perturbations, uncertainty from matched predictions, and EigenCAM-based explanation stability. A condition-aware calibrator converts these measurements into a reliability score. Validation-selected thresholds then assign each detection to one of three groups: **trusted**, **uncertain**, or **rejected**.

The repository evaluates **pretrained detectors without fine-tuning their backbones**. It provides the data-preparation, inference, reliability-scoring, threshold-selection, ablation, and evaluation scripts used in the pipeline.

## Method overview

The implementation follows five stages:

1. **Detection:** Run COCO-pretrained detectors on the configured ExDark and BDD100K subsets.
2. **Correctness and calibration:** Associate predictions with ground-truth boxes and estimate confidence–accuracy discrepancies using validation data.
3. **Reliability features:** Measure detection consistency across brightness, blur, resize, and flip perturbations; compute confidence dispersion and EigenCAM stability.
4. **Reliability scoring:** Compare individual signals and a fixed equal-weight score with a condition-aware logistic-regression calibrator fitted on validation detections.
5. **Risk-aware gating:** Select two thresholds on validation detections and apply them unchanged to the test detections.

For a detection with reliability score $s$, the implemented decision rule is:

$$
G(s)=\begin{cases}
\text{trusted}, & s\geq\tau_{\mathrm{trusted}},\\
\text{rejected}, & s<\tau_{\mathrm{rejected}},\\
\text{uncertain}, & \text{otherwise},
\end{cases}
$$

where $\tau_{\mathrm{rejected}}\leq\tau_{\mathrm{trusted}}$. These are empirical operating thresholds, **not formal statistical guarantees of risk**.

### Reliability signals

| Signal | What is measured |
| --- | --- |
| Detection confidence | Detector-reported confidence for a predicted box |
| Calibration error | Absolute confidence–accuracy gap from the corresponding validation confidence bin |
| Prediction consistency | Agreement of same-class boxes across perturbed inference passes |
| Prediction uncertainty | Dispersion of confidence across matched predictions |
| Explanation stability | Similarity of EigenCAM saliency regions |
| Runtime overhead | Additional computation recorded as an input to the condition-aware calibrator |

The fixed-score baseline equally averages confidence, inverse calibration error, consistency, inverse uncertainty, and explanation stability. The condition-aware model uses numerical features and condition identifiers. See `src/reliability_components.py` and `src/condition_calibrated_score.py` for the exact implementations.

## Datasets and detectors

| Dataset | Evaluation configuration |
| --- | --- |
| [ExDark](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) | Custom 70/15/15 split within class folders; validation and test partitions are used for evaluation |
| [BDD100K](https://bdd-data.berkeley.edu/) | Condition-specific subsets derived from the **official validation set**, each divided into custom validation and test halves |

BDD100K conditions are selected using time-of-day and weather attributes, including daytime, night, rainy, and foggy. 
| Detector | Pretrained checkpoint / weights |
| --- | --- |
| YOLOv8n | `yolov8n.pt` |
| YOLOv10n | `yolov10n.pt` |
| RT-DETR | `rtdetr-l.pt` |
| Faster R-CNN (ResNet-50-FPN) | `FasterRCNN_ResNet50_FPN_Weights.COCO_V1` |

Models retain their upstream pretrained weights. The detector checkpoints and datasets are not distributed with this repository.

## Installation

A CUDA-capable GPU is required by the current inference scripts. Install a mutually compatible CUDA-enabled PyTorch / torchvision combination for your system.

```bash
git clone https://github.com/Sajjan-Sam/TrustDet-AD.git
cd TrustDet-AD

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Save the exact installed versions used for your run.
python -m pip freeze > environment-freeze.txt
```

The dependency file is **not an environment lockfile** from the original experiments. Package updates may affect predictions, timing, and the resulting reliability scores.

Edit the dataset roots and `gpu_id` in [`configs/paths.yaml`](configs/paths.yaml). Run all commands below from the repository root.

## Reproducing the experiments

**1. Prepare the datasets.** Download ExDark and BDD100K from their respective providers, following [DATASETS.md](docs/DATASETS.md), and run:

```bash
python src/check_dataset_paths.py
python src/prepare_exdark.py
python src/prepare_bdd100k.py
python src/dataset_stats.py
```

**2. Run pretrained detectors and evaluate the baseline predictions.**

```bash
python src/run_yolov8.py
python src/run_yolov10.py
python src/run_rtdetr.py
python src/run_fasterrcnn.py
python src/evaluate_detection.py
```

**3. Create detection-level labels and calibration diagnostics.**

```bash
python src/create_detection_level_dataset.py
python src/calibration_metrics.py
```

**4. Compute reliability scores and fit the condition-aware calibrator.**

```bash
python src/run_trustdet_scoring.py
```

**5. Select gating thresholds and evaluate the frozen settings.**

```bash
python src/tune_thresholds.py
python src/evaluate_gating.py
python src/run_ablation.py
python src/bootstrap_confidence_intervals.py
```

**6. Generate optional plots and qualitative examples.**

```bash
python src/make_plots.py
python src/select_qualitative_examples.py
python src/make_qualitative_panels.py
```

For the execution order, intermediate files, caching behavior, and script-specific settings, read the full [reproduction guide](docs/REPRODUCING.md) **before rerunning experiments**.

## Evaluation protocol

- **Correctness:** Detection-level labels use an IoU threshold of 0.5, with the matching behavior documented in the reproduction guide.
- **Confidence calibration:** Ten equal-width confidence bins are computed; reliability features use validation-derived calibration information.
- **Reliability subset:** Scoring uses predictions with confidence at least 0.25, sampling at most **150 images per dataset–condition–detector combination** across validation and test combined (seed 42). This differs from the full baseline detection evaluation.
- **Perturbations:** Brightness adjustment, Gaussian blur, resize to 0.85 scale, and horizontal flip.
- **Threshold selection:** The trusted threshold maximizes validation trusted precision subject to coverage of at least 50%. The rejected threshold maximizes rejected-set purity subject to a rejection rate of at least 10% and the threshold-order constraint.
- **Test evaluation:** The fitted calibrator and selected gate thresholds are applied without refitting on test detections.
- **Uncertainty intervals:** The implementation uses 1,000 detection-level bootstrap resamples with percentile 95% intervals (seed 42).

Evaluation outputs include detector metrics, calibration diagnostics, reliability scores, gating summaries, ablations, confidence intervals, and optional qualitative panels. **No pretrained calibrator, generated predictions, figures, or numerical result tables are bundled in this release.**

## Repository structure

```text
TrustDet-AD/
├── configs/              # Dataset paths and experiment settings
├── docs/
│   ├── DATASETS.md       # Dataset setup and custom partitions
│   ├── REPRODUCING.md    # Execution order and evaluation details
│   └── THIRD_PARTY.md    # Upstream resources and licenses
├── src/
│   ├── prepare_*.py      # Dataset preparation
│   ├── run_yolo*.py      # YOLO inference
│   ├── run_rtdetr.py     # RT-DETR inference
│   ├── run_fasterrcnn.py # Faster R-CNN inference
│   ├── reliability_components.py
│   ├── run_trustdet_scoring.py
│   ├── tune_thresholds.py
│   ├── risk_aware_gate.py
│   ├── evaluate_*.py
│   └── run_ablation.py
├── requirements.txt
└── README.md
```

The scripts generate `data/processed/`, `results/`, `logs/`, and `figures/` locally. The tree above highlights principal files rather than every source module.

## Implementation notes and limitations

This release is intended to make the implemented evaluation **inspectable and reproducible**, not to imply that every limitation of the protocol has been resolved.

1. **Validation reuse:** The same validation population is used both to fit the condition-aware calibrator and to select the gating thresholds. There is no independent calibration holdout.
2. **Nonstandard evaluation partitions:** ExDark uses a custom split; BDD100K uses custom subdivisions of its official validation set. BDD100K condition groups can share underlying images, so pooled observations may not be independent.
3. **Class mapping:** COCO-to-BDD100K mapping does not cover the BDD100K *rider* and *traffic sign* classes. Per-class results should accompany aggregate detector comparisons.
4. **Metric interpretation:** The gating evaluator and bootstrap script use different denominators for a metric labeled false-trusted rate (FTR); compare them only after harmonizing the definition. Detection-level bootstrap resampling also does not account for within-image dependence.
5. **Evaluation implementation:** The detection-correctness matcher and the precision/recall evaluator do not use identical matching procedures. Some configuration fields and thresholds are fixed inside scripts rather than controlled uniformly through YAML.
6. **Cost and portability:** Repeated perturbation inference and EigenCAM computation incur additional latency. Runtime overhead is also a model feature, so hardware and package versions can influence the fitted score.
7. **Rerun isolation:** Some stages reuse cached files or append outputs. Archive or remove old outputs and use a clean run directory or checkout when changing experimental settings.

Further details are documented in [REPRODUCING.md](docs/REPRODUCING.md).

## Citation and third-party acknowledgments

Citation metadata for the associated TrustDet-AD manuscript will be added when its publication record is available. Until then, reference the repository URL and commit identifier if you need to specify the exact code version used.

ExDark, BDD100K, Ultralytics, torchvision, COCO API, and the other dependencies remain subject to their original licenses and citation requirements. Refer to [THIRD_PARTY.md](docs/THIRD_PARTY.md) and the respective upstream projects.
