# TrustDet-AD

Detection reliability and risk-aware gating for object detection in adverse conditions.
The pipeline evaluates pretrained detectors on ExDark and BDD100K, combines confidence,
calibration, prediction consistency, uncertainty and EigenCAM stability, and assigns
trusted, uncertain or rejected labels to detections.

## Setup

Use a Python environment with a CUDA-enabled PyTorch and torchvision installation
compatible with your GPU. The inference scripts select `cuda:<gpu_id>` from the config.
They do not train the detector backbones.

```bash
git clone https://github.com/Sajjan-Sam/TrustDet-AD.git
cd TrustDet-AD
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency list is not a lockfile from the original experiment environment.
Record your installed versions with `python -m pip freeze` when running experiments.

1. Obtain the datasets using [the dataset guide](docs/DATASETS.md).
2. Edit dataset locations and `gpu_id` in `configs/paths.yaml`.
3. Run the commands in [the reproduction guide](docs/REPRODUCING.md) from this directory.

Pretrained checkpoints are `yolov8n.pt`, `yolov10n.pt`, `rtdetr-l.pt`, and
`FasterRCNN_ResNet50_FPN_Weights.COCO_V1`. Model libraries may download them on first use.
Datasets, checkpoints, fitted calibrators and experiment outputs are obtained or generated
locally and are not bundled here.

## Repository

| Location | Contents |
| --- | --- |
| `configs/` | Dataset locations, inference settings and experiment defaults |
| `src/prepare_*.py` | Annotation conversion and custom evaluation splits |
| `src/run_yolo*.py`, `src/run_rtdetr.py`, `src/run_fasterrcnn.py` | Pretrained detector inference |
| `src/*score*.py`, `src/reliability_components.py` | Reliability features and calibration |
| `src/tune_thresholds.py`, `src/risk_aware_gate.py` | Validation threshold selection and gating |
| `src/evaluate_*.py`, `src/run_ablation.py`, `src/bootstrap_confidence_intervals.py` | Evaluation and analysis |
| `src/make_plots.py`, `src/*qualitative*.py` | Plots and qualitative examples generated locally |

## Evaluation conventions

Calibration and gating thresholds are fitted on validation detections. Frozen settings
are then evaluated on test detections. BDD100K uses custom halves of its official validation
set, and ExDark uses a custom split within class folders. Reliability scoring samples at
most 150 images per dataset/condition/detector at confidence >= 0.25. These conventions and
other implementation limits are detailed in the reproduction guide.

External datasets, model weights and dependencies retain their own licenses and usage
terms. See [third-party resources](docs/THIRD_PARTY.md).
