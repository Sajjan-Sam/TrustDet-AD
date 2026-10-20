# Datasets

Download the datasets from their maintainers and follow their access and usage terms.
The preparation scripts read local files; they do not download datasets.

## ExDark

Source: [Exclusively Dark Image Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset).
Obtain the images and bounding-box annotations and arrange them as follows:

```text
data/raw/ExDark/
  images/
    Bicycle/
    Boat/
    ...
  annotations/
    Bicycle/
    Boat/
    ...
```

Annotation filenames include the image extension, for example
`images/Bicycle/example.jpg` and `annotations/Bicycle/example.jpg.txt`.
The parser expects the annotation header followed by rows beginning with
`class x y width height`.

`prepare_exdark.py` shuffles valid samples within sorted class folders using the
configured random seed, then creates a custom 70/15/15 train/validation/test split.
Only validation and test annotations are used for detector evaluation. The train
partition is written for completeness; this pipeline does not train detectors.

## BDD100K

Source: [BDD100K](https://bdd-data.berkeley.edu/), with
[download documentation](https://doc.bdd100k.com/download.html).
Obtain the 100K image collection and detection annotations with condition metadata.

```text
data/raw/BDD100K/
  images/100k/
    train/
    val/
  labels/det_20/
    det_train.json
    det_val.json
```

The path checker validates both train and validation downloads. Subset preparation
reads `det_val.json` and `images/100k/val/` only. It uses `attributes.timeofday`
(`daytime`, `night`) and `attributes.weather` (`rainy`, `foggy`). Each subset is shuffled
with the configured seed and split into custom validation and test halves.
These are not the official BDD100K test partitions. A single image may appear in
more than one condition subset.

## Check and prepare

Set the dataset roots in `configs/paths.yaml`, then run from the repository root:

```bash
python src/check_dataset_paths.py
python src/prepare_exdark.py
python src/prepare_bdd100k.py
python src/dataset_stats.py
```

Processed COCO annotations are written to `data/processed/`, preparation issues to
`logs/`, and dataset statistics to `results/metrics/`. Review the logs for skipped
images or invalid annotations before running inference.
