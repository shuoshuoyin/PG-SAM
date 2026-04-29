# Data Card

## Purpose

The bundled data are included only to verify that the PG-SAM release package can run end-to-end after download. They provide a small sanity-check set for inference and final mask export.

## Contents

```text
data/
`-- images/   # RGB input images
```

The current release contains bundled input images to be segmented, for example:

```text
data/images/image1065.png
```

## Package Check

Run:

```bash
python check_package.py
```

to verify that required source files and bundled input images are present.

## Intended Use

These samples are not intended to replace the full training or evaluation dataset. They are a compact reproducibility check so readers can confirm that:

- the folder is complete,
- the checkpoints can be found,
- the inference script writes final segmentation masks.
