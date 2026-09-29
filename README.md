# PG-SAM: Main Map Area Extraction Demo

This repository contains the inference-only PG-SAM V6.1 Full demo: the model code, an inference checkpoint, and a small set of input maps. It does not contain training data, ground-truth masks, evaluation tables, or experimental outputs. The checkpoint retains only inference settings and trained weights; training state and evaluation records were removed.

The default PG-SAM checkpoint is the V6.1 Full `manual400` seed-42 model at `checkpoints/pgsam_best_model.pt`. Its SHA-256 hash and inference settings are recorded in `model_info.json`.

The SAM2.1 Hiera-Large base checkpoint is not redistributed here. Download `sam2.1_hiera_large.pt` from the [official model page](https://huggingface.co/facebook/sam2.1-hiera-large) and place it at `checkpoints/sam2.1_hiera_large.pt`.

## Files

```text
PG-SAM_demo/
|-- infer.py               # inference entry point
|-- check_package.py       # package and checkpoint check
|-- model_info.json        # checkpoint identity and inference settings
|-- setup.py
|-- requirements.txt
|-- run_inference.bat       # Windows example
|-- run_inference.sh        # Linux/macOS example
|-- checkpoints/           # PG-SAM weight; SAM2 weight downloaded separately
|-- map_sam2/              # PG-SAM model and inference code
|-- sam2/                  # SAM2 model code and configuration
|-- tools/                 # inference-checkpoint export utility
`-- data/images/          # input maps for a run check
```

## Setup

Python 3.10 or newer is required. A CUDA GPU is recommended because the model uses SAM2.1 Hiera-Large. Install a PyTorch build compatible with your CUDA driver first, then run:

```bash
pip install -e .
```

After downloading the SAM2.1 base checkpoint, verify the package and both weights:

```bash
python check_package.py --check-weights
```

## Run inference

On Windows, run `run_inference.bat`; on Linux/macOS, run `bash run_inference.sh`. The equivalent command is:

```bash
python infer.py
```

The default command processes `data/images/`. Final binary masks (0 for background, 255 for the main map area) are written to `outputs/inference_run/masks/`; run metadata is written to `outputs/inference_run/metadata/`. Generated files under `outputs/` are ignored by Git.

To process your own map images:

```bash
python infer.py --image-dir path/to/images --output-dir outputs/my_images
```

Use `--image path/to/image.png` for a single image. Optional `--save-raw`, `--save-prob`, and `--save-overlay` flags export additional visualizations. `--sam2-checkpoint` and `--pgsam-checkpoint` select custom checkpoint paths.

The default protocol follows the released checkpoint: resize to 1024 × 1024 by letterboxing, use SGN-generated point prompts and the learned implicit spatial prior, then apply decoder and boundary/full-resolution refinement. Masks are restored to each image's original dimensions. No box prompts are used.

## Data and license

The bundled images are input-only samples for checking that the demo runs; they are not an evaluation dataset. See `DATA_CARD.md` for details. Review the included licenses and the SAM2 model page before using or redistributing third-party components or weights.
