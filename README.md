# PG-SAM: A Prior-Guided Segmentation Framework for Main Map Area Extraction

This repository provides the reproducible inference code for PG-SAM. A reader can download the repository, install the Python dependencies, download the required SAM2.1 base checkpoint, and run segmentation on the bundled input images.

The package includes the inference source code, SAM2 configuration files, sample input images, and the PG-SAM checkpoint used by default:

```text
checkpoints/pgsam_best_model.pt
```

The SAM2.1 Hiera-L base checkpoint is not included in this GitHub repository. Download it from the Hugging Face model page and place it at:

```text
checkpoints/sam2.1_hiera_large.pt
```

Official download source:

https://huggingface.co/facebook/sam2.1-hiera-large

Download `sam2.1_hiera_large.pt` from the Hugging Face page above, then place it at:

```text
checkpoints/sam2.1_hiera_large.pt
```

Do not remove the `checkpoints/`, `sam2/`, or `map_sam2/` directories after placing the checkpoint if you want the package to run offline.

The package ships with a few input images in `data/images/` so readers can verify that image loading, model inference, post-processing, and final mask export are wired correctly. `model_info.json` records information for the supplied checkpoint.

## Files

```text
release-folder/
|-- infer.py            # model inference
|-- check_package.py    # package sanity check
|-- model_info.json     # supplied checkpoint information
|-- DATA_CARD.md        # bundled input-data description
|-- setup.py            # editable local package installer
|-- pyproject.toml
|-- requirements.txt    # Python dependencies
|-- run_inference.bat   # Windows example command
|-- run_inference.sh    # Linux/macOS example command
|-- checkpoints/        # PG-SAM weight; SAM2 base weight is downloaded separately
|-- map_sam2/           # PG-SAM model code required for inference
|-- sam2/               # SAM2 model code and configs required for inference
`-- data/
    `-- images/         # bundled input images to segment
```

## Setup

The package requires Python 3.10 or newer. A CUDA GPU is recommended for practical runtime because the default model is SAM2.1 Hiera-L.

Open a terminal inside the release folder, then install the local package and dependencies:

```bash
cd path/to/release-folder
pip install -e .
```

If PyTorch is not installed yet, install the PyTorch build that matches the local CUDA driver first, then run the command above. You can also pass custom weight paths with `--sam2-checkpoint` and `--pgsam-checkpoint`.

Before running inference, download `sam2.1_hiera_large.pt` as described above.

## Run The Bundled Inference

Windows:

```bat
run_inference.bat
```

Linux/macOS:

```bash
bash run_inference.sh
```

Direct Python command:

```bash
python infer.py \
  --image-dir data/images \
  --output-dir outputs/inference_run
```

Outputs are written to:

```text
outputs/inference_run/
`-- masks/
```

## Use Your Own Images

```bash
python infer.py \
  --image-dir path/to/images \
  --output-dir outputs/my_images
```

## Data Check

```bash
python check_package.py
python check_package.py --check-weights
```

The second command also verifies that the default SAM2 and PG-SAM checkpoint files are present.

## License And Third-Party Checkpoints

The SAM2.1 Hiera-L checkpoint is distributed by Meta. Please review the SAM2 license and model page before using or redistributing it. This repository does not upload `sam2.1_hiera_large.pt`; users should download it from the official source and keep it at `checkpoints/sam2.1_hiera_large.pt`.
