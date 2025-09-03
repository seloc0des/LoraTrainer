# LoRaTrainer

A simple interactive CLI to help you:

- Discover or specify your Automatic1111 (stable-diffusion-webui) and kohya_ss installations
- Select your training image folder and preprocess images to 512x512 (letterbox with padding)
- Enter key LoRA training parameters with sensible defaults
- Build and (optionally) run a kohya_ss training command
- Save/reuse your answers in a JSON config
- Optionally generate .txt captions next to processed images (constant/filename/template)
- Auto-discover pretrained models from your Automatic1111 models directory with interactive selection

## Quick Start

1) Create a virtual environment (recommended):

```
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:

```
pip install -r requirements.txt
```

3) Run the CLI:

```
python train_lora.py
```

The tool will guide you through discovery and setup, resize your images to 512x512 (without distortion), and then construct a `kohya_ss` training command. You can choose to run it immediately or just print the command.

### Captions Generation (optional)

You can opt-in to generate `.txt` captions for each processed image:

- **constant**: same text for all images
- **filename**: tokens derived from the filename (configurable delimiter)
- **template**: format with placeholders `{filename}`, `{class}`, `{instance}`

Captions are written next to processed images as `.txt` files. Existing captions can be preserved or overwritten. Training will add `--caption_extension .txt` automatically when captions are enabled.

### AI Captioners (optional)

- **CLIP Interrogator** (in-Python): Generates natural language captions with CLIP+BLIP.
  - You can choose presets: `fast` (recommended for ~8GB VRAM like RTX 3070) or `best` (more VRAM).
  - If installed via `requirements.txt`, the CLI will offer to run it.
  - Captions are written as `.txt` files; you may choose to overwrite existing captions.

- **WD14 Tagger via kohya_ss**: Generates tag-style captions.
  - The CLI will attempt to locate `tag_images_by_wd14_tagger.py` inside your `kohya_ss` repo and run it.
  - Configurable repo (default `SmilingWolf/wd-v1-4-convnext-tagger-v2`), thresholds, and batch size.
  - Writes `.txt` captions next to images.

Tip for ~8GB GPUs (e.g., RTX 3070 8GB):

- Prefer CLIP Interrogator preset `fast`.
- Keep WD14 batch size low (e.g., 2-4) to avoid OOM.

## Defaults and Config

- Config is stored at `~/.lora_trainer/config.json`.
- You can rerun the tool to reuse and edit previous answers.

## Notes

- This project does not install Automatic1111 or kohya_ss for you. It will try to discover them and then ask for paths if not found.
- Training requires a working kohya_ss installation with `train_network.py` (or equivalent) available.
- Image preprocessing writes into a `processed/` folder next to your original images.
