#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from PIL import Image, ImageOps
from rich import print
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from tqdm import tqdm

console = Console()

CONFIG_DIR = Path.home() / ".lora_trainer"
CONFIG_PATH = CONFIG_DIR / "config.json"


# ---------------------- Config Models ----------------------
@dataclass
class PathsConfig:
    automatic1111_path: Optional[str] = None
    kohya_path: Optional[str] = None
    venv_python: Optional[str] = None  # optional python path for kohya venv


@dataclass
class TrainConfig:
    training_images_dir: Optional[str] = None
    processed_images_dir: Optional[str] = None
    output_dir: Optional[str] = None
    output_name: Optional[str] = None
    resolution: int = 512
    learning_rate: float = 1e-4
    text_encoder_lr: Optional[float] = None
    unet_lr: Optional[float] = None
    network_dim: int = 64
    network_alpha: int = 32
    train_batch_size: int = 2
    max_train_steps: int = 2000
    save_every_n_steps: int = 500
    lr_scheduler: str = "cosine"
    mixed_precision: str = "bf16"
    pretrained_model_name_or_path: Optional[str] = None
    # captions
    captions_enabled: bool = False
    caption_mode: Optional[str] = None  # "constant" | "filename" | "template"
    caption_constant_text: Optional[str] = None
    caption_template: Optional[str] = None  # e.g. "{instance} {class}, {filename}"
    caption_class_token: Optional[str] = None
    caption_instance_token: Optional[str] = None
    caption_delimiter: str = "_"  # used for filename tokenization
    caption_overwrite: bool = False


@dataclass
class AppConfig:
    paths: PathsConfig
    train: TrainConfig


# ---------------------- Utilities ----------------------

def load_config() -> AppConfig:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text())
            paths = PathsConfig(**data.get("paths", {}))
            train = TrainConfig(**data.get("train", {}))
            return AppConfig(paths=paths, train=train)
        except Exception:
            pass
    return AppConfig(paths=PathsConfig(), train=TrainConfig())


def save_config(cfg: AppConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"paths": asdict(cfg.paths), "train": asdict(cfg.train)}
    CONFIG_PATH.write_text(json.dumps(payload, indent=2))


def path_contains_any(path: Path, names: List[str]) -> bool:
    return any((path / n).exists() for n in names)


def is_automatic1111_root(path: Path) -> bool:
    return path_contains_any(path, ["webui.sh", "webui-user.sh", "launch.py"]) and (
        (path / "modules").exists() or (path / ".git").exists()
    )


def is_kohya_root(path: Path) -> bool:
    return (path / "train_network.py").exists() or (path / "sd-scripts" / "train_network.py").exists()


def search_common_paths(candidates: List[Path], predicate) -> Optional[Path]:
    for p in candidates:
        try:
            if p and p.exists() and predicate(p):
                return p
        except Exception:
            continue
    return None


def discover_automatic1111(existing: Optional[str]) -> Optional[str]:
    # 1) env var
    env = os.getenv("LORA_AUTOMATIC1111_PATH")
    if env and is_automatic1111_root(Path(env)):
        return str(Path(env).resolve())

    # 2) user-provided existing path
    if existing and is_automatic1111_root(Path(existing)):
        return str(Path(existing).resolve())

    # 3) common locations
    home = Path.home()
    common = [
        home / "stable-diffusion-webui",
        home / "Repositories" / "stable-diffusion-webui",
        Path("/opt") / "stable-diffusion-webui",
        Path("/srv") / "stable-diffusion-webui",
    ]
    found = search_common_paths(common, is_automatic1111_root)
    return str(found.resolve()) if found else None


def a1111_models_dir(auto_root: Path) -> Path:
    return auto_root / "models" / "Stable-diffusion"


def discover_pretrained_models(auto_root: Path) -> List[Path]:
    models_dir = a1111_models_dir(auto_root)
    if not models_dir.exists():
        return []
    exts = {".safetensors", ".ckpt", ".pt"}
    candidates = [p for p in models_dir.rglob("*") if p.suffix.lower() in exts]
    # Exclude SDXL models when focusing on SD 1.5
    def is_sdxl(path: Path) -> bool:
        s = str(path).lower()
        return "sdxl" in s
    filtered = [p for p in candidates if not is_sdxl(p)]
    return sorted(filtered)


def discover_kohya(existing: Optional[str]) -> Optional[str]:
    # 1) env var
    env = os.getenv("LORA_KOHYA_PATH")
    if env and is_kohya_root(Path(env)):
        return str(Path(env).resolve())

    # 2) user-provided existing path
    if existing and is_kohya_root(Path(existing)):
        return str(Path(existing).resolve())

    # 3) common locations
    home = Path.home()
    common = [
        home / "kohya_ss",
        home / "sd-scripts",  # some forks/layouts
        Path("/opt") / "kohya_ss",
        Path("/srv") / "kohya_ss",
    ]
    found = search_common_paths(common, is_kohya_root)
    return str(found.resolve()) if found else None


def prompt_path(question: str, default: Optional[str], validator) -> str:
    while True:
        ans = Prompt.ask(question, default=default or "")
        p = Path(ans).expanduser()
        if validator(p):
            return str(p.resolve())
        print(f"[red]Invalid path:[/red] {p}")


def ensure_processed_dir(images_dir: Path) -> Path:
    processed = images_dir.parent / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    return processed


def list_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]


def letterbox_to_square(img: Image.Image, size: int = 512) -> Image.Image:
    # Preserve aspect ratio using padding, without distortion
    return ImageOps.pad(img, (size, size), method=Image.Resampling.LANCZOS, color=(0, 0, 0))


def preprocess_images(src_dir: Path, dst_dir: Path, size: int = 512) -> Tuple[int, int]:
    images = list_images(src_dir)
    if not images:
        return 0, 0
    converted, skipped = 0, 0
    for src in tqdm(images, desc="Preprocessing images", unit="img"):
        rel = src.relative_to(src_dir)
        out_path = dst_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                if im.width == size and im.height == size:
                    # copy as-is (ensure format)
                    im.save(out_path.with_suffix(".png"))
                    converted += 1
                else:
                    out = letterbox_to_square(im, size)
                    out.save(out_path.with_suffix(".png"))
                    converted += 1
        except Exception:
            skipped += 1
    return converted, skipped


def build_caption_for_image(
    img_path: Path,
    mode: str,
    constant_text: Optional[str],
    template: Optional[str],
    class_token: Optional[str],
    instance_token: Optional[str],
    delimiter: str,
) -> str:
    stem = img_path.stem
    if mode == "constant":
        return (constant_text or "").strip()
    if mode == "filename":
        tokens = [t for t in stem.replace("-", delimiter).split(delimiter) if t]
        return ", ".join(tokens)
    if mode == "template":
        safe = {
            "filename": stem,
            "class": class_token or "",
            "instance": instance_token or "",
        }
        try:
            return (template or "{filename}").format(**safe).strip()
        except Exception:
            return stem
    return stem


def generate_captions(
    images_root: Path,
    mode: str,
    constant_text: Optional[str],
    template: Optional[str],
    class_token: Optional[str],
    instance_token: Optional[str],
    delimiter: str,
    overwrite: bool,
) -> Tuple[int, int]:
    images = list_images(images_root)
    if not images:
        return 0, 0
    written, skipped = 0, 0
    for img in tqdm(images, desc="Generating captions", unit="img"):
        txt_path = img.with_suffix(".txt")
        if txt_path.exists() and not overwrite:
            skipped += 1
            continue
        caption = build_caption_for_image(
            img, mode, constant_text, template, class_token, instance_token, delimiter
        )
        try:
            txt_path.write_text(caption + "\n", encoding="utf-8")
            written += 1
        except Exception:
            skipped += 1
    return written, skipped


def generate_captions_clip_interrogator(
    images_root: Path,
    preset: str = "fast",
    overwrite: bool = False,
) -> Tuple[int, int]:
    try:
        from clip_interrogator import Config, Interrogator  # type: ignore
    except Exception as e:
        print("[yellow]clip-interrogator not installed. Install it to use AI captioning:[/yellow] pip install clip-interrogator")
        return 0, 0

    # Choose models that fit in ~8GB VRAM
    if preset == "fast":
        cfg = Config(clip_model_name="ViT-L-14/openai")
        # Default BLIP is usually base for speed
    else:  # best
        cfg = Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k")

    try:
        ci = Interrogator(cfg)
    except Exception as e:
        print(f"[red]Failed to initialize CLIP Interrogator:[/red] {e}")
        return 0, 0

    images = list_images(images_root)
    if not images:
        return 0, 0
    written, skipped = 0, 0
    for img_path in tqdm(images, desc=f"AI captioning ({preset})", unit="img"):
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists() and not overwrite:
            skipped += 1
            continue
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                caption = ci.interrogate(im)
            txt_path.write_text(caption.strip() + "\n", encoding="utf-8")
            written += 1
        except Exception:
            skipped += 1
    return written, skipped


def which_python_for_kohya(kohya_path: Path, existing: Optional[str]) -> Optional[str]:
    # If user previously set a venv python, prefer it if it exists
    if existing and Path(existing).exists():
        return existing

    # try kohya venvs
    candidates = [
        kohya_path / ".venv" / "bin" / "python",
        kohya_path / "venv" / "bin" / "python",
        kohya_path / "env" / "bin" / "python",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # fallback to current python
    return sys.executable


def build_kohya_command(kohya_root: Path, python_exec: str, train_cfg: TrainConfig) -> List[str]:
    # Support both layouts: train_network.py at root or under sd-scripts/
    train_script = kohya_root / "train_network.py"
    if not train_script.exists():
        alt = kohya_root / "sd-scripts" / "train_network.py"
        train_script = alt if alt.exists() else train_script

    cmd = [
        python_exec,
        str(train_script),
        "--network_module", "networks.lora",
        "--pretrained_model_name_or_path", train_cfg.pretrained_model_name_or_path or "",
        "--train_data_dir", train_cfg.processed_images_dir or "",
        "--resolution", f"{train_cfg.resolution},{train_cfg.resolution}",
        "--output_dir", train_cfg.output_dir or "",
        "--output_name", train_cfg.output_name or "lora_out",
        "--learning_rate", str(train_cfg.learning_rate),
        "--network_dim", str(train_cfg.network_dim),
        "--network_alpha", str(train_cfg.network_alpha),
        "--train_batch_size", str(train_cfg.train_batch_size),
        "--max_train_steps", str(train_cfg.max_train_steps),
        "--save_every_n_steps", str(train_cfg.save_every_n_steps),
        "--lr_scheduler", train_cfg.lr_scheduler,
        "--mixed_precision", train_cfg.mixed_precision,
    ]
    if train_cfg.text_encoder_lr:
        cmd += ["--text_encoder_lr", str(train_cfg.text_encoder_lr)]
    if train_cfg.unet_lr:
        cmd += ["--unet_lr", str(train_cfg.unet_lr)]
    if train_cfg.captions_enabled:
        cmd += ["--caption_extension", ".txt"]
    return [c for c in cmd if c != ""]


def find_wd14_script(kohya_root: Path) -> Optional[Path]:
    candidates = [
        kohya_root / "tag_images_by_wd14_tagger.py",
        kohya_root / "sd-scripts" / "tag_images_by_wd14_tagger.py",
        kohya_root / "finetune" / "tag_images_by_wd14_tagger.py",
        kohya_root / "sd-scripts" / "finetune" / "tag_images_by_wd14_tagger.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def run_wd14_tagger(
    kohya_root: Path,
    python_exec: str,
    images_root: Path,
    repo_id: str = "SmilingWolf/wd-v1-4-convnext-tagger-v2",
    batch_size: int = 2,
    general_threshold: float = 0.35,
    character_threshold: float = 0.85,
    recursive: bool = True,
    overwrite: bool = False,
) -> Tuple[bool, str]:
    script = find_wd14_script(kohya_root)
    if not script:
        return False, "wd14 tagger script not found in kohya_ss repository"
    cmd = [
        python_exec,
        str(script),
        "--repo_id", repo_id,
        "--batch_size", str(batch_size),
        "--general_threshold", str(general_threshold),
        "--character_threshold", str(character_threshold),
        "--caption_extension", ".txt",
        "--max_data_loader_n_workers", "2",
    ]
    if recursive:
        cmd.append("--recursive")
    if overwrite:
        cmd.append("--force")
    cmd += ["--train_data_dir", str(images_root)]
    try:
        proc = subprocess.Popen(cmd, cwd=str(kohya_root))
        proc.wait()
        ok = proc.returncode == 0
        return ok, f"returncode={proc.returncode}"
    except Exception as e:
        return False, str(e)


# ---------------------- Main Flow ----------------------

def main():
    cfg = load_config()

    print("[bold cyan]LoRa Trainer Setup[/bold cyan]")

    # Discover Automatic1111
    auto1111 = discover_automatic1111(cfg.paths.automatic1111_path)
    if not auto1111:
        print("Could not auto-discover Automatic1111 (stable-diffusion-webui).")
        auto1111 = prompt_path(
            "Enter path to your Automatic1111 (stable-diffusion-webui) root",
            cfg.paths.automatic1111_path,
            is_automatic1111_root,
        )
    print(f"Using Automatic1111 at: [green]{auto1111}[/green]")
    cfg.paths.automatic1111_path = auto1111

    # Discover kohya_ss
    kohya = discover_kohya(cfg.paths.kohya_path)
    if not kohya:
        print("Could not auto-discover kohya_ss.")
        kohya = prompt_path(
            "Enter path to your kohya_ss (root containing train_network.py)",
            cfg.paths.kohya_path,
            is_kohya_root,
        )
    print(f"Using kohya_ss at: [green]{kohya}[/green]")
    cfg.paths.kohya_path = kohya

    # Python for kohya (prefer kohya venv)
    venv_python = which_python_for_kohya(Path(kohya), cfg.paths.venv_python)
    print(f"Python for kohya: [green]{venv_python}[/green]")
    cfg.paths.venv_python = venv_python

    # Training images dir
    default_images = cfg.train.training_images_dir or str(Path.cwd() / "training_images")
    images_dir_str = Prompt.ask("Path to your training images folder", default=default_images)
    images_dir = Path(images_dir_str).expanduser().resolve()
    if not images_dir.exists() or not images_dir.is_dir():
        print(f"[red]Training images directory not found:[/red] {images_dir}")
        sys.exit(1)
    cfg.train.training_images_dir = str(images_dir)

    # Preprocess
    processed_dir = ensure_processed_dir(images_dir)
    cfg.train.processed_images_dir = str(processed_dir)
    print(f"Processed images will be written to: [green]{processed_dir}[/green]")

    # Resize to 512x512 with letterbox padding
    do_prep = Confirm.ask("Resize images to 512x512 (letterbox)?", default=True)
    if do_prep:
        converted, skipped = preprocess_images(images_dir, processed_dir, size=cfg.train.resolution)
        print(f"Preprocessing complete. Converted: [green]{converted}[/green], Skipped: [yellow]{skipped}[/yellow]")

    # Optional: generate captions
    if Confirm.ask("Generate captions (.txt) for each processed image?", default=False):
        cfg.train.captions_enabled = True
        mode = Prompt.ask("Caption mode", choices=["constant", "filename", "template"], default=(cfg.train.caption_mode or "filename"))
        cfg.train.caption_mode = mode
        if mode == "constant":
            cfg.train.caption_constant_text = Prompt.ask("Constant caption text", default=cfg.train.caption_constant_text or "")
        elif mode == "template":
            cfg.train.caption_template = Prompt.ask(
                "Template (use {filename}, {class}, {instance})",
                default=cfg.train.caption_template or "{instance} {class}, {filename}",
            )
        if mode in ("filename", "template"):
            cfg.train.caption_class_token = Prompt.ask("Class token (optional)", default=cfg.train.caption_class_token or "") or None
            cfg.train.caption_instance_token = Prompt.ask("Instance token (optional)", default=cfg.train.caption_instance_token or "") or None
        if mode == "filename":
            cfg.train.caption_delimiter = Prompt.ask("Filename delimiter", default=cfg.train.caption_delimiter or "_")
        cfg.train.caption_overwrite = Confirm.ask("Overwrite existing .txt captions?", default=cfg.train.caption_overwrite)

        written, skipped_c = generate_captions(
            Path(cfg.train.processed_images_dir),
            cfg.train.caption_mode,
            cfg.train.caption_constant_text,
            cfg.train.caption_template,
            cfg.train.caption_class_token,
            cfg.train.caption_instance_token,
            cfg.train.caption_delimiter,
            cfg.train.caption_overwrite,
        )
        print(f"Captions generated. Written: [green]{written}[/green], Skipped: [yellow]{skipped_c}[/yellow]")
    else:
        cfg.train.captions_enabled = False

    # Optional: AI captioning with CLIP Interrogator
    if Confirm.ask("Use AI captioner (CLIP Interrogator) to generate captions?", default=False):
        preset = Prompt.ask("Preset", choices=["fast", "best"], default="fast")
        overwrite_ai = Confirm.ask("Overwrite existing .txt captions with AI captions?", default=False)
        ai_written, ai_skipped = generate_captions_clip_interrogator(Path(cfg.train.processed_images_dir), preset=preset, overwrite=overwrite_ai)
        if ai_written or ai_skipped:
            cfg.train.captions_enabled = True  # ensure training uses .txt
        print(f"AI captions: Written [green]{ai_written}[/green], Skipped [yellow]{ai_skipped}[/yellow]")

    # Optional: WD14 tagger via kohya_ss
    if Confirm.ask("Use WD14 tagger (kohya_ss) to generate tag-style captions?", default=False):
        repo_id = Prompt.ask(
            "WD14 model repo",
            default="SmilingWolf/wd-v1-4-convnext-tagger-v2",
        )
        bsz = int(Prompt.ask("Batch size", default="2"))
        g_thr = float(Prompt.ask("General threshold", default="0.35"))
        c_thr = float(Prompt.ask("Character threshold", default="0.85"))
        overwrite_wd = Confirm.ask("Overwrite existing .txt captions with WD14 tags?", default=False)
        ok, msg = run_wd14_tagger(
            Path(cfg.paths.kohya_path),
            cfg.paths.venv_python or sys.executable,
            Path(cfg.train.processed_images_dir),
            repo_id=repo_id,
            batch_size=bsz,
            general_threshold=g_thr,
            character_threshold=c_thr,
            recursive=True,
            overwrite=overwrite_wd,
        )
        if ok:
            cfg.train.captions_enabled = True
            print("[green]WD14 tagging completed successfully[/green]")
        else:
            print(f"[red]WD14 tagging failed:[/red] {msg}")

    # Output dir
    default_out_dir = cfg.train.output_dir or str(Path.cwd() / "outputs")
    out_dir_str = Prompt.ask("Output directory for LoRA", default=default_out_dir)
    out_dir = Path(out_dir_str).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.train.output_dir = str(out_dir)

    # Output name
    cfg.train.output_name = Prompt.ask("LoRA output name (no extension)", default=cfg.train.output_name or "my_lora")

    # Base model
    # Offer auto-discovered models from Automatic1111
    discovered = []
    try:
        discovered = discover_pretrained_models(Path(cfg.paths.automatic1111_path)) if cfg.paths.automatic1111_path else []
    except Exception:
        discovered = []
    chosen_model: Optional[str] = None
    if discovered:
        print("Found pretrained models in Automatic1111:")
        for i, p in enumerate(discovered, 1):
            print(f"  [{i}] {p}")
        pick = Prompt.ask("Select a model by number or press Enter to type a custom path", default="")
        if pick.strip().isdigit():
            idx = int(pick.strip())
            if 1 <= idx <= len(discovered):
                chosen_model = str(discovered[idx - 1])
    if not chosen_model:
        default_model = cfg.train.pretrained_model_name_or_path or (
            str(a1111_models_dir(Path(cfg.paths.automatic1111_path))) if cfg.paths.automatic1111_path else ""
        )
        chosen_model = Prompt.ask("Path to pretrained model (ckpt/safetensors)", default=default_model)
    cfg.train.pretrained_model_name_or_path = chosen_model

    # Hyperparameters
    cfg.train.resolution = int(Prompt.ask("Resolution (square)", default=str(cfg.train.resolution)))
    cfg.train.learning_rate = float(Prompt.ask("Learning rate", default=str(cfg.train.learning_rate)))
    te_lr = Prompt.ask("Text encoder LR (optional)", default=str(cfg.train.text_encoder_lr or ""))
    cfg.train.text_encoder_lr = float(te_lr) if te_lr.strip() else None
    unet_lr = Prompt.ask("UNet LR (optional)", default=str(cfg.train.unet_lr or ""))
    cfg.train.unet_lr = float(unet_lr) if unet_lr.strip() else None
    cfg.train.network_dim = int(Prompt.ask("network_dim", default=str(cfg.train.network_dim)))
    cfg.train.network_alpha = int(Prompt.ask("network_alpha", default=str(cfg.train.network_alpha)))
    cfg.train.train_batch_size = int(Prompt.ask("train_batch_size", default=str(cfg.train.train_batch_size)))
    cfg.train.max_train_steps = int(Prompt.ask("max_train_steps", default=str(cfg.train.max_train_steps)))
    cfg.train.save_every_n_steps = int(Prompt.ask("save_every_n_steps", default=str(cfg.train.save_every_n_steps)))
    cfg.train.lr_scheduler = Prompt.ask("LR scheduler", default=cfg.train.lr_scheduler)
    cfg.train.mixed_precision = Prompt.ask("Mixed precision (fp16|bf16|no)", default=cfg.train.mixed_precision)

    # Save config
    save_config(cfg)
    print(f"Config saved to [green]{CONFIG_PATH}[/green]")

    # Build command
    cmd = build_kohya_command(Path(kohya), cfg.paths.venv_python or sys.executable, cfg.train)

    # Show summary
    table = Table(title="Training Summary")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    pairs = {
        "Automatic1111": cfg.paths.automatic1111_path,
        "kohya_ss": cfg.paths.kohya_path,
        "Python": cfg.paths.venv_python,
        "Images": cfg.train.training_images_dir,
        "Processed": cfg.train.processed_images_dir,
        "Output Dir": cfg.train.output_dir,
        "Output Name": cfg.train.output_name,
        "Model": cfg.train.pretrained_model_name_or_path,
        "Resolution": f"{cfg.train.resolution}x{cfg.train.resolution}",
        "LR": str(cfg.train.learning_rate),
        "TE LR": str(cfg.train.text_encoder_lr or "-"),
        "UNet LR": str(cfg.train.unet_lr or "-"),
        "Dim/Alpha": f"{cfg.train.network_dim}/{cfg.train.network_alpha}",
        "Batch": str(cfg.train.train_batch_size),
        "Max Steps": str(cfg.train.max_train_steps),
        "Save Every": str(cfg.train.save_every_n_steps),
        "Scheduler": cfg.train.lr_scheduler,
        "Precision": cfg.train.mixed_precision,
        "Captions": "on" if cfg.train.captions_enabled else "off",
        "Caption Mode": cfg.train.caption_mode or "-",
    }
    for k, v in pairs.items():
        table.add_row(k, str(v))
    console.print(table)

    # Print command
    print("\n[bold]Constructed kohya_ss command:[/bold]")
    print(" ")
    print(" ".join(f'"{c}"' if ' ' in c else c for c in cmd))
    print(" ")

    # Optionally run
    if Confirm.ask("Run training now?", default=False):
        try:
            proc = subprocess.Popen(cmd, cwd=str(Path(kohya).resolve()))
            print("Training started. Follow logs in this terminal.")
            proc.wait()
            print(f"Process exited with code {proc.returncode}")
        except FileNotFoundError as e:
            print(f"[red]Failed to start process:[/red] {e}")
            sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
