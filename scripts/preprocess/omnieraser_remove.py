from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    args = parse_args()
    config = load_yaml(resolve_project_path(args.path_config))

    omnieraser_root = resolve_project_path(
        args.omnieraser_root or require_path(config, "third_party.Omnieraser_root")
    )
    lora_checkpoint = resolve_project_path(
        args.lora_checkpoint or require_path(config, "third_party.Omnieraser_checkpoint")
    )
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(args)
    if not pairs:
        raise ValueError("No input image/mask pairs were found.")

    inpaint_pairs = prepare_output_placeholders(pairs, output_dir, args) if args.skip_empty_masks else pairs
    if not inpaint_pairs:
        print("No non-empty masks found; copied all frames without loading OmniEraser.")
        return

    controlnet_dir = omnieraser_root / "ControlNet_version"
    sys.path.insert(0, str(controlnet_dir))

    # These imports intentionally happen after sys.path setup because the third-party
    # pipeline uses local sibling imports such as transformer_flux.
    from controlnet_flux import FluxControlNetModel  # noqa: PLC0415
    from pipeline_flux_controlnet_removal import FluxControlNetInpaintingPipeline  # noqa: PLC0415
    from transformer_flux import FluxTransformer2DModel  # noqa: PLC0415

    pipe = build_pipeline(
        base_model=args.base_model,
        controlnet_model=args.controlnet_model,
        lora_checkpoint=lora_checkpoint,
        device=args.device,
        controlnet_cls=FluxControlNetModel,
        transformer_cls=FluxTransformer2DModel,
        pipeline_cls=FluxControlNetInpaintingPipeline,
    )

    import torch

    for image_path, mask_path in inpaint_pairs:
        image, mask, width, height = load_inputs(
            image_path,
            mask_path,
            resolution=args.resolution,
            invert_mask=args.invert_mask,
        )
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        result = pipe(
            prompt=args.prompt,
            height=height,
            width=width,
            control_image=image,
            control_mask=mask,
            num_inference_steps=args.steps,
            true_guidance_scale=args.true_guidance_scale,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
            generator=generator,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        ).images[0]

        output_path = make_output_path(image_path, output_dir, args.suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove masked people/objects with OmniEraser ControlNet FLUX."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="Single RGB image path.")
    input_group.add_argument("--image-dir", help="Directory of RGB images.")
    parser.add_argument("--mask", help="Single mask path for --image.")
    parser.add_argument("--mask-dir", help="Directory of masks for --image-dir.")
    parser.add_argument("--output-dir", default="outputs/preprocess/omnieraser")
    parser.add_argument("--suffix", default="_omnieraser")
    parser.add_argument("--path-config", default="configs/path.yaml")
    parser.add_argument("--omnieraser-root", default="")
    parser.add_argument("--lora-checkpoint", default="")
    parser.add_argument("--base-model", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument(
        "--controlnet-model",
        default="alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta",
    )
    parser.add_argument("--prompt", default="There is nothing here.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--true-guidance-scale", type=float, default=1.0)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-empty-masks",
        action="store_true",
        help="Copy frames with empty masks to the output without loading/running OmniEraser for them.",
    )
    parser.add_argument(
        "--invert-mask",
        action="store_true",
        help="Use this when black pixels mark the region to remove.",
    )
    args = parser.parse_args()

    if args.image and not args.mask:
        parser.error("--mask is required with --image.")
    if args.image_dir and not args.mask_dir:
        parser.error("--mask-dir is required with --image-dir.")
    return args


def build_pipeline(
    *,
    base_model: str,
    controlnet_model: str,
    lora_checkpoint: Path,
    device: str,
    controlnet_cls: Any,
    transformer_cls: Any,
    pipeline_cls: Any,
) -> Any:
    import torch

    if not lora_checkpoint.is_file():
        raise FileNotFoundError(f"Missing OmniEraser LoRA checkpoint: {lora_checkpoint}")

    controlnet = controlnet_cls.from_pretrained(
        controlnet_model,
        torch_dtype=torch.bfloat16,
    )
    transformer = transformer_cls.from_pretrained(
        base_model,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    pipe = pipeline_cls.from_pretrained(
        base_model,
        controlnet=controlnet,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to(device)
    pipe.load_lora_weights(str(lora_checkpoint.parent), weight_name=lora_checkpoint.name)
    pipe.transformer.to(torch.bfloat16)
    pipe.controlnet.to(torch.bfloat16)
    return pipe


def load_inputs(
    image_path: Path,
    mask_path: Path,
    *,
    resolution: int,
    invert_mask: bool,
) -> tuple[Image.Image, Image.Image, int, int]:
    from PIL import Image

    image = open_srgb(image_path)
    mask = Image.open(mask_path)
    if mask.mode == "RGBA":
        mask = mask.getchannel("A")
    else:
        mask = mask.convert("L")
    if invert_mask:
        import numpy as np

        mask = Image.fromarray(255 - np.array(mask, dtype=np.uint8))

    width, height = image.size
    if width < height:
        new_width = resolution
        new_height = int(height / width * resolution)
    else:
        new_width = int(width / height * resolution)
        new_height = resolution

    new_width = max(16, new_width - new_width % 16)
    new_height = max(16, new_height - new_height % 16)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST).convert("RGB")
    return image, mask, new_width, new_height


def open_srgb(path: Path) -> Image.Image:
    from PIL import Image, ImageCms

    image = Image.open(path).convert("RGB")
    icc_profile = image.info.get("icc_profile")
    if not icc_profile:
        return image
    import io

    srgb_profile = ImageCms.createProfile("sRGB")
    src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
    converted = ImageCms.profileToProfile(image, src_profile, srgb_profile)
    converted.info.pop("icc_profile", None)
    return converted.convert("RGB")


def collect_pairs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.image:
        return [(resolve_project_path(args.image), resolve_project_path(args.mask))]

    image_dir = resolve_project_path(args.image_dir)
    mask_dir = resolve_project_path(args.mask_dir)
    mask_by_stem = {
        path.stem: path
        for path in sorted(mask_dir.iterdir())
        if path.suffix.lower() in image_suffixes()
    }
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in image_suffixes():
            continue
        mask_path = mask_by_stem.get(image_path.stem)
        if mask_path is None:
            print(f"Skip {image_path}: no same-stem mask in {mask_dir}", file=sys.stderr)
            continue
        pairs.append((image_path, mask_path))
    return pairs


def prepare_output_placeholders(
    pairs: list[tuple[Path, Path]],
    output_dir: Path,
    args: argparse.Namespace,
) -> list[tuple[Path, Path]]:
    inpaint_pairs: list[tuple[Path, Path]] = []
    for image_path, mask_path in pairs:
        if mask_has_foreground(mask_path, invert_mask=bool(args.invert_mask)):
            inpaint_pairs.append((image_path, mask_path))
            continue
        output_path = make_output_path(image_path, output_dir, args.suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == image_path.suffix.lower():
            shutil.copy2(image_path, output_path)
        else:
            from PIL import Image

            Image.open(image_path).convert("RGB").save(output_path)
        print(f"Copied unchanged {output_path}")
    return inpaint_pairs


def mask_has_foreground(mask_path: Path, *, invert_mask: bool) -> bool:
    import numpy as np
    from PIL import Image

    mask = Image.open(mask_path)
    if mask.mode == "RGBA":
        arr = np.array(mask.getchannel("A"), dtype=np.uint8)
    else:
        arr = np.array(mask.convert("L"), dtype=np.uint8)
    if invert_mask:
        return bool((arr < 255).any())
    return bool((arr > 0).any())


def make_output_path(image_path: Path, output_dir: Path, suffix: str) -> Path:
    return output_dir / f"{image_path.stem}{suffix}.png"


def image_suffixes() -> set[str]:
    return {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def require_path(config: dict[str, Any], dotted_key: str) -> str:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Missing path config key: {dotted_key}")
        current = current[part]
    if not isinstance(current, str) or not current:
        raise ValueError(f"Invalid path config key: {dotted_key}")
    return current


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    main()
