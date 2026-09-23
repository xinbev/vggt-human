from pathlib import Path
from PIL import Image


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Crop the VGGT overview into independently editable PPT components.")
    parser.add_argument("image", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGBA")
    if image.size != (1600, 455):
        raise ValueError(f"Expected the supplied 1600x455 figure, got {image.size}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Pixel rectangles map directly to the 16 x 4.55 inch slide (100 px per inch).
    crops = {
        "left_body": (0, 48, 320, 455),
        "stage01_upper": (320, 62, 830, 250),
        "stage01_lower": (320, 250, 830, 455),
        "stage02_upper": (830, 62, 1320, 250),
        "stage02_lower": (830, 250, 1320, 455),
        "right_body": (1320, 62, 1600, 455),
    }
    for name, box in crops.items():
        image.crop(box).save(args.output_dir / f"{name}.png")


if __name__ == "__main__":
    main()
