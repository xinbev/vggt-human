from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "outputs/vis/tsmr_paper_main_figure/tsmr_method_overview_v1.png"
OUT = ROOT / "outputs/vis/tsmr_paper_main_figure/assets/simple_tsmr"

source = Image.open(SOURCE).convert("RGB")
OUT.mkdir(parents=True, exist_ok=True)

crops = {
    "tracked_rgb.png": (35, 90, 300, 245),
    "aligned_human_scene.png": (35, 250, 300, 610),
    "support_surface.png": (410, 180, 570, 530),
    "body_probes.png": (445, 535, 568, 733),
    "selector_tokens.png": (800, 185, 950, 415),
    "grounded_before_after.png": (1260, 120, 1532, 575),
}

for name, box in crops.items():
    source.crop(box).save(OUT / name)

print(OUT)
