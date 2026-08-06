from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
OVERVIEW = ROOT / "outputs/vis/tsmr_paper_main_figure/tsmr_full_pipeline_abc_paper.png"
DETAIL = ROOT / "outputs/vis/tsmr_paper_main_figure/tsmr_detail_logic_panel.png"
TARGET = ROOT / "outputs/vis/tsmr_paper_main_figure/tsmr_full_method_with_detail.png"

overview = Image.open(OVERVIEW).convert("RGB").crop((0, 0, 2400, 620))
detail = Image.open(DETAIL).convert("RGB").resize((2400, 1170), Image.Resampling.LANCZOS)

canvas = Image.new("RGB", (2400, 1800), "white")
canvas.paste(overview, (0, 0))
canvas.paste(detail, (0, 630))

TARGET.parent.mkdir(parents=True, exist_ok=True)
canvas.save(TARGET, quality=100)
print(TARGET)
