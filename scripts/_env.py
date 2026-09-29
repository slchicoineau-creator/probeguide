"""Path configuration for the reproduction scripts (import first).
  VILLA_SRC          path to villa/vesuvius/src           (default: ../villa/vesuvius/src next to this repo)
  PROBEGUIDE_ASSETS  folder with the HF checkpoints        (default: <repo>/assets)
  PROBEGUIDE_DATA    working folder for points/, emb/, results (default: <repo>/scripts)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
VILLA_SRC = os.path.abspath(os.environ.get("VILLA_SRC", os.path.join(ROOT, "..", "villa", "vesuvius", "src")))
ASSETS = os.path.abspath(os.environ.get("PROBEGUIDE_ASSETS", os.path.join(ROOT, "assets")))
DATA = os.path.abspath(os.environ.get("PROBEGUIDE_DATA", HERE))
for p in (ROOT, VILLA_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)
os.makedirs(DATA, exist_ok=True)
os.chdir(DATA)


def backbone_path() -> str:
    """Dinovol teacher backbone under ASSETS, accepting the HF file name (no renaming needed)."""
    for name in ("dinovol_v2_ps8_paris4_step352500_teacher_backbone.pt", "dinovol_teacher_backbone.pt"):
        if os.path.exists(os.path.join(ASSETS, name)):
            return os.path.join(ASSETS, name)
    raise FileNotFoundError(f"put dinovol_v2_ps8_paris4_step352500_teacher_backbone.pt (HF "
                            f"scrollprize/dinovol_v2_ps8_with_paris4_352500) in {ASSETS}")
