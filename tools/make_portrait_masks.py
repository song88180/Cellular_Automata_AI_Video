#!/usr/bin/env python3
"""Dev-only: cut the people out of the portrait photos (rembg person
segmentation) and store the alpha masks as assets/images/<name>_mask.png.
Playback only reads the PNGs, so rembg is never a runtime dependency.

  .venv/bin/python tools/make_portrait_masks.py
"""
from pathlib import Path

from PIL import Image
from rembg import new_session, remove

IMAGES = Path(__file__).resolve().parent.parent / "assets" / "images"
PEOPLE = ("neumann", "conway", "wolfram", "langton", "conus")


def main() -> None:
    # BiRefNet-portrait keeps hair and patterned clothing (u2net_human_seg
    # punched holes in Langton's plaid shirt).
    session = new_session("birefnet-portrait")
    for name in PEOPLE:
        img = Image.open(IMAGES / f"{name}.jpg").convert("RGB")
        k = max(1, 1024 // max(img.size))  # upscale small photos before segmenting
        big = img.resize((img.width * k, img.height * k), Image.LANCZOS)
        mask = remove(big, session=session, only_mask=True, post_process_mask=True)
        mask = mask.convert("L").resize(img.size, Image.LANCZOS)
        mask.save(IMAGES / f"{name}_mask.png")
        print("wrote", IMAGES / f"{name}_mask.png", mask.size)


if __name__ == "__main__":
    main()
