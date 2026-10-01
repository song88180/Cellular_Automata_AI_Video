"""Ordered list of chapters."""
from . import (
    chapter1_intro,
    chapter2_elementary_ca,
    chapter3_game_of_life,
    chapter4_turmites,
    chapter5_langton_loops,
    chapter6_growing_nca,
    chapter7_outro,
)

CHAPTERS = {
    "chapter1": chapter1_intro,
    "chapter2": chapter2_elementary_ca,
    "chapter3": chapter3_game_of_life,
    "chapter4": chapter4_turmites,
    "chapter5": chapter5_langton_loops,
    "chapter6": chapter6_growing_nca,
    "chapter7": chapter7_outro,
}

CHAPTER_ORDER = ["chapter1", "chapter2", "chapter3", "chapter4", "chapter5", "chapter6", "chapter7"]
