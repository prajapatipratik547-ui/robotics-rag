"""The text each chunk is embedded and reranked as.

Kept free of heavy imports so the fine-tuning scripts (train/), which run in
their own environment, format passages exactly as the pipeline does.
"""

import re
from pathlib import Path


def doc_title(source_file: str) -> str:
    """'rplidar_a1m8_datasheet.pdf' -> 'rplidar a1m8 datasheet'."""
    return re.sub(r"[_\-]+", " ", Path(source_file).stem)


def embedding_input(chunk: dict) -> str:
    """What actually gets embedded: the chunk prefixed with its document title
    and section. A datasheet page saying 'Scan frequency: 5.5 Hz' often never
    names the part; the prefix puts 'rplidar a1m8' into both vectors."""
    return f"{doc_title(chunk['source_file'])} | {chunk['section']}\n\n{chunk['text']}"
