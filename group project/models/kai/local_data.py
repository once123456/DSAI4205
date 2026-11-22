import csv
import hashlib
import io
import os
from pathlib import Path


def processed_directory(start=None):
    override = os.environ.get("DSAI4205_DATA_DIR")
    if override:
        return Path(override).expanduser()
    start = Path(start or Path.cwd()).resolve()
    for base in (start, *start.parents):
        for folder in (base / "data/processed", base / "group project/data/processed"):
            if (folder / "train_cleaned.csv").is_file():
                return folder
    raise FileNotFoundError("Set DSAI4205_DATA_DIR to the folder containing the course cleaned CSV files.")


def read_projected_data(split, manifest, start=None):
    path = processed_directory(start) / (split + "_cleaned.csv")
    columns = ["row_id", "Phrase", "Sentiment", "text_clean", "cross_split_overlap"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        writer.writerows(csv.DictReader(stream))
    raw = buffer.getvalue().encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != manifest[split]["sha256"]:
        raise ValueError(f"Dataset fingerprint mismatch for {split}")
    return raw
