#!/usr/bin/env python3
"""Add content hashes to comment JSON files for sync detection.

Reads config.json, computes SHA-256 hashes for each code line, and
updates the comment files in the mirror structure (comments/<lab_id>/<file>.json).
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = (ROOT / (sys.argv[1] if len(sys.argv) > 1 else "config.json")).resolve()
CONFIG_DIR = CONFIG_PATH.parent
config = json.loads(CONFIG_PATH.read_text())
CODE_DIR = (CONFIG_DIR / config.get("code_dir", ".")).resolve()
COMMENTS_DIR = CONFIG_DIR / "comments"


def line_hash(text):
    return hashlib.sha256(text.strip().encode()).hexdigest()[:8]


for lab in config.get("labs", []):
    lab_id = lab["id"]
    file_list = [f["path"] for f in lab["files"]] if lab.get("files") else [lab["file"]]

    for code_file in file_list:
        code_path = CODE_DIR / lab_id / code_file
        stem = Path(code_file).stem
        exp_path = COMMENTS_DIR / lab_id / f"{stem}.json"

        if not code_path.exists():
            print(f"  {lab_id}/{code_file}: code file not found, skipping")
            continue

        code_lines = code_path.read_text().split("\n")
        exp_path.parent.mkdir(parents=True, exist_ok=True)

        if exp_path.exists():
            explanations = json.loads(exp_path.read_text())
        else:
            explanations = {}
            print(f"  {lab_id}/{code_file}: no comment file, creating empty")

        sample_val = next(iter(explanations.values()), None) if explanations else None
        already_hashed = isinstance(sample_val, dict)

        new_exp = {}
        for line_num_str, entry in explanations.items():
            line_idx = int(line_num_str) - 1
            h = line_hash(code_lines[line_idx]) if 0 <= line_idx < len(code_lines) else ""
            if already_hashed:
                entry["hash"] = h
                new_exp[line_num_str] = entry
            else:
                new_exp[line_num_str] = {"text": entry, "hash": h}

        exp_path.write_text(json.dumps(new_exp, indent=2))
        print(f"  {lab_id}/{code_file}: {len(new_exp)} entries updated")

print("Done!")
