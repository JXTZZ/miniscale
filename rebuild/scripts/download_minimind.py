"""Download the MiniMind training chain and verify upstream checksums.

Run from rebuild/: python scripts/download_minimind.py --profile full
Only Python's standard library is required. Interrupted files use a .part suffix.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

REPOSITORY = "https://www.modelscope.cn/datasets/gongjy/minimind_dataset"
TREE_URL = "https://www.modelscope.cn/api/v1/datasets/gongjy/minimind_dataset/repo/tree"
FILES = {
    "pretrain_t2t_mini.jsonl": "pretrain",
    "pretrain_t2t.jsonl": "pretrain",
    "sft_t2t_mini.jsonl": "sft",
    "sft_t2t.jsonl": "sft",
    "dpo.jsonl": "preference",
    "rlaif.jsonl": "rl",
    "agent_rl_math.jsonl": "agent",
    "agent_rl.jsonl": "agent",
}
COMMON_FILES = ("dpo.jsonl", "rlaif.jsonl", "agent_rl_math.jsonl", "agent_rl.jsonl")
PROFILES = {
    "mini": ("pretrain_t2t_mini.jsonl", "sft_t2t_mini.jsonl", *COMMON_FILES),
    "full": ("pretrain_t2t.jsonl", "sft_t2t.jsonl", *COMMON_FILES),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(metadata: dict, output: Path) -> dict:
    name = metadata["Name"]
    target = output / FILES[name] / name
    target.parent.mkdir(parents=True, exist_ok=True)
    size, expected_hash = metadata["Size"], metadata["Sha256"]
    url = f"{REPOSITORY}/resolve/{metadata['Revision']}/{name}"
    if target.exists():
        if target.stat().st_size != size or sha256(target) != expected_hash:
            raise RuntimeError(f"Existing file does not match upstream: {target}")
        print(f"Verified existing: {name}", flush=True)
    else:
        partial = target.with_suffix(target.suffix + ".part")
        for attempt in range(4):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == size:
                break
            if offset > size:
                raise RuntimeError(f"Partial file is larger than upstream: {partial}")
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                with urlopen(Request(url, headers=headers), timeout=60) as response:
                    append = offset > 0 and response.status == 206
                    if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError(f"Invalid resume response: {name}")
                    done = offset if append else 0
                    last_report = time.monotonic()
                    with partial.open("ab" if append else "wb") as stream:
                        while block := response.read(1024 * 1024):
                            stream.write(block)
                            done += len(block)
                            if time.monotonic() - last_report >= 20:
                                print(f"{name}: {done / size:.1%} ({done / 1e6:.0f}/{size / 1e6:.0f} MB)", flush=True)
                                last_report = time.monotonic()
                if partial.stat().st_size == size:
                    break
                raise RuntimeError(f"Incomplete response: {name}")
            except Exception:
                if attempt == 3:
                    raise
                print(f"Retrying {name} (attempt {attempt + 2}/4)", flush=True)
                time.sleep(2)
        actual_hash = sha256(partial)
        if actual_hash != expected_hash:
            raise RuntimeError(f"SHA-256 mismatch: {partial}; remove the partial file before retrying")
        partial.replace(target)
        print(f"Downloaded and verified: {name} ({size:,} bytes)", flush=True)
    return {
        "path": str(target.relative_to(output)),
        "source": url,
        "revision": metadata["Revision"],
        "bytes": size,
        "sha256": expected_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/raw/minimind")
    parser.add_argument("--profile", choices=PROFILES, default="mini",
                        help="choose full or mini pretrain/SFT files; default: mini")
    args = parser.parse_args()
    with urlopen(TREE_URL + "?" + urlencode({"Revision": "master", "Root": ""}), timeout=30) as response:
        listing = json.load(response)
    available = {entry["Name"]: entry for entry in listing["Data"]["Files"]}
    names = PROFILES[args.profile]
    missing = set(names) - available.keys()
    if missing:
        raise RuntimeError(f"Upstream files missing: {sorted(missing)}")
    selected = [available[name] for name in names]
    print(f"Downloading {len(selected)} files, {sum(item['Size'] for item in selected) / 1e9:.2f} GB", flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda item: download(item, args.output), selected))
    manifest = args.output / "download_manifest.json"
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"repository": REPOSITORY, "files": results}, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest)
    print(f"Manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
