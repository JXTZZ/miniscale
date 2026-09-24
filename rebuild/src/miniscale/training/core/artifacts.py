from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from miniscale.integrity import atomic_output_path, atomic_write_text


def append_metric(path: str | Path, metric: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as output:
        output.write(json.dumps(metric, ensure_ascii=False) + "\n")


def truncate_metrics_after(path: str | Path, step: int) -> None:
    target = Path(path)
    if not target.exists():
        return
    retained: list[str] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        try:
            metric = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(metric.get("step", -1)) <= step:
            retained.append(json.dumps(metric, ensure_ascii=False))
    atomic_write_text(target, "".join(f"{line}\n" for line in retained))


def prune_periodic_checkpoints(path: str | Path, keep_last: int) -> None:
    checkpoint_dir = Path(path)
    for checkpoint in sorted(checkpoint_dir.glob("step_*.pt"))[:-keep_last]:
        checkpoint.unlink()


def mirror_checkpoint(source: str | Path, destination: str | Path) -> Path:
    """Atomically expose a second checkpoint name without duplicating disk use when possible."""

    source_path = Path(source)
    target = Path(destination)
    with atomic_output_path(target) as temporary:
        temporary.unlink()
        try:
            os.link(source_path, temporary)
        except OSError:
            shutil.copyfile(source_path, temporary)
    return target
