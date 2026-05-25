#!/usr/bin/env python3
"""Inspect mmap hot-cache residency, faults, and worker RSS."""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from torch.utils.data import DataLoader

from utils.dataset.crop_volume import CropTifVolumeHotDataset
from utils.runtime_factory import load_yaml_config
from utils.sanitize.data_config import CropTifVolumeHotDatasetParams


def _format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"


def _load_dataset(data_config_path: str, section_name: str) -> CropTifVolumeHotDataset:
    config = load_yaml_config(data_config_path)
    section = config.get(section_name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing dataset section: {section_name}")
    params = CropTifVolumeHotDatasetParams.model_validate(section.get("params", {}))
    return CropTifVolumeHotDataset(params)


def _parse_kb_field(path: Path, key: str) -> int:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}:"):
            return int(line.split()[1]) * 1024
    return 0


def _smaps_stats_for_file(pid: int, target_path: Path) -> dict[str, int]:
    stats = {"rss": 0, "pss": 0, "shared_clean": 0, "private_clean": 0, "private_dirty": 0}
    current_path = None
    smaps_path = Path(f"/proc/{pid}/smaps")
    if not smaps_path.exists():
        return stats

    for line in smaps_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line and line[0].isalnum() and "-" in line:
            parts = line.split()
            current_path = parts[-1] if len(parts) >= 6 else None
            continue
        if current_path != str(target_path):
            continue
        for key, stat_name in (
            ("Rss", "rss"),
            ("Pss", "pss"),
            ("Shared_Clean", "shared_clean"),
            ("Private_Clean", "private_clean"),
            ("Private_Dirty", "private_dirty"),
        ):
            if line.startswith(f"{key}:"):
                stats[stat_name] += int(line.split()[1]) * 1024
                break
    return stats


def _smaps_rollup(pid: int) -> dict[str, int]:
    rollup_path = Path(f"/proc/{pid}/smaps_rollup")
    if not rollup_path.exists():
        return {"rss": 0, "pss": 0, "shared_clean": 0, "private_dirty": 0}
    return {
        "rss": _parse_kb_field(rollup_path, "Rss"),
        "pss": _parse_kb_field(rollup_path, "Pss"),
        "shared_clean": _parse_kb_field(rollup_path, "Shared_Clean"),
        "private_dirty": _parse_kb_field(rollup_path, "Private_Dirty"),
    }


def _fault_counts() -> tuple[int, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return int(usage.ru_minflt), int(usage.ru_majflt)


def _print_mapping_stats(pid: int, label: str, mapped_path: Path) -> None:
    proc_stats = _smaps_rollup(pid)
    file_stats = _smaps_stats_for_file(pid, mapped_path)
    resident_ratio = 0.0
    file_size = mapped_path.stat().st_size if mapped_path.exists() else 0
    if file_size > 0:
        resident_ratio = file_stats["rss"] / float(file_size)
    print(
        f"{label}: pid={pid} "
        f"rss={_format_bytes(proc_stats['rss'])} "
        f"pss={_format_bytes(proc_stats['pss'])} "
        f"shared_clean={_format_bytes(proc_stats['shared_clean'])} "
        f"private_dirty={_format_bytes(proc_stats['private_dirty'])}"
    )
    print(
        f"{label}: mapped_file_rss={_format_bytes(file_stats['rss'])} "
        f"mapped_file_shared_clean={_format_bytes(file_stats['shared_clean'])} "
        f"mapped_file_private_dirty={_format_bytes(file_stats['private_dirty'])} "
        f"resident_ratio={resident_ratio:.3f}"
    )


def _benchmark_loader(dataset: CropTifVolumeHotDataset, batch_size: int, num_workers: int, batches: int) -> None:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    iterator = iter(loader)
    worker_pids = [worker.pid for worker in getattr(iterator, "_workers", []) if worker is not None]
    before_faults = _fault_counts()
    start_time = time.perf_counter()
    sample_count = 0

    try:
        for batch_idx, batch in enumerate(iterator):
            sample_count += int(batch["target"].shape[0])
            if batch_idx + 1 >= batches:
                break
        duration = time.perf_counter() - start_time
        after_faults = _fault_counts()

        print(
            f"benchmark: samples={sample_count} batches={batches} "
            f"duration={duration:.3f}s throughput={sample_count / max(duration, 1e-9):.2f} samples/s"
        )
        print(
            f"faults: minor_delta={after_faults[0] - before_faults[0]} "
            f"major_delta={after_faults[1] - before_faults[1]}"
        )
        _print_mapping_stats(os.getpid(), "main", dataset._crops_path())
        for pid in worker_pids:
            _print_mapping_stats(pid, "worker", dataset._crops_path())
    finally:
        if hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect mmap hot-cache residency and worker RSS")
    parser.add_argument("--data-config", type=str, required=True)
    parser.add_argument("--section", type=str, default="train_dataset", choices=("train_dataset", "val_dataset"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batches", type=int, default=16)
    args = parser.parse_args()

    dataset = _load_dataset(args.data_config, args.section)
    print(f"cache_dir={dataset._crop_cache_dir()}")
    print(f"total_crops={len(dataset)}")
    print(f"crops_file={dataset._crops_path()} size={_format_bytes(dataset._crops_path().stat().st_size)}")
    print(f"starts_file={dataset._starts_path()} size={_format_bytes(dataset._starts_path().stat().st_size)}")
    print(f"full_sizes_file={dataset._full_sizes_path()} size={_format_bytes(dataset._full_sizes_path().stat().st_size)}")

    _print_mapping_stats(os.getpid(), "main", dataset._crops_path())
    _benchmark_loader(dataset, args.batch_size, args.num_workers, args.batches)


if __name__ == "__main__":
    main()
