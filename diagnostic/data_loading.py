"""Dataset split loading and normalized query/gallery records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Tuple

import numpy as np
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class QueryRecord:
    query_id: int
    text: str
    pid: int


@dataclass(frozen=True)
class GalleryRecord:
    image_id: int
    path: str
    pid: int


@dataclass
class SplitData:
    dataset: Any
    query_records: list[QueryRecord]
    gallery_records: list[GalleryRecord]
    query_pids: np.ndarray
    gallery_pids: np.ndarray
    gallery_paths: list[str]
    img_loader: DataLoader
    txt_loader: DataLoader
    num_classes: int


@dataclass
class SplitMetadata:
    query_records: list[QueryRecord]
    gallery_records: list[GalleryRecord]
    query_pids: np.ndarray
    gallery_pids: np.ndarray
    gallery_paths: list[str]


def load_split_metadata(repo_args: SimpleNamespace, split: str) -> SplitMetadata:
    dataset_name = repo_args.dataset_name
    root_dir = Path(repo_args.root_dir)
    if dataset_name == "RSTPReid":
        dataset_dir = root_dir / "RSTPReid"
        anno_path = dataset_dir / "data_captions.json"
        image_key = "img_path"
    elif dataset_name == "CUHK-PEDES":
        dataset_dir = root_dir / "CUHK-PEDES"
        anno_path = dataset_dir / "reid_raw.json"
        image_key = "file_path"
    elif dataset_name == "ICFG-PEDES":
        dataset_dir = root_dir / "ICFG-PEDES"
        anno_path = dataset_dir / "ICFG-PEDES.json"
        image_key = "file_path"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if not anno_path.exists():
        raise FileNotFoundError(f"Dataset annotation file not found: {anno_path}")

    with anno_path.open("r", encoding="utf-8") as handle:
        annos = json.load(handle)
    query_records: list[QueryRecord] = []
    gallery_records: list[GalleryRecord] = []
    query_id = 0
    for image_id, anno in enumerate(annos):
        if anno.get("split") != split:
            continue
        pid = int(anno["id"])
        path = str(dataset_dir / "imgs" / anno[image_key])
        gallery_records.append(GalleryRecord(image_id=len(gallery_records), path=path, pid=pid))
        for caption in anno.get("captions", []):
            query_records.append(QueryRecord(query_id=query_id, text=str(caption), pid=pid))
            query_id += 1
    if not query_records or not gallery_records:
        raise RuntimeError(f"No records found for {dataset_name} split {split}")
    return SplitMetadata(
        query_records=query_records,
        gallery_records=gallery_records,
        query_pids=np.asarray([record.pid for record in query_records], dtype=np.int64),
        gallery_pids=np.asarray([record.pid for record in gallery_records], dtype=np.int64),
        gallery_paths=[record.path for record in gallery_records],
    )


def load_split(repo_args: SimpleNamespace, split: str) -> SplitData:
    from datasets.bases import ImageDataset, TextDataset
    from datasets.build import build_transforms
    from datasets.cuhkpedes import CUHKPEDES
    from datasets.icfgpedes import ICFGPEDES
    from datasets.rstpreid import RSTPReid

    factories = {
        "CUHK-PEDES": CUHKPEDES,
        "ICFG-PEDES": ICFGPEDES,
        "RSTPReid": RSTPReid,
    }
    if repo_args.dataset_name not in factories:
        raise ValueError(f"Unsupported dataset: {repo_args.dataset_name}")
    dataset = factories[repo_args.dataset_name](root=repo_args.root_dir)
    if not hasattr(dataset, split):
        raise ValueError(f"Dataset {repo_args.dataset_name} has no split '{split}'")
    split_ds = getattr(dataset, split)

    transform = build_transforms(img_size=repo_args.img_size, is_train=False)
    img_set = ImageDataset(split_ds["image_pids"], split_ds["img_paths"], transform)
    txt_set = TextDataset(
        split_ds["caption_pids"],
        split_ds["captions"],
        text_length=repo_args.text_length,
    )
    img_loader = DataLoader(
        img_set,
        batch_size=repo_args.test_batch_size,
        shuffle=False,
        num_workers=repo_args.num_workers,
    )
    txt_loader = DataLoader(
        txt_set,
        batch_size=repo_args.test_batch_size,
        shuffle=False,
        num_workers=repo_args.num_workers,
    )

    query_records = [
        QueryRecord(query_id=index, text=str(text), pid=int(pid))
        for index, (pid, text) in enumerate(zip(txt_set.caption_pids, txt_set.captions))
    ]
    gallery_records = [
        GalleryRecord(image_id=index, path=str(Path(path)), pid=int(pid))
        for index, (pid, path) in enumerate(zip(img_set.image_pids, img_set.img_paths))
    ]
    if not query_records or not gallery_records:
        raise RuntimeError(f"No records found for {repo_args.dataset_name} split {split}")

    return SplitData(
        dataset=dataset,
        query_records=query_records,
        gallery_records=gallery_records,
        query_pids=np.asarray([record.pid for record in query_records], dtype=np.int64),
        gallery_pids=np.asarray([record.pid for record in gallery_records], dtype=np.int64),
        gallery_paths=[record.path for record in gallery_records],
        img_loader=img_loader,
        txt_loader=txt_loader,
        num_classes=len(dataset.train_id_container),
    )
