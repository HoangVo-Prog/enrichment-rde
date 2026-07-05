"""Embedding extraction helpers for retrievers and cue scorer."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diagnostic.retriever_loading import RetrieverAdapter
from diagnostic.scoring import RetrieverEmbeddingCache


def extract_retriever_embeddings(
    adapter: RetrieverAdapter,
    img_loader: DataLoader,
    txt_loader: DataLoader,
    use_grab: bool,
    logger: logging.Logger,
) -> RetrieverEmbeddingCache:
    query_global = []
    query_grab = []
    gallery_global = []
    gallery_grab = []
    adapter.eval()
    logger.info("Encoding %d text queries with evaluated retriever", len(txt_loader.dataset))
    with torch.no_grad():
        for _, captions in txt_loader:
            feats = adapter.encode_text(captions)
            query_global.append(F.normalize(feats.float(), p=2, dim=1).cpu())
            if use_grab:
                query_grab.append(F.normalize(adapter.encode_text_grab(captions).float(), p=2, dim=1).cpu())
        logger.info("Encoding %d gallery images with evaluated retriever", len(img_loader.dataset))
        for _, images in img_loader:
            feats = adapter.encode_image(images)
            gallery_global.append(F.normalize(feats.float(), p=2, dim=1).cpu())
            if use_grab:
                gallery_grab.append(F.normalize(adapter.encode_image_grab(images).float(), p=2, dim=1).cpu())
    return RetrieverEmbeddingCache(
        query_global=torch.cat(query_global, dim=0),
        gallery_global=torch.cat(gallery_global, dim=0),
        query_grab=torch.cat(query_grab, dim=0) if use_grab else None,
        gallery_grab=torch.cat(gallery_grab, dim=0) if use_grab else None,
    )

