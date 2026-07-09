"""Off-the-shelf CLIP cue scorer for cue construction and Cue Shift."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diagnostic.constants import DEFAULT_PROMPT_TEMPLATES


@dataclass
class CueScorerOutput:
    gallery_features: torch.Tensor
    cue_features: dict[str, torch.Tensor]
    affinities: dict[str, np.ndarray]
    prompts_by_cue: dict[str, list[str]]

    @property
    def scores(self) -> dict[str, np.ndarray]:
        return self.affinities

    @property
    def prompts(self) -> dict[str, list[str]]:
        return self.prompts_by_cue

    @property
    def image_features_shape(self) -> tuple[int, int]:
        return (int(self.gallery_features.shape[0]), int(self.gallery_features.shape[1]))


class OffTheShelfCLIPCueScorer:
    """Frozen CLIP scorer loaded without TBPS fine-tuned weights."""

    def __init__(
        self,
        model_name: str,
        repo_args: SimpleNamespace,
        device: torch.device,
        logger: logging.Logger,
        prompt_templates: Sequence[str] = DEFAULT_PROMPT_TEMPLATES,
    ) -> None:
        from diagnostic.prototype_clip_model import build_CLIP_from_openai_pretrained

        self.model_name = model_name
        self.device = device
        self.prompt_templates = tuple(prompt_templates)
        logger.info("Loading off-the-shelf CLIP cue scorer: %s", model_name)
        try:
            self.model, self.base_cfg = build_CLIP_from_openai_pretrained(
                model_name,
                repo_args.img_size,
                repo_args.stride_size,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load off-the-shelf CLIP cue scorer. "
                "If the model weights are not cached, this may require network access."
            ) from exc
        if device.type == "cpu":
            self.model = self.model.float()
        self.model.to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _encode_text_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        x, _ = self.model.encode_text(tokens.long().to(self.device))
        features = x[torch.arange(x.shape[0], device=x.device), tokens.to(self.device).argmax(dim=-1)]
        return F.normalize(features.float(), p=2, dim=1)

    def encode_cues(self, cues: Sequence[str], text_length: int) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
        from datasets.bases import tokenize
        from utils.simple_tokenizer import SimpleTokenizer

        tokenizer = SimpleTokenizer()
        cue_features: dict[str, torch.Tensor] = {}
        prompts_by_cue: dict[str, list[str]] = {}
        with torch.no_grad():
            for cue in cues:
                prompts = [template.format(cue=cue) for template in self.prompt_templates]
                tokens = torch.stack(
                    [tokenize(prompt, tokenizer=tokenizer, text_length=text_length) for prompt in prompts],
                    dim=0,
                )
                prompt_features = self._encode_text_tokens(tokens)
                cue_features[cue] = F.normalize(prompt_features.mean(dim=0, keepdim=True), p=2, dim=1)[0].cpu()
                prompts_by_cue[cue] = prompts
        return cue_features, prompts_by_cue

    def encode_gallery_images(self, img_loader: DataLoader, logger: logging.Logger) -> torch.Tensor:
        features = []
        logger.info("Encoding %d gallery images with off-the-shelf CLIP cue scorer", len(img_loader.dataset))
        with torch.no_grad():
            for _, images in img_loader:
                x, _ = self.model.encode_image(images.to(self.device))
                image_features = x[:, 0, :].float()
                features.append(F.normalize(image_features, p=2, dim=1).cpu())
        return torch.cat(features, dim=0)

    def score(self, cues: Sequence[str], img_loader: DataLoader, text_length: int, logger: logging.Logger) -> CueScorerOutput:
        gallery_features = self.encode_gallery_images(img_loader, logger)
        cue_features, prompts_by_cue = self.encode_cues(cues, text_length)
        affinities = {
            cue: (gallery_features @ cue_features[cue]).cpu().numpy()
            for cue in cues
        }
        return CueScorerOutput(
            gallery_features=gallery_features,
            cue_features=cue_features,
            affinities=affinities,
            prompts_by_cue=prompts_by_cue,
        )
