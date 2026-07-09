"""Fixed cue ontology and cue vocabulary loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class CueSpec:
    name: str
    aliases: tuple[str, ...]
    category: str = ""
    region: str = ""
    exclude_aliases: tuple[str, ...] = ()


def _color_aliases(color: str) -> list[str]:
    if color == "black":
        return ["black", "dark"]
    if color == "gray":
        return ["gray", "grey"]
    return [color]


def _garment_aliases(garment: str) -> list[str]:
    if garment == "t-shirt":
        return ["t-shirt", "tshirt", "t shirt"]
    if garment == "pants":
        return ["pants", "trousers"]
    if garment == "sneakers":
        return ["sneakers", "trainers"]
    return [garment]


def _build_colored_cues(colors: Sequence[str], garments: Sequence[str], category: str, region: str) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    seen: set[str] = set()
    for color in colors:
        canonical_color = "gray" if color == "grey" else color
        for garment in garments:
            canonical_garment = "t-shirt" if garment in {"tshirt", "t-shirt"} else garment
            canonical = f"{canonical_color} {canonical_garment}"
            if canonical in seen:
                continue
            seen.add(canonical)
            aliases = sorted(
                {
                    f"{color_alias} {garment_alias}"
                    for color_alias in _color_aliases(canonical_color)
                    for garment_alias in _garment_aliases(canonical_garment)
                }
            )
            cues.append(
                {
                    "name": canonical,
                    "category": category,
                    "region": region,
                    "aliases": aliases,
                    "exclude_aliases": [],
                }
            )
    return cues


def built_in_cue_ontology() -> list[dict[str, Any]]:
    colors = [
        "black",
        "white",
        "red",
        "blue",
        "green",
        "yellow",
        "gray",
        "grey",
        "brown",
        "pink",
        "purple",
        "orange",
    ]
    ontology: list[dict[str, Any]] = []
    ontology.extend(
        _build_colored_cues(
            colors,
            ["jacket", "coat", "hoodie", "shirt", "t-shirt", "sweater", "vest", "uniform", "top"],
            "upper_color_garment",
            "upper_body",
        )
    )
    ontology.extend(
        _build_colored_cues(
            colors,
            ["pants", "trousers", "jeans", "shorts", "skirt", "dress"],
            "lower_color_garment",
            "lower_body",
        )
    )
    ontology.extend(
        _build_colored_cues(
            colors,
            ["shoes", "sneakers", "boots", "sandals"],
            "footwear_color",
            "feet",
        )
    )
    ontology.extend(
        [
            {"name": "bag", "category": "accessory_object", "region": "carried", "aliases": ["bag", "bags", "carrying a bag", "with a bag"], "exclude_aliases": []},
            {"name": "backpack", "category": "accessory_object", "region": "carried", "aliases": ["backpack", "back pack", "rucksack"], "exclude_aliases": []},
            {"name": "handbag", "category": "accessory_object", "region": "carried", "aliases": ["handbag", "hand bag", "purse"], "exclude_aliases": []},
            {"name": "shoulder bag", "category": "accessory_object", "region": "carried", "aliases": ["shoulder bag"], "exclude_aliases": []},
            {"name": "hat", "category": "accessory_object", "region": "head", "aliases": ["hat", "wearing a hat"], "exclude_aliases": []},
            {"name": "cap", "category": "accessory_object", "region": "head", "aliases": ["cap", "baseball cap"], "exclude_aliases": []},
            {"name": "glasses", "category": "accessory_object", "region": "head", "aliases": ["glasses", "sunglasses", "wearing glasses"], "exclude_aliases": []},
            {"name": "umbrella", "category": "accessory_object", "region": "carried", "aliases": ["umbrella"], "exclude_aliases": []},
            {"name": "striped", "category": "pattern", "region": "clothing", "aliases": ["striped", "stripe", "stripes"], "exclude_aliases": []},
            {"name": "plaid", "category": "pattern", "region": "clothing", "aliases": ["plaid"], "exclude_aliases": []},
            {"name": "checked", "category": "pattern", "region": "clothing", "aliases": ["checked", "checkered"], "exclude_aliases": []},
            {"name": "printed", "category": "pattern", "region": "clothing", "aliases": ["printed", "print"], "exclude_aliases": []},
            {"name": "logo", "category": "pattern", "region": "clothing", "aliases": ["logo"], "exclude_aliases": []},
        ]
    )
    return ontology


def normalize_text(text: str) -> str:
    import re

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_normalized_phrase(haystack: str, needle: str) -> bool:
    if not needle:
        return True
    return f" {needle} " in f" {haystack} "


def normalized_unique(values: Iterable[str]) -> tuple[str, ...]:
    normalized = {normalize_text(value) for value in values if isinstance(value, str)}
    normalized.discard("")
    return tuple(sorted(normalized, key=lambda value: (-len(value.split()), -len(value), value)))


def _cue_spec_from_raw(raw: Mapping[str, Any], index: int, source: str) -> CueSpec:
    prefix = f"Cue spec at index {index} in {source}"
    name = raw.get("name")
    aliases = raw.get("aliases")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{prefix} must have a non-empty string 'name'")
    if not isinstance(aliases, list) or not aliases:
        raise ValueError(f"{prefix} must have a non-empty list 'aliases'")
    if not all(isinstance(alias, str) and alias.strip() for alias in aliases):
        raise ValueError(f"{prefix} aliases must be non-empty strings")
    exclude_aliases = raw.get("exclude_aliases", [])
    if not isinstance(exclude_aliases, list):
        raise ValueError(f"{prefix} optional 'exclude_aliases' must be a list")
    category = raw.get("category", "")
    region = raw.get("region", "")
    alias_values = list(aliases)
    if name not in alias_values:
        alias_values.append(name)
    return CueSpec(
        name=name.strip(),
        aliases=tuple(alias.strip() for alias in alias_values),
        category=str(category).strip(),
        region=str(region).strip(),
        exclude_aliases=tuple(str(alias).strip() for alias in exclude_aliases),
    )


def load_cue_specs(cue_vocab_file: Path | None) -> list[CueSpec]:
    if cue_vocab_file is None:
        raw_specs = built_in_cue_ontology()
        source = "builtin"
    else:
        with cue_vocab_file.open("r", encoding="utf-8") as handle:
            raw_specs = json.load(handle)
        source = str(cue_vocab_file)
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError(f"Cue vocabulary {source} must contain a non-empty JSON list")

    specs: list[CueSpec] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_specs):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Cue spec at index {index} in {source} must be an object")
        spec = _cue_spec_from_raw(raw, index, source)
        if spec.name in seen:
            raise ValueError(f"Duplicate cue name '{spec.name}' in {source}")
        seen.add(spec.name)
        specs.append(spec)
    return sorted(specs, key=lambda spec: spec.name)


def detect_cues_in_query(query_text: str, cue_specs: Sequence[CueSpec]) -> list[str]:
    normalized_query = normalize_text(query_text)
    detected: set[str] = set()
    for spec in cue_specs:
        excludes = normalized_unique(spec.exclude_aliases)
        if any(contains_normalized_phrase(normalized_query, alias) for alias in excludes):
            continue
        aliases = normalized_unique(spec.aliases)
        if any(contains_normalized_phrase(normalized_query, alias) for alias in aliases):
            detected.add(spec.name)
    return sorted(detected)

