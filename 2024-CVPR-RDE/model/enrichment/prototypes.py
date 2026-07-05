import torch
import torch.nn.functional as F


LAYOUT_EXTRACTOR_MODES = (
    "global",
    "horizontal",
    "vertical",
    "grid",
)
LEARNED_EXTRACTOR_MODES = (
    "retrieval_backbone",
)
TARGET_RELATIVE_MODES = (
    "cluster",
    "cluster_residual",
    "cluster_density",
    "cluster_rarity",
)
SCALAR_TARGET_RELATIVE_MODES = (
    "cluster_density",
    "cluster_rarity",
)
EXTRACTOR_MODES = LAYOUT_EXTRACTOR_MODES + LEARNED_EXTRACTOR_MODES + TARGET_RELATIVE_MODES

_LEGACY_EXTRACTOR_MODE_ALIASES = {
    "global_horizontal": "global,horizontal",
    "global_vertical": "global,vertical",
    "global_grid": "global,grid",
}


def _normalize_extractor_modes(mode):
    if isinstance(mode, str):
        raw_tokens = [token.strip().lower() for token in mode.split(",") if token.strip()]
    elif isinstance(mode, (list, tuple)):
        raw_tokens = [str(token).strip().lower() for token in mode if str(token).strip()]
    else:
        raise ValueError("--extractor_mode must be a comma-separated string or sequence of mode names")

    expanded_tokens = []
    for token in raw_tokens:
        alias = _LEGACY_EXTRACTOR_MODE_ALIASES.get(token)
        if alias is not None:
            expanded_tokens.extend(alias.split(","))
        else:
            expanded_tokens.append(token)

    modes = []
    seen = set()
    for token in expanded_tokens:
        if token not in EXTRACTOR_MODES:
            raise ValueError(
                f"--extractor_mode supports comma-separated values from {EXTRACTOR_MODES}, got {mode}"
            )
        if token not in seen:
            seen.add(token)
            modes.append(token)

    if not modes:
        raise ValueError("--extractor_mode must contain at least one mode")
    return tuple(modes)


def canonicalize_extractor_mode(mode):
    return ",".join(_normalize_extractor_modes(mode))


def prototype_slot_count(mode, num_parts):
    modes = _normalize_extractor_modes(mode)
    if num_parts < 1:
        raise ValueError("--num_parts must be a positive integer")

    slots = 0
    for extractor in modes:
        if extractor in ("global", "retrieval_backbone", *TARGET_RELATIVE_MODES):
            slots += 1
        elif extractor == "grid":
            slots += num_parts * num_parts
        else:
            slots += num_parts
    return slots


def has_target_relative_modes(mode):
    return any(extractor in TARGET_RELATIVE_MODES for extractor in _normalize_extractor_modes(mode))


def _slot_count_for_mode(extractor, num_parts):
    if extractor in ("global", "retrieval_backbone", *TARGET_RELATIVE_MODES):
        return 1
    if extractor == "grid":
        return num_parts * num_parts
    return num_parts


def evidence_slot_indices(mode, num_parts):
    modes = _normalize_extractor_modes(mode)
    if num_parts < 1:
        raise ValueError("--num_parts must be a positive integer")

    indices = {}
    cursor = 0
    for extractor in modes:
        count = _slot_count_for_mode(extractor, num_parts)
        indices[extractor] = tuple(range(cursor, cursor + count))
        cursor += count
    return indices


def _balanced_bounds(size, num_parts, device):
    boundaries = torch.linspace(0, size, steps=num_parts + 1, device=device)
    boundaries = boundaries.round().long().tolist()
    bounds = []
    for part_idx in range(num_parts):
        start = boundaries[part_idx]
        end = boundaries[part_idx + 1]
        if end <= start:
            end = min(start + 1, size)
            start = max(0, end - 1)
        bounds.append((start, end))
    return bounds


def _resolve_patch_grid(patch_features, grid_size, modes):
    batch_size, num_patches, dim = patch_features.shape
    if grid_size is not None:
        grid_h, grid_w = grid_size
        if grid_h * grid_w == num_patches:
            return patch_features.reshape(batch_size, grid_h, grid_w, dim)

    requires_grid = any(extractor in ("vertical", "grid") for extractor in modes)
    if requires_grid:
        raise ValueError(
            f"--extractor_mode {','.join(modes)} requires a valid patch grid_size "
            "for vertical/grid extractors"
        )
    return None


def _horizontal_prototypes(patch_features, patch_grid, num_parts):
    parts = []
    device = patch_features.device
    if patch_grid is not None:
        grid_h = patch_grid.shape[1]
        for start, end in _balanced_bounds(grid_h, num_parts, device):
            part = patch_grid[:, start:end, :, :].mean(dim=(1, 2))
            parts.append(F.normalize(part, p=2, dim=-1))
        return parts

    num_patches = patch_features.shape[1]
    for start, end in _balanced_bounds(num_patches, num_parts, device):
        part = patch_features[:, start:end, :].mean(dim=1)
        parts.append(F.normalize(part, p=2, dim=-1))
    return parts


def _vertical_prototypes(patch_grid, num_parts):
    parts = []
    grid_w = patch_grid.shape[2]
    for start, end in _balanced_bounds(grid_w, num_parts, patch_grid.device):
        part = patch_grid[:, :, start:end, :].mean(dim=(1, 2))
        parts.append(F.normalize(part, p=2, dim=-1))
    return parts


def _grid_prototypes(patch_grid, num_parts):
    parts = []
    row_bounds = _balanced_bounds(patch_grid.shape[1], num_parts, patch_grid.device)
    col_bounds = _balanced_bounds(patch_grid.shape[2], num_parts, patch_grid.device)
    for row_start, row_end in row_bounds:
        for col_start, col_end in col_bounds:
            part = patch_grid[:, row_start:row_end, col_start:col_end, :].mean(dim=(1, 2))
            parts.append(F.normalize(part, p=2, dim=-1))
    return parts


def build_part_prototypes(token_features, num_parts, grid_size=None, mode="global,horizontal"):
    modes = _normalize_extractor_modes(mode)
    if num_parts < 1:
        raise ValueError("--num_parts must be a positive integer")
    unsupported = [extractor for extractor in modes if extractor not in LAYOUT_EXTRACTOR_MODES]
    if unsupported:
        raise ValueError(
            "build_part_prototypes only supports layout/global modes; "
            f"got {unsupported}"
        )

    token_features = token_features.float()
    global_feature = F.normalize(token_features[:, 0, :], p=2, dim=-1)
    if modes == ("global",):
        return global_feature.unsqueeze(1)

    patch_features = token_features[:, 1:, :]
    patch_grid = _resolve_patch_grid(patch_features, grid_size, modes)

    parts = []
    for extractor in modes:
        if extractor == "global":
            parts.append(global_feature)
        elif extractor == "horizontal":
            parts.extend(_horizontal_prototypes(patch_features, patch_grid, num_parts))
        elif extractor == "vertical":
            parts.extend(_vertical_prototypes(patch_grid, num_parts))
        else:
            parts.extend(_grid_prototypes(patch_grid, num_parts))

    return torch.stack(parts, dim=1)


def _zero_token(batch_size, dim, device, dtype):
    return torch.zeros(batch_size, dim, device=device, dtype=dtype)


def build_evidence_bank(
    token_features,
    num_parts,
    grid_size=None,
    mode="global,horizontal",
    retrieval_features=None,
):
    modes = _normalize_extractor_modes(mode)
    if num_parts < 1:
        raise ValueError("--num_parts must be a positive integer")

    token_features = token_features.float()
    batch_size, _, evidence_dim = token_features.shape
    device = token_features.device
    dtype = token_features.dtype
    global_feature = F.normalize(token_features[:, 0, :], p=2, dim=-1)
    patch_features = token_features[:, 1:, :]
    patch_grid = _resolve_patch_grid(patch_features, grid_size, modes)

    parts = []
    for extractor in modes:
        if extractor == "global":
            parts.append(global_feature)
        elif extractor == "horizontal":
            parts.extend(_horizontal_prototypes(patch_features, patch_grid, num_parts))
        elif extractor == "vertical":
            parts.extend(_vertical_prototypes(patch_grid, num_parts))
        elif extractor == "grid":
            parts.extend(_grid_prototypes(patch_grid, num_parts))
        elif extractor == "retrieval_backbone":
            if retrieval_features is not None and retrieval_features.shape[0] != batch_size:
                raise ValueError("retrieval_features batch size must match token_features")
            if retrieval_features is not None and retrieval_features.shape[-1] == evidence_dim:
                parts.append(F.normalize(retrieval_features.float(), p=2, dim=-1))
            else:
                parts.append(_zero_token(batch_size, evidence_dim, device, dtype))
        elif extractor in TARGET_RELATIVE_MODES:
            parts.append(_zero_token(batch_size, evidence_dim, device, dtype))
        else:
            raise ValueError(
                f"--extractor_mode supports comma-separated values from {EXTRACTOR_MODES}, got {mode}"
            )

    return torch.stack(parts, dim=1)


def _target_relative_features(cache, space):
    if space == "host_global":
        key = "host_image_features"
    elif space == "retrieval":
        key = "retrieval_features"
    else:
        raise ValueError("--target_relative_space must be either 'host_global' or 'retrieval'")
    if key not in cache:
        raise ValueError(f"Target-relative evidence requires pool cache key '{key}'")
    return F.normalize(cache[key].float(), p=2, dim=-1)


def _cluster_features(features, num_clusters, iterations=10):
    if features.dim() != 2:
        raise ValueError("Target-relative feature bank must have shape [K, d]")
    pool_size = features.shape[0]
    if pool_size < 1:
        raise ValueError("Cannot build target-relative evidence from an empty target pool")
    clusters = min(max(1, int(num_clusters)), pool_size)
    if clusters == pool_size:
        assignments = torch.arange(pool_size, device=features.device)
        centroids = features
        counts = torch.ones(pool_size, device=features.device, dtype=torch.long)
        return assignments, centroids, counts

    init = torch.linspace(0, pool_size - 1, steps=clusters, device=features.device)
    centroids = features.index_select(0, init.round().long()).clone()
    assignments = None
    for _ in range(max(1, int(iterations))):
        scores = features @ centroids.t()
        next_assignments = scores.argmax(dim=1)
        if assignments is not None and torch.equal(next_assignments, assignments):
            break
        assignments = next_assignments
        updated = []
        for cluster_idx in range(clusters):
            mask = assignments.eq(cluster_idx)
            if mask.any():
                center = features[mask].mean(dim=0)
                updated.append(F.normalize(center, p=2, dim=0))
            else:
                updated.append(centroids[cluster_idx])
        centroids = torch.stack(updated, dim=0)

    counts = torch.bincount(assignments, minlength=clusters).long()
    return assignments, centroids, counts


def _standardize_scalar(values):
    values = values.float()
    if values.numel() <= 1:
        return values - values.mean()
    std = values.std(unbiased=False)
    return (values - values.mean()) / std.clamp_min(1e-6)


def finalize_target_evidence_cache(cache, args, evidence_dim):
    if "evidence_bank" not in cache:
        if "prototypes" not in cache:
            return cache
        cache["evidence_bank"] = cache["prototypes"]

    cache["prototypes"] = cache["evidence_bank"]
    mode = getattr(args, "extractor_mode", "global,horizontal")
    if not has_target_relative_modes(mode):
        return cache

    cluster_method = getattr(args, "target_relative_cluster_method", "kmeans")
    if cluster_method != "kmeans":
        raise ValueError("--target_relative_cluster_method currently supports only 'kmeans'")

    num_parts = getattr(args, "num_parts", 6)
    slots = evidence_slot_indices(mode, num_parts)
    features = _target_relative_features(
        cache,
        getattr(args, "target_relative_space", "host_global"),
    )
    assignments, centroids, counts = _cluster_features(
        features,
        getattr(args, "target_relative_num_clusters", 16),
    )

    evidence_bank = cache["evidence_bank"].float()
    assigned_centroids = centroids.index_select(0, assignments)
    assigned_counts = counts.index_select(0, assignments)
    pool_size = features.shape[0]

    def write_vector(mode_name, values):
        if mode_name not in slots:
            return
        slot = slots[mode_name][0]
        values = F.normalize(values.float(), p=2, dim=-1)
        if values.shape[-1] == evidence_dim:
            evidence_bank[:, slot, :] = values.to(
                device=evidence_bank.device,
                dtype=evidence_bank.dtype,
            )
        else:
            if getattr(args, "evidence_projection", "auto") == "none":
                raise ValueError(
                    f"--extractor_mode {mode_name} requires evidence projection when "
                    "the target-relative feature dim differs from the shared evidence dim"
                )
            cache[f"{mode_name}_features"] = values

    write_vector("cluster", assigned_centroids)
    residual = features - assigned_centroids
    zero_safe_residual = torch.where(
        residual.norm(dim=-1, keepdim=True).gt(1e-8),
        residual,
        torch.zeros_like(residual),
    )
    write_vector("cluster_residual", zero_safe_residual)

    density = assigned_counts.float() / float(pool_size)
    density_scalar = _standardize_scalar(torch.log(density.clamp_min(1e-8)))
    rarity_scalar = _standardize_scalar(-torch.log(density.clamp_min(1e-8)))
    if "cluster_density" in slots:
        cache["cluster_density_scalar"] = density_scalar.view(-1, 1)
    if "cluster_rarity" in slots:
        cache["cluster_rarity_scalar"] = rarity_scalar.view(-1, 1)

    cache["target_relative_cluster_ids"] = assignments.long()
    cache["target_relative_cluster_counts"] = assigned_counts.long()
    cache["evidence_bank"] = evidence_bank
    cache["prototypes"] = evidence_bank
    return cache
