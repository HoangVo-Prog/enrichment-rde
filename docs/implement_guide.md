# Target-Aware Text Enrichment Plug-and-Play Implementation Guide

This document is a standalone implementation specification for the target-aware
text-enrichment utilities in this repository. It is written so that an engineer
can reproduce the same behavior in another image-text retrieval project without
reading the source code.

The source implementation is attached to an ITSELF/CLIP-style text-based person
search model, but the method is backbone-agnostic. The only hard requirement is
that the host retrieval model can encode text queries and target images into
compatible feature spaces, and can expose one reusable target-image cache.

The method enriches a text query by looking at the current target image pool.
For each text query, it ranks the full target pool with label-free host scores,
gathers evidence from the top-M ranked images, mixes that evidence with a
query-conditioned rank-slot mixer, and uses the mixed context to apply a
residual update to the query feature. Final retrieval is then ordinary cosine
similarity between the enriched query feature and target image features.

The current implementation intentionally does not include the older sampled
shared-K target pool, adaptive K pool selection, label-forced top-M selection,
robust guard/gain target losses, separately supervised evidence branches, or
checkpointed target caches.

## 1. High-Level Contract

The complete pipeline is:

```text
text query
-> host text feature and optional alternate-space text feature
-> full target-image cache
-> label-free top-M target selection over the full cache
-> gather top-M evidence slot table
-> project evidence into the active enrichment space
-> query-conditioned rank-slot mixer
-> residual query fusion
-> target-pool retrieval loss during training
-> enriched-query similarity against target retrieval features at evaluation
```

Identity labels may be used for:

- target retrieval supervision;
- evaluation metrics;
- diagnostics such as positive-in-top-M rate.

Identity labels must not be used for:

- selecting top-M target images;
- forcing positives into the gathered evidence table;
- constructing query-specific target pools;
- target-relative clustering.

## 2. Notation

Let a training record be:

```text
(x_i, t_i, y_i, image_id_i, query_index_i)
```

where `x_i` is an image, `t_i` is a caption, `y_i` is an identity label,
`image_id_i` is a stable image identifier, and `query_index_i` is the stable
caption/query row index in the dataset.

Let the target image pool be:

```text
G = {x_j}_{j=1..K}
```

During training, `G` is the full set of unique training images, deduplicated by
`image_id`. During evaluation, `G` is the full evaluation gallery.

The host model exposes a global CLIP-style feature space:

```text
h_j = norm(E_I(x_j)) in R^{D_g}
z_q = norm(E_T(t_q)) in R^{D_g}
```

Optionally, the host exposes an alternate retrieval space. In this repository it
is called `grab`; in another project it can be any retrieval-backbone or local
aggregation feature space:

```text
r_j = norm(R_I(x_j)) in R^{D_r}
w_q = norm(R_T(t_q)) in R^{D_r}
```

The active enrichment space is selected by configuration:

```text
if enrichment_space == "global":
    u_q = z_q, a_j = h_j, D_a = D_g

if enrichment_space == "grab":
    u_q = w_q, a_j = r_j, D_a = D_r
```

Here `u_q` is the query feature that will be enriched, and `a_j` is the target
image feature used for target-aware retrieval loss and final target-aware
scoring.

All cosine features are L2-normalized with:

```text
norm(v) = v / max(||v||_2, eps)
```

In the source code, PyTorch `F.normalize(..., p=2, dim=-1)` is used. For loss
temperature, the denominator is clamped as `max(tau, 1e-6)`.

The evidence bank has shape:

```text
B_cache in R^{K x S x D_e}
```

where:

- `K` is the target-pool size;
- `S` is the number of evidence slots implied by `extractor_mode`;
- `D_e` is the shared evidence dimension, equal to the host visual token width in this repository.

After top-M selection and projection into the active space, the mixer input has
shape:

```text
B_q in R^{B x M_actual x S x D_a}
```

where `B` is the query batch size and `M_actual = min(top_m, K)`.

The enriched query feature is:

```text
u_tilde_q in R^{D_a}
```

## 3. Configuration Surface

Use the following options or equivalent config fields. Defaults are the source
implementation defaults.

### 3.1 Core Options

| Option | Default | Allowed values | Meaning |
| --- | ---: | --- | --- |
| `target_enrichment` | `False` | boolean | Build and use the target-aware branch. |
| `enrichment_start` | `1` | integer `>= 1` | First epoch that passes a target cache into training forward. |
| `enrichment_space` | `global` | `global`, `grab` | Active query/image space to enrich. |
| `top_m` | `32` | integer `>= 1` | Number of target-pool images gathered per query. |
| `topm_rank_space` | `host_global` | `host_global`, `retrieval`, `hybrid_global_grab` | Feature space used for top-M selection. |
| `topm_rank_lambda` | `0.5` | float in `[0, 1]` | Global-score weight for hybrid ranking. |
| `lambda_ret` | `1.0` | float `> 0` | Weight on target-pool retrieval loss. |
| `freeze_host` | `False` | boolean | Freeze all non-enrichment parameters. Requires target enrichment. |
| `use_host_loss` | `True` | boolean | Include host losses in optimized objective. |
| `lambda_host` | `1.0` | float | Weight applied to the host loss. |

Recommended frozen plug-and-play baseline for a strong pretrained host:

```text
target_enrichment = true
enrichment_space = global
freeze_host = true
use_host_loss = false
use_freeze_indices = true
pnp_text_only = true
```

### 3.2 Evidence Options

| Option | Default | Allowed values | Meaning |
| --- | ---: | --- | --- |
| `extractor_mode` | `global,horizontal` | comma-separated provider list | Evidence providers and slot order. |
| `num_parts` | `6` | integer `>= 1` | Number of horizontal/vertical parts; grid uses `num_parts x num_parts`. |
| `target_relative_space` | `host_global` | `host_global`, `retrieval` | Feature bank used for label-free target-relative evidence. |
| `target_relative_num_clusters` | `16` | integer `>= 1` | K-means cluster count before clipping to pool size. |
| `target_relative_cluster_method` | `kmeans` | `kmeans` | Current target-relative clustering method. |
| `evidence_token_budget` | `0` | integer `>= 0` | `0` disables the budget check; positive value rejects too many slots. |
| `evidence_projection` | `auto` | `auto`, `linear`, `none` | Policy when evidence source dimension differs from `D_e`. |

Supported evidence providers are:

```text
global
horizontal
vertical
grid
retrieval_backbone
cluster
cluster_residual
cluster_density
cluster_rarity
```

Legacy aliases are accepted and expanded before deduplication:

```text
global_horizontal -> global,horizontal
global_vertical   -> global,vertical
global_grid       -> global,grid
```

### 3.3 Cache and Frozen-Index Options

| Option | Default | Allowed values | Meaning |
| --- | ---: | --- | --- |
| `recompute_level` | `epoch` | `epoch`, `step` | Unit used by `recompute_interval`. |
| `recompute_interval` | `1` | `-1` or integer `>= 1` | `-1` builds once; otherwise refresh every N units. |
| `pool_interval` | alias | integer | Alias for `recompute_interval`. |
| `use_freeze_indices` | `False` | boolean | Precompute top-M rows once and reuse them by query index. |
| `freeze_indices` | alias | boolean | Alias for `use_freeze_indices`. |
| `pnp_text_only` | `False` | boolean | Frozen-host mode that encodes only batch text online. |

`pnp_text_only` requires:

```text
freeze_host = true
use_host_loss = false
use_freeze_indices = true
enrichment_space = global
```

The current frozen-index implementation uses ranking depth:

```text
rank_depth = min(top_m, K)
```

There is no current `pool_k` or sampled-pool ranking depth.

### 3.4 Mixer and Fusion Options

| Option | Default | Allowed values | Meaning |
| --- | ---: | --- | --- |
| `context_module` | `mixer` | `mixer` | Only supported context module. |
| `mixer_dim` | `256` | integer `>= 1` | Hidden width of the rank-slot mixer. |
| `mixer_depth` | `2` | integer `>= 1` | Number of mixer blocks. |
| `mixer_hidden_part` | `32` | integer `>= 1` | Slot-mixing MLP hidden width. |
| `mixer_hidden_rank` | `64` | integer `>= 1` | Rank-mixing MLP hidden width. |
| `mixer_hidden_channel` | `512` | integer `>= 1` | Channel-mixing MLP hidden width. |
| `mixer_hidden_readout` | `128` | integer `>= 1` | MLP readout hidden width. |
| `context_pooling` | `mlp` | `mlp` | MLP context pooling after mixer blocks. |
| `mixer_context_pooling` | alias | same | Alias for `context_pooling`. |
| `residual_gate` | `residual` | `residual`, `static` | Learned per-query gate or static scalar gate. |
| `gate_mode` | alias | same | Alias for `residual_gate`. |
| `enrich_gamma` | `None` | float or null | Static gate value; required only when `residual_gate=static`. |
| `residual_gate_hidden_dim` | `128` | integer `>= 1` | Hidden width of learned residual gate MLP. |

Validation rules:

```text
context_module must be mixer
residual_gate=static requires enrich_gamma
residual_gate=residual forbids enrich_gamma
enrichment_space=grab requires alternate/GRAB features
hybrid_global_grab ranking requires alternate/GRAB features
topm_rank_lambda must be in [0, 1]
lambda_ret must be > 0
```

### 3.5 Removed Historical Options

Do not implement or expose these older options for the current method:

```text
use_shared_k
pool_k
pool_k_mode
pool_k_candidates
pool_coverage_epochs
pool_clusters
positive_ratio_max / eta
pool_dist_metric
pool_dist_threshold / epsilon
use_target_retrieval_loss
use_target_robust_loss
hard_neg_k / robust_hard_k
lambda_rob
lambda_gain
gain_margin
```

The current target retrieval loss is always computed when target enrichment is
active in training. There is no robust no-harm, guard, gain, or margin-gain loss.

## 4. Required Host Adapter

A target repository should implement a small adapter around its host model. The
adapter can be methods on the model class or a separate wrapper.

### 4.1 Basic Encoders

The minimum global-space API is:

```python
def encode_text(caption_ids) -> Tensor[B, D_g]
def encode_image(images) -> Tensor[N, D_g]
```

If the host has an alternate retrieval space, add:

```python
def encode_text_alt(caption_ids) -> Tensor[B, D_r]
def encode_image_alt(images) -> Tensor[N, D_r]
```

In this repository these are named `encode_text_grab` and `encode_image_grab`.

### 4.2 Target Cache Encoder

The target-pool manager calls an image-cache encoder in evaluation mode and
under `torch.no_grad()`:

```python
def encode_target_image_cache(images, cache_prototypes=True) -> dict:
    image_tokens = visual_encoder(images)  # [N, 1 + Patches, D_e]
    host_image_features = image_tokens[:, 0, :]

    cache = {
        "host_image_features": host_image_features,
    }

    if not cache_prototypes:
        return cache

    if enrichment_space == "grab":
        retrieval_features = encode_image_alt_from_tokens(image_tokens)
    else:
        retrieval_features = host_image_features

    cache["retrieval_features"] = retrieval_features

    if topm_rank_space == "hybrid_global_grab":
        cache["grab_image_features"] = encode_image_alt_from_tokens(image_tokens)

    evidence_bank = build_evidence_bank(
        token_features=image_tokens,
        num_parts=num_parts,
        grid_size=patch_grid_size_or_none,
        mode=extractor_mode,
        retrieval_features=retrieval_features,
    )
    cache["evidence_bank"] = evidence_bank
    cache["prototypes"] = evidence_bank

    if "retrieval_backbone" is enabled and retrieval_features.shape[-1] != D_e:
        if evidence_projection == "none":
            raise ValueError
        cache["retrieval_backbone_features"] = retrieval_features

    return cache
```

The key names are part of the compatibility contract:

| Key | Shape | Required when | Meaning |
| --- | --- | --- | --- |
| `host_image_features` | `[K, D_g]` | always | Global image features used by `host_global` ranking. |
| `retrieval_features` | `[K, D_a]` | `cache_prototypes=True` | Image features compared against enriched queries. |
| `evidence_bank` | `[K, S, D_e]` | `cache_prototypes=True` | Evidence slots before active-space projection. |
| `prototypes` | `[K, S, D_e]` | compatibility | Alias of `evidence_bank`. |
| `grab_image_features` | `[K, D_r]` | hybrid ranking | Alternate-space image features for hybrid ranking. |
| `retrieval_backbone_features` | `[K, D_a]` | dimension-mismatched retrieval evidence | Raw retrieval evidence projected after top-M gather. |
| `pids` | `[K]` | training/eval cache final assembly | Identity labels for loss/metrics/diagnostics. |
| `image_ids` | `[K]` | training cache | Stable unique target image IDs. |
| `top_indices` | `[B, M]` | frozen-index training only | Precomputed top-M rows for the current query batch. |
| `diagnostics` | dict | training cache | Cache health scalar diagnostics. |

### 4.3 Target Cache Finalizer

A host wrapper should expose:

```python
def finalize_target_cache(cache):
    return finalize_target_evidence_cache(cache, args, evidence_dim=D_e)
```

This is required whenever `extractor_mode` includes any of:

```text
cluster
cluster_residual
cluster_density
cluster_rarity
```

It must be called after all target image chunks have been concatenated, because
these providers depend on the distribution of the full current target pool.

### 4.4 Enrich-Only Helper

For evaluation, expose:

```python
def enrich_text_features(query_features, host_text_features, target_cache, alt_text_features=None):
    return target_enricher.enrich_only(
        query_features=query_features,
        host_text_features=host_text_features,
        pool_cache=target_cache,
        space=enrichment_space,
        grab_text_features=alt_text_features,
    )
```

If your project does not use the name `grab`, treat `grab_text_features` as
`alternate_text_features`. The behavior is the same.

## 5. Dataset and Batch Contract

The target-pool manager expects the training dataset to be iterable as records:

```text
(pid, image_id, img_path, caption)
```

The online training batch must contain:

| Batch key | Shape/type | Required when | Meaning |
| --- | --- | --- | --- |
| `caption_ids` | `LongTensor[B, L]` | always | Tokenized caption/query input. |
| `images` | `Tensor[B, C, H, W]` | normal training | Online host image input. |
| `pids` | `LongTensor[B]` | target loss and host losses | Query identity labels. |
| `image_ids` | `LongTensor[B]` | recommended | Stable image IDs for diagnostics and compatibility. |
| `index` | `LongTensor[B]` | frozen-index mode | Stable query/caption row indices. |

If a dataset has multiple captions per image, `image_id` must be identical for
those captions. The target pool deduplicates by `image_id`; the frozen ranking
table is still built per caption/query row.

In `pnp_text_only` mode, `images` does not need to be moved to the GPU for the
online training batch, because all image-side features come from the frozen
cache.

## 6. Target Pool Manager

The target pool is a reusable gallery cache. It is not sampled per query and is
not centered on positives.

### 6.1 Unique Training Image Records

Build unique image records by scanning the training dataset in order:

```python
records_by_image_id = {}
for pid, image_id, img_path, caption in train_dataset:
    if image_id not in records_by_image_id:
        records_by_image_id[int(image_id)] = {
            "pid": int(pid),
            "image_id": int(image_id),
            "img_path": img_path,
        }
records = list(records_by_image_id.values())
```

Also build query records for frozen ranking:

```python
query_records = []
for query_index, (pid, image_id, img_path, caption) in enumerate(train_dataset):
    query_records.append({
        "pid": int(pid),
        "query_index": int(query_index),
        "caption": caption,
    })
```

### 6.2 Pool Image Transform

The source manager uses the same CLIP-style normalization for cache encoding:

```text
Resize(img_size)
ToTensor()
Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
          std=[0.26862954, 0.26130258, 0.27577711])
```

A target repository should use the preprocessing expected by its host encoder.
The important requirement is consistency between target-cache images and normal
evaluation gallery images.

### 6.3 Cache Refresh Schedule

Define the refresh unit:

```text
unit(epoch, step) = epoch, if recompute_level == "epoch"
unit(epoch, step) = step,  if recompute_level == "step"
```

Define the interval ID:

```text
if recompute_interval == -1:
    interval_id = 0
else:
    interval_id = floor((unit(epoch, step) - 1) / recompute_interval)
```

Validation:

```text
recompute_interval must be -1 or >= 1
```

A full training cache is refreshed when it does not exist, or when
`recompute_interval != -1` and the current interval ID differs from the previous
cache interval ID.

### 6.4 Full Training Cache Algorithm

Default training mode uses the full unique training image set.

```python
def build_full_training_cache(model, records, epoch, step):
    if len(records) == 0:
        raise ValueError("empty image pool")

    cache = encode_records(model, records, cache_prototypes=True)
    device = cache["host_image_features"].device
    cache["image_ids"] = tensor([r["image_id"] for r in records], device=device)
    cache["pids"] = tensor([r["pid"] for r in records], device=device)
    cache["diagnostics"] = {
        "pool_interval_id": float(interval_id(epoch, step)),
        "pool_interval_reused": 0.0,
        "pool_cache_size": float(len(records)),
    }
    return cache
```

When the cached object is reused, update only the returned diagnostics:

```text
pool_interval_reused = 1.0 after the first request in an interval, else 0.0
```

The cache object itself contains full-pool tensors, so every query in the batch
sees the same target pool. Query specificity is introduced only by top-M ranking.

### 6.5 Frozen-Index Cache Algorithm

Frozen-index mode precomputes top-M rankings once and reuses them. It is meant
for frozen-host plug-and-play training.

Requirements:

```text
use_freeze_indices = true
target_enrichment = true
batch contains index
host should be frozen for ranking consistency
```

Algorithm:

```python
def build_frozen_index_cache(model):
    cache = encode_records(model, unique_image_records, cache_prototypes=True)
    cache["image_ids"] = full_gallery_image_ids
    cache["pids"] = full_gallery_pids

    host_images = norm(cache["host_image_features"])
    retrieval_images = norm(cache["retrieval_features"])

    if topm_rank_space == "host_global":
        query_features = encode_text_records(query_records)
        rank_images = host_images
    elif topm_rank_space == "retrieval":
        if enrichment_space == "grab":
            query_features = encode_alt_text_records(query_records)
        else:
            query_features = encode_text_records(query_records)
        rank_images = retrieval_images
    elif topm_rank_space == "hybrid_global_grab":
        query_features = encode_text_records(query_records)
        alt_query_features = encode_alt_text_records(query_records)
        alt_image_features = cache["grab_image_features"] or cache["retrieval_features"]

    rank_depth = min(top_m, pool_size)

    for query_chunk in query_features:
        if topm_rank_space == "hybrid_global_grab":
            scores = lambda * norm(query_chunk) @ host_images.T \
                   + (1 - lambda) * norm(alt_query_chunk) @ norm(alt_image_features).T
        else:
            scores = norm(query_chunk) @ rank_images.T
        append topk(scores, k=rank_depth, sorted=True).indices

    frozen_rank_indices = concat(rank_chunks, dim=0)  # [num_queries, rank_depth]
    frozen_cache = cache
```

For a training batch:

```python
query_indices = batch["index"].detach().long().cpu()
validate 0 <= query_indices < frozen_rank_indices.shape[0]
top_indices = frozen_rank_indices[query_indices, :min(top_m, rank_depth)]
return frozen_cache plus top_indices and diagnostics
```

Frozen cache diagnostics:

```text
pool_interval_id = 0
pool_interval_reused = 1 after first frozen-cache request, else 0
pool_cache_size = K
frozen_indices_used = 1
frozen_index_depth = rank_depth
```

### 6.6 Evaluation Gallery Cache

Evaluation target enrichment builds the target cache from the full gallery:

```python
def compute_target_gallery_cache(model, img_loader):
    chunks = []
    gids = []
    for pid, image in img_loader:
        cache_chunk = model.encode_target_image_cache(image)
        chunks.append(detach_to_cpu(cache_chunk))
        gids.append(pid)

    cache = concat_each_key(chunks).to(device)
    cache["pids"] = concat(gids).to(device)
    cache = model.finalize_target_cache(cache)  # if available
    return cache, gids_cpu
```

No identity label is used to select evidence during evaluation. Labels are used
only to compute retrieval metrics.

## 7. Evidence Providers

Evidence providers produce a fixed slot table per target image:

```text
B_j = [b_{j,1}; ...; b_{j,S}] in R^{S x D_e}
```

The full target evidence bank is:

```text
B_cache = [B_1; ...; B_K] in R^{K x S x D_e}
```

Provider order is the order in `extractor_mode` after alias expansion and
duplicate removal. Do not sort providers alphabetically.

### 7.1 Extractor Mode Parsing

Normalize a provider list as follows:

```python
raw_tokens = split comma-separated string, lowercase, trim blanks
expand aliases:
    global_horizontal -> global,horizontal
    global_vertical   -> global,vertical
    global_grid       -> global,grid
remove duplicates while keeping first occurrence
validate every token is supported
```

Slot counts:

```text
count(global) = 1
count(horizontal) = num_parts
count(vertical) = num_parts
count(grid) = num_parts * num_parts
count(retrieval_backbone) = 1
count(cluster) = 1
count(cluster_residual) = 1
count(cluster_density) = 1
count(cluster_rarity) = 1
```

Thus:

```text
S = sum_provider count(provider)
```

The slot index map is built with a running cursor:

```python
cursor = 0
for provider in normalized_providers:
    indices[provider] = range(cursor, cursor + count(provider))
    cursor += count(provider)
```

### 7.2 Visual Token Inputs

The evidence builder receives visual token features:

```text
T in R^{B x (1 + N_p) x D_e}
```

`T[:, 0, :]` is the global/CLS token. `T[:, 1:, :]` are patch/local tokens. If a
valid grid exists, reshape patch tokens to:

```text
P_grid in R^{B x H_p x W_p x D_e}
```

where:

```text
H_p * W_p = N_p
```

Vertical and grid evidence require a valid patch grid. Horizontal evidence can
use either the grid or the flat patch sequence.

### 7.3 Balanced Region Bounds

For a one-dimensional axis length `L` and `num_parts = P`, compute balanced
bounds exactly as:

```python
boundaries = round(linspace(0, L, P + 1))
for part p in 0..P-1:
    start = boundaries[p]
    end = boundaries[p + 1]
    if end <= start:
        end = min(start + 1, L)
        start = max(0, end - 1)
    bounds.append((start, end))
```

The same rule is used for rows, columns, and flat patch positions. This covers
all positions and prevents empty regions even when `L < P`.

### 7.4 Layout Evidence Formulas

Let `N(v) = norm(v)`.

Global evidence:

```text
b_global = N(T_cls)
```

Horizontal evidence with a valid grid:

```text
Omega_h(p) = rows start_p:end_p across all columns
b_horizontal,p = N(mean_{(row,col) in Omega_h(p)} P_grid[row,col])
```

Horizontal evidence without a grid:

```text
Omega_h(p) = flat patch indices start_p:end_p
b_horizontal,p = N(mean_{n in Omega_h(p)} T_patch,n)
```

Vertical evidence:

```text
Omega_v(p) = all rows across columns start_p:end_p
b_vertical,p = N(mean_{(row,col) in Omega_v(p)} P_grid[row,col])
```

Grid evidence:

```text
Omega_grid(a,b) = rows row_start_a:row_end_a and cols col_start_b:col_end_b
b_grid,a,b = N(mean_{(row,col) in Omega_grid(a,b)} P_grid[row,col])
```

Grid slots are emitted row-major: for each row part, iterate over all column
parts.

### 7.5 Retrieval-Backbone Evidence

If `retrieval_backbone` is enabled and the retrieval feature dimension equals
`D_e`, write:

```text
b_retrieval = norm(retrieval_features)
```

If the retrieval feature dimension differs from `D_e`, the evidence bank stores
a zero placeholder in that slot. The raw bank is stored separately as:

```text
cache["retrieval_backbone_features"] = retrieval_features
```

During top-M gather, selected raw retrieval features are projected to the shared
evidence dimension if needed, normalized, and written into the gathered slot.

### 7.6 Target-Relative Evidence

Target-relative evidence is computed after the full target cache is merged. It
is label-free and describes the distribution of the current target pool.

Choose the target-relative feature bank:

```text
phi_j = norm(host_image_features_j), if target_relative_space == "host_global"
phi_j = norm(retrieval_features_j),  if target_relative_space == "retrieval"
```

Let `Phi = [phi_1; ...; phi_K] in R^{K x D_phi}`.

Cluster count:

```text
C = min(max(1, target_relative_num_clusters), K)
```

If `C == K`, use one image per cluster:

```text
assignment_j = j
centroid_j = phi_j
count_j = 1
```

Otherwise, initialize centroids by evenly spaced image indices:

```text
init_indices = round(linspace(0, K - 1, C))
mu_c = phi_{init_indices_c}
```

Run at most 10 iterations of spherical k-means:

```text
assignment_j = argmax_c phi_j^T mu_c
mu_c = norm(mean_{j: assignment_j = c} phi_j), if cluster c is nonempty
mu_c unchanged, if cluster c is empty
stop early if assignments do not change
```

Cluster counts:

```text
n_c = |{j: assignment_j = c}|
assigned_count_j = n_{assignment_j}
assigned_centroid_j = mu_{assignment_j}
```

Cluster centroid evidence:

```text
b_cluster,j = norm(assigned_centroid_j)
```

Cluster residual evidence:

```text
residual_j = phi_j - assigned_centroid_j
if ||residual_j||_2 <= 1e-8:
    residual_j = 0
b_cluster_residual,j = norm(residual_j)
```

Density and rarity scalars:

```text
density_raw_j = log(max(assigned_count_j / K, 1e-8))
rarity_raw_j  = -log(max(assigned_count_j / K, 1e-8))
```

Standardize over the target pool with population standard deviation:

```text
stdz(v_j) = (v_j - mean_l v_l) / max(std_l(v_l), 1e-6)
```

If there is only one scalar value, return `v - mean(v)`, which is zero.

```text
cluster_density_scalar_j = stdz(density_raw_j)
cluster_rarity_scalar_j  = stdz(rarity_raw_j)
```

Writing target-relative evidence:

- If vector evidence dimension equals `D_e`, write normalized vectors directly into `evidence_bank`.
- If vector evidence dimension differs from `D_e` and `evidence_projection != none`, store raw vectors under `cluster_features` or `cluster_residual_features`.
- If vector evidence dimension differs and `evidence_projection == none`, raise an error.
- Scalar evidence is always stored as `[K, 1]` under `cluster_density_scalar` or `cluster_rarity_scalar` and projected after top-M gather.

The finalized cache must also contain:

```text
target_relative_cluster_ids: LongTensor[K]
target_relative_cluster_counts: LongTensor[K]
```

### 7.7 Applying Auxiliary Evidence After Top-M Gather

After top-M ranking, gather the base evidence tensor:

```text
gathered = evidence_bank[top_indices]  # [B, M_actual, S, D_e]
```

Then overwrite auxiliary slots as needed:

```python
for provider, raw_key in {
    "retrieval_backbone": "retrieval_backbone_features",
    "cluster": "cluster_features",
    "cluster_residual": "cluster_residual_features",
}.items():
    if provider is enabled and raw_key exists:
        raw_selected = raw_bank[top_indices]
        projected = project_raw_vector_evidence(raw_selected)
        gathered[:, :, slot(provider), :] = norm(projected)

for scalar_provider in ["cluster_density", "cluster_rarity"]:
    if scalar_provider is enabled:
        scalar_selected = scalar_bank[top_indices]  # [B, M, 1]
        projected = scalar_projector[scalar_provider](scalar_selected)
        gathered[:, :, slot(scalar_provider), :] = norm(projected)
```

If any target-relative provider is enabled, the cache must have been finalized.
Otherwise the implementation raises an error before mixing.

## 8. Top-M Target Selection

For every query, select target evidence by ranking the full current target pool.
Let:

```text
M_actual = min(top_m, K)
```

### 8.1 Supplied Frozen Indices

If the cache contains `top_indices`, validate:

```text
top_indices has rank 2
top_indices.shape[0] == batch_size
top_indices.shape[1] >= 1
0 <= min(top_indices)
max(top_indices) < K
```

Then use:

```text
I_q = top_indices[q, :M_actual]
```

### 8.2 Online Ranking Scores

If no frozen indices are supplied, compute ranking scores without gradient.

Host-global ranking:

```text
score(q,j) = norm(host_text_q)^T norm(host_image_j)
```

Retrieval ranking:

```text
score(q,j) = norm(query_feature_q)^T norm(retrieval_feature_j)
```

Retrieval ranking requires matching feature dimensions.

Hybrid global+alternate ranking:

```text
score_global(q,j) = norm(host_text_q)^T norm(host_image_j)
score_alt(q,j)    = norm(alt_text_q)^T norm(alt_image_j)
score(q,j)        = lambda * score_global(q,j) + (1 - lambda) * score_alt(q,j)
```

`lambda = topm_rank_lambda`. Hybrid ranking requires alternate text features and
alternate image features. If alternate image features are absent but
`retrieval_features` have the alternate dimension, `retrieval_features` may be
used as the alternate image bank.

Top-M indices:

```text
I_q = argsort_desc(score(q, :))[:M_actual]
```

Use sorted descending top-k.

## 9. Target Enricher Forward Pass

The training forward API is:

```python
out = target_enricher(
    query_features,      # [B, D_a]
    host_text_features,  # [B, D_g]
    query_pids,          # [B]
    pool_cache,          # target cache dict
    space,               # "global" or "grab"
    grab_text_features=None,
)
```

The eval-only API is the same except it omits labels and losses:

```python
enriched = target_enricher.enrich_only(
    query_features,
    host_text_features,
    pool_cache,
    space,
    grab_text_features=None,
)
```

Forward algorithm:

```python
host_images = norm(pool_cache["host_image_features"])
retrieval_images = norm(pool_cache["retrieval_features"])
evidence_bank = pool_cache.get("evidence_bank", pool_cache["prototypes"])
pool_pids = pool_cache["pids"]

q = norm(query_features)
g = norm(host_text_features)

top_indices = select_top_indices(q, g, host_images, retrieval_images, pool_cache)

gathered = evidence_bank[top_indices]
gathered = apply_auxiliary_evidence(gathered, top_indices, pool_cache)
selected = project_to_active_space(gathered, space)

context = mixer(q, selected)
delta = fusion_mlp(q, context)
gate = residual_gate(q, context)
enriched = norm(q + gate * delta)

losses = target_retrieval_loss(enriched, retrieval_images, query_pids, pool_pids)
diagnostics = compute_diagnostics(...)
return enriched, top_indices, losses, diagnostics
```

Active-space projection:

```text
if space == "global":
    selected = norm(gathered)
if space == "grab":
    selected = norm(proto_to_grab(gathered))
```

The current implementation instantiates only the active branch. Global
enrichment does not build GRAB mixer/fusion modules. GRAB enrichment does not
build global mixer/fusion modules.

## 10. Rank-Slot Query-Conditioned Mixer

The source class is named `RankPartQueryConditionedMixerAdapter`; in a general
implementation, read "part" as evidence slot. The mixer input is:

```text
z in R^{B x D_a}
E in R^{B x M_actual x S x D_a}
```

The configured rank count is `M = top_m`, slot count is `S`, and mixer width is
`D_m = mixer_dim`.

### 10.1 Input Validation and Rank Padding

Validate:

```text
z rank is 2
E rank is 4
z.shape[0] == E.shape[0]
z.shape[1] == D_a
E.shape[-1] == D_a
E.shape[2] == S
E.shape[1] <= M
```

If `M_actual < M`, pad ranks with zeros:

```text
E_padded = concat(E, zeros[B, M - M_actual, S, D_a], axis=rank)
rank_mask[rank] = 1 for rank < M_actual, else 0
rank_mask shape = [1, M, 1, 1]
```

If `M_actual == M`, no mask is needed.

### 10.2 Input Projection and FiLM Conditioning

Project evidence:

```text
X_pre = W_in(E_padded) + R + S_emb
```

where:

```text
W_in: R^{D_a} -> R^{D_m}
R in R^{1 x M x 1 x D_m}      # learned rank embedding
S_emb in R^{1 x 1 x S x D_m}  # learned slot embedding
```

Rank and slot embeddings are initialized with truncated normal standard
deviation `0.02`.

Project the query:

```text
q_m = W_q z
```

Generate FiLM parameters:

```text
[scale_raw, shift] = MLP_film(q_m)
scale = tanh(scale_raw)
```

`MLP_film` is:

```text
Linear(D_m, D_m) -> GELU -> Linear(D_m, 2D_m)
```

Apply FiLM:

```text
X_0 = LN(X_pre) * (1 + scale[:, None, None, :]) + shift[:, None, None, :]
```

Apply `rank_mask` after input projection and after FiLM when padding is present.

### 10.3 Mixer Block

A mixer block has three residual sublayers: slot mixing, rank mixing, and channel
mixing.

Slot mixing operates along the slot axis. For each batch, rank, and channel:

```text
Delta_slot = MLP_slot(permute(LN_slot(X), [B, M, D_m, S]))
Delta_slot = inverse_permute(Delta_slot)
Y = X + Delta_slot
```

`MLP_slot` is:

```text
Linear(S, mixer_hidden_part) -> GELU -> Linear(mixer_hidden_part, S)
```

Rank mixing operates along the rank axis. For each batch, slot, and channel:

```text
Delta_rank = MLP_rank(permute(LN_rank(Y), [B, S, D_m, M]))
Delta_rank = inverse_permute(Delta_rank)
Z = Y + Delta_rank
```

`MLP_rank` is:

```text
Linear(M, mixer_hidden_rank) -> GELU -> Linear(mixer_hidden_rank, M)
```

Channel mixing operates along the channel axis:

```text
X_next = Z + MLP_channel(LN_channel(Z))
```

`MLP_channel` is:

```text
Linear(D_m, mixer_hidden_channel) -> GELU -> Linear(mixer_hidden_channel, D_m)
```

Apply `rank_mask` after each residual sublayer when padding is present. Repeat
for `mixer_depth` blocks.

After the final block:

```text
H = LN_final(X_D) in R^{B x M x S x D_m}
H = H * rank_mask, if rank padding is present
H_flat = flatten(H over rank and slot) in R^{B x (M*S) x D_m}
```

### 10.4 Context Pooling

The mixer supports one pooling mode in the current source implementation.

#### MLP Pooling

MLP pooling reads across the flattened token axis independently for each
channel:

```text
H_t = transpose(H_flat, [B, D_m, M*S])
h = MLP_readout(H_t).squeeze(-1)
```

`MLP_readout` is:

```text
Linear(M*S, mixer_hidden_readout) -> GELU -> Linear(mixer_hidden_readout, 1)
```

The pooled mixer vector is:

```text
h in R^{B x D_m}
```

### 10.5 Mixer Output

The final context vector is projected back into the active enrichment space:

```text
c = W_out h in R^{B x D_a}
```

`W_out` is linear `D_m -> D_a`.

## 11. Residual Query Fusion

Fusion uses the normalized active query `u`, mixer context `c`, and elementwise
interaction `u * c`.

Fusion MLP:

```text
delta = F([u; c; u * c])
```

where `F` is:

```text
Linear(3D_a, D_a) -> LayerNorm(D_a) -> GELU -> Linear(D_a, D_a)
```

### 11.1 Static Gate

If `residual_gate == "static"`:

```text
g = enrich_gamma
```

Broadcast `g` to shape `[B, 1]`.

### 11.2 Learned Residual Gate

If `residual_gate == "residual"`:

```text
g = sigmoid(G([u; c; u * c]))
```

where `G` is:

```text
Linear(3D_a, residual_gate_hidden_dim)
-> LayerNorm(residual_gate_hidden_dim)
-> GELU
-> Linear(residual_gate_hidden_dim, 1)
```

Initialize the last layer as:

```text
last_weight = 0
last_bias = log(p / (1 - p)), where p = 0.1 clipped to [1e-4, 1 - 1e-4]
```

Thus the initial gate is approximately `0.1` for every query.

### 11.3 Enriched Query

The enriched query is:

```text
u_tilde = norm(u + g * delta)
```

The design intentionally preserves the original query through a residual path;
the context calibrates the query instead of replacing it.

## 12. Target-Pool Retrieval Loss

The current target-specific loss is a supervised target-pool retrieval loss.

Given a batch of enriched queries:

```text
U_tilde in R^{B x D_a}
A in R^{K x D_a}    # normalized retrieval_features from the target cache
query_pids in R^B
pool_pids in R^K
```

Positive mask:

```text
P_{qj} = 1[y_q == y_j]
```

Every training query must have at least one positive target image in the full
training target pool:

```text
sum_j P_{qj} >= 1 for all q
```

The implementation raises an error if this condition fails.

Logits:

```text
L_{qj} = (u_tilde_q^T a_j) / max(tau, 1e-6)
```

Positive log-sum-exp:

```text
pos_lse_q = log sum_{j: P_{qj}=1} exp(L_{qj})
```

All-pool log-sum-exp:

```text
all_lse_q = log sum_{j=1..K} exp(L_{qj})
```

Loss:

```text
L_ret = - mean_q (pos_lse_q - all_lse_q)
```

Weighted target-enrichment loss:

```text
L_target = lambda_ret * L_ret
```

If host losses are enabled, the optimized training loss is:

```text
L_total = lambda_host * L_host + L_target
```

If host losses are disabled:

```text
L_total = L_target
```

Identity labels are used in this loss only after top-M evidence has already been
selected label-free.

## 13. Diagnostics

Diagnostics are scalar tensors or floats used for logging and health checks.
They should not affect forward behavior.

### 13.1 Target and Top-M Diagnostics

Let `I_q` be the selected top-M indices and `top_pid_{qm} = pool_pids[I_{qm}]`.

```text
top_positive_{qm} = 1[top_pid_{qm} == query_pid_q]
positive_in_pool_q = any_j P_{qj}
positive_in_topm_q = any_m top_positive_{qm}
num_positive_in_pool_q = sum_j P_{qj}
num_positive_in_topm_q = sum_m top_positive_{qm}
host_topm_recall_q = num_positive_in_topm_q / max(num_positive_in_pool_q, 1)
```

First positive rank uses 1-based ranks and an absent rank of `M_actual + 1`:

```text
rank_m = m + 1
first_rank_q = min_m where top_positive_{qm}=1 else M_actual + 1
```

Scores for selected top-M rows are recomputed with host-global scores:

```text
S_host = norm(host_text) @ norm(host_image).T
selected_scores_qm = S_host[q, I_{qm}]
```

Logged target diagnostics:

```text
target_positive_in_pool_rate = mean_q positive_in_pool_q
target_positive_in_topm_rate = mean_q positive_in_topm_q
target_num_positive_in_pool = mean_q num_positive_in_pool_q
target_num_positive_in_topm = mean_q num_positive_in_topm_q
target_host_topm_recall = mean_q host_topm_recall_q
target_first_positive_rank = mean first_rank over queries with positive_in_topm
target_first_positive_rank_with_absent = mean_q first_rank_q
target_missing_topm_rate = mean_q (not positive_in_topm_q)
target_host_top1_score = mean_q selected_scores_{q1}
target_host_topm_score = mean_{q,m} selected_scores_{qm}
target_host_top1_topm_gap = mean_q (selected_scores_{q1} - selected_scores_{qM})
target_raw_enriched_cosine = mean_q u_q^T u_tilde_q
target_raw_context_cosine = mean_q u_q^T norm(c_q)
target_context_norm = mean_q ||c_q||_2
target_enrichment_shift_norm = mean_q ||u_tilde_q - u_q||_2
target_residual_gate_mean/std/min/max = statistics of g
```

If no positive appears in top-M for any query, `target_first_positive_rank` is
reported as zero.

### 13.2 Mixer Diagnostics

Core mixer diagnostics:

```text
mixer/context_norm = mean ||c||_2
mixer/context_delta_cosine = mean cosine(c, delta)
mixer/output_delta_norm = mean ||delta||_2
mixer/film_scale_mean/std
mixer/film_shift_mean/std
mixer/H_mean
mixer/H_std
mixer/H_flat_token_std
mixer/readout_output_norm
mixer/rank_mixing_weight_norm
mixer/part_mixing_weight_norm
mixer/channel_mixing_weight_norm
mixer/readout_weight_norm, when MLP readout exists
```

Weight norms sum the Frobenius norms of matrix parameters in the corresponding
modules and ignore bias/vector parameters.

### 13.3 Cache Diagnostics

Full training cache:

```text
pool_interval_id
pool_interval_reused
pool_cache_size
```

Frozen-index cache:

```text
pool_interval_id
pool_interval_reused
pool_cache_size
frozen_indices_used
frozen_index_depth
```

## 14. Training Integration

A generic training loop should do the following.

### 14.1 Build the Target Pool Manager

```python
if args.target_enrichment:
    target_pool = TargetPoolManager(train_dataset, args, logger)
else:
    target_pool = None
```

### 14.2 Enable Enrichment by Epoch

```python
use_target = (
    target_pool is not None
    and epoch >= args.enrichment_start
)
```

### 14.3 Per-Batch Flow

```python
for batch in train_loader:
    if args.pnp_text_only:
        move every tensor except batch["images"] to device
    else:
        move full batch to device

    target_cache = None
    if use_target:
        target_cache = target_pool.get_train_cache(model, batch, epoch, global_step)

    ret = model(batch, epoch=epoch, current_step=global_step, target_cache=target_cache)

    if target_cache and "diagnostics" in target_cache:
        copy scalar cache diagnostics into ret

    loss = ret["loss"]
    loss.backward()
    optimizer.step()
```

### 14.4 Model Forward Integration

For global enrichment:

```python
target_ret = target_enricher(
    query_features=t_feats,
    host_text_features=t_feats,
    grab_text_features=t_grab_f if topm_rank_space == "hybrid_global_grab" else None,
    query_pids=batch["pids"],
    pool_cache=target_cache,
    space="global",
)
t_feats = target_ret["enriched_features"]
```

For alternate/GRAB enrichment:

```python
target_ret = target_enricher(
    query_features=t_alt_feats,
    host_text_features=t_feats,
    grab_text_features=t_alt_feats,
    query_pids=batch["pids"],
    pool_cache=target_cache,
    space="grab",
)
t_alt_feats = target_ret["enriched_features"]
```

Expose training keys:

```python
ret["target_enrichment_loss"] = target_ret["total_loss"]
ret["target_retrieval_loss"] = detach(target_ret["target_retrieval_loss"])
ret["_loss_grad_sources"] = {
    "target_enrichment_loss": target_ret["total_loss"],
    "target_retrieval_loss": target_ret["target_retrieval_loss"],
}
copy scalar target_* and mixer/* diagnostics into ret
```

Then compute host losses, if enabled, using the enriched active text feature.

### 14.5 Frozen-Host Plug-and-Play Training

For a strict plug-and-play adaptation:

```text
freeze all host parameters
train only target_enricher.*
disable host losses
use frozen indices
optionally use pnp_text_only
```

The optimization then changes only:

```text
evidence projection layers
scalar evidence projectors
proto_to_grab projection, when GRAB enrichment is active
rank-slot mixer
context pooling layers
fusion MLP
learned residual gate, when residual_gate=residual
```

The target image cache features and frozen rankings are treated as fixed inputs.

## 15. Optimizer, Freezing, and Checkpoint Policy

### 15.1 Freezing

Freezing rule:

```python
for name, parameter in model.named_parameters():
    parameter.requires_grad = name.startswith("target_enricher.")
```

Raise an error if this leaves zero trainable target-enrichment parameters.

### 15.2 Learning Rate

The source optimizer gives `target_enricher` parameters:

```text
lr = base_lr * lr_factor
```

It skips all parameters with `requires_grad=False`.

### 15.3 Checkpoints

Checkpoint normal model and optimizer state:

```text
model state dict
optimizer state
scheduler state
epoch/global-step metadata
config
```

Do not checkpoint target-pool caches or frozen top-M tables as model state. They
are runtime artifacts and should be rebuilt from the current config, dataset,
and host checkpoint after resume.

## 16. Evaluation Integration

Evaluation can run host-only or with target enrichment enabled.

### 16.1 Baseline Global Scores

Encode all text queries and gallery images:

```text
Z = norm(encode_text(all_queries)) in R^{Q x D_g}
H = norm(encode_image(all_gallery)) in R^{K x D_g}
S_global = Z H^T
```

### 16.2 Optional Alternate/GRAB Scores

If the host has alternate features and `only_global` is false:

```text
W = norm(encode_text_alt(all_queries)) in R^{Q x D_r}
R = norm(encode_image_alt(all_gallery)) in R^{K x D_r}
S_grab = W R^T
```

The source evaluator also reports global+grab score-fusion rows for weights:

```text
[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.68, 0.32]
```

Score range matching is row-wise:

```text
ScaleLike(A; B)_{qj} =
    (A_{qj} - min_l A_{ql}) / max(max_l A_{ql} - min_l A_{ql}, eps)
    * (max_l B_{ql} - min_l B_{ql})
    + min_l B_{ql}
```

The helper `_scaled_fuse(primary, secondary, primary_weight)` computes:

```text
primary_weight * primary + (1 - primary_weight) * ScaleLike(secondary; primary)
```

### 16.3 Target-Aware Evaluation Scores

When target enrichment is enabled:

1. Build the full gallery target cache.
2. Encode every text query into host-global features.
3. Encode alternate text features only when `enrichment_space == "grab"` or `topm_rank_space == "hybrid_global_grab"`.
4. Enrich each text query using the full gallery cache.
5. Compare enriched text features with `target_cache["retrieval_features"]`.

Formula:

```text
U_tilde = norm(enrich(all_queries, target_gallery_cache)) in R^{Q x D_a}
A = norm(target_cache["retrieval_features"]) in R^{K x D_a}
S_target = U_tilde A^T
```

The source evaluator reports target fusion rows for prototype weights:

```text
proto_lambda in [0.0, 0.1, 0.2, ..., 1.0]
```

For each base score matrix `S_base`:

```text
S_base+proto(lambda) =
    (1 - lambda) * ScaleLike(S_base; S_target) + lambda * S_target
```

When `lambda = 1.0`, the row is pure target-aware retrieval:

```text
S_base+proto(1) = S_target
```

For global-only evaluation, inspect the row named:

```text
global+proto(1)-t2i
```

Fusion rows are ablations. They are not extra trainable branches.

### 16.4 Retrieval Metrics

For text-to-image evaluation, sort gallery indices by descending score for each
query. Let `match_{qk}` be 1 if the gallery item at rank `k` has the same
identity as query `q`.

Rank-k accuracy:

```text
R@k = 100 * mean_q 1[sum_{r=1..k} match_{qr} > 0]
```

Average precision for query `q`:

```text
AP_q = (1 / num_rel_q) * sum_k precision_q(k) * match_{qk}
precision_q(k) = (sum_{r=1..k} match_{qr}) / k
```

Mean average precision:

```text
mAP = 100 * mean_q AP_q
```

INP for query `q` uses the rank of the last relevant item:

```text
last_q = max{k: match_{qk}=1}
INP_q = (sum_{r=1..last_q} match_{qr}) / last_q
mINP = 100 * mean_q INP_q
```

The source also reports:

```text
rSum = R@1 + R@5 + R@10
```

Optional image-to-text rows are computed by transposing the similarity matrix
and swapping query/gallery labels.

## 17. Logging Keys

Log scalar keys if they satisfy any of:

```text
key contains "loss"
key ends with "grad_norm"
key starts with "pool_"
key starts with "target_"
key starts with "mixer/"
```

Core training keys:

```text
train/loss
train/host_loss
train/cid_loss
train/tal_loss
train/target_enrichment_loss
train/target_retrieval_loss
train/grad_norm
train/loss_grad_norm
train/host_loss_grad_norm
train/cid_loss_grad_norm
train/tal_loss_grad_norm
train/target_enrichment_loss_grad_norm
train/target_retrieval_loss_grad_norm
```

Cache keys:

```text
train/pool_interval_id
train/pool_interval_reused
train/pool_cache_size
train/frozen_indices_used
train/frozen_index_depth
```

Target diagnostics:

```text
train/target_positive_in_pool_rate
train/target_positive_in_topm_rate
train/target_num_positive_in_pool
train/target_num_positive_in_topm
train/target_host_topm_recall
train/target_first_positive_rank
train/target_first_positive_rank_with_absent
train/target_missing_topm_rate
train/target_host_top1_score
train/target_host_topm_score
train/target_host_top1_topm_gap
train/target_raw_enriched_cosine
train/target_raw_context_cosine
train/target_context_norm
train/target_enrichment_shift_norm
train/target_residual_gate_mean
train/target_residual_gate_std
train/target_residual_gate_min
train/target_residual_gate_max
```

Mixer keys:

```text
train/mixer/context_norm
train/mixer/context_delta_cosine
train/mixer/output_delta_norm
train/mixer/rank_mixing_weight_norm
train/mixer/part_mixing_weight_norm
train/mixer/channel_mixing_weight_norm
train/mixer/readout_weight_norm
train/mixer/film_scale_mean
train/mixer/film_scale_std
train/mixer/film_shift_mean
train/mixer/film_shift_std
train/mixer/H_mean
train/mixer/H_std
train/mixer/H_flat_token_std
train/mixer/readout_output_norm
```

Evaluation keys follow the row names:

```text
eval/<task>/R1
eval/<task>/R5
eval/<task>/R10
eval/<task>/mAP
eval/<task>/mINP
eval/<task>/rSum
eval/top_R1
eval/best_R1
eval/ablation_best_R1
eval/ablation_best_lambda
eval/delta_R1_target_vs_global
eval/delta_mAP_target_vs_global
eval/delta_mINP_target_vs_global
```

## 18. Minimal Porting Recipes

### 18.1 Minimal Single-Space Port

If the target host exposes only global image/text embeddings and no patch grid,
implement the minimal behavior first:

```text
enrichment_space = global
topm_rank_space = host_global
extractor_mode = global
retrieval_features = host_image_features
evidence_bank = norm(host_image_features)[:, None, :]
```

This gives one evidence slot per target image and exercises the full top-M,
mixer, fusion, loss, and evaluation path.

### 18.2 Adding Layout Evidence

If the visual encoder exposes tokens:

```text
[CLS, patch_1, ..., patch_N]
```

add:

```text
extractor_mode = global,horizontal
```

If the patch grid is known, vertical and grid evidence can also be enabled:

```text
extractor_mode = global,horizontal,vertical
extractor_mode = global,grid
```

Use the exact balanced bounds algorithm from Section 7.3.

### 18.3 Adding an Alternate Retrieval Space

If the host has an alternate retrieval space:

```text
enrichment_space = grab
retrieval_features = alternate image features
query_features = alternate text features
host_text_features = global text features
host_image_features = global image features
```

If evidence tokens remain in global visual-token dimension `D_e`, project them
to alternate dimension with `proto_to_grab: D_e -> D_r` before the mixer.

### 18.4 Adding Hybrid Ranking

Hybrid ranking needs both global and alternate text/image features:

```text
topm_rank_space = hybrid_global_grab
score = lambda * global_score + (1 - lambda) * alternate_score
```

Cache `grab_image_features` for the target pool. At evaluation and online
training, compute `grab_text_features` for the query batch.

### 18.5 Adding Target-Relative Evidence

To add cluster evidence:

```text
extractor_mode = global,horizontal,cluster,cluster_residual
```

To add scalar density/rarity:

```text
extractor_mode = global,horizontal,cluster_density,cluster_rarity
```

Make sure `finalize_target_cache` runs after all target images are encoded and
before the cache is used by the enricher.

## 19. Validation Test Checklist

A robust port should include tests for the following.

Configuration:

```text
top_m must be positive
lambda_ret must be positive
topm_rank_lambda must be in [0, 1]
num_parts must be positive
residual_gate static/residual validation
grab enrichment rejects global-only hosts
hybrid ranking rejects missing alternate features
removed historical flags are not accepted
```

Evidence:

```text
extractor aliases expand correctly
duplicate extractor tokens are removed in first-seen order
slot counts match every provider combination
global evidence returns [B, 1, D]
global,horizontal with P parts returns [B, 1+P, D]
grid with P parts returns [B, P*P, D] plus any other enabled slots
all vector evidence slots are normalized
vertical/grid reject invalid grid_size
retrieval_backbone stores raw features when dimensions differ
target-relative finalizer writes cluster IDs/counts and scalar fields
```

Cache:

```text
training cache has one row per unique image_id
full cache includes host_image_features, retrieval_features, evidence_bank/prototypes, pids, image_ids
recompute_interval=-1 reuses the first cache
recompute by epoch/step changes interval_id correctly
frozen indices require batch index
frozen_index_depth equals min(top_m, pool_size)
frozen batch cache injects top_indices with shape [B, frozen_index_depth]
```

Top-M:

```text
online top-M uses configured scores only
labels do not affect top-M
top_indices are sorted descending by score
precomputed top_indices are batch-size and range checked
M_actual = min(top_m, pool_size)
```

Mixer and fusion:

```text
mixer pads when pool_size < top_m
rank mask zeros padded ranks after each sublayer
MLP pooling returns [B, D_active]
learned residual gate initializes near 0.1
static gate uses enrich_gamma
enriched_features are L2-normalized and match query feature shape
```

Loss and training:

```text
target retrieval loss is finite
every query must have at least one positive in full target pool
lambda_ret scales total target loss
gradients flow into target_enricher parameters
freeze_host leaves only target_enricher.* trainable
pnp_text_only skips online image encoding and requires frozen indices
```

Evaluation:

```text
evaluation target cache spans the full gallery
pure target row +proto(1) equals enriched-query similarity
score fusion uses row-wise ScaleLike
metrics report R1, R5, R10, mAP, mINP, rSum
```

## 20. Common Implementation Mistakes

- Selecting top-M with identity labels or forcing positives into top-M.
- Reintroducing sampled shared-K/adaptive-pool behavior from older experiments.
- Building a different target pool for each query.
- Forgetting to deduplicate training target images by `image_id`.
- Forgetting that frozen rankings are per caption/query index, not per image.
- Training in one feature space and evaluating against another.
- Omitting `grab_image_features` or alternate text features for hybrid ranking.
- Passing an evidence bank whose slot count does not match `extractor_mode`.
- Using vertical or grid evidence without a valid patch grid.
- Enabling target-relative evidence without finalizing the full target cache.
- Treating target-pool caches or frozen indices as model checkpoint state.
- Leaving host losses enabled during strict frozen plug-and-play training.
- Judging the method only from fused evaluation rows instead of the pure `+proto(1)` row.

## 21. Source-Compatible Names

When porting, different names are fine, but preserving these conceptual names
makes parity easier to check:

| Source name | General meaning |
| --- | --- |
| `TargetPrototypeEnricher` | Top-M selection, evidence gather, context, fusion, loss, diagnostics. |
| `RankPartQueryConditionedMixerAdapter` | Query-conditioned rank-slot mixer. |
| `build_evidence_bank` | Builds layout, retrieval, and placeholder evidence slots. |
| `finalize_target_evidence_cache` | Computes target-relative full-pool evidence. |
| `TargetPoolManager` | Owns full training cache and frozen-index cache. |
| `host_image_features` | Global image ranking features. |
| `retrieval_features` | Active target image features for target loss and final target scoring. |
| `evidence_bank` / `prototypes` | Cached evidence slot tensor. |
| `top_indices` | Precomputed frozen top-M rows for the current batch. |
| `grab` | Source name for the optional alternate retrieval space. |

The algorithm is plug-and-play when replacing the host model changes only the
feature/cache adapter. The target-aware behavior itself remains: full-pool cache,
label-free top-M selection, evidence-slot gathering, query-conditioned rank-slot
mixing, residual query enrichment, and target-pool retrieval supervision.
