import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mixer import RankPartQueryConditionedMixerAdapter, _FusionMLP
from .prototypes import (
    SCALAR_TARGET_RELATIVE_MODES,
    TARGET_RELATIVE_MODES,
    evidence_slot_indices,
    prototype_slot_count,
)


DEFAULT_RESIDUAL_GATE_INIT = 0.1
TOPM_RANK_SPACES = ("host_global", "retrieval", "hybrid_global_grab", "hybrid_global_tse")


def canonical_enrichment_space(space):
    space = str(space).lower()
    if space == "tse":
        return "grab"
    return space


def canonical_rank_space(space):
    space = str(space).lower()
    if space == "hybrid_global_tse":
        return "hybrid_global_grab"
    return space


def _masked_logsumexp(values, mask, dim):
    neg_inf = torch.finfo(values.dtype).min
    return torch.logsumexp(values.masked_fill(~mask, neg_inf), dim=dim)


class _ResidualGateMLP(nn.Module):
    def __init__(self, dim, hidden_dim, initial_value=DEFAULT_RESIDUAL_GATE_INIT):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("--residual_gate_hidden_dim must be a positive integer")
        initial_value = min(max(float(initial_value), 1e-4), 1.0 - 1e-4)
        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(initial_value / (1.0 - initial_value)))

    def forward(self, query, context):
        gate_input = torch.cat([query, context, query * context], dim=-1)
        return torch.sigmoid(self.net(gate_input))


class TargetPrototypeEnricher(nn.Module):
    def __init__(self, embed_dim, grab_embed_dim, args):
        super().__init__()
        self.embed_dim = embed_dim
        self.grab_embed_dim = grab_embed_dim
        if args.top_m < 1:
            raise ValueError("--top_m must be a positive integer")
        self.top_m = args.top_m
        self.gamma = getattr(args, "enrich_gamma", None)
        self.residual_gate_mode = getattr(args, "residual_gate", "residual")
        if self.residual_gate_mode not in ("static", "residual"):
            raise ValueError("--residual_gate must be either 'static' or 'residual'")
        if self.residual_gate_mode == "static" and self.gamma is None:
            raise ValueError("--residual_gate static requires --enrich_gamma")
        if self.residual_gate_mode == "residual" and self.gamma is not None:
            raise ValueError("--enrich_gamma is only valid when --residual_gate is static")
        self.tau = args.tau
        self.lambda_ret = getattr(args, "lambda_ret", 1.0)
        if self.lambda_ret <= 0:
            raise ValueError("--lambda_ret must be positive")
        self.context_module = getattr(args, "context_module", "mixer")
        if self.context_module != "mixer":
            raise ValueError("--context_module must be mixer")
        self.enrichment_space = canonical_enrichment_space(getattr(args, "enrichment_space", "global"))
        if self.enrichment_space not in ("global", "grab"):
            raise ValueError("--enrichment_space must be global, tse, or grab")
        self.topm_rank_space = canonical_rank_space(getattr(args, "topm_rank_space", "host_global"))
        if self.topm_rank_space not in ("host_global", "retrieval", "hybrid_global_grab"):
            raise ValueError("--topm_rank_space must be one of host_global, retrieval, hybrid_global_grab")
        self.topm_rank_lambda = float(getattr(args, "topm_rank_lambda", 0.5))
        if self.topm_rank_lambda < 0.0 or self.topm_rank_lambda > 1.0:
            raise ValueError("--topm_rank_lambda must be in [0, 1]")
        self.enable_global = self.enrichment_space == "global"
        self.enable_grab = self.enrichment_space == "grab"

        self.extractor_mode = getattr(args, "extractor_mode", "global,horizontal")
        self.num_parts = getattr(args, "num_parts", 6)
        num_slots = prototype_slot_count(self.extractor_mode, self.num_parts)
        self.evidence_slot_indices = evidence_slot_indices(self.extractor_mode, self.num_parts)
        self.requires_target_relative = any(
            mode in self.evidence_slot_indices for mode in TARGET_RELATIVE_MODES
        )
        self.scalar_evidence_projectors = nn.ModuleDict()
        for mode in SCALAR_TARGET_RELATIVE_MODES:
            if mode in self.evidence_slot_indices:
                self.scalar_evidence_projectors[mode] = nn.Linear(1, embed_dim)
        needs_raw_retrieval_projection = (
            "retrieval_backbone" in self.evidence_slot_indices
            and self.enrichment_space == "grab"
        )
        needs_raw_target_relative_projection = (
            getattr(args, "target_relative_space", "host_global") == "retrieval"
            and self.enrichment_space == "grab"
            and any(mode in self.evidence_slot_indices for mode in ("cluster", "cluster_residual"))
        )
        if needs_raw_retrieval_projection or needs_raw_target_relative_projection:
            self.raw_vector_evidence_to_proto = nn.Linear(grab_embed_dim, embed_dim)
        mixer_kwargs = dict(
            num_ranks=self.top_m,
            num_slots=num_slots,
            mixer_dim=getattr(args, "mixer_dim", 256),
            depth=getattr(args, "mixer_depth", 2),
            hidden_part=getattr(args, "mixer_hidden_part", 32),
            hidden_rank=getattr(args, "mixer_hidden_rank", 64),
            hidden_channel=getattr(args, "mixer_hidden_channel", 512),
            hidden_readout=getattr(args, "mixer_hidden_readout", 128),
            context_pooling=getattr(args, "context_pooling", "mlp"),
        )
        if self.enable_global:
            self.global_context = RankPartQueryConditionedMixerAdapter(embed_dim, **mixer_kwargs)
            self.global_fusion = _FusionMLP(embed_dim)
            if self.residual_gate_mode == "residual":
                self.global_residual_gate = _ResidualGateMLP(
                    embed_dim,
                    getattr(args, "residual_gate_hidden_dim", 128),
                )

        if self.enable_grab:
            self.proto_to_grab = nn.Linear(embed_dim, grab_embed_dim)
            self.grab_context = RankPartQueryConditionedMixerAdapter(grab_embed_dim, **mixer_kwargs)
            self.grab_fusion = _FusionMLP(grab_embed_dim)
            if self.residual_gate_mode == "residual":
                self.grab_residual_gate = _ResidualGateMLP(
                    grab_embed_dim,
                    getattr(args, "residual_gate_hidden_dim", 128),
                )

    def _require_global(self):
        if not self.enable_global:
            raise ValueError("Global enrichment is disabled because --enrichment_space is set to tse/grab")

    def _require_grab(self):
        if not self.enable_grab:
            raise ValueError("TSE/GRAB enrichment is disabled because --enrichment_space is global")

    def _project_prototypes(self, prototypes, space):
        space = canonical_enrichment_space(space)
        if space == "grab":
            self._require_grab()
            prototypes = self.proto_to_grab(prototypes.float())
        return F.normalize(prototypes.float(), p=2, dim=-1)

    def _gather_bank(self, bank, top_indices, trailing_shape):
        bank = bank.to(device=top_indices.device)
        gathered = bank.index_select(0, top_indices.reshape(-1))
        return gathered.view(*top_indices.shape, *trailing_shape)

    def _project_raw_vector_evidence(self, values):
        values = values.float()
        if values.shape[-1] == self.embed_dim:
            return values
        if not hasattr(self, "raw_vector_evidence_to_proto"):
            raise ValueError("Raw vector evidence requires a projection to the shared evidence dim")
        if values.shape[-1] != self.grab_embed_dim:
            raise ValueError("Raw vector evidence dim must match either embed_dim or alternate embed_dim")
        return self.raw_vector_evidence_to_proto(values)

    def _apply_auxiliary_evidence(self, gathered, top_indices, pool_cache):
        if self.requires_target_relative and "target_relative_cluster_ids" not in pool_cache:
            raise ValueError(
                "Target-relative evidence requires a finalized target cache; "
                "call finalize_target_cache after merging the full target pool"
            )

        updated = gathered.clone()
        vector_keys = {
            "retrieval_backbone": "retrieval_backbone_features",
            "cluster": "cluster_features",
            "cluster_residual": "cluster_residual_features",
        }
        for mode, key in vector_keys.items():
            if mode not in self.evidence_slot_indices or key not in pool_cache:
                continue
            slot = self.evidence_slot_indices[mode][0]
            bank = pool_cache[key].float()
            selected = self._gather_bank(bank, top_indices, (bank.shape[-1],))
            projected = self._project_raw_vector_evidence(selected)
            updated[:, :, slot, :] = F.normalize(projected.float(), p=2, dim=-1)

        for mode, projector in self.scalar_evidence_projectors.items():
            key = f"{mode}_scalar"
            if key not in pool_cache:
                raise ValueError(f"--extractor_mode {mode} requires finalized target cache key '{key}'")
            slot = self.evidence_slot_indices[mode][0]
            bank = pool_cache[key].float()
            selected = self._gather_bank(bank, top_indices, (1,))
            projected = projector(selected.float())
            updated[:, :, slot, :] = F.normalize(projected.float(), p=2, dim=-1)
        return updated

    def _context(self, query_features, selected_prototypes, space):
        space = canonical_enrichment_space(space)
        if space == "grab":
            self._require_grab()
            return self.grab_context(query_features.float(), selected_prototypes.float())
        self._require_global()
        return self.global_context(query_features.float(), selected_prototypes.float())

    def _context_diagnostics(self, space):
        space = canonical_enrichment_space(space)
        if space == "grab":
            self._require_grab()
            return dict(self.grab_context.last_diagnostics)
        self._require_global()
        return dict(self.global_context.last_diagnostics)

    def _fusion_delta(self, query_features, context, space):
        space = canonical_enrichment_space(space)
        if space == "grab":
            self._require_grab()
            delta = self.grab_fusion(query_features.float(), context.float())
        else:
            self._require_global()
            delta = self.global_fusion(query_features.float(), context.float())
        return delta

    def _residual_gate(self, query_features, context, space):
        space = canonical_enrichment_space(space)
        if self.residual_gate_mode == "static":
            return query_features.new_full((query_features.shape[0], 1), float(self.gamma))
        if space == "grab":
            self._require_grab()
            return self.grab_residual_gate(query_features.float(), context.float())
        self._require_global()
        return self.global_residual_gate(query_features.float(), context.float())

    def _fuse(self, query_features, context, space):
        delta = self._fusion_delta(query_features, context, space)
        residual_gate = self._residual_gate(query_features, context, space)
        return F.normalize(query_features.float() + residual_gate * delta, p=2, dim=-1)

    def _fuse_with_delta(self, query_features, context, space):
        delta = self._fusion_delta(query_features, context, space)
        residual_gate = self._residual_gate(query_features, context, space)
        enriched = F.normalize(query_features.float() + residual_gate * delta, p=2, dim=-1)
        return enriched, delta, residual_gate

    def _rank_scores(
        self,
        query_features,
        host_text_features,
        host_image_features,
        retrieval_features,
        pool_cache,
        space,
        grab_text_features=None,
    ):
        if self.topm_rank_space == "host_global":
            return host_text_features @ host_image_features.t()

        if self.topm_rank_space == "retrieval":
            if query_features.shape[-1] != retrieval_features.shape[-1]:
                raise ValueError(
                    "--topm_rank_space retrieval requires query_features and "
                    "pool_cache['retrieval_features'] to have the same dimension"
                )
            return query_features @ retrieval_features.t()

        if grab_text_features is None:
            if canonical_enrichment_space(space) == "grab":
                grab_text_features = query_features
            else:
                raise ValueError("--topm_rank_space hybrid_global_grab requires TSE/GRAB text features")
        grab_image_features = pool_cache.get("grab_image_features")
        if grab_image_features is None:
            if retrieval_features.shape[-1] == grab_text_features.shape[-1]:
                grab_image_features = retrieval_features
            else:
                raise ValueError("--topm_rank_space hybrid_global_grab requires pool_cache['grab_image_features']")
        grab_text_features = F.normalize(grab_text_features.float(), p=2, dim=-1)
        grab_image_features = F.normalize(grab_image_features.float(), p=2, dim=-1)
        if grab_text_features.shape[-1] != grab_image_features.shape[-1]:
            raise ValueError("--topm_rank_space hybrid_global_grab requires alternate text/image dimensions to match")

        global_scores = host_text_features @ host_image_features.t()
        grab_scores = grab_text_features @ grab_image_features.t()
        return self.topm_rank_lambda * global_scores + (1.0 - self.topm_rank_lambda) * grab_scores

    def _top_indices(
        self,
        query_features,
        host_text_features,
        host_image_features,
        retrieval_features,
        pool_cache,
        space,
        grab_text_features=None,
    ):
        supplied = pool_cache.get("top_indices")
        if supplied is not None:
            top_indices = supplied.long()
            if top_indices.dim() != 2:
                raise ValueError("pool_cache['top_indices'] must have shape [batch, top_m]")
            if top_indices.shape[0] != host_text_features.shape[0]:
                raise ValueError("pool_cache['top_indices'] batch size must match query batch size")
            top_m = min(self.top_m, host_image_features.shape[0], top_indices.shape[1])
            if top_m < 1:
                raise ValueError("pool_cache['top_indices'] must contain at least one column")
            top_indices = top_indices[:, :top_m].to(host_image_features.device)
            if int(top_indices.min().item()) < 0 or int(top_indices.max().item()) >= host_image_features.shape[0]:
                raise ValueError("pool_cache['top_indices'] contains indices outside the target pool")
            return top_indices

        with torch.no_grad():
            host_scores = self._rank_scores(
                query_features=query_features,
                host_text_features=host_text_features,
                host_image_features=host_image_features,
                retrieval_features=retrieval_features,
                pool_cache=pool_cache,
                space=space,
                grab_text_features=grab_text_features,
            )
            top_m = min(self.top_m, host_image_features.shape[0])
            return host_scores.topk(k=top_m, dim=1, largest=True, sorted=True).indices

    def forward(self, query_features, host_text_features, query_pids, pool_cache, space, grab_text_features=None):
        space = canonical_enrichment_space(space)
        host_image_features = F.normalize(pool_cache["host_image_features"].float(), p=2, dim=-1)
        retrieval_features = F.normalize(pool_cache["retrieval_features"].float(), p=2, dim=-1)
        prototypes = pool_cache.get("evidence_bank", pool_cache["prototypes"]).float()
        pool_pids = pool_cache["pids"].long()

        normalized_query = F.normalize(query_features.float(), p=2, dim=-1)
        host_text_features = F.normalize(host_text_features.float(), p=2, dim=-1)
        top_indices = self._top_indices(
            query_features=normalized_query,
            host_text_features=host_text_features,
            host_image_features=host_image_features,
            retrieval_features=retrieval_features,
            pool_cache=pool_cache,
            space=space,
            grab_text_features=grab_text_features,
        )

        gathered = prototypes[top_indices]
        gathered = self._apply_auxiliary_evidence(gathered, top_indices, pool_cache)
        selected_prototypes = self._project_prototypes(gathered, space)

        context = self._context(normalized_query, selected_prototypes, space)
        enriched, delta, residual_gate = self._fuse_with_delta(normalized_query, context, space)

        losses = self.compute_losses(
            enriched_query=enriched,
            retrieval_features=retrieval_features,
            query_pids=query_pids.long(),
            pool_pids=pool_pids,
        )
        diagnostics = self.compute_diagnostics(
            raw_query=normalized_query,
            enriched_query=enriched,
            context=context,
            host_text_features=host_text_features,
            host_image_features=host_image_features,
            delta=delta,
            residual_gate=residual_gate,
            top_indices=top_indices,
            query_pids=query_pids.long(),
            pool_pids=pool_pids,
            mixer_diagnostics=self._context_diagnostics(space),
        )

        return {
            "enriched_features": enriched,
            "top_indices": top_indices,
            **losses,
            **diagnostics,
        }

    def enrich_only(self, query_features, host_text_features, pool_cache, space, grab_text_features=None):
        space = canonical_enrichment_space(space)
        host_image_features = F.normalize(pool_cache["host_image_features"].float(), p=2, dim=-1)
        retrieval_features = F.normalize(pool_cache["retrieval_features"].float(), p=2, dim=-1)
        prototypes = pool_cache.get("evidence_bank", pool_cache["prototypes"]).float()
        normalized_query = F.normalize(query_features.float(), p=2, dim=-1)
        host_text_features = F.normalize(host_text_features.float(), p=2, dim=-1)
        top_indices = self._top_indices(
            query_features=normalized_query,
            host_text_features=host_text_features,
            host_image_features=host_image_features,
            retrieval_features=retrieval_features,
            pool_cache=pool_cache,
            space=space,
            grab_text_features=grab_text_features,
        )

        gathered = prototypes[top_indices]
        gathered = self._apply_auxiliary_evidence(gathered, top_indices, pool_cache)
        selected_prototypes = self._project_prototypes(gathered, space)
        context = self._context(normalized_query, selected_prototypes, space)
        return self._fuse(normalized_query, context, space)

    def compute_losses(self, enriched_query, retrieval_features, query_pids, pool_pids):
        positive_mask = query_pids.view(-1, 1).eq(pool_pids.view(1, -1))
        valid_positive = positive_mask.any(dim=1)
        if not valid_positive.all():
            missing = int((~valid_positive).sum().item())
            raise ValueError(
                "Target retrieval loss requires every query to have at least one "
                f"positive image in the target pool; missing positives for {missing} query(s)."
            )

        retrieval_scores = enriched_query @ retrieval_features.t() / max(self.tau, 1e-6)
        pos_lse = _masked_logsumexp(retrieval_scores, positive_mask, dim=1)
        all_lse = torch.logsumexp(retrieval_scores, dim=1)
        target_retrieval_loss = -(pos_lse - all_lse).mean()
        total = self.lambda_ret * target_retrieval_loss
        return {
            "target_retrieval_loss": target_retrieval_loss,
            "total_loss": total,
        }

    def compute_diagnostics(
        self,
        raw_query,
        enriched_query,
        context,
        host_text_features,
        host_image_features,
        delta,
        residual_gate,
        top_indices,
        query_pids,
        pool_pids,
        mixer_diagnostics,
    ):
        with torch.no_grad():
            zero = raw_query.new_tensor(0.0)
            top_m = top_indices.shape[1]
            top_pids = pool_pids[top_indices]
            top_positive = top_pids.eq(query_pids.view(-1, 1))
            positive_mask = query_pids.view(-1, 1).eq(pool_pids.view(1, -1))

            positive_in_pool = positive_mask.any(dim=1)
            positive_in_topm = top_positive.any(dim=1)
            top_positive_count = top_positive.sum(dim=1).float()
            pool_positive_count = positive_mask.sum(dim=1).float()
            host_topm_recall = top_positive_count / pool_positive_count.clamp_min(1.0)

            ranks = torch.arange(1, top_m + 1, device=top_indices.device).view(1, -1)
            absent_rank = raw_query.new_full((top_indices.shape[0], top_m), float(top_m + 1))
            first_rank = torch.where(top_positive, ranks.float(), absent_rank).min(dim=1).values
            if positive_in_topm.any():
                first_rank_when_present = first_rank[positive_in_topm].mean()
            else:
                first_rank_when_present = zero

            host_scores = host_text_features @ host_image_features.t()
            selected_scores = host_scores.gather(1, top_indices)
            host_topm_gap = selected_scores[:, 0] - selected_scores[:, -1]

            raw_enriched_cosine = (raw_query * enriched_query).sum(dim=1)
            raw_context_cosine = (raw_query * F.normalize(context.float(), p=2, dim=-1)).sum(dim=1)
            enrichment_shift = (enriched_query - raw_query).norm(dim=1)
            context_delta_cosine = F.cosine_similarity(context.float(), delta.float(), dim=-1)
            residual_gate = residual_gate.detach().float()

            diagnostics = {
                "target_positive_in_pool_rate": positive_in_pool.float().mean(),
                "target_positive_in_topm_rate": positive_in_topm.float().mean(),
                "target_num_positive_in_pool": pool_positive_count.mean(),
                "target_num_positive_in_topm": top_positive_count.mean(),
                "target_host_topm_recall": host_topm_recall.mean(),
                "target_first_positive_rank": first_rank_when_present,
                "target_first_positive_rank_with_absent": first_rank.mean(),
                "target_missing_topm_rate": (~positive_in_topm).float().mean(),
                "target_host_top1_score": selected_scores[:, 0].mean(),
                "target_host_topm_score": selected_scores.mean(),
                "target_host_top1_topm_gap": host_topm_gap.mean(),
                "target_raw_enriched_cosine": raw_enriched_cosine.mean(),
                "target_raw_context_cosine": raw_context_cosine.mean(),
                "target_context_norm": context.norm(dim=1).mean(),
                "target_enrichment_shift_norm": enrichment_shift.mean(),
                "target_residual_gate_mean": residual_gate.mean(),
                "target_residual_gate_std": residual_gate.std(unbiased=False),
                "target_residual_gate_min": residual_gate.min(),
                "target_residual_gate_max": residual_gate.max(),
                "mixer/context_delta_cosine": context_delta_cosine.mean(),
                "mixer/output_delta_norm": delta.norm(dim=1).mean(),
            }
            diagnostics.update(mixer_diagnostics)
            return diagnostics
