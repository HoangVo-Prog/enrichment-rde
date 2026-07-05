import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .enricher import canonical_enrichment_space, canonical_rank_space
from .pool_common import _PoolImageDataset, _PoolTextDataset, _unwrap_model


def _seed_worker(_worker_id):
    return None


def _seeded_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


class TargetPoolCacheMixin:
    def _frozen_index_ready(self):
        return (
            self.frozen_rank_indices is not None
            and self.frozen_index_depth is not None
            and self.frozen_cache is not None
        )

    def _build_frozen_index_cache(self, model):
        if not self.records:
            raise ValueError("Cannot build frozen indices from an empty training image pool")
        if not self.query_records:
            raise ValueError("Cannot build frozen indices from an empty training query set")

        cache = self._encode_records(model, self.records, cache_prototypes=True)
        device = cache["host_image_features"].device
        gallery_image_ids = torch.tensor(
            [record["image_id"] for record in self.records],
            dtype=torch.long,
            device="cpu",
        )
        cache["image_ids"] = gallery_image_ids.to(device)
        cache["pids"] = torch.tensor(
            [record["pid"] for record in self.records],
            dtype=torch.long,
            device=device,
        )

        rank_space = canonical_rank_space(getattr(self.args, "topm_rank_space", "host_global"))
        rank_lambda = float(getattr(self.args, "topm_rank_lambda", 0.5))
        if rank_lambda < 0.0 or rank_lambda > 1.0:
            raise ValueError("--topm_rank_lambda must be in [0, 1]")

        host_image_features = F.normalize(cache["host_image_features"].float(), p=2, dim=-1)
        rank_image_features = host_image_features
        grab_image_features = None
        if rank_space == "retrieval":
            rank_image_features = F.normalize(cache["retrieval_features"].float(), p=2, dim=-1)
        elif rank_space == "hybrid_global_grab":
            grab_image_features = cache.get("grab_image_features")
            if grab_image_features is None:
                if canonical_enrichment_space(getattr(self.args, "enrichment_space", "global")) == "grab":
                    grab_image_features = cache["retrieval_features"]
                else:
                    raise ValueError(
                        "--topm_rank_space hybrid_global_grab requires "
                        "pool cache key 'grab_image_features'"
                    )
            grab_image_features = F.normalize(grab_image_features.float(), p=2, dim=-1)
        elif rank_space != "host_global":
            raise ValueError("--topm_rank_space must be one of host_global, retrieval, hybrid_global_grab")

        if rank_space == "retrieval" and canonical_enrichment_space(getattr(self.args, "enrichment_space", "global")) == "grab":
            query_features = self._encode_grab_text_records(model, self.query_records)
        else:
            query_features = self._encode_text_records(model, self.query_records)

        grab_query_features = None
        if rank_space == "hybrid_global_grab":
            grab_query_features = self._encode_grab_text_records(model, self.query_records)

        requested_depth = int(getattr(self.args, "top_m", 1))
        if requested_depth < 1:
            raise ValueError("Frozen index depth must be positive")
        rank_depth = min(requested_depth, host_image_features.shape[0])

        rank_chunks = []
        batch_size = min(
            max(1, getattr(self.args, "test_batch_size", 1)),
            max(1, query_features.shape[0]),
        )
        with torch.no_grad():
            for start in range(0, query_features.shape[0], batch_size):
                query_chunk = query_features[start:start + batch_size].to(device)
                query_chunk = F.normalize(query_chunk.float(), p=2, dim=-1)
                if rank_space == "hybrid_global_grab":
                    grab_query_chunk = grab_query_features[start:start + batch_size].to(device)
                    grab_query_chunk = F.normalize(grab_query_chunk.float(), p=2, dim=-1)
                    global_scores = query_chunk @ host_image_features.t()
                    grab_scores = grab_query_chunk @ grab_image_features.t()
                    scores = rank_lambda * global_scores + (1.0 - rank_lambda) * grab_scores
                else:
                    scores = query_chunk @ rank_image_features.t()
                rank_chunks.append(scores.topk(k=rank_depth, dim=1, largest=True, sorted=True).indices.cpu())

        self.frozen_rank_indices = torch.cat(rank_chunks, dim=0)
        self.frozen_gallery_image_ids = gallery_image_ids
        self.frozen_cache = cache
        self.frozen_index_depth = rank_depth
        self.frozen_cache_requests = 0

        if self.logger is not None:
            self.logger.info(
                "Frozen host retrieval index built: queries={} gallery={} rank_depth={}".format(
                    self.frozen_rank_indices.shape[0],
                    len(self.records),
                    rank_depth,
                )
            )

    def _coerce_frozen_query_indices(self, query_indices):
        if not torch.is_tensor(query_indices):
            query_indices = torch.tensor(query_indices, dtype=torch.long)
        query_indices = query_indices.detach().long().cpu()
        if query_indices.numel() == 0:
            raise ValueError("Cannot gather frozen indices for an empty batch")
        if int(query_indices.max().item()) >= self.frozen_rank_indices.shape[0]:
            raise ValueError("Batch query index exceeds the frozen retrieval index size")
        if int(query_indices.min().item()) < 0:
            raise ValueError("Batch query index must be non-negative")
        return query_indices

    def _frozen_query_indices(self, batch):
        if "index" not in batch:
            raise ValueError("--use_freeze_indices requires training batches to include dataset indices")
        return self._coerce_frozen_query_indices(batch["index"])

    def _frozen_batch_cache(self, batch):
        if self.frozen_cache is None:
            raise ValueError("Frozen full-gallery cache is not available")
        device = self.frozen_cache["host_image_features"].device
        query_indices = self._frozen_query_indices(batch)

        top_m = min(int(getattr(self.args, "top_m", 1)), self.frozen_index_depth)
        top_indices = self.frozen_rank_indices.index_select(0, query_indices)[:, :top_m].to(device)

        diagnostics = {
            "pool_interval_id": 0.0,
            "pool_interval_reused": float(self.frozen_cache_requests > 0),
            "pool_cache_size": float(self.frozen_cache["pids"].numel()),
            "frozen_indices_used": 1.0,
            "frozen_index_depth": float(self.frozen_index_depth),
        }

        self.frozen_cache_requests += 1
        cache = {key: value for key, value in self.frozen_cache.items()}
        cache["top_indices"] = top_indices
        cache["diagnostics"] = diagnostics
        return cache

    def _full_training_set_cache(self, model, epoch, step):
        if self._should_refresh_full_training_set_cache(epoch, step):
            self._build_full_training_set_cache(model, epoch, step)

        diagnostics = dict(self.full_training_cache["diagnostics"])
        diagnostics["pool_interval_reused"] = float(self.full_training_cache_requests > 0)
        cache = {
            key: value
            for key, value in self.full_training_cache.items()
            if key != "diagnostics"
        }
        cache["diagnostics"] = diagnostics
        self.full_training_cache_requests += 1
        return cache

    def _should_refresh_full_training_set_cache(self, epoch, step):
        if self.full_training_cache is None:
            return True
        if getattr(self.args, "recompute_interval", 1) == -1:
            return False
        return self._interval_id(epoch, step) != self.full_training_interval_id

    def _build_full_training_set_cache(self, model, epoch, step):
        if not self.records:
            raise ValueError("Cannot build a full-training-set target cache from an empty image pool")

        interval_id = self._interval_id(epoch, step)
        cache = self._encode_records(model, self.records, cache_prototypes=True)
        device = cache["host_image_features"].device
        cache_size = len(self.records)
        cache["image_ids"] = torch.tensor(
            [record["image_id"] for record in self.records],
            dtype=torch.long,
            device=device,
        )
        cache["pids"] = torch.tensor(
            [record["pid"] for record in self.records],
            dtype=torch.long,
            device=device,
        )
        cache["diagnostics"] = {
            "pool_interval_id": float(interval_id),
            "pool_interval_reused": 0.0,
            "pool_cache_size": float(cache_size),
        }

        self.full_training_cache = cache
        self.full_training_interval_id = interval_id
        self.full_training_cache_requests = 0
        if self.logger is not None:
            self.logger.info(
                "Target-pool full training cache built: interval={} images={} top_m={}".format(
                    interval_id,
                    cache_size,
                    getattr(self.args, "top_m", 1),
                )
            )

    def _encode_records(self, model, records, cache_prototypes=True):
        core_model = _unwrap_model(model)
        device = next(core_model.parameters()).device
        was_training = core_model.training
        core_model.eval()
        dataset = _PoolImageDataset(records, self.transform)
        loader = DataLoader(
            dataset,
            batch_size=min(max(1, getattr(self.args, "test_batch_size", 1)), max(1, len(records))),
            shuffle=False,
            num_workers=getattr(self.args, "num_workers", 0),
            worker_init_fn=_seed_worker,
            generator=_seeded_generator(self.seed + 801),
        )

        chunks = []
        with torch.no_grad():
            for _, _, images in loader:
                images = images.to(device)
                chunks.append(core_model.encode_target_image_cache(images, cache_prototypes=cache_prototypes))

        if was_training:
            core_model.train()

        merged = {}
        for key in chunks[0].keys():
            merged[key] = torch.cat([chunk[key] for chunk in chunks], dim=0)
        if cache_prototypes and hasattr(core_model, "finalize_target_cache"):
            merged = core_model.finalize_target_cache(merged)
        return merged

    def _encode_text_records(self, model, records):
        core_model = _unwrap_model(model)
        device = next(core_model.parameters()).device
        was_training = core_model.training
        core_model.eval()

        tokenizer = getattr(self.train_dataset, "tokenizer", None)
        if tokenizer is None:
            from utils.simple_tokenizer import SimpleTokenizer
            tokenizer = SimpleTokenizer()
        dataset = _PoolTextDataset(
            records,
            tokenizer=tokenizer,
            text_length=getattr(self.train_dataset, "text_length", getattr(self.args, "text_length", 77)),
            truncate=getattr(self.train_dataset, "truncate", True),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(max(1, getattr(self.args, "test_batch_size", 1)), max(1, len(records))),
            shuffle=False,
            num_workers=getattr(self.args, "num_workers", 0),
            worker_init_fn=_seed_worker,
            generator=_seeded_generator(self.seed + 901),
        )

        chunks = []
        with torch.no_grad():
            for _, _, captions in loader:
                captions = captions.to(device)
                chunks.append(core_model.encode_text(captions).detach().cpu())

        if was_training:
            core_model.train()

        return torch.cat(chunks, dim=0)

    def _encode_grab_text_records(self, model, records):
        core_model = _unwrap_model(model)
        device = next(core_model.parameters()).device
        was_training = core_model.training
        core_model.eval()

        tokenizer = getattr(self.train_dataset, "tokenizer", None)
        if tokenizer is None:
            from utils.simple_tokenizer import SimpleTokenizer
            tokenizer = SimpleTokenizer()
        dataset = _PoolTextDataset(
            records,
            tokenizer=tokenizer,
            text_length=getattr(self.train_dataset, "text_length", getattr(self.args, "text_length", 77)),
            truncate=getattr(self.train_dataset, "truncate", True),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(max(1, getattr(self.args, "test_batch_size", 1)), max(1, len(records))),
            shuffle=False,
            num_workers=getattr(self.args, "num_workers", 0),
            worker_init_fn=_seed_worker,
            generator=_seeded_generator(self.seed + 902),
        )

        chunks = []
        with torch.no_grad():
            for _, _, captions in loader:
                captions = captions.to(device)
                chunks.append(core_model.encode_text_grab(captions).detach().cpu())

        if was_training:
            core_model.train()

        return torch.cat(chunks, dim=0)
