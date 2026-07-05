from .pool_cache import TargetPoolCacheMixin
from .pool_common import _pool_transform


class TargetPoolManager(TargetPoolCacheMixin):
    def __init__(self, train_dataset, args, logger=None):
        self.args = args
        self.seed = int(getattr(args, "seed", 1))
        self.logger = logger
        self.train_dataset = train_dataset
        self.records = self._build_unique_records(train_dataset.dataset)
        self.query_records = self._build_query_records(train_dataset.dataset)
        self.transform = _pool_transform(args.img_size)
        self.full_training_cache = None
        self.full_training_interval_id = None
        self.full_training_cache_requests = 0
        self.frozen_cache = None
        self.frozen_rank_indices = None
        self.frozen_gallery_image_ids = None
        self.frozen_index_depth = None
        self.frozen_cache_requests = 0

        if logger is not None:
            logger.info("Target enrichment will select top-M directly from the full training set.")
        if getattr(args, "use_freeze_indices", False) and logger is not None:
            if not getattr(args, "freeze_host", False):
                logger.warning(
                    "--use_freeze_indices is intended for a frozen host. If host weights "
                    "change, the precomputed retrieval ranking will become stale."
                )
            if getattr(args, "txt_aug", False):
                logger.warning(
                    "--use_freeze_indices builds rankings from unaugmented captions; "
                    "consider disabling text augmentation for exact frozen-host training."
                )

    def _build_unique_records(self, dataset):
        records = {}
        for pid, image_id, img_path, _ in dataset:
            if image_id not in records:
                records[int(image_id)] = {
                    "pid": int(pid),
                    "image_id": int(image_id),
                    "img_path": img_path,
                }
        return list(records.values())

    def _build_query_records(self, dataset):
        records = []
        for query_index, (pid, _, _, caption) in enumerate(dataset):
            records.append({
                "pid": int(pid),
                "query_index": int(query_index),
                "caption": caption,
            })
        return records

    def get_train_cache(self, model, batch, epoch, step):
        if getattr(self.args, "use_freeze_indices", False):
            if not self._frozen_index_ready():
                self._build_frozen_index_cache(model)
            return self._frozen_batch_cache(batch)
        return self._full_training_set_cache(model, epoch, step)

    def _interval_unit(self, epoch, step):
        return epoch if getattr(self.args, "recompute_level", "epoch") == "epoch" else step

    def _interval_id(self, epoch, step):
        interval = getattr(self.args, "recompute_interval", 1)
        if interval == -1:
            return 0
        if interval < 1:
            raise ValueError("--recompute_interval must be -1 or a positive integer")
        unit = self._interval_unit(epoch, step)
        return (unit - 1) // interval
