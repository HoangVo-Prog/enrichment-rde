from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
import re
 
from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging


def _canonical_enrichment_space(space):
    return "grab" if str(space).lower() == "tse" else str(space).lower()


def _canonical_rank_space(space):
    return "hybrid_global_grab" if str(space).lower() == "hybrid_global_tse" else str(space).lower()


def _scale_scores_like(scores, reference, eps=1e-12):
    score_min = scores.min(dim=1, keepdim=True).values
    score_max = scores.max(dim=1, keepdim=True).values
    ref_min = reference.min(dim=1, keepdim=True).values
    ref_max = reference.max(dim=1, keepdim=True).values

    score_range = (score_max - score_min).clamp_min(eps)
    ref_range = ref_max - ref_min
    return (scores - score_min) / score_range * ref_range + ref_min


def _scaled_fuse(primary_scores, secondary_scores, primary_weight):
    scaled_secondary = _scale_scores_like(secondary_scores, primary_scores)
    return primary_weight * primary_scores + (1.0 - primary_weight) * scaled_secondary


def _prototype_lambdas():
    return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _format_lambda(value):
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return "{:.2f}".format(value).rstrip("0").rstrip(".")


def _core_model(model):
    return model.module if hasattr(model, "module") else model


def _score_matrix(query_features, gallery_features, chunk_size=0):
    if chunk_size is None or int(chunk_size) <= 0:
        return query_features @ gallery_features.t()

    chunks = []
    chunk_size = int(chunk_size)
    for start in range(0, query_features.shape[0], chunk_size):
        query_chunk = query_features[start:start + chunk_size]
        chunks.append(query_chunk @ gallery_features.t())
    return torch.cat(chunks, dim=0)


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices

def get_metrics(similarity, qids, gids, n_, retur_indices=False):
    t2i_cmc, t2i_mAP, t2i_mINP, indices = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
    if retur_indices:
        return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0]+ t2i_cmc[4]+ t2i_cmc[9]], indices
    else:
        return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0]+ t2i_cmc[4]+ t2i_cmc[9]]


def _metric_task_name(task):
    task = str(task).replace("+", "_plus_")
    task = task.replace("(", "_").replace(")", "")
    task = task.replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_/-]+", "_", task).strip("_")


def _row_to_eval_metrics(row):
    task = _metric_task_name(row[0])
    return {
        f"eval/{task}/R1": float(row[1]),
        f"eval/{task}/R5": float(row[2]),
        f"eval/{task}/R10": float(row[3]),
        f"eval/{task}/mAP": float(row[4]),
        f"eval/{task}/mINP": float(row[5]),
        f"eval/{task}/rSum": float(row[6]) if len(row) > 6 else 0.0,
    }


def _ablation_lambda_from_key(key):
    match = re.search(r"\(([-+]?\d*\.?\d+)\)$", str(key))
    return float(match.group(1)) if match else 0.0


class Evaluator():
    def __init__(self, img_loader, txt_loader, args=None):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("RDE.eval")
        self.args = args
        self.last_metrics = {}
        self.last_best_task = None

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption).cpu()
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img).cpu()
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)
        return qfeats.cpu(), gfeats.cpu(), qids.cpu(), gids.cpu()
    
    def _compute_embedding_tse(self, model):
        model = model.eval() 
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text_tse(caption).cpu()
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image_tse(img).cpu()
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0) 
        return qfeats.cpu(), gfeats.cpu(), qids.cpu(), gids.cpu()

    def _compute_target_gallery_cache(self, model):
        model = model.eval()
        core_model = _core_model(model)
        device = next(core_model.parameters()).device
        gids, cache_chunks = [], []

        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                cache = core_model.encode_target_image_cache(img)
            gids.append(pid.view(-1))
            cache_chunks.append({k: v.detach().cpu() for k, v in cache.items()})

        if not cache_chunks:
            raise ValueError("Cannot build target gallery cache from an empty image loader")

        gids = torch.cat(gids, 0)
        target_cache = {}
        for key in cache_chunks[0].keys():
            target_cache[key] = torch.cat([chunk[key] for chunk in cache_chunks], dim=0).to(device)
        target_cache["pids"] = gids.to(device)
        if hasattr(core_model, "finalize_target_cache"):
            target_cache = core_model.finalize_target_cache(target_cache)
        return target_cache, gids.cpu()

    def _compute_enriched_text_embedding(self, model, target_cache):
        model = model.eval()
        core_model = _core_model(model)
        device = next(core_model.parameters()).device

        args = self.args
        qids, qfeats = [], []
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                host_text_feat = core_model.encode_text(caption)
                grab_text_feat = None
                if (
                    _canonical_enrichment_space(getattr(args, "enrichment_space", "global")) == "grab"
                    or _canonical_rank_space(getattr(args, "topm_rank_space", "host_global")) == "hybrid_global_grab"
                ):
                    grab_text_feat = core_model.encode_text_grab(caption)
                if _canonical_enrichment_space(getattr(args, "enrichment_space", "global")) == "grab":
                    query_feat = grab_text_feat
                else:
                    query_feat = host_text_feat
                text_feat = core_model.enrich_text_features(
                    query_feat,
                    host_text_feat,
                    target_cache,
                    grab_text_features=grab_text_feat,
                ).cpu()
            qids.append(pid.view(-1))
            qfeats.append(text_feat)

        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)
        return qfeats.cpu(), qids.cpu()

    def _target_eval_tasks(self, base_tasks, sims_target):
        for proto_lambda in _prototype_lambdas():
            proto_value = _format_lambda(proto_lambda)
            for base_name, base_scores in base_tasks:
                fused_name = "{}+proto({})".format(base_name, proto_value)
                scaled_base_scores = _scale_scores_like(base_scores, sims_target)
                yield fused_name, (
                    (1.0 - proto_lambda) * scaled_base_scores
                    + proto_lambda * sims_target
                )
    
    def eval(self, model, i2t_metric=False, use_target_enrichment=None):
        if use_target_enrichment is None:
            use_target_enrichment = bool(getattr(self.args, "target_enrichment", False)) if self.args is not None else False
        qfeats, gfeats, qids, gids = self._compute_embedding(model)
        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features
        score_chunk_size = getattr(self.args, "eval_score_chunk_size", 0) if self.args is not None else 0
        sims_bse = _score_matrix(qfeats, gfeats, score_chunk_size)
  
        vq_feats, vg_feats, _, _ = self._compute_embedding_tse(model)
        vq_feats = F.normalize(vq_feats, p=2, dim=1) # text features
        vg_feats = F.normalize(vg_feats, p=2, dim=1) # image features
        sims_tse = _score_matrix(vq_feats, vg_feats, score_chunk_size)
        
        sims_dict = {
            'BGE': sims_bse,
            'TSE': sims_tse,
            'BGE+TSE': (sims_bse+sims_tse)/2
        }

        if use_target_enrichment:
            if self.args is None:
                raise ValueError("Target enrichment evaluation requires evaluator args")
            core_model = _core_model(model)
            if not hasattr(core_model, "target_enricher"):
                raise ValueError("Target enrichment is enabled, but the model has no target_enricher module")
            self.logger.info("Building target-aware cache from the evaluation gallery")
            target_cache, target_gids = self._compute_target_gallery_cache(model)
            target_qfeats, target_qids = self._compute_enriched_text_embedding(model, target_cache)
            target_qfeats = F.normalize(target_qfeats, p=2, dim=1)
            target_gfeats = F.normalize(target_cache["retrieval_features"].detach().cpu(), p=2, dim=1)
            sims_target = _score_matrix(target_qfeats, target_gfeats, score_chunk_size)
            qids = target_qids
            gids = target_gids
        else:
            sims_target = None

        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP","rSum"])
        top1 = 0
        eval_metrics = {}
        rows_by_task = {}
        best_task = None
        best_row = None
        best_ablation_task = None
        best_ablation_row = None
        
        for key in sims_dict.keys():
            sims = sims_dict[key]
            rs = get_metrics(sims, qids, gids, f'{key}-t2i',False)
            table.add_row(rs)
            rows_by_task[key] = rs
            eval_metrics.update(_row_to_eval_metrics(rs))
            if i2t_metric:
                i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=sims.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
                i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
                i2t_row = [f'{key}-i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP, i2t_cmc[0] + i2t_cmc[4] + i2t_cmc[9]]
                table.add_row(i2t_row)
                eval_metrics.update(_row_to_eval_metrics(i2t_row))

            if best_row is None or rs[1] >= best_row[1]:
                best_task = key
                best_row = rs

        if sims_target is not None:
            for key, sims in self._target_eval_tasks(list(sims_dict.items()), sims_target):
                rs = get_metrics(sims, qids, gids, f'{key}-t2i',False)
                table.add_row(rs)
                rows_by_task[key] = rs
                eval_metrics.update(_row_to_eval_metrics(rs))
                if i2t_metric:
                    i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=sims.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
                    i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
                    i2t_row = [f'{key}-i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP, i2t_cmc[0] + i2t_cmc[4] + i2t_cmc[9]]
                    table.add_row(i2t_row)
                    eval_metrics.update(_row_to_eval_metrics(i2t_row))

                if best_row is None or rs[1] >= best_row[1]:
                    best_task = key
                    best_row = rs
                if "+proto(" in key and (best_ablation_row is None or rs[1] >= best_ablation_row[1]):
                    best_ablation_task = key
                    best_ablation_row = rs

            target_key = "BGE+proto(1)"
            if "BGE" in rows_by_task and target_key in rows_by_task:
                base_row = rows_by_task["BGE"]
                target_row = rows_by_task[target_key]
                eval_metrics["eval/delta_R1_target_vs_BGE"] = float(target_row[1] - base_row[1])
                eval_metrics["eval/delta_R5_target_vs_BGE"] = float(target_row[2] - base_row[2])
                eval_metrics["eval/delta_R10_target_vs_BGE"] = float(target_row[3] - base_row[3])
                eval_metrics["eval/delta_mAP_target_vs_BGE"] = float(target_row[4] - base_row[4])
                eval_metrics["eval/delta_mINP_target_vs_BGE"] = float(target_row[5] - base_row[5])
                eval_metrics["eval/delta_rSum_target_vs_BGE"] = float(target_row[6] - base_row[6])
                self.logger.info(
                    "Target-aware delta vs BGE: R1 {:+.2f}, mAP {:+.2f}, mINP {:+.2f}".format(
                        target_row[1] - base_row[1],
                        target_row[4] - base_row[4],
                        target_row[5] - base_row[5],
                    )
                )

        if best_ablation_row is not None:
            top1 = float(best_ablation_row[1])
            best_task = best_ablation_task
            eval_metrics["eval/ablation_best_R1"] = float(best_ablation_row[1])
            eval_metrics["eval/ablation_best_R5"] = float(best_ablation_row[2])
            eval_metrics["eval/ablation_best_R10"] = float(best_ablation_row[3])
            eval_metrics["eval/ablation_best_mAP"] = float(best_ablation_row[4])
            eval_metrics["eval/ablation_best_mINP"] = float(best_ablation_row[5])
            eval_metrics["eval/ablation_best_rSum"] = float(best_ablation_row[6])
            eval_metrics["eval/ablation_best_lambda"] = _ablation_lambda_from_key(best_ablation_task)
        elif best_row is not None:
            top1 = float(best_row[1])

        self.last_metrics = eval_metrics
        self.last_best_task = best_task

        table.custom_format["R1"] = lambda f, v: f"{v:.2f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.2f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.2f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.2f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.2f}"
        table.custom_format["rSum"] = lambda f, v: f"{v:.2f}"
        self.logger.info('\n' + str(table))
        self.logger.info('\n' + "best R1 = " + str(top1))
        if best_task is not None:
            self.logger.info("best R1 row = {}".format(best_task))
        return top1
