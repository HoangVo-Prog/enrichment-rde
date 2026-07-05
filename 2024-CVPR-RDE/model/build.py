from model import objectives

from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
from .enrichment import (
    TargetPrototypeEnricher,
    build_evidence_bank,
    canonical_enrichment_space,
    canonical_rank_space,
    evidence_slot_indices,
    finalize_target_evidence_cache,
)
import torch
import torch.nn as nn 
import torch.nn.functional as F

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def _set_default(args, name, value):
    if not hasattr(args, name):
        setattr(args, name, value)


def ensure_enrichment_defaults(args):
    defaults = {
        "seed": 1,
        "target_enrichment": False,
        "enrichment_start": 1,
        "enrichment_space": "global",
        "top_m": 32,
        "topm_rank_space": "host_global",
        "topm_rank_lambda": 0.5,
        "extractor_mode": "global,horizontal",
        "num_parts": 6,
        "target_relative_space": "host_global",
        "target_relative_num_clusters": 16,
        "target_relative_cluster_method": "kmeans",
        "evidence_token_budget": 0,
        "evidence_projection": "auto",
        "use_freeze_indices": False,
        "pnp_text_only": False,
        "freeze_host": False,
        "use_host_loss": True,
        "lambda_host": 1.0,
        "lambda_ret": 1.0,
        "context_module": "mixer",
        "mixer_dim": 256,
        "mixer_depth": 2,
        "mixer_hidden_part": 32,
        "mixer_hidden_rank": 64,
        "mixer_hidden_channel": 512,
        "mixer_hidden_readout": 128,
        "context_pooling": "mlp",
        "residual_gate": "residual",
        "enrich_gamma": None,
        "residual_gate_hidden_dim": 128,
        "recompute_level": "epoch",
        "recompute_interval": 1,
    }
    for name, value in defaults.items():
        _set_default(args, name, value)

    if args.top_m < 1:
        raise ValueError("--top_m must be a positive integer")
    if args.num_parts < 1:
        raise ValueError("--num_parts must be a positive integer")
    if args.lambda_ret <= 0:
        raise ValueError("--lambda_ret must be positive")
    if args.topm_rank_lambda < 0.0 or args.topm_rank_lambda > 1.0:
        raise ValueError("--topm_rank_lambda must be in [0, 1]")
    if args.recompute_interval != -1 and args.recompute_interval < 1:
        raise ValueError("--recompute_interval must be -1 or a positive integer")
    if args.recompute_level not in ("epoch", "step"):
        raise ValueError("--recompute_level must be epoch or step")
    if canonical_enrichment_space(args.enrichment_space) not in ("global", "grab"):
        raise ValueError("--enrichment_space must be global, tse, or grab")
    if canonical_rank_space(args.topm_rank_space) not in ("host_global", "retrieval", "hybrid_global_grab"):
        raise ValueError("--topm_rank_space must be host_global, retrieval, hybrid_global_grab, or hybrid_global_tse")
    if args.context_module != "mixer":
        raise ValueError("--context_module must be mixer")
    if args.residual_gate == "static" and args.enrich_gamma is None:
        raise ValueError("--residual_gate static requires --enrich_gamma")
    if args.residual_gate == "residual" and args.enrich_gamma is not None:
        raise ValueError("--enrich_gamma is only valid with --residual_gate static")
    if args.freeze_host and not args.target_enrichment:
        raise ValueError("--freeze_host requires --target_enrichment")
    if args.use_freeze_indices and not args.target_enrichment:
        raise ValueError("--use_freeze_indices requires --target_enrichment")
    if args.pnp_text_only:
        if not args.freeze_host:
            raise ValueError("--pnp_text_only requires --freeze_host")
        if args.use_host_loss:
            raise ValueError("--pnp_text_only requires --no_use_host_loss")
        if not args.use_freeze_indices:
            raise ValueError("--pnp_text_only requires --use_freeze_indices")
        if canonical_enrichment_space(args.enrichment_space) != "global":
            raise ValueError("--pnp_text_only requires --enrichment_space global")


def freeze_host_parameters(model, trainable_prefix="target_enricher."):
    frozen_params = 0
    trainable_params = 0
    for name, parameter in model.named_parameters():
        keep_trainable = name.startswith(trainable_prefix)
        parameter.requires_grad = keep_trainable
        if keep_trainable:
            trainable_params += parameter.numel()
        else:
            frozen_params += parameter.numel()
    if trainable_params == 0:
        raise ValueError("--freeze_host requires target enrichment parameters to train")
    return frozen_params, trainable_params

class RDE(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        ensure_enrichment_defaults(args)
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
 
        self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)
        self.grab_embed_dim = self.visul_emb_layer.embed_dim
        if getattr(args, "target_enrichment", False):
            self.target_enricher = TargetPrototypeEnricher(self.embed_dim, self.grab_embed_dim, args)
 
        if 'TAL' in self.current_task:
            loss_type = 'TAL'
        elif 'TRL' in self.current_task:
            loss_type = 'TRL'
        elif 'InfoNCE' in self.current_task:
            loss_type = 'InfoNCE'
        elif 'SDM' in self.current_task:
            loss_type = 'SDM'
        else:
            exit()
        self.loss_type = loss_type
        self.freeze_host_stats = None
        if getattr(args, "freeze_host", False):
            self.freeze_host_stats = freeze_host_parameters(self)
 
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
      
    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)].float()

    def encode_image_tse(self, image):
        x,atten_i = self.base_model.encode_image(image)
        i_tse_f = self.visul_emb_layer(x, atten_i)   
        return i_tse_f.float()
 
    def encode_text_tse(self, text):
        x,atten_t = self.base_model.encode_text(text.long())
        t_tse_f = self.texual_emb_layer(x, text, atten_t)
        return t_tse_f.float()

    def encode_image_grab(self, image):
        return self.encode_image_tse(image)

    def encode_text_grab(self, text):
        return self.encode_text_tse(text)

    def _text_eot_features(self, text_feats, caption_ids):
        batch_indices = torch.arange(text_feats.shape[0], device=text_feats.device)
        return text_feats[batch_indices, caption_ids.argmax(dim=-1)].float()

    def _needs_tse_text_for_target(self, target_cache=None):
        if target_cache is None or not getattr(self.args, "target_enrichment", False):
            return False
        if canonical_enrichment_space(self.args.enrichment_space) == "grab":
            return True
        return canonical_rank_space(self.args.topm_rank_space) == "hybrid_global_grab"

    def encode_target_image_cache(self, image, cache_prototypes=True):
        image_feats, atten_i = self.base_model.encode_image(image)
        host_features = image_feats[:, 0, :].float()
        cache = {"host_image_features": host_features}
        if not cache_prototypes:
            return cache

        rank_space = canonical_rank_space(getattr(self.args, "topm_rank_space", "host_global"))
        enrichment_space = canonical_enrichment_space(getattr(self.args, "enrichment_space", "global"))
        needs_tse_rank = rank_space == "hybrid_global_grab"
        tse_features = None
        if enrichment_space == "grab" or needs_tse_rank:
            tse_features = self.visul_emb_layer(image_feats, atten_i).float()

        if enrichment_space == "grab":
            cache["retrieval_features"] = tse_features
        else:
            cache["retrieval_features"] = host_features
        if needs_tse_rank:
            cache["grab_image_features"] = tse_features

        grid_size = None
        if hasattr(self.base_model.visual, "num_y") and hasattr(self.base_model.visual, "num_x"):
            grid_size = (self.base_model.visual.num_y, self.base_model.visual.num_x)
        evidence_bank = build_evidence_bank(
            image_feats,
            getattr(self.args, "num_parts", 6),
            grid_size=grid_size,
            mode=getattr(self.args, "extractor_mode", "global,horizontal"),
            retrieval_features=cache["retrieval_features"],
        )
        cache["evidence_bank"] = evidence_bank
        cache["prototypes"] = evidence_bank

        slots = evidence_slot_indices(
            getattr(self.args, "extractor_mode", "global,horizontal"),
            getattr(self.args, "num_parts", 6),
        )
        if "retrieval_backbone" in slots and cache["retrieval_features"].shape[-1] != self.embed_dim:
            if getattr(self.args, "evidence_projection", "auto") == "none":
                raise ValueError(
                    "--extractor_mode retrieval_backbone requires evidence projection "
                    "when retrieval feature dim differs from the shared evidence dim"
                )
            cache["retrieval_backbone_features"] = cache["retrieval_features"]
        return cache

    def finalize_target_cache(self, cache):
        return finalize_target_evidence_cache(cache, self.args, self.embed_dim)

    def enrich_text_features(self, query_features, host_text_features, target_cache, grab_text_features=None):
        if not hasattr(self, "target_enricher"):
            raise ValueError("Target enrichment is not enabled on this model")
        self.target_enricher = self.target_enricher.float()
        return self.target_enricher.enrich_only(
            query_features=query_features,
            host_text_features=host_text_features,
            pool_cache=target_cache,
            space=canonical_enrichment_space(getattr(self.args, "enrichment_space", "global")),
            grab_text_features=grab_text_features,
        )

    def compute_per_loss(self, batch):
        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = self._text_eot_features(text_feats, caption_ids)

        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)

        lossA, simsA = objectives.compute_per_loss(i_feats, t_feats, batch['pids'], \
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        lossB, simsB = objectives.compute_per_loss(i_tse_f, t_tse_f, batch['pids'],\
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        
        return lossA.detach().cpu(), lossB.detach().cpu(), simsA, simsB

    def forward(self, batch, epoch=None, current_step=None, target_cache=None):
        ret = dict()
        ret.update({'temperature': 1 / self.logit_scale})

        caption_ids = batch['caption_ids']
        use_host_loss = getattr(self.args, "use_host_loss", True)
        text_only = getattr(self.args, "pnp_text_only", False) or not use_host_loss

        i_feats = i_tse_f = None
        t_tse_f = None
        if text_only:
            text_feats, atten_t = self.base_model.encode_text(caption_ids.long())
            t_feats = self._text_eot_features(text_feats, caption_ids)
            if self._needs_tse_text_for_target(target_cache):
                t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            zero_source = t_feats
        else:
            images = batch['images']
            image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
            i_feats = image_feats[:, 0, :].float()
            t_feats = self._text_eot_features(text_feats, caption_ids)
            i_tse_f = self.visul_emb_layer(image_feats, atten_i)
            t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            zero_source = i_feats

        if getattr(self.args, "target_enrichment", False) and target_cache is not None:
            self.target_enricher = self.target_enricher.float()
            enrichment_space = canonical_enrichment_space(self.args.enrichment_space)
            if enrichment_space == "grab":
                if t_tse_f is None:
                    text_feats, atten_t = self.base_model.encode_text(caption_ids.long())
                    t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
                target_ret = self.target_enricher(
                    query_features=t_tse_f,
                    host_text_features=t_feats,
                    grab_text_features=t_tse_f,
                    query_pids=batch["pids"],
                    pool_cache=target_cache,
                    space="grab",
                )
                t_tse_f = target_ret["enriched_features"]
            else:
                grab_text_features = None
                if canonical_rank_space(getattr(self.args, "topm_rank_space", "host_global")) == "hybrid_global_grab":
                    if t_tse_f is None:
                        text_feats, atten_t = self.base_model.encode_text(caption_ids.long())
                        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
                    grab_text_features = t_tse_f
                target_ret = self.target_enricher(
                    query_features=t_feats,
                    host_text_features=t_feats,
                    grab_text_features=grab_text_features,
                    query_pids=batch["pids"],
                    pool_cache=target_cache,
                    space="global",
                )
                t_feats = target_ret["enriched_features"]

            ret["target_enrichment_loss"] = target_ret["total_loss"]
            ret["target_retrieval_loss"] = target_ret["target_retrieval_loss"].detach()
            ret["_loss_grad_sources"] = {
                "target_enrichment_loss": target_ret["total_loss"],
                "target_retrieval_loss": target_ret["target_retrieval_loss"],
            }
            for metric_key, metric_value in target_ret.items():
                if not (metric_key.startswith("target_") or metric_key.startswith("mixer/")):
                    continue
                if torch.is_tensor(metric_value) and metric_value.numel() == 1:
                    ret[metric_key] = metric_value.detach()

        zero = zero_source.float().sum() * 0.0
        host_loss = zero
        if use_host_loss:
            label_hat = batch['label_hat'].to(i_feats.device)
            loss1, loss2 = objectives.compute_rbs(i_feats, t_feats, i_tse_f, t_tse_f, batch['pids'], \
                                                  label_hat=label_hat, margin=self.args.margin,tau=self.args.tau,\
                                                    loss_type=self.loss_type,logit_scale=self.logit_scale)
            ret.update({'bge_loss':loss1})
            ret.update({'tse_loss':loss2})
            host_loss = getattr(self.args, "lambda_host", 1.0) * (loss1 + loss2)

        ret["host_loss"] = host_loss
        ret["loss"] = host_loss + ret.get("target_enrichment_loss", zero)
  
        return ret


def build_model(args, num_classes=11003):
    model = RDE(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
