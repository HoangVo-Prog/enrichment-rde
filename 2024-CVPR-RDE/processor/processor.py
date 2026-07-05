import logging
import os
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from utils.wandb_utils import log_wandb
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import numpy as np
from matplotlib import pyplot as plt
from pylab import xticks,yticks,np
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from model.enrichment import TargetPoolManager


################### CODE FOR THE BETA MODEL  ########################

import scipy.stats as stats
def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta

class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end

    def look_lookup(self, x):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)


def split_prob(prob, threshld):
    if prob.min() > threshld:
        """From https://github.com/XLearning-SCU/2021-NeurIPS-NCR"""
        # If prob are all larger than threshld, i.e. no noisy data, we enforce 1/100 unlabeled data
        print('No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled.')
        threshld = np.sort(prob)[len(prob)//100]
    pred = (prob > threshld)
    return (pred+0)

def get_loss(model, data_loader):
    logger = logging.getLogger("RDE.train")
    model.eval()
    device = "cuda"
    data_size = data_loader.dataset.__len__()
    real_labels = data_loader.dataset.real_correspondences
    lossA, lossB, simsA,simsB = torch.zeros(data_size), torch.zeros(data_size), torch.zeros(data_size),torch.zeros(data_size)
    for i, batch in enumerate(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        index = batch['index']
        with torch.no_grad(): 
            la, lb, sa, sb = model.compute_per_loss(batch)
            for b in range(la.size(0)):
                lossA[index[b]]= la[b]
                lossB[index[b]]= lb[b]
                simsA[index[b]]= sa[b]
                simsB[index[b]]= sb[b]
            if i % 100 == 0:
                logger.info(f'compute loss batch {i}')

    losses_A = (lossA-lossA.min())/(lossA.max()-lossA.min())    
    losses_B = (lossB-lossB.min())/(lossB.max()-lossB.min())
    
    input_loss_A = losses_A.reshape(-1,1) 
    input_loss_B = losses_B.reshape(-1,1)
 
    logger.info('\nFitting GMM ...') 
 
    if model.args.noisy_rate > 0.4 or model.args.dataset_name=='RSTPReid':
        # should have a better fit 
        gmm_A = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
        gmm_B = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
    else:
        gmm_A = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm_B = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)

    gmm_A.fit(input_loss_A.cpu().numpy())
    prob_A = gmm_A.predict_proba(input_loss_A.cpu().numpy())
    prob_A = prob_A[:, gmm_A.means_.argmin()]

    gmm_B.fit(input_loss_B.cpu().numpy())
    prob_B = gmm_B.predict_proba(input_loss_B.cpu().numpy())
    prob_B = prob_B[:, gmm_B.means_.argmin()]
 
 
    pred_A = split_prob(prob_A, 0.5)
    pred_B = split_prob(prob_B, 0.5)
  
    return torch.Tensor(pred_A), torch.Tensor(pred_B)


def _meter_value(value):
    if torch.is_tensor(value):
        return value.detach().item()
    return float(value)


def _scalar_value(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return value.detach().float().item()
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _is_trainable_loss(value):
    return torch.is_tensor(value) and value.numel() == 1 and value.requires_grad


def _is_loss_key(key):
    return key == "loss" or key.endswith("_loss")


def _should_track_log_scalar(key):
    return "loss" in key or key.endswith("grad_norm")


def _should_track_wandb_scalar(key):
    return (
        "loss" in key
        or key.endswith("grad_norm")
        or key.startswith("pool_")
        or key.startswith("target_")
        or key.startswith("mixer/")
    )


def _grad_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        param_norm = parameter.grad.detach().data.float().norm(2).item()
        total += param_norm ** 2
    return total ** 0.5


def _loss_grad_norm(loss, parameters):
    if not _is_trainable_loss(loss):
        return 0.0
    if not parameters:
        return 0.0
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    total = 0.0
    for grad in grads:
        if grad is None:
            continue
        grad_norm = grad.detach().float().norm(2).item()
        total += grad_norm ** 2
    return total ** 0.5


def _iter_loss_grad_sources(ret):
    sources = {}
    explicit_sources = ret.get("_loss_grad_sources", {})
    if isinstance(explicit_sources, dict):
        for key, value in explicit_sources.items():
            if _is_loss_key(key):
                sources[key] = value

    for key, value in ret.items():
        if key == "_loss_grad_sources":
            continue
        if _is_loss_key(key) and key not in sources:
            sources[key] = value
    return sources.items()


def _update_meter(meters, key, value, batch_size):
    if key not in meters:
        meters[key] = AverageMeter()
    meters[key].update(value, batch_size)


def _set_loader_epoch(loader, epoch):
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    batch_sampler = getattr(loader, "batch_sampler", None)
    batch_sampler_inner = getattr(batch_sampler, "sampler", None)
    if hasattr(batch_sampler_inner, "set_epoch") and batch_sampler_inner is not sampler:
        batch_sampler_inner.set_epoch(epoch)


def _move_train_batch_to_device(batch, device, pnp_text_only=False):
    if pnp_text_only:
        return {key: (value if key == "images" else value.to(device)) for key, value in batch.items()}
    return {key: value.to(device) for key, value in batch.items()}


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _set_wandb_summary(wandb_run, summary_values):
    if wandb_run is None or not hasattr(wandb_run, "summary"):
        return
    for key, value in summary_values.items():
        wandb_run.summary[key] = int(value)


def _count_module_tensors(module, name_filter=None):
    def _include(name):
        return name_filter is None or name_filter(name)

    total_params = 0
    trainable_params = 0
    for name, parameter in module.named_parameters():
        if not _include(name):
            continue
        total_params += parameter.numel()
        if parameter.requires_grad:
            trainable_params += parameter.numel()

    buffer_params = sum(
        buffer.numel()
        for name, buffer in module.named_buffers()
        if _include(name)
    )
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "buffers": buffer_params,
    }


def _flatten_param_summary(prefix, stats):
    return {
        f"{prefix}_total_params": stats["total_params"],
        f"{prefix}_trainable_params": stats["trainable_params"],
        f"{prefix}_frozen_params": stats["frozen_params"],
        f"{prefix}_buffers": stats["buffers"],
    }


def _log_param_scope(logger, label, stats):
    logger.info(
        "%s params: total=%s (%.3fM), trainable=%s (%.3fM), frozen=%s (%.3fM), "
        "buffers=%s (%.3fM)",
        label,
        f"{stats['total_params']:,}",
        stats["total_params"] / 1_000_000.0,
        f"{stats['trainable_params']:,}",
        stats["trainable_params"] / 1_000_000.0,
        f"{stats['frozen_params']:,}",
        stats["frozen_params"] / 1_000_000.0,
        f"{stats['buffers']:,}",
        stats["buffers"] / 1_000_000.0,
    )


def _log_enrichment_branch_size(model, logger, wandb_run=None):
    model = _unwrap_model(model)
    target_enricher = getattr(model, "target_enricher", None)
    model_stats = _count_module_tensors(model)
    host_stats = _count_module_tensors(
        model,
        name_filter=lambda name: not name.startswith("target_enricher."),
    )
    summary = {}
    summary.update(_flatten_param_summary("model", model_stats))
    summary.update(_flatten_param_summary("host", host_stats))

    _log_param_scope(logger, "Model", model_stats)
    _log_param_scope(logger, "Host", host_stats)

    if target_enricher is None:
        logger.info("Target enrichment branch params: disabled")
        summary.update(_flatten_param_summary("target_enrichment_branch", {
            "total_params": 0,
            "trainable_params": 0,
            "frozen_params": 0,
            "buffers": 0,
        }))
        _set_wandb_summary(wandb_run, summary)
        return

    target_stats = _count_module_tensors(target_enricher)
    _log_param_scope(logger, "Target enrichment branch", target_stats)
    summary.update(_flatten_param_summary("target_enrichment_branch", target_stats))
    _set_wandb_summary(wandb_run, summary)


def _target_enrichment_active(args, epoch):
    enrichment_start = getattr(args, "enrichment_start", 1)
    if enrichment_start < 1:
        raise ValueError("--enrichment_start must be a positive integer")
    return getattr(args, "target_enrichment", False) and epoch >= enrichment_start


def _should_run_eval(args, epoch):
    eval_after_epoch = getattr(args, "eval_after_epoch", 0)
    if eval_after_epoch < 0:
        raise ValueError("--eval_after_epoch must be a non-negative integer")
    return epoch >= eval_after_epoch



def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer, target_pool=None, wandb_run=None):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("RDE.train")
    logger.info('start training')
    if get_rank() == 0:
        _log_enrichment_branch_size(model, logger, wandb_run=wandb_run)
    if target_pool is None and getattr(args, "target_enrichment", False):
        target_pool = TargetPoolManager(train_loader.dataset, args, logger)
    if target_pool is not None and getattr(args, "enrichment_start", 1) > 1:
        logger.info(
            "Target enrichment delayed until epoch {}; earlier epochs use host training only".format(
                args.enrichment_start
            )
        )
    if getattr(args, "eval_after_epoch", 0) > 0:
        logger.info(
            "Evaluation delayed until epoch {}; earlier epochs skip validation".format(
                args.eval_after_epoch
            )
        )

    meters = {
        "loss": AverageMeter(),
        "bge_loss": AverageMeter(),
        "tse_loss": AverageMeter(),
        "host_loss": AverageMeter(),
        "target_enrichment_loss": AverageMeter(),
        "target_retrieval_loss": AverageMeter(),
        "grad_norm": AverageMeter(),
        "loss_grad_norm": AverageMeter(),
        "bge_loss_grad_norm": AverageMeter(),
        "tse_loss_grad_norm": AverageMeter(),
        "host_loss_grad_norm": AverageMeter(),
        "target_enrichment_loss_grad_norm": AverageMeter(),
        "target_retrieval_loss_grad_norm": AverageMeter(),
    }
    wandb_meters = {}

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0
    best_epoch = None
    now_top1 = 0.0
    current_steps = 0
    if _should_run_eval(args, 0):
        initial_top1 = evaluator.eval(
            model.eval(),
            use_target_enrichment=_target_enrichment_active(args, start_epoch),
        )
        if get_rank() == 0:
            initial_metrics = dict(getattr(evaluator, "last_metrics", {}))
            initial_metrics["eval/top_R1"] = initial_top1
            if "eval/ablation_best_R1" in initial_metrics:
                initial_metrics["eval/best_ablation_R1"] = initial_top1
            log_wandb(wandb_run, initial_metrics, step=0, epoch=start_epoch - 1)
    # evaluator.eval(model.eval())
    # train
    sims = []
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        for meter in wandb_meters.values():
            meter.reset()

        # model.train()
        model.epoch = epoch
        _set_loader_epoch(train_loader, epoch)
        # data_size = train_loader.dataset.__len__()
        # pred_A, pred_B  =  torch.ones(data_size), torch.ones(data_size)
    
        if getattr(args, "use_host_loss", True):
            pred_A, pred_B = get_loss(model, train_loader)
        
            consensus_division = pred_A + pred_B # 0,1,2 
            consensus_division[consensus_division==1] += torch.randint(0, 2, size=(((consensus_division==1)+0).sum(),))
            label_hat = consensus_division.clone()
            label_hat[consensus_division>1] = 1
            label_hat[consensus_division<=1] = 0 
        else:
            label_hat = None
        
        model.train() 
        use_target_enrichment = target_pool is not None and _target_enrichment_active(args, epoch)
        if target_pool is not None and epoch == getattr(args, "enrichment_start", 1):
            logger.info("Target enrichment starts at epoch {}".format(epoch))
        for n_iter, batch in enumerate(train_loader):
            current_steps += 1
            batch = _move_train_batch_to_device(
                batch,
                device,
                pnp_text_only=getattr(args, "pnp_text_only", False),
            )
            index = batch['index']
            
            if label_hat is not None:
                batch['label_hat'] = label_hat[index.cpu()]
 
            target_cache = None
            if use_target_enrichment:
                target_cache = target_pool.get_train_cache(model, batch, epoch, current_steps)

            ret = model(batch, epoch=epoch, current_step=current_steps, target_cache=target_cache)
            if target_cache is not None and "diagnostics" in target_cache:
                for diag_key, diag_value in target_cache["diagnostics"].items():
                    if isinstance(diag_value, (int, float)):
                        ret[diag_key] = diag_value
            total_loss = ret.get("loss")
            if total_loss is None:
                total_loss = sum([v for k, v in ret.items() if "loss" in k and _is_trainable_loss(v)])

            batch_size = batch['images'].shape[0]
            loss_scalar = _scalar_value(total_loss)
            if loss_scalar is None:
                raise ValueError("Training forward must return a scalar loss")
            meters['loss'].update(loss_scalar, batch_size)
            _update_meter(wandb_meters, "loss", loss_scalar, batch_size)
            for key, value in ret.items():
                if key == "loss":
                    continue
                scalar = _scalar_value(value)
                if scalar is None:
                    continue
                if _should_track_log_scalar(key):
                    _update_meter(meters, key, scalar, batch_size)
                if _should_track_wandb_scalar(key):
                    _update_meter(wandb_meters, key, scalar, batch_size)
         
            optimizer.zero_grad()
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            loss_grad_sources = dict(_iter_loss_grad_sources(ret))
            loss_grad_sources["loss"] = total_loss
            for loss_key, loss_value in loss_grad_sources.items():
                grad_norm_key = f"{loss_key}_grad_norm"
                grad_norm_value = _loss_grad_norm(loss_value, trainable_params)
                _update_meter(meters, grad_norm_key, grad_norm_value, batch_size)
                _update_meter(wandb_meters, grad_norm_key, grad_norm_value, batch_size)
            total_loss.backward()
            grad_norm_value = _grad_norm(model.parameters())
            meters['grad_norm'].update(grad_norm_value, batch_size)
            _update_meter(wandb_meters, "grad_norm", grad_norm_value, batch_size)
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.count > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
                if get_rank() == 0:
                    train_metrics = {
                        "train/{}".format(k): v.avg
                        for k, v in wandb_meters.items()
                        if v.count > 0
                    }
                    train_metrics["train/lr"] = scheduler.get_lr()[0]
                    train_metrics["train/temperature"] = _scalar_value(ret.get("temperature"))
                    log_wandb(wandb_run, train_metrics, step=current_steps, epoch=epoch)
        
 
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        temperature = _scalar_value(ret.get('temperature'))
        if temperature is not None:
            tb_writer.add_scalar('temperature', temperature, epoch)
        for k, v in meters.items():
            if v.count > 0:
                tb_writer.add_scalar(k, v.avg, epoch)
        if get_rank() == 0:
            epoch_metrics = {
                "train_epoch/{}".format(k): v.avg
                for k, v in wandb_meters.items()
                if v.count > 0
            }
            epoch_metrics["train_epoch/lr"] = scheduler.get_lr()[0]
            epoch_metrics["train_epoch/temperature"] = temperature
            log_wandb(wandb_run, epoch_metrics, step=current_steps, epoch=epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0 and _should_run_eval(args, epoch):
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                eval_model = model.module.eval() if args.distributed else model.eval()
                top1 = evaluator.eval(
                    eval_model,
                    use_target_enrichment=_target_enrichment_active(args, epoch),
                )
                now_top1 = max(now_top1, top1)
                eval_metrics = dict(getattr(evaluator, "last_metrics", {}))
                eval_metrics["eval/top_R1"] = top1
                eval_metrics["eval/best_R1"] = now_top1
                if "eval/ablation_best_R1" in eval_metrics:
                    eval_metrics["eval/best_ablation_R1"] = now_top1
                log_wandb(wandb_run, eval_metrics, step=current_steps, epoch=epoch)

                torch.cuda.empty_cache()
                top1 = float(top1)
                if best_top1 < top1:
                    best_top1 = top1
                    best_epoch = epoch
                    arguments["epoch"] = epoch
                    arguments["best_top1"] = best_top1
                    if wandb_run is not None:
                        wandb_run.summary["best_R1"] = float(best_top1)
                        wandb_run.summary["best_R1_row"] = str(getattr(evaluator, "last_best_task", ""))
                    checkpointer.save("best", **arguments)
 
    if get_rank() == 0:
        if best_epoch is None:
            logger.info("No validation results were produced; best checkpoint was not saved.")
        else:
            logger.info(f"best R1: {best_top1} at epoch {best_epoch}")

    arguments["epoch"] = epoch
    checkpointer.save("last", **arguments)
                    
def do_inference(model, test_img_loader, test_txt_loader, args=None, use_target_enrichment=None):

    logger = logging.getLogger("RDE.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader, args)
    top1 = evaluator.eval(model.eval(), use_target_enrichment=use_target_enrichment)
    return top1
