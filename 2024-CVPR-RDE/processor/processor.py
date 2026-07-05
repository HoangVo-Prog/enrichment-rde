import logging
import os
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
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



def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("RDE.train")
    logger.info('start training')
    target_pool = None
    if getattr(args, "target_enrichment", False):
        target_pool = TargetPoolManager(train_loader.dataset, args, logger)

    meters = {
        "loss": AverageMeter(),
        "bge_loss": AverageMeter(),
        "tse_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
    }
    if getattr(args, "target_enrichment", False):
        meters["target_enrichment_loss"] = AverageMeter()
        meters["target_retrieval_loss"] = AverageMeter()

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0
    # evaluator.eval(model.eval())
    # train
    sims = []
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()

        # model.train()
        model.epoch = epoch
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
        for n_iter, batch in enumerate(train_loader):
            if getattr(args, "pnp_text_only", False):
                batch = {k: (v if k == "images" else v.to(device)) for k, v in batch.items()}
            else:
                batch = {k: v.to(device) for k, v in batch.items()}
            index = batch['index']
            
            if label_hat is not None:
                batch['label_hat'] = label_hat[index.cpu()]
 
            global_step = (epoch - 1) * len(train_loader) + n_iter + 1
            target_cache = None
            if target_pool is not None and epoch >= getattr(args, "enrichment_start", 1):
                target_cache = target_pool.get_train_cache(model, batch, epoch, global_step)

            ret = model(batch, epoch=epoch, current_step=global_step, target_cache=target_cache)
            if target_cache is not None and "diagnostics" in target_cache:
                ret.update(target_cache["diagnostics"])
            total_loss = ret["loss"] if "loss" in ret else sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['bge_loss'].update(_meter_value(ret.get('bge_loss', 0)), batch_size)
            meters['tse_loss'].update(_meter_value(ret.get('tse_loss', 0)), batch_size)
            if "target_enrichment_loss" in meters:
                meters['target_enrichment_loss'].update(_meter_value(ret.get('target_enrichment_loss', 0)), batch_size)
                meters['target_retrieval_loss'].update(_meter_value(ret.get('target_retrieval_loss', 0)), batch_size)
         
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
 
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
 
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")

    arguments["epoch"] = epoch
    checkpointer.save("last", **arguments)
                    
def do_inference(model, test_img_loader, test_txt_loader, args=None, use_target_enrichment=None):

    logger = logging.getLogger("RDE.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader, args)
    top1 = evaluator.eval(model.eval(), use_target_enrichment=use_target_enrichment)
    return top1
