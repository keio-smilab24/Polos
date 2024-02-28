from typing import *
import random

import torch
from torch import Tensor

from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_info
import clip
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

def read_image(imgid):
    from pathlib import Path
    vanilla = Path(imgid)
    fixed = Path(f"data_en/images/{imgid}")
    assert not (vanilla.exists() == fixed.exists()) # 両者共に存在/不在だと困る

    path = vanilla if vanilla.exists() else fixed
    return Image.open(path).convert("RGB")


class MID():
    def __init__(self,device="cuda"):
        self.clip, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.device = device

    def batchify(self, targets, batch_size):
        return [targets[i:i+batch_size] for i in range(0,len(targets),batch_size)]

    def __call__(self, mt_list, refs_list, img_list, no_ref=False):
        B = 32
        mt_list, refs_list, img_list = [self.batchify(x,B) for x in [mt_list,refs_list,img_list]]
        scores = []
        assert len(mt_list) == len(refs_list) == len(img_list)
        for mt, refs, imgs in (pbar:= tqdm(zip(mt_list,refs_list, img_list),total=len(mt_list))):
            pbar.set_description(f"MID")
            imgs = [read_image(imgid) for imgid in imgs]
            refs_token = []
            for ref_list in refs:
                refs_token.append([clip.tokenize(ref,truncate=True).to(self.device) for ref in ref_list])

            refs = torch.cat([torch.cat(ref,dim=0) for ref in refs_token], dim=0)
            mts = clip.tokenize([x for x in mt],truncate=True).to(self.device)
            imgs = torch.cat([self.clip_preprocess(img).unsqueeze(0) for img in imgs],dim=0).to(self.device)

            imgs = self.clip.encode_image(imgs)
            mts = self.clip.encode_text(mts)
            refs = self.clip.encode_text(refs)
            compute_pmi(imgs,refs,mts)
        
        return scores



def log_det(X):
    eigenvalues = X.svd()[1]
    return eigenvalues.log().sum()


def robust_inv(x, eps=0):
    Id = torch.eye(x.shape[0]).to(x.device)
    return (x + eps * Id).inverse()


def exp_smd(a, b, reduction=True):
    a_inv = robust_inv(a)
    if reduction:
        assert b.shape[0] == b.shape[1]
        return (a_inv @ b).trace()
    else:
        return (b @ a_inv @ b.t()).diag()


def compute_pmi(x: Tensor, y: Tensor, x0: Tensor, limit: int = 30000,
                 reduction: bool = True, full: bool = False) -> Tensor:
    r"""
    A numerical stable version of the MID score.

    Args:
        x (Tensor): features for real samples
        y (Tensor): features for text samples
        x0 (Tensor): features for fake samples
        limit (int): limit the number of samples
        reduction (bool): returns the expectation of PMI if true else sample-wise results
        full (bool): use full samples from real images

    Returns:
        Scalar value of the mutual information divergence between the sets.
    """
    N = x.shape[0]
    excess = N - limit
    if 0 < excess:
        if not full:
            x = x[:-excess]
            y = y[:-excess]
        x0 = x0[:-excess]
    N = x.shape[0]
    M = x0.shape[0]

    assert N >= x.shape[1], "not full rank for matrix inversion!"
    if x.shape[0] < 30000:
        rank_zero_info("if it underperforms, please consider to use "
                       "the epsilon of 5e-4 or something else.")

    z = torch.cat([x, y], dim=-1)
    z0 = torch.cat([x0, y[:x0.shape[0]]], dim=-1)
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    z_mean = torch.cat([x_mean, y_mean], dim=-1)
    x0_mean = x0.mean(dim=0, keepdim=True)
    z0_mean = z0.mean(dim=0, keepdim=True)

    X = (x - x_mean).t() @ (x - x_mean) / (N - 1)
    Y = (y - y_mean).t() @ (y - y_mean) / (N - 1)
    Z = (z - z_mean).t() @ (z - z_mean) / (N - 1)
    X0 = (x0 - x_mean).t() @ (x0 - x_mean) / (M - 1)  # use the reference mean
    Z0 = (z0 - z_mean).t() @ (z0 - z_mean) / (M - 1)  # use the reference mean

    alternative_comp = False
    # notice that it may have numerical unstability. we don't use this.
    if alternative_comp:
        def factorized_cov(x, m):
            N = x.shape[0]
            return (x.t() @ x - N * m.t() @ m) / (N - 1)
        X0 = factorized_cov(x0, x_mean)
        Z0 = factorized_cov(z0, z_mean)

    # assert double precision
    for _ in [X, Y, Z, X0, Z0]:
        assert _.dtype == torch.float64

    # Expectation of PMI
    mi = (log_det(X) + log_det(Y) - log_det(Z)) / 2
    rank_zero_info(f"MI of real images: {mi:.4f}")

    # Squared Mahalanobis Distance terms
    if reduction:
        smd = (exp_smd(X, X0) + exp_smd(Y, Y) - exp_smd(Z, Z0)) / 2
    else:
        smd = (exp_smd(X, x0 - x_mean, False) + exp_smd(Y, y - y_mean, False)
               - exp_smd(Z, z0 - z_mean, False)) / 2
        mi = mi.unsqueeze(0)  # for broadcasting

    return mi + smd
