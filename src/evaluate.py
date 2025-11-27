import torch
import numpy as np
from tqdm import tqdm

def extract_features(model, loader):
    model.eval()
    feats, pids = [], []

    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader):
            imgs = imgs.to(next(model.parameters()).device)
            f = model(imgs, return_feature=True)
            feats.append(f.cpu().numpy())
            pids.extend(labels.numpy())

    feats = np.vstack(feats)
    return feats, np.array(pids)


def compute_distance_matrix(qf, gf, metric='cosine'):
    """ Cosine / Euclidean distance """
    if metric=="cosine":
        # normalized vectors -> cosine similarity
        qf_norm = qf / np.linalg.norm(qf, axis=1, keepdims=True)
        gf_norm = gf / np.linalg.norm(gf, axis=1, keepdims=True)
        distmat = 1 - np.dot(qf_norm, gf_norm.T)
    
    elif metric=="euclidean":
        distmat = np.sqrt(np.sum((qf[:, None] - gf[None, :])**2, axis=2))
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")
    return distmat


def evaluate_metrics(distmat, pids):
    idx = np.argsort(distmat, axis=1)
    matches = (pids[idx] == pids[:, None])

    all_cmc, all_AP = [], []
    for i in range(len(pids)):
        m = matches[i]
        if not np.any(m):
            continue
        pos = np.where(m)[0]
        first = pos[0]

        cmc = np.zeros(len(pids))
        cmc[first:] = 1
        all_cmc.append(cmc)

        precision = np.cumsum(m) / (np.arange(len(pids)) + 1)
        all_AP.append((precision * m).sum() / m.sum())

    cmc = np.mean(all_cmc, axis=0)
    mAP = np.mean(all_AP)
    return mAP, cmc

"""def evaluate(qf, qpid, gf, gpid):
    distmat = compute_distmat(qf, gf)
    idx = np.argsort(distmat, axis=1)

    matches = (gpid[idx] == qpid[:, None])

    # CMC + mAP
    all_cmc, all_AP = [], []

    for i in range(len(qpid)):
        m = matches[i]
        if not np.any(m):
            continue
        
        # CMC
        pos = np.where(m == True)[0]
        first = pos[0]

        cmc = np.zeros(len(gpid))
        cmc[first:] = 1
        all_cmc.append(cmc)

        # AP
        precision = np.cumsum(m) / (np.arange(len(gpid)) + 1)
        all_AP.append((precision * m).sum() / m.sum())

    cmc = np.mean(all_cmc, axis=0)
    mAP = np.mean(all_AP)

    return mAP, cmc[0], cmc
"""


