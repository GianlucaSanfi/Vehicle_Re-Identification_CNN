import torch
import numpy as np
from tqdm import tqdm

def extract_features(model, loader):
    print("Extracting features...")

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
        #distmap = euclidean_chunked(qf, gf)
        distmat = np.sqrt(np.sum((qf[:, None] - gf[None, :])**2, axis=2))
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")
    return distmat

## memory safe euclidean distance (chunked distance)
def euclidean_chunked(qf, gf, chunk=500):
    dist = []
    for i in range(0, len(qf), chunk):
        d = np.sqrt(np.sum((qf[i:i+chunk, None] - gf[None, :])**2, axis=2))
        dist.append(d)
    return np.vstack(dist)


def evaluate_metrics(distmat, q_pids, g_pids):
    num_q = distmat.shape[0]
    cmc_list = []
    ap_list = []

    for i in range(num_q):
        order = np.argsort(distmat[i])
        matches = (g_pids[order] == q_pids[i])

        if not np.any(matches):
            continue

        # CMC
        cmc = np.zeros(len(g_pids))
        first_match = np.where(matches)[0][0]
        cmc[first_match:] = 1
        cmc_list.append(cmc)

        # AP
        precision = np.cumsum(matches) / (np.arange(len(matches)) + 1)
        ap = (precision * matches).sum() / matches.sum()
        ap_list.append(ap)

    return np.mean(ap_list), np.mean(cmc_list, axis=0)
    
    """ 
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
 """


