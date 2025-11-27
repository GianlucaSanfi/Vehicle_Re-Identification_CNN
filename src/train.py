import torch
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, ce_loss, tri_loss, epoch):
    model.train()
    total_loss = 0
    for imgs, pids, _ in tqdm(loader, desc=f"Epoch {epoch+1}"):
        imgs = imgs.to(model.backbone[0].weight.device)
        pids = pids.to(model.backbone[0].weight.device)

        logits, feat = model(imgs)
        loss_id = ce_loss(logits, pids)
        loss_tri = tri_loss(torch.nn.functional.normalize(feat, p=2, dim=1), pids)
        loss = loss_id + loss_tri

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


""" def train(model, loader, optimizer, ce_loss, tri_loss):
    model.train()
    total_loss = 0

    for imgs, pids, _ in tqdm(loader):
        imgs, pids = imgs.to(DEVICE), pids.to(DEVICE)

        logits, feat = model(imgs, return_feature=False)

        loss_id = ce_loss(logits, pids)
        loss_tri = tri_loss(F.normalize(feat, p=2, dim=1), pids)

        loss = loss_id + loss_tri

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total / len(loader)

 """