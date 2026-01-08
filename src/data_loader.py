from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import random
from collections import defaultdict
import numpy as np
from torchvision import transforms
from globals import IMG_HEIGHT, IMG_WIDTH, BATCH_P, BATCH_K

# ------------------------
# Dataset
# ------------------------
class ReIDDataset(Dataset):
    def __init__(self, samples, transform=None):
        """samples : list of (image_path, pid)"""
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, pid = self.samples[idx]
        #print(path)
        img = Image.open(path).convert("RGB")
        if self.transform: 
            img = self.transform(img)
        return img, pid, path

# ------------------------
# P-K Sampler
# ------------------------
class PKSampler(Sampler):
    """ P identities × K samples per identity """
    def __init__(self, samples, P=BATCH_P, K=BATCH_K):
        self.P = P
        self.K = K

        self.index_dict = {}
        for idx, (_, pid) in enumerate(samples):
            self.index_dict.setdefault(pid, []).append(idx)

        self.pids = list(self.index_dict.keys())

    def __iter__(self):
        pid_list = self.pids.copy()
        random.shuffle(pid_list)
        batch = []

        for pid in pid_list:
            idxs = self.index_dict[pid]

            if len(idxs) >= self.K:
                chosen = random.sample(idxs, self.K)
            else:
                chosen = np.random.choice(idxs, self.K, replace=True).tolist()

            batch.extend(chosen)

            if len(batch) == self.P * self.K:
                #yield from batch
                yield batch
                batch = []

    def __len__(self):
        return len(self.pids) * self.K

# ------------------------
# Transforms
# ------------------------
train_tf = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1,0.1,0.1,0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_tf = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ------------------------
# Dataset loaders
# ------------------------
def split_query_gallery(samples, num_query_per_id=1):
    """
    Split test samples into query and gallery sets
    """
    pid_dict = defaultdict(list)
    for path, pid in samples:
        pid_dict[pid].append(path)

    query, gallery = [], []

    for pid, paths in pid_dict.items():
        random.shuffle(paths)
        query_paths = paths[:num_query_per_id]
        gallery_paths = paths[num_query_per_id:]

        for p in query_paths:
            query.append((p, pid))
        for p in gallery_paths:
            gallery.append((p, pid))

    return query, gallery
    
def load_dataset(dataset_name):
    """
    Returns train_samples, test_samples
    Each sample: (img_path, pid)
    """
    train_file = f"datasets/{dataset_name}/train_list.txt"
    test_file  = f"datasets/{dataset_name}/test_list.txt"

    # create unique pids to solve index out of bound #
    train_raw = []
    with open(train_file) as f:
        for line in f:
            path, pid = line.strip().split()
            train_raw.append((path, int(pid)))

    # CREATE PID MAPPING
    unique_pids = sorted(set(pid for _, pid in train_raw))
    pid2label = {pid: idx for idx, pid in enumerate(unique_pids)}

    # REMAP TRAIN
    train_samples = []
    for path, pid in train_raw:
        path = path.replace("\\", "/")
        full_path = f"datasets/{dataset_name}/{path}"
        train_samples.append((full_path, pid2label[pid]))

    # REMAP TEST USING SAME MAPPING 
    test_samples = []
    with open(test_file) as f:
        for line in f:
            path, pid = line.strip().split()
            pid = int(pid)

            #if pid not in pid2label:
            #    continue  # unseen identity (optional)

            path = path.replace("\\", "/")
            full_path = f"datasets/{dataset_name}/{path}"
            test_samples.append((full_path, pid))

    """
    train_samples = []
    with open(train_file) as f:
        for line in f:
            path, pid = line.strip().split()
            #path = "datasets/"+dataset_name+"/"+path
            path = path.replace("\\", "/")
            full_path = f"datasets/{dataset_name}/{path}"
            #print(full_path)
            train_samples.append((full_path, int(pid)))

    test_samples = []
    with open(test_file) as f:
        for line in f:
            path, pid = line.strip().split()
            #path = "datasets/"+dataset_name+"/"+path         ### adjustment to work with fake dataset (TEST)
            path = path.replace("\\", "/")
            full_path = f"datasets/{dataset_name}/{path}"
            #print(full_path)
            test_samples.append((full_path, int(pid)))

    """
    return train_samples, test_samples

def build_train_loader(samples, batch_p=BATCH_P, batch_k=BATCH_K):
    dataset = ReIDDataset(samples, transform=train_tf)
    sampler = PKSampler(samples, P=batch_p, K=batch_k)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4, pin_memory=True)
    return loader

def build_test_loader(samples, batch_size=64):
    dataset = ReIDDataset(samples, transform=test_tf)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return loader
