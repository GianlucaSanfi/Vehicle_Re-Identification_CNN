# create_veri776_lists.py
import os
import random

#################
#structure of images in dataset: vehicle ID from the first 4 digits of the filename
#################


# ------------------------------
# CONFIG
# ------------------------------
dataset_root = "datasets/VeRi776"   # Change if your VeRi776 folder is elsewhere
train_dir = os.path.join(dataset_root, "image_train")
query_dir = os.path.join(dataset_root, "image_query")
gallery_dir = os.path.join(dataset_root, "image_test")  # called image_test in your structure

output_train_list = os.path.join(dataset_root, "train_list.txt")
output_test_list = os.path.join(dataset_root, "test_list.txt")

# ------------------------------
# HELPER
# ------------------------------
def create_list(image_dir, output_file):
    samples = []
    for fname in os.listdir(image_dir):
        if fname.endswith(".jpg") or fname.endswith(".png"):
            # Assuming VeRi filenames like: 0001_c001_000123.jpg
            # First 4 digits = vehicle ID
            pid = int(fname[:4])
            path = os.path.join("image_train" if "train" in output_file else os.path.basename(image_dir), fname)
            samples.append((path, pid))

    # Shuffle samples
    random.shuffle(samples)

    # Write file
    with open(output_file, "w") as f:
        for path, pid in samples:
            f.write(f"{path} {pid}\n")
    print(f"Saved {len(samples)} samples to {output_file}")

# ------------------------------
# Create train_list.txt
# ------------------------------
create_list(train_dir, output_train_list)

# ------------------------------
# Create test_list.txt (query + gallery)
# ------------------------------
# Combine query and gallery for evaluation
test_samples = []

for folder in [query_dir, gallery_dir]:
    for fname in os.listdir(folder):
        if fname.endswith(".jpg") or fname.endswith(".png"):
            pid = int(fname[:4])
            path = os.path.join(os.path.basename(folder), fname)
            test_samples.append((path, pid))

random.shuffle(test_samples)

with open(output_test_list, "w") as f:
    for path, pid in test_samples:
        f.write(f"{path} {pid}\n")
print(f"Saved {len(test_samples)} test samples to {output_test_list}")
