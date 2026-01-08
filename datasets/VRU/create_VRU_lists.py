import os

SPLIT_DIR = "train_test_split"

def process_file(split_file, output_file):
    with open(split_file, "r") as f, open(output_file, "w") as out:
        for line in f:
            img_id, pid = line.strip().split()
            pid = int(pid)

            img_name = f"{img_id}.jpg"
            img_path = os.path.join("Pic", img_name)

            if not os.path.exists(img_path):
                print(f"[WARNING] Missing image: {img_path}")
                break

            out.write(f"{img_path} {pid}\n")


def main():
    print("Creating VRU list files using explicit PIDs...")

    train_txt = os.path.join(SPLIT_DIR, "train_list.txt")
    test_txt = os.path.join(SPLIT_DIR, "test_list_1200.txt")
    assert os.path.exists(train_txt), "train not found"
    assert os.path.exists(test_txt), "test not found"

    process_file(train_txt, "train_list.txt")
    process_file(test_txt, "test_list.txt")

    print("Done!")
    

if __name__ == "__main__":
    main()