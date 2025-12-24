# Vehicle_Re-Identification_CNN

* Install requirements:
pip install -r requirements.txt
NOTE (if GPU available):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118



* train CNN on VRU dataset with ResNet-18 and ResNet-50 to extract vehicle class features and distance metrics
USE (python src/main.py --dataset [DATASET] --[train | evaluate]) {--attention}:
python src/main.py --dataset VRU --train --attention
python src/main.py --dataset VRU --evaluate

* Datasets are expected to have the structure:
DATASET_NAME
    train_list.txt
    test_list.txt
    images (also splitted for train/test)

both lists.txt are like follows:
    <relative_path> <pid>
