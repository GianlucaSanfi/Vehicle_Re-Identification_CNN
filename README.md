# Vehicle_Re-Identification_CNN

* Install requirements:
pip install -r requirements.txt
NOTE (if GPU available):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118



* train CNN on VRU dataset with ResNet-18 and ResNet-50 to extract vehicle class features and distance metrics
USE (python src/main.py --dataset [DATASET] --[train | evaluate]):
python src/main.py --dataset VRU --train
python src/main.py --dataset VRU --evaluate

