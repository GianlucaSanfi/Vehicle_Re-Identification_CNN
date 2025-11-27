# Vehicle_Re-Identification_CNN

* Install requirements:\n
pip install -r requirements.txt\n
NOTE (if GPU available):\n
    \tpip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n



* train CNN on VRU dataset with ResNet-18 and ResNet-50 to extract vehicle class features and distance metrics\n
USE (python src/main.py --dataset [DATASET] --[train | evaluate]):\n
python src/main.py --dataset VRU --train\n
python src/main.py --dataset VRU --evaluate\n


