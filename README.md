# Vehicle_Re-Identification_CNN

* Install requirements:<br>
pip install -r requirements.txt<br>
NOTE (if GPU available):<br>
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118<br>

<br>

* train CNN on VRU dataset with ResNet-18 and ResNet-50 to extract vehicle class features and distance metrics<br>
USE (python src/main.py --dataset [DATASET] --[train | evaluate]):<br>
python src/main.py --dataset VRU --train<br>
python src/main.py --dataset VRU --evaluate<br>



