# Vehicle_Re-Identification_CNN

* ## INFO  
The program automatically performs a training on both backbone imagenet neural networks: ResNet-18 and ResNet-50.  
The parameters can set on which dataset (in the **datasets** folder) operate and whether or not use attention mechanism.  

* ## Install requirements  
```
pip install -r requirements.txt  
```
### NOTE (if GPU available):  
```
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118<br>
```

* ## train CNN on VRU dataset with ResNet-18 and ResNet-50 to extract vehicle class features and distance metrics<br>
USE (python src/main.py --dataset [DATASET] --[train | evaluate]) {--attention}:  
```
python src/main.py --dataset VRU --train --attention
python src/main.py --dataset VRU --evaluate
```

* ## Datasets are expected to have the structure:    
```
DATASET_NAME
    train_list.txt
    test_list.txt
    images (also splitted for train/test)
```

every line of both lists.txt must be of the form:  
```
    <relative_image_path> <pid> 
```

* ## NOTE:  
Datasets available at the following links
### VRU: [github link to google open sourced data](https://github.com/GeoX-Lab/ReID/tree/main/VRU)  
### VeRi776: [keggle site](https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset)