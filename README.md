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

* ## Use of the system  
(**NOTE**: the number of epochs is set in __globals.py__ file)  

__python src/main.py --dataset **`<DATASET>`** --[train | evaluate] {--attention} {--no_eval} {--PROGRESSIVE --EPOCHS **`<EPOCHS>`**} {--ON_DATASET **`<EVAL_DATASET>`**}__  

to train (evaluate) the ResNet-18 and ResNet-50 models on a specific dataset with the fixed number of epochs, use:  
```
python src/main.py --dataset VRU --train --attention --no_eval
python src/main.py --dataset VRU --evaluate --attention
```
The flag **no_eval** disengages the __Extended Evaluation__ during training. The evaluation metrics will be generated only during evaluation phase.  

If we have a model trained with a number of epochs (must be set in __globals.py__) and we want to train such model with more epochs, use **PROGRESSIVE** specifying the number of additional EPOCHS to train it.  
__(If the model does not exist it generates an error)__  
__(If the parameters of the existing model do not exist, it restart the original parameters)__  
e.g. train with 5 more epochs:  
```
python src/main.py --dataset VRU --train --attention --no_eval --PROGRESSIVE --EPOCHS 5
```
**Cross-Dataset Evaluation function**    
If we want to evaluate the trained model on a specific dataset, use **ON_DATASET** listing the desired set to use.  
e.g. evaluate on VeRi776 the model trained on VRU:  
```
python src/main.py --dataset VRU --evaluate --attention --ON_DATASET VeRi776
``` 


* ## Datasets are expected to have the structure:    
```
DATASET_NAME
    train_list.txt
    test_list.txt
    images (folder with images used for train/test)
```

every line of both lists.txt must be of the form:  
```
<relative_image_path> <pid> 
```

The suite is provided with two scripts that take as input the available datasets and make them compatible with the expected structure, creating the train and test lists.  
### VRU  
```
python create_VRU_lists.py
```  
### VeRi-776  
```
python create_veri776_lists.py
```  

* ## NOTE:  
Datasets available at the following links
### VRU: [github link to google open sourced data](https://github.com/GeoX-Lab/ReID/tree/main/VRU)  
### VeRi776: [keggle site](https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset)