# LAB2
EEG Classification with BCI competition dataset

## Datasets
The training and testing data in BCI Competition III – IIIb.

---

## 📁 Project Structure
```
LAB2/
│
├── models              # EEGNet and DeepConvNet
│    ├── model_ELU
│    ├── model_ReLU
│    └── model_LeakyReLU
├── data_LAB2            # Datasets
│    ├── S4b_train.npz
│    ├── S4b_test.npz
│    ├── X11b_train.npz
│    └── X11b_test.npz
│ 
├── dataloader.py        # Image preprocessing (CLAHE, resize to 512×512)
├── main.py              # Training pipeline for classification models
│
├── plots_LAB2/          # Accuracy and Loss curves
├── pkls_LAB2/          # Accuracy and Loss curves
└── Source/              # Trained model weights (.pkl) -->　In Google Cloud
```
- **Because of the file size limit, you can download the model weights (.pkl files) from Google Cloud :** [pkls](https://drive.google.com/drive/folders/1MaRhkFk5fxD5Tn6RfLvimDjYHDQ80gMe?usp=sharing)
---
## Code main.py
Different models choose different ==> model_ft

For resnet model:
- num_ftrs = model_ft.fc.in_features \n
- model_ft.fc = nn.Linear(num_ftrs, n_class) 

For not resnet model:
- in_features = model_ft.get_classifier().in_features
- model_ft.reset_classifier(num_classes=n_class)
```
# select model
model_select = 'vgg16'

# For vgg efficientnet densenet  resnet
model_ft = timm.create_model(model_select, pretrained=True)

# For 'vit_base_patch16_224'
#model_ft = timm.create_model(model_select, pretrained=True,img_size=512) 

# For resnet model
'''
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, n_class)
'''

# For not resnet model
in_features = model_ft.get_classifier().in_features
model_ft.reset_classifier(num_classes=n_class)
```
---
## Folder Descriptions

| Folder | Description |
|:--|:--|
| `csvs/` | Training and validation logs | 
| `cm_plot/` | Confusion matrix heatmaps | 
| `plots/` | Accuracy & F1-score curves |
| `pkls/` | Model weights (.pkl) | 
---
## Results
### Hyper Parameters - Set1
- **Batch size:** 64  
- **Learning rate:** 1e-2  
- **Epochs:** 150  
- **Optimizer:** Adam  
- **Loss function:** `torch.nn.CrossEntropyLoss()`  
- **Activation function:** `ELU()`
###  Accuracy under Hyper Parameters - Set 1
| Activation Function |  EEGNet  | DeepConvNet |
|:--------------------:|:----------:|:------------:|
| **ELU**        | 85.93% | 81.11% |
### Hyper Parameters - Set2
- **Batch size:** 64  
- **Learning rate:** 1e-3 (EEGNet) / 1e-4 (DeepConvNet)
- **Epochs:** 1000 (EEGNet) / 3000 (DeepConvNet) 
- **Optimizer:** Adam  
- **Loss function:** `torch.nn.CrossEntropyLoss()`  
- **Activation function:** `ELU()` / `ReLU()` / `LeakyReLU()`
### Accuracy under Hyper Parameters - Set 2

| Activation Function |   EEGNet   | DeepConvNet |
|:--------------------:|:----------:|:------------:|
| **ELU**             | **87.13%** |   82.59% |
| **ReLU**        | **87.13%** |   83.15% |
| **LeakyReLU**              |   86.39%   |   82.22% |



