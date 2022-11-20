# CE6190 Project 1: Literature Review on Transformer-based Semantic Segmentation
In this project, we experiment with two recent transformer-based methods: [Segformer](https://github.com/NVlabs/SegFormer) (NeurIPS 2021) and [Segmenter](https://github.com/rstrudel/segmenter) (ICCV 2021)

## Data Preparation
We use a new, uncommon and yet challenging dataset from this [link](https://download.visinf.tu-darmstadt.de/data/from_games/). Due to resource limit, we only use its part 1 data split.
```sh
cd dataset/games
wget https://download.visinf.tu-darmstadt.de/data/from_games/data/01_images.zip
wget https://download.visinf.tu-darmstadt.de/data/from_games/data/01_labels.zip
unzip -q 01_images.zip 
unzip -q 01_labels.zip
mv images source_images
mv labels source_labels
mkdir trainId_labels images labels
python data_processing.py
```
You should see 

```
project
│   README.md
│       
└───dataset
    └───games
        │───images
        │   │───train
        │   │   │   train_00001_image.png
        │   │   │   ...
        │   │
        │   └───val
        │       │   val_02300_image.png
        │       │   ...
        │ 
        │───labels
        │   │───train
        │   │   │   train_00001_labelIds.png
        │   │   │   ...
        │   │
        │   └───val
        │       │   val_02300_labelIds.png
        │       │   ...
        │    
        │───...  
```

## Experiments
under project's root directory, run:
```sh
git clone -b segmenter --single-branch git@github.com:Chen-Hailin/CE6190.git segmenter
git clone -b segformer --single-branch git@github.com:Chen-Hailin/CE6190.git segformer
```
You should have the following file structure:
```
project
│   README.md
│       
└───dataset
│       
└───segmenter
│       
└───segformer
```
Go to `segmenter` or `segformer` directory and follow the readme.md instructions there to run experiments.



