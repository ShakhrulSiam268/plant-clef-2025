# plant-clef-2025
![Project Logo](sample_data/new_banner_plantclef2025.png)

![https://www.kaggle.com/competitions/plantclef-2025/overview](https://kaggle.com/static/images/open-in-kaggle.svg)



## Installation Instruction
![Python versions](https://img.shields.io/badge/python-3.10%20-3776AB?logo=python&logoColor=white)

![](https://img.shields.io/badge/torch-2.9.1-blue.svg)
![](https://img.shields.io/badge/torchvision-0.24.1-blue.svg)
![](https://img.shields.io/badge/transformers-4.57.3-blue.svg)
![](https://img.shields.io/badge/scikit--learn-1.7.2-blue.svg)

Installing essential packages
```shell
conda create -n env_name python=3.10
conda activate env_name
pip install -r requirements.txt
```

Installing GroundingDINO
```shell
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e .
```

## Datasets
We used the PlantCLEF dataset in this project. They can be downloaded from the official [competition page](https://www.imageclef.org/PlantCLEF2025).

- Training data (160 GB) [(Download Link)](https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/PlantCLEF2024singleplanttrainingdata_800_max_side_size.tar)
- Test data [(Download Link)](https://lab.plantnet.org/LifeCLEF/PlantCLEF2025/vegetation_plot_test_data/PlantCLEF2025test.tar)

## Feature Extraction
In this project, we used BioCLIP-2 feature extractor. 

Run the following command to Extract features from all training data.
```shell
python data/feature_extract_bioclip2.py
```

## Advanced Algorithm
![Project Logo](sample_data/model.png)
Run the following command to train the MLP classifier.
```shell
python src/train_mlp.py
```
Our trained model weight can be found in [This Link](https://drive.google.com/drive/folders/1FspoSKnjp56iVzfgGC4q13qSOe4KG4E4?usp=drive_link).

## Evaluation
First run `segment_bbox.py` to generate bounding boxes from all the test images using GroudingDINO model.

Then, Run the following command to get classification result using `BioCLIP-2 + GroundingDINO`.
```shell
python src/test_grid_sam.py
```
This will create .csv file containing all predicted class. Submit it to [Kaggle](https://www.kaggle.com/competitions/plantclef-2025/overview) Submission to get official score for evaluation.

## Test Examples
Detecting Bounding Box
![Sample Test Image](sample_data/test_image_1.png)

Generating Prediction
![Sample Test Image](sample_data/example_test.png)

## Results
![Results](sample_data/results.png)