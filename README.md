# Automatic-License-Plate-Detection YOLOv10 + easyOCR
## Demo

[![License Plate](https://github.com/drago467/Automatic-License-Plate-Detection/blob/main/web/static/upload/image1.jpg)]

[![License Plate](https://github.com/drago467/Automatic-License-Plate-Detection/blob/main/web/static/upload/image2.jpg)]

## Data
The video I used in this project can be downloaded [here](https://www.kaggle.com/datasets/duydieunguyen/licenseplates)

## Model
A YOLOv10 pre-trained model (YOLOv10) was used to detect license plate.


## Project Setup

* Make an environment with python=3.8 using the following command 
``` bash
conda create --prefix ./env python==3.8 -y
```
* Activate the environment
``` bash
conda activate ./env
``` 

* Install the project dependencies using the following command 
```bash
pip install -r requirements.txt
```
* Run app.py 
``` python
python app.py
```
