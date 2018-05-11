# Retinal Vessel Segmentation

# Table of Contents
1. [Introduction](README.md#Introduction)
2. [Approach](README.md#Approach)
3. [Dependencies](README.md#Dependencies)
4. [Instructions](README.md#Instructions)

# Introduction

# Approach
1. Data preprocessing
2. Label extraction
3. FCNN classifier generation
4. Train model
5. Predict data

# Dependencies
1. Valid python 2.7+/3.5+
2. Install Tensorflow with low level instruction for CPU/GPU [please refer](https://www.tensorflow.org/install/)
3. Install required libraries 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `pip install -r requirements.txt`


# Dependencies 
1. Check python style or static checks (>8.5 score)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `pylint python/*.py`

2. Run train phase for different classifiers
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `python3 python/dev2_model.py --classification 4 --cache`

3. Run train phase for different classifiers
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `python3 python/dev2_model.py --classification 4 --predict --cache`
