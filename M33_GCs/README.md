## Summary
This project serves as an extension of my [undergraduate thesis](https://github.com/raymundo-mora/Projects/tree/main/Undergraduate_Thesis). Each step in my thesis that produced the final results has been neatly documented in a module stored in `m33.py`. `m33.pipeline` takes in a list of sources (e.g. the 10 clusters from Ma 2015 described in my undergraduate thesis.) and performs the entire analaysis we are interested in. The results from `m33.pipeline` are then put into `figures/` and `resutls/`. 
## Current Objective(s)
#### 1. 
The first objective is to run a source detection algorithm through my data to return a list of sources that can be analyzed by `m33.pipeline`. Not all of these sources will be clusters and it will be my job to find characteristics that will help me classify the newly found sources. The goal is to provide a much larger list of globular clusters and their structural parameters in the Triangulum Galaxy that has ever been created. 
#### 2. 
The second objective that I am focusing on is finetuning the functions described in `m33.py`. All of the functions are working as expected and it is great news to move forward with the project. However, I will be comparing a select number of clusters in my data with existing literature to make sure that my algorithm is performing with scientific integrity. This part of the project will be tedious but essential nonetheless to ensure that our findings are trustworthy and sound enough to report to the rest of the scientific community. 
## Structure
### `m33.py`
This file contains all of the functions used to produce our analysis. Having this file makes scripting and the editing of initial parameters much easier for the ensurance of sound results. 
### `m33_demo.ipynb` 
This notebook serves as a demonstration of how to use each function described in `m33.py` and displays their output. 
### `datasets/` 
All datasets relevant to the project are kept in this directory. 
### `figures/`
Running `m33.pipeline` with `inspect=True` produces a PDF for each object processed by our pipeline. The figures in each PDF help us visualize what occurred at each step and find inconsistencies. 
### `results/`
The `.csv` files that store the structual parameters for each object processed by our pipeline are stored in this directory. 
