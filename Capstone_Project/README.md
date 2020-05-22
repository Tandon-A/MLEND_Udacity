# Machine Learning Engineer Nanodegree

## Capstone Project Reamde 

## Introduction 

This project explores the use of Convolutional Neural Networks for the task of facial emotion recognition. The final model obtained is then deployed in a simple web application. 
This project is completed as part of the Udacity Machine Learning Engineer nanodegree. 

## Requirements

Following libraries are required for the CNN development part:  
1. Pytorch (1.5.0)
2. torchvision (0.6)
3. torchsummary
4. Pandas, for accessing the CSV file.
5. Matplotlib, for visualization
6. scikit-plot, for result plots 
7. sklearn (scikit-learn) for metric calculation
8. Numpy 

## Environment

The modelling step is done on Google Colaboratory, which provides access to GPU for model training. A Google Colaboratory notebook is then linked with the Google drive, which allows storing the CSV file.

## Setup 

### Model training 

First, the dataset, in this case, the icml_face_data.csv is downloaded from the [Kaggle website](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). ([Alternative Link](https://drive.google.com/open?id=1VIPLzqy1qIwY-ssY2QoNLT6ak5sWsiLU))

This CSV file is then uploaded to Google drive which is then linked with the Colaboratory notebook.   
The CSV file needs to be provided if using a different environment. 

To train models, one of the python notebooks can be used.   
1. FER_Experiments: This notebook contains code for various experiments performed in this project.  
2. FinalModel: This notebook contains code only for the final model.  

Running the final model notebook should produce similar results, as described in the project report.  
Both of these notebooks require the CSV file. The CVS file path needs to be provided as per directory structure. (**Change variable FER_DATA_PATH**, defined in the 'data loading' section in the notebooks) 

The final model can be restored using torch.jit.load. 

### Web Application 

The present web application is a rather simple one and can only perform emotion recognition on images which are like that of the dataset.   
Images are named by their emotion category and are available in the model_data folder in the submission and are also available [here](https://github.com/Tandon-A/MLEND_Udacity/tree/master/Capstone_Project/Webapp/ferapp/model_data). These images can be used to test the model deployed in the web application. 

Additional functionality to perform inference on natural images can be added to improve the project further. 

## Notes 

In this project, the random seed is not fixed, so the results would slightly differ from one model training to another. 

