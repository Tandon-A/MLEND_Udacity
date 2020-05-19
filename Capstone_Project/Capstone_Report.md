# Machine Learning Engineer Nanodegree
## Capstone Project
Abhishek Tandon  
22nd July, 2020

## I. Definition

### Project Overview 

Humans use a mix of voice, gestures and facial expressions to communicate and convey their emotions. By representing up to 55% of human communication, facial expressions are the dominant way by which humans express their feelings [1]. An automated system to recognize emotions can improve human-computer interaction and make applications such as patient emotion monitoring in care facilities come true. Analyzing facial images using Computer Vision techniques can help build such automated systems.
Many studies follow the categorical Ekman emotion model [2], dividing emotions into six basic emotion classes: anger, disgust, fear, happiness, sadness and surprise. In [3], researchers use dimensionality reduction and eigenspaces to recognize emotions. In [4], the authors use a multilayer perceptron (MLP) to learn 'good' features automatically. Following this, in the present era of deep learning, authors from [5] and [6] have applied convolutional neural networks for this task.

![Sample Images](https://raw.githubusercontent.com/Tandon-A/MLEND_Udacity/master/Capstone_Project/assets/sample_images.jpg "Sample Images")   
###### Figure 1: Sample images from FER 2013 dataset.

### Problem Statement

The problem of facial emotion recognition is posed as an image classification problem, i.e. classifying an image in different emotion categories. 	
Machine Learning approaches using Convolutional Neural Networks (CNN) have shown high performance in image classification tasks. [7]
A CNN can learn features automatically in an efficient manner and are ideal to use for a dataset consisting of images.

This project explores the use of CNN for the task of recognizing emotions using facial expressions.

### Metrics 

This project uses accuracy as the primary evaluation metric to facilitate model comparison. Accuracy calculates the fraction of correct predictions for a model.  
_**Accuracy = (TP + TN) / (TP + FP + TN + FN)**_,   
where _TP = True Positives, FP = False Positives, TN = True Negatives and FN = False Negative_.

Accuracy as a metric is easy to interpret and implement, but it paints a different picture for a model in case of imbalanced datasets. Consider an example of a binary classification problem with the negative class samples having a ratio of 1:100 with the positive class samples. In such a case, a majority class classifier which always predicts positive class for every image would have 99% accuracy even though the model hasn't learnt any features. Metrics such as precision and recall can help evaluate a model’s performance in such cases.

FER 2013, the dataset used in this project, is a bit imbalanced as it contains only 547 images for disgust class as compared to the 8989 images for the happiness class. Hence, precision and recall are used as additional evaluation metrics. Mathematically,   
_**Precision = TP / (TP + FP)**_ and _**Recall = TP / (TP + FN)**_,   
where _TP = True Positives, FP = False Positives, TN = True Negatives and FN = False Negative_.

## Analysis 

### Data Exploration 

This project uses the Facial Emotion Recognition 2013 (FER 2013) dataset introduced in ICML 2013 workshop on recognizing facial expressions challenge [8]. This dataset is available on Kaggle. (https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
The dataset consists of a total of 35887 facial images. The dataset labels each image into one of anger, disgust, fear, happiness, sadness, surprise and neutral emotion categories, having 4953, 547, 5121, 8989, 6077, 4002 and 6198 images for the emotion categories respectively.

![Dataset Image Statistics](https://raw.githubusercontent.com/Tandon-A/MLEND_Udacity/master/Capstone_Project/assets/data.jpg "Dataset Statistics")  
###### Figure 2: FER 2013 dataset statistics 

The dataset is further divided into parts/splits. The training split consists of a total of 28709 images, the public test data has 3589 images, and the private test data consists of 3589 images. The public test data is used as validation dataset. Each image in this dataset is a grayscale image of 48 X 48 pixels.

The dataset, as hosted in the Kaggle competition, is a CSV file having a column for pixels. This column needs to be processed and converted to image representation before training. 

### Exploratory Visualization

As an exploratory step, the average face for the whole dataset is calculated.   

![Average Face](https://raw.githubusercontent.com/Tandon-A/MLEND_Udacity/master/Capstone_Project/assets/avg_face.jpg "Average Face")
###### Figure 3: Average Face   

The dataset has the highest samples for the 'happy' class, and as can be seen in the above figure, the average face looks like smiling and is close to the images in the 'happy' category. 

As an extra step, the average face for every emotion label is also calculated. 

![Average face per category](https://raw.githubusercontent.com/Tandon-A/MLEND_Udacity/master/Capstone_Project/assets/avg_face_cat.jpg "Average Face per emotion category")  
###### Figure 4: Average face per emotion category

As shown in the above figure, the average face presents an idea about how the images distinguish between different emotion categories.

### Algorithms and Techniques

This section gives an overview of the techniques used in the project.   

A neural network (NN), as inspired by the functioning of the human brain, applies a linear transformation (multiplication by weights and addition of bias) to the input and generates an output representation. Activation functions such as ReLU are nonlinear and allow a model to perform complex nonlinear changes to the input data.     
A loss function compares the predictions of the model with an expected output. The gradient of this function is backpropagated to adjust parameters to decrease the loss function. This process repeated over multiple turns allows a NN model to learn the weights to produce better output representation. 

In a NN, each neuron in a layer is connected to every neuron in the previous layer. These dense connections in a NN lead to remarkably high memory usage while learning to classify images. 

In a convolutional neural network (CNN), the neurons are arranged in 3 dimensions (width, height and depth) and are connected only to some neurons in the previous layer.   
This ordering of neurons is known as a convolution filter.  This filter is convolved across the width and height of the input space to learn features.  The parameters of the convolution filter remain same or shared for every location of the input space.   
The local connectivity between neurons and parameter sharing reduce memory consumption significantly, making them highly efficient to learn features from images. 

A CNN is a sequence of layers where every layer transforms the data through a differentiable function [9]. The three main layers are: 
1. Convolutional Layer (Conv Layer) : This layer represents the collection of convolution filters, transforming the input from one volume representation to another. For example, a Conv layer may change an input image of [48 X 48 X 1] into [48 X 48 X 32], if the layer has 32 filters. 
2. Pooling Layer (Pool layer) : This layer performs a downsampling operation along the spatial dimensions (width, height). For example, a Pool layer would change an input space of [48 X 48 X 32] into [24 X 24 X 32].   
3. Fully Connected Layer (FC) : This layer is like a standard neural network where each neuron is connected to all neurons in the previous volume. This layer is generally used towards the end of a CNN model to classify obtained features into different categories. 

This project uses CNNs to recognize emotions in facial images. 

### Benchmark 

Both the shallow model and the deep model, as introduced in [10] are used as benchmark models in this project. The shallow model has two convolutional layers and one fully connected layer and achieves 55% accuracy on the validation set and 54% accuracy on the test set. The deep model has four convolutional layers and two fully connected layers at the end and achieves 65% accuracy on the validation set and 64% accuracy on the test set.

![Shallow Model](https://raw.githubusercontent.com/Tandon-A/MLEND_Udacity/master/Capstone_Project/assets/shallow_model.png "Shallow Model")   
###### Figure 5 (a): Shallow Model architecture 
![Deep Model](https://raw.githubusercontent.com/Tandon-A/MLEND_Udacity/master/Capstone_Project/assets/deep_model.png "Deep Model")  
###### Figure 5 (b): Deep Model architecture 

## Methodology 

### Data Preprocessing 

As mentioned in the analysis section, this dataset is a CSV file which has a pixels column representing the images. The pixel information is first extracted and is converted to proper image representation to facilitate the CNN training procedure.  For every image, the linear pixel array from the CSV file is converted to a multidimensional array of size 48 X 48.

CNN, in general, are data-intensive, i.e. they require a large amount of data to generalize better. A bunch of data augmentation techniques such as randomly flipping the image along the horizontal axis and randomly rotating the image are applied to increase the dataset size on which the model trains. 
These techniques are employed at the runtime, and so the memory consumed by the dataset remains low. 

The following augmentation strategies are used to train the final model:  
1. RandomHorizontalFlip: Flips an image along the horizontal axis randomly. 
2. Translate: Translate an image along the width and height dimension using a randomly sampled shift.
3. Scale: Scales an image using a randomly sampled scaling factor.
4. RandomErasing: Randomly erases small rectangular regions of an image

Along with the augmentation techniques, the images are mean, and standard deviation normalized using the mean and standard deviation value calculated from the training images before the training process starts. 
The images present in the validation and testing split are also normalized using the training split mean and standard deviation values.

### Implementation 
 
This project uses PyTorch[10] for building CNN models. 

Initially, a simple CNN model composed of three convolutional layers and max-pooling layers is built. 
Figure -- Basic Model 

This model achieves a validation accuracy of **(fill in value here)** 


## References
1. Mehrabian, Albert. "Communication without words." Communication theory (2008): 193-200.
2. Ekman, Paul. "Basic emotions." Handbook of cognition and emotion 98.45-60 (1999): 16.
3. Murthy, G. R. S., and R. S. Jadon. "Recognizing facial expressions using eigenspaces." International Conference on Computational Intelligence and Multimedia Applications (ICCIMA 2007). Vol. 3. IEEE, 2007.
4. Perikos, Isidoros, Epaminondas Ziakopoulos, and Ioannis Hatzilygeroudis. "Recognizing emotions from facial expressions using neural network." IFIP International Conference on Artificial Intelligence Applications and Innovations. Springer, Berlin, Heidelberg, 2014.
5. Tang, Yichuan. "Deep learning using linear support vector machines." arXiv preprint arXiv:1306.0239 (2013).
6. Giannopoulos, Panagiotis, Isidoros Perikos, and Ioannis Hatzilygeroudis. "Deep learning approaches for facial emotion recognition: A case study on fer-2013." Advances in Hybridization of Intelligent Methods. Springer, Cham, 2018. 1-16.
7. Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.
8. Goodfellow, Ian J., et al. "Challenges in representation learning: A report on three machine learning contests." International Conference on Neural Information Processing. Springer, Berlin, Heidelberg, 2013.
9. CS231n Convolutional Neural Networks for Visual Recognition, cs231n.github.io/convolutional-networks/.
10. Alizadeh, Shima and Azar Fazel. “Convolutional Neural Networks for Facial Expression Recognition.” ArXiv abs/1704.06756 (2017): n. pag.
11. Paszke, Adam, et al. "PyTorch: An imperative style, high-performance deep learning library." Advances in Neural Information Processing Systems. 2019.
