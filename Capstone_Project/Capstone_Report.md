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


## References
1. Mehrabian, Albert. "Communication without words." Communication theory (2008): 193-200.
2. Ekman, Paul. "Basic emotions." Handbook of cognition and emotion 98.45-60 (1999): 16.
3. Murthy, G. R. S., and R. S. Jadon. "Recognizing facial expressions using eigenspaces." International Conference on Computational Intelligence and Multimedia Applications (ICCIMA 2007). Vol. 3. IEEE, 2007.
4. Perikos, Isidoros, Epaminondas Ziakopoulos, and Ioannis Hatzilygeroudis. "Recognizing emotions from facial expressions using neural network." IFIP International Conference on Artificial Intelligence Applications and Innovations. Springer, Berlin, Heidelberg, 2014.
5. Tang, Yichuan. "Deep learning using linear support vector machines." arXiv preprint arXiv:1306.0239 (2013).
6. Giannopoulos, Panagiotis, Isidoros Perikos, and Ioannis Hatzilygeroudis. "Deep learning approaches for facial emotion recognition: A case study on fer-2013." Advances in Hybridization of Intelligent Methods. Springer, Cham, 2018. 1-16.
7. Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.

