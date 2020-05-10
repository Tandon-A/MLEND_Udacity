# Machine Learning Engineer Nanodegree
## Capstone Proposal
Abhishek Tandon   
10th May, 2020

## Proposal

### Domain Background 

Humans use a mix of voice, gestures and facial expressions to communicate and convey their emotions. By representing up to 55% of human communication, facial expressions are the dominant way by which humans express their feelings [1]. An automated system to recognize emotions can improve human-computer interaction and make applications such as patient emotion monitoring in care facilities come true. Analyzing facial images using Computer Vision techniques can help build such automated systems. 

Many studies follow the categorical Ekman emotion model [2], dividing emotions into six basic emotion classes: anger, disgust, fear, happiness, sadness and surprise. In [3], researchers use dimensionality reduction and eigenspaces to recognize emotions. In [4], the authors use a multilayer perceptron (MLP) to learn 'good' features automatically.  Following this, in the present era of deep learning, authors from [5] and [6] have applied convolutional neural networks for this task. 

### Problem Statement 

This project explores the use of convolutional neural networks for the task of recognizing emotions using facial expressions. Convolutional Neural Networks can learn features automatically in an efficient manner and are ideal to use for a dataset consisting of images.

### Dataset and Inputs 

This project uses the Facial Emotion Recognition 2013 (FER 2013) dataset introduced in ICML 2013 workshop on recognizing facial expressions challenge [7].  This dataset is available on Kaggle.  (https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). 

The dataset consists of a total of 35887 facial images. The dataset labels each image into one of anger, disgust, fear, happiness, sadness, surprise and neutral emotion categories, having 4953, 547, 5121, 8989, 6077, 4002 and 6198 images for the emotion categories respectively.  
The dataset is further divided into parts/splits. The training split consists of a total of 28709 images, the public test data has 3589 images, and the private test data consists of 3589 images. The public test data is used as validation dataset. Each image in this dataset is a grayscale image of 48 X 48 pixels. 


### Solution Statement

As described above, this project explores the use of Convolutional Neural Networks to recognize emotions in images. The potential solution is to use techniques such as learning rate schedulers, deeper model architectures, and many augmentation techniques to obtain high performance on this task. 

### Benchmark Model 

The shallow model, having two convolutional layers and one fully connected layer as introduced in [8] is used as a benchmark model. This model achieves 55% accuracy on the validation set and 54% on the test set. 

### Evaluation Metrics 

The Kaggle competition using the FER dataset uses accuracy as the evaluation metric. This project also uses accuracy as the main evaluation metric to facilitate model comparison. Accuracy calculates the fraction of correct predictions for a model.    
_Accuracy = (TP + TN) / (TP + FP + TN + FN)_,    
where TP = True Positives, FP = False Positives, TN = True Negatives and FN = False Negative. 

Accuracy as a metric is easy to interpret and implement, but it paints a different picture for a model in case of imbalanced datasets. Consider an example of a binary classification problem with the negative class samples having a ratio of 1:100 with the positive class samples. In such a case, a majority class classifier which always predicts positive class for every image would have 99% accuracy even though the model hasn't learnt any features for classification.  

Metrics such as precision and recall help in evaluating model's performance in case of imbalanced datasets. FER 2013, the dataset used in this project, contains only 547 images for disgust class as compared to the 8989 images for the happiness class. Mathematically,   
_Precision = TP / (TP + FP)_ 
_Recall = TP / (TP + FN)_,   
where TP = True Positives, FP = False Positives, TN = True Negatives and FN = False Negative.

### Project Design 

The first step of this project is to explore the data by visualizing class images and the average face for each emotion class.   
The next step is the modelling step. Starting with a simple convolutional neural network and then moving on to deeper architectures to build a better model. Along with iterating on the model, data augmentation techniques and learning rate schedulers would be used to increase the performance on this task. The model having the highest accuracy on the validation dataset would be used as the final model.  
The last step would be to evaluate the final model on the test set. 

### References 
1. Mehrabian, Albert. "Communication without words." Communication theory (2008): 193-200.
2. Ekman, Paul. "Basic emotions." Handbook of cognition and emotion 98.45-60 (1999): 16.
3. Murthy, G. R. S., and R. S. Jadon. "Recognizing facial expressions using eigenspaces." International Conference on Computational Intelligence and Multimedia Applications (ICCIMA 2007). Vol. 3. IEEE, 2007.
4. Perikos, Isidoros, Epaminondas Ziakopoulos, and Ioannis Hatzilygeroudis. "Recognizing emotions from facial expressions using neural network." IFIP International Conference on Artificial Intelligence Applications and Innovations. Springer, Berlin, Heidelberg, 2014.
5. Tang, Yichuan. "Deep learning using linear support vector machines." arXiv preprint arXiv:1306.0239 (2013).
6. Giannopoulos, Panagiotis, Isidoros Perikos, and Ioannis Hatzilygeroudis. "Deep learning approaches for facial emotion recognition: A case study on fer-2013." Advances in Hybridization of Intelligent Methods. Springer, Cham, 2018. 1-16.
7. Goodfellow, Ian J., et al. "Challenges in representation learning: A report on three machine learning contests." International Conference on Neural Information Processing. Springer, Berlin, Heidelberg, 2013.
8. Alizadeh, Shima and Azar Fazel. “Convolutional Neural Networks for Facial Expression Recognition.” ArXiv abs/1704.06756 (2017): n. pag.
