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

Some sample images of all the classes are plotted. As can be seen in the following figure (figure 5), many images have some watermark text such as the fifth image for 'angry' emotion. The facial pose differs a lot from one image to another, and in many cases, the full face is not visible. 
Some of the images are very similar, making it difficult for the model to predict correctly just one emotion class. A model might get confused by the fifth image for the anger emotion, as it is very similar to images in the surprise class. 

Data augmentation techniques can help overcome some of these challenges. For example, scaling can help generate more images where the whole face is not visible, supporting the model to learn a better representation. These techniques are explored in further sections. 

![EDA sample class images](https://raw.githubusercontent.com/Tandon-A/MLEND_Udacity/master/Capstone_Project/assets/EDA_crop.png "EDA Sample class images")  
###### Figure 5: Sample image per emotion category

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
###### Figure 6 (a): Shallow Model architecture 
![Deep Model](https://raw.githubusercontent.com/Tandon-A/MLEND_Udacity/master/Capstone_Project/assets/deep_model.png "Deep Model")  
###### Figure 6 (b): Deep Model architecture 

## Methodology 

### Data Preprocessing 

As mentioned in the analysis section, this dataset is a CSV file which has a pixels column representing the images. The pixel information is first extracted and is converted to proper image representation to facilitate the CNN training procedure.  For every image, the linear pixel array from the CSV file is converted to a multidimensional array of size 48 X 48.

CNN, in general, are data-intensive, i.e. they require a large amount of data to generalize better. A bunch of data augmentation techniques such as randomly flipping the image along the horizontal axis and randomly rotating the image are applied to increase the dataset size on which the model trains. 
These techniques are employed at the runtime, and so the memory consumed by the dataset remains low.

Combinations of the following augmentation strategies are used to train the models:    
1. [RandomHorizontalFlip](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.RandomHorizontalFlip): Flips an image along the horizontal axis randomly. 
2. [RandomTranslate](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.RandomAffine): Translate an image along the width and height dimension using a randomly sampled shift. It is implemented by specifying only translate agruments in RandomAffine transform in Pytorch library. 
3. [RandomRotation](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.RandomRotation): Rotates an image by a random angle. 
4. [RandomScale](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.RandomAffine): Scales an image using a randomly sampled scaling factor. It is implemented by specifying only scaling agruments in RandomAffine transform in Pytorch library. 
5. [RandomErasing](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.RandomErasing): Randomly erases small rectangular regions of an image

Along with the augmentation techniques, the images are mean, and standard deviation normalized using the mean and standard deviation value calculated from the training images before the training process starts. 
The images present in the validation and testing split are also normalized using the training split mean and standard deviation values.

### Implementation 

The implementation consists of two main steps: 
1. CNN Model training  
2. Web App development  

During the first step, the CNN model is trained on the dataset following the below steps: 
1. Load the CSV file in memory.  
2. Convert images to proper representation.  
3. Extract training, validation and testing split.   
4. Create Dataloader, to apply augmentations on the fly while training.   
5. Define model architecture.   
6. Setup loss function, optimize.   
7. Set values for hyperparameters (learning rate, weight decay).   
8. Train and validate the network.  
9. Refine the network, hyperparameter values and augmentation strategies.  
10. Test the final model.  
11. Serve the final model.   

PyTorch[11]  is used for building and training CNN models. 
Initially, a simple CNN model composed of three convolutional layers and max-pooling layers is built. All models use cross-entropy loss as the loss criterion and adam optimizer as the optimizer. 

![Basic Model](https://raw.githubusercontent.com/Tandon-A/MLEND_Udacity/master/Capstone_Project/assets/basic_model.png "Basic Model")  
###### Figure 7: Model 1 architecture 
This model achieves a validation accuracy of 46.78%. (Only RandomHorizontalFlip used as augmentation strategy in this case)

The modelling step is done on Google Colaboratory, which provides access to GPU for model training. A Google Colaboratory notebook is then linked with the Google drive, which allows storing the CSV file. 

The backend of the web app is written in python using the Flask library. The frontend is developed using HTML, CSS and Javascript using bootstrap and jquery libraries. The app is deployed to the web using Heroku.

_TODO_
**Figure -- Web App** 

### Refinement 

The basic model in the earlier section is refined upon by using different augmentation strategies, CNN models and learning rate schedulers. 

Firstly two more models are developed by adding more convolutional layers and using dropout in one model. The models are as depicted in the below figure.   
![Model 2, 3](https://raw.githubusercontent.com/Tandon-A/MLEND_Udacity/master/Capstone_Project/assets/model_2_3.png "Model 2, 3")  
###### Figure 8: Model 2 and model 3 architecture 

| Model Type  | Validation accuracy |
|------------ | --------------------| 
Model 1       | 43.42               | 
Model 2       | 59.35               | 
Model 3       | 59.74               | 
###### Table 1: Baisc Model 1, 2 and 3 comparison 

Only random horizontal flip is used for data augmentation to test the models. Model 2 and 3 are comparable to each other and much better than the first model. These models are utilized in further steps. 

Next, the models are trained using different augmentation strategies such as translating, scaling, rotating, flipping. The following data augmentation strategies are defined:  
1. Data Augmentation 1 = RandomHorizontalFlip   -- trans 1  
2. Data Augmentation 2 = RandomHorizontalFlip, RandomRotation(degrees=10), RandomErasing -- trans 3   
3. Data Augmentation 3 = RandomHorizontalFlip, RandomRotation(degrees=10), RandomAffine(translate=(0.1, 0.1)), RandomErasing -- trans 4     
4. Data Augmentation 4 = RandomHorizontalFlip, RandomRotation(degrees=10), RandomAffine(translate=(0.1, 0.1), scale=(0.8, 0.9)), RandomErasing -- trans 7   
5. Data Augmentation 5 = RandomOrdering(RandomHorizontalFlip, RandomAffine(translate=(0.1, 0.1), One of(RandomRotation(degrees=10), RandomAffine(scale=(0.8, 0.9)))) -- trans10     


| Model Type  | Augmentation Techniques | Validation accuracy |
|------------ | ------------------------| ------------------- | 
Model 2       | Data Augmentation 1     | 59.35               | 
Model 3       | Data Augmentation 1     | 59.74               |  
Model 2       | Data Augmentation 2     | 63.30               | 
Model 3       | Data Augmentation 2     | 64.28               |
Model 2       | Data Augmentation 3     | **63.28**           | 
Model 3       | Data Augmentation 3     | **65.34**           |   
Model 2       | Data Augmentation 4     | 61.38               | 
Model 3       | Data Augmentation 4     | 63.25               |
Model 2       | Data Augmentation 5     | 62.69               | 
Model 3       | Data Augmentation 5     | 64.86               |  

###### Table 2: Augmentation strategies comparison (models are run only for 60 epochs)     

Some of the essential comparisons from the above table are between data augmentation strategy two and three, and strategy four and five. Adding the translate augmentation to strategy two helps both the models achieve very high accuracy on the validation set. (Comparison of strategy two and three). Upon adding the random scaling augmentation, the approach of choosing only one of rotation and scaling at a time works better. (Comparison of method four and five).     
Data Augmentation strategy three and five are selected for further experiments. 

Another refinement is done by dynamically changing the learning rate using learning rate schedulers. Experiments are conducted with three learning rate schedulers:   
1. [ReduceLROnPlateau](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau): This scheduler decreases the learning rate by a factor of 0.1 whenever the loss function on the validation set plateaus, i.e. does not fall for more than ten epochs.    
2. [StepLR](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.StepLR): This scheduler decreases the learning rate by a factor of 0.1 after every step size epochs. (step_size is set to 20 epochs)  
3. [CosineAnnealingWarmRestarts](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts): This scheduler decreases the learning rate using a cosine annealing policy and also using warm restarts, i.e. setting the learning rate to be the initial learning rate in between training. Two parameters are experimented with, T_0: Number of epochs for first restart, T_mult: A factor which increases the number of epochs between two restarts. This is referred to as cos_warm(T_0, T_mult) in the project. 

| Model Type  | Learning Rate Scheduler | Validation accuracy |
|------------ | ------------------------| ------------------- | 
Model 2       | ReduceLROnPlateau       | 64.70               | 
Model 2       | StepLR(step_size=20)    | 60.71               |  
Model 2       | cos_warm(2,10)          | **65.34**           | 
Model 2       | cos_warm(1, 10)         | 63.75               |
Model 2       | cos_warm(2, 20)         | 64.56               | 

###### Table 3: Learning rate schedulers comparison (models are run for 150 epochs, using data augmentation strategy five)     

ReduceLROnPlateau and CosineAnnealingWarmRestarts scheduler with T_0 = 2 and T_mult=10 are selected for further experiments. 

One more iteration of the modelling procedure is performed to boost the performance. Specifically, the filter of the convolutional layers of the two models are changed from the filter size of two to three and zero padding is changed from zero to one. 
ResNet models are also taken into consideration to see if deeper architectures can help. 

| Model Type       |  Validation accuracy |
|----------------- |  ------------------- | 
Model 2 (2X2)      |  64.92               | 
Model 2 (3X3)      |  67.09               |
Model 3 (2X2)      |  66.59               | 
Model 3 (3X3)      |  **68.63**           |
Resnet18 pretrained|  65.45               | 
Resnet34 pretrained|  65.81               |

###### Table 4: Learning rate schedulers comparison (models are run for 150 epochs, using data augmentation strategy five)

The filter size of 3 X 3 has improved the performance by almost three percentage points. Model 3 with 3X3 filter size and model 2 with 3X3 filter size are selected for further experiments. 

Experiments are now conducted with two models - Model 2 (3X3 filter), Model 3 (3X3 filter), two data augmentation strategies - data augmentation strategy three and five and two learning rate schedulers - cos_warm(2, 10) and ReduceLROnPlateau. 

_TODO_
Mention the final model and training strategy used. Mention accuracy. 

## Results 

### Model Evaluation and validation 

After the refinement step, the model which performs best on the validation set is selected as the final model. 

Figure -- final model 
The model architecture is as shown in the above figure. The weights of the model are initialized by default in the Pytorch library using the Xavier initialization method. [12]. The model uses a learning rate of 0.001 and is trained for a total of 150 epochs.  Early stopping is used to avoid overfitting, and the model is stopped at epoch (fill here). 

Figure -- loss curve final model 

Performance of the model on the test set is used as a final evaluation step to check how well does the model generalize to unseen data. In this case, the model achieves a test set accuracy of (fill here)%. 

### Justification 

The final model achieves a validation set accuracy of (fill here)% and testing set accuracy of (fill here)%. This model has shown higher performance as compared to the benchmark model's accuracy on both validation and testing set. 

Human scores on this dataset are in the range of [65%, 70%] [8] which is comparable to the final model's performance, showing that the model accurately predicts emotions in facial images. 

Figure -- PR Curve 
Possibly -- Metric PR 
Para -- about PR 

## Conclusion 

### Free-Form Visualization

### Reflection 


The following steps review the entire project flow: 
1. Interested in emotion recognition, I researched on the work done in the field. Relevant papers and public datasets were found.   
2. The problem was finalized, and FER 2013 dataset was selected for the project.  
3. The dataset was downloaded from the Kaggle website and preprocessed.   
4. CNN models were developed, trained and refined using the PyTorch library.  
5. The model achieving the highest performance on the validation set was selected as the final model and was further evaluated on the testing set.  
6. A web application was developed using Flask, HTML, Javascript and Heroku. The final model was deployed in this application.  

I found the steps of training and refining models and developing a web app to be most challenging. I had never worked on the deployment of ML models before this project, and so it was a bit difficult for me. Though I was already familiar with the modelling step, it is a time-taking step as their many parameters to experiment with, making it challenging to complete on time.   

The most exciting aspect was the use of Pytorch JIT (Just in time) compiler for converting the model to a representation which can be used for deployment purposes in the web application. While researching the topic, I found some techniques to extend the FER dataset further to do a multi-label emotion classification. I am sure these techniques would be helpful for some future projects.   

Overall, I am satisfied with the model's performance on the dataset, and it can be used to recognize emotions in static images. 
The code for this project is available at https://github.com/Tandon-A/MLEND_Udacity/tree/master/Capstone_Project, and the web application is available at **(provide web app link)**. 

### Improvement 

While the final model obtained is better than the benchmark model and overall has high performance, there is scope for improvement. 

From a general perspective, adding more data can help the model generalize and distinguish between different classes better. 
At present, the modelling procedure was focussed on achieving a better performing model, but building a better model also depends on the application context in which the model will be used. Applications such as robots require a model to run on a device with memory and power consumption constraints. In such a case, the focus should be to build an efficient model which achieves high performance while keeping the overhead small.  Techniques such as quantization aware training and creating an efficient architecture by using depthwise separable convolution can be explored. 
Other techniques such as MixUp [13], AutoAugment[14] and Knowledge distillation[15] can be utilized to increase model performance. 

The deployed web app in this project is a fairly simplistic one. Improvements in the frontend can help boost user experience. Additional functionality, such as recognizing emotions in real-time using camera feed, can be added. 

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
12. Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." Proceedings of the thirteenth international conference on artificial intelligence and statistics. 2010.
13. Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization." arXiv preprint arXiv:1710.09412 (2017).
14. Cubuk, Ekin D., et al. "Autoaugment: Learning augmentation strategies from data." Proceedings of the IEEE conference on computer vision and pattern recognition. 2019.
15. Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).


