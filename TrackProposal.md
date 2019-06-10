#Evaluating and Optimizing Deep Learning Performance with Python

Preparing your data and setting up your model is only half the process! This track takes an in-depth look at the iterative process of tuning and evaluating your deep learning models. 

## Track Objectives

1. Learner will be able to evaluate and describe a  deep neural network's performance using concepts, such as overfitting, data imbalance, and fairness.
2. Given a deep neural network suffering from overfitting, learner will be able to improve its performance using appropriate regularization and data augmentation techniques.
3. Learner will be able to improve a deep neural network's performance and run-time using optimization algorithms and/or a pre-trained model.
4. Leaner will be able to tune a model's hyperparameters effectively in terms of performance and run-time.



## Course 1: Regularization Techniques for Deep Neural Networks 

#### Description:

Regularization adresses one of the most likely culprits hurting your model performance:  overfitting (when your model is memorzing rather than learning). This course will not only teach you how to diagnose overfitting and underfitting but arm you with effective regularization techniques. You will navigate the [UCI Breast Cancer Diagnostic](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) dataset to explore hands-on how the bias-variance tradeoff relates to overfitting and underfitting. With TensorFlow, you will implement preventative strategies, such as weight penalties, dropout, and early stopping, on deep neural networks to improve performance. 

#### Learning Objectives:

1. Given a model's performance metrics, learner will be able to distinguish whether a model is overfitting or underfitting.
2. Learner will be able to explain why regularization techniques work in preventing overfitting in relation to the bias-variance tradeoff.
3. Learner will be able to implement regularization techniques (specifically L1/L2 regularization, dropout, and early stopping) on deep neural networks with TensorFlow. 

#### Prerequsites:

- [Deep Learning in Python](https://www.datacamp.com/courses/deep-learning-in-python#!) because this course assumes fundamental knowledge in deep neural networks. 

- [Introduction to TensorFlow in Python](https://www.datacamp.com/courses/introduction-to-tensorflow-in-python) because this course uses TensorFlow to implement neural networks and regularization techniques. 


##Course 2: Data Imbalance and Data Augmentation 

Ever thought: "if only I had more data…"? Data augmentation answers that by making new data from the data you have! You will be using  [The Nature Conservancy Fisheries Monitoring](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring) image dataset to classify six types of fish species. The catch is that three of the species make up less than 10% of the total data. This type of issue is called imbalanced data and it comes up often with real-world problems, where data is limited and imperfect. In this course, you will learn how tackle this problem with sampling and data augmentation methods. 

##### Learning Objectives:

1. Learner will be able to evaluate data imablance and its affect on model performance using the F1-score and a confusion-matrix.
2. Learner will be able to explain the different advantages and disadvantages between oversampling and undersampling.
3. Learner will be able to list different data augmentation strategies and implement them

#### Prerequsites:

- [Deep Learning in Python](https://www.datacamp.com/courses/deep-learning-in-python#!) because this course assumes fundamental knowledge in deep neural networks and Keras. 




## Course 3: Optimizing Deep Neural Networks

Training a deep neural network takes hours, even days. That's why a lot of research has gone into developing algorithms to optimize the training neural networks. In this course, you will gain practical knowledge on the most effective optimization algorithms and experience implementing them with [Keras Optimizers](https://keras.io/optimizers/) to get faster and better results. 

#### Learning Objectives:

1. Learner will be able to explain the goal of optimization algorithms in relation to the concept of learning rate.
2. Learner will be able to compare and contrast the following optimization algorithms: mini-batch gradient descent, RMSprop, and Adam.
3. Learner will be able to evaluate the effectiveness of an optimization algorithm based on time to train and model performance results.

#### Prerequsites:

- [Deep Learning in Python](https://www.datacamp.com/courses/deep-learning-in-python#!) because this course assumes fundamental knowledge in deep neural networks and uses Keras 2.0. 


## Course 4: Tuning and Optimizing Hyperparameters 

A deep neural network can come with many hyperparameters: learning rate, number of hidden units, activation function, batch size, and the list goes on! It would be too painful and time-consuming to manually hand-tune these parameters. Instead, machine learning pros use hyperparameter optimization methods to automate the tuning of hyperparameters. In this course, you will learn about these methods and implement them with the [skopt](https://scikit-optimize.github.io/) library. This will all be done hands-on through the challenge of tuning a deep neural network for MNIST digit recognition.

#### Learning Objectives:

1. Learner will be able to identify the hyperparameters of different types of deep neural networks.
2. Learner will be able to select and justify hyperparameters for a model based on the results of a hyperparameter optimization method.
3. Learner will be able to compare different hyperparameter optimization methods.

#### Prerequsites:

- [Deep Learning in Python](https://www.datacamp.com/courses/deep-learning-in-python#!) because this course assumes fundamental knowledge in deep neural networks. 


## Course 5: Introduction to Transfer Learning and Pre-Trained Models 

As you know deep neural networks take a lot of data, resources, and time to train. So, why start from scratch each time we train a new model? This is the essence of transfer learning which leverages pre-trained models to solve new problems in similar domains. In this course, you will use Keras to download a [convolutional neural net pretrained on the ImageNet dataset](https://keras.io/applications/#inceptionresnetv2) and re-train it to classify to [fruits](https://www.kaggle.com/moltean/fruits).  

#### Learning Objectives:

1. Learner will be able to describe the process of transfer learning and how it differs from traditional machine learning.
2. Learner will be able use a pre-trained ImageNet convolutional neural network to solve an image classification problem.

#### Prerequsites:

- [Convolutional Neural Networks for Image Processing](https://www.datacamp.com/courses/convolutional-neural-networks-for-image-processing) because this course requires an understanding in CNNs.


## Course 6: Evaluating Fairness and Bias in Machine Learning Models 

Because of the black-box nature of machine learning and the risk of algorithmic bias, fairness has become an important performance metric to consider. Fairness in machine learning has been cited as a growing issues by the governments, big tech companies, and research instituitions. Through real-world examples, you will learn about different ways machine learning models can perpetuate harmful discriminatory biases, and, in each of these examples, you will implement methods that could have prevented such biases. 

#### Learning Objectives:

1. Learner will be able to describe several real-world instances of algorithmic bias. 
2. Given a scenario and model, learner will be able to recognize potential features of the model  that could cause algorithmic bias.
3. Learner will be able to implement basic auditing methods to evaluate a machien learning model's fairness. 

#### Prerequsites:

- [Supervised Learning with scikit-learn](https://www.datacamp.com/courses/supervised-learning-with-scikit-learn) because some of the examples used in this course are supervised models.
- [Unsupervised Learning in Python](https://www.datacamp.com/courses/unsupervised-learning-in-python)because some of the examples used in this course are unsupervised models.

