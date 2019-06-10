

# Sample Exercise

This is a sample exercise for Lesson 1.2 in the Imbalanced Data and Data Augmentation course.

## Context

 In this exercise, you will fit a neural network classifier to the [Digits dataset](https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits), which has already been pre-loaded and split into the variables `X_train`, `X_test`, `y_train`, and `y_test`.  We'll use scikit-learn's ``confusion_matrix``  to evaluate the classifier's performance.

## Instruction

- Fit the classifier with the training data
- Predict the labels of the test data
- Compute the confusion matrix using the correct labels and the predicted labels

## Code

Assumes the following has been preloaded: 

```
from sklearn import datasets, metrics, neural_network, model_selection
digits = datasets.load_digits()
X_train, X_test, y_train, y_test = model_selection.train_test_split(digits.data, digits.target, test_size=0.5)
```

```
# Instantiate a neural network classifier and fit it to the training data
nn = neural_network.MLPClassifier()
nn.fit(___, ___)

# Predict the labels for the test data 
y_pred = nn.predict(___)

# Compute the confusion matrix
confusion_matrix = metrics.confusion_matrix(___, ___)
print(confusion_matrix)
```

## Solution Code

```
# Instantiate a neural network classifier and fit it to the training data
nn = neural_network.MLPClassifier()
nn.fit(X_train, y_train)

# Predict the labels for the test data 
y_pred = nn.predict(X_test)

# Compute the confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print(confusion_matrix)
```

