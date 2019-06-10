# Course:  Imbalanced Data and Data Augmentation

## Chapter 1 - Problem of Imbalanced Data 

- Lesson 1.1 - Introduction to Imbalanced Data

  - A learning objective: Assess the class distribution of a dataset for data imbalance.

- Lesson 1.2 - Confusion-Matrix 

  - A learning objective: Create a confusion-matrix for a multiclass classifier to identify the most mis-classified classes.

- Lesson 1.3 -  F1 Score

  - A learning objective: Demonstrate why the F1-Score is a better metric than accuracy for classfiying an imbalanced data set.


## Chapter 2 - Undersampling and Oversampling 

- Lesson 2.1 - Undersampling

  - A learning objective: Implement undersampling on the majority classes using the [imbalanced-learn API](https://imbalanced-learn.readthedocs.io/en/stable/index.html).

- Lesson 2.2 - Oversampling

  - A learning objective: Implement undersampling on the minority classes using the [imbalanced-learn API](https://imbalanced-learn.readthedocs.io/en/stable/index.html).

- Lesson 2.3 - Synthetic Minority Over-sampling Technique (SMOTE)

  - A learning objective: Implement SMOTE using the [imbalanced-learn API](https://imbalanced-learn.readthedocs.io/en/stable/index.html).

- Lesson 2.4 - Comparing Sampling Methods

  - A learning objective: Evaluate the best sampling method for a given imbalanced dataset.


## Chapter 3 - Data Augmentation

- Lesson 3.1 - Introduction to Data Augmentation

  - A learning objective: Define data augmentation and its uses.

- Lesson 3.2 - Basic Methods

  - A learning objective: Flip, rotate, scale, and blur images to create new data.

- Lesson 3.3 - More Advanced Methods

  - A learning objective: Perform feature standardization and ZCA whitening  to create new data.

- Lesson 3.4 - Keras's [ImageDataGenerator](https://keras.io/preprocessing/image/)

  - A learning objective:  Create an [ImageDataGenerator](https://keras.io/preprocessing/image/) class for real-time data augmentation.


## Chapter 4 - Putting it all Together - Classifying Fish Species

- Lesson 4.1 - Assessing [The Nature Conservancy Fisheries Monitoring](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring) Dataset

  - A learning objective: Explore the dataset to diagnose the imbalance on the given data set.

- Lesson 4.2 - Addressing the Imbalance

  - A learning objective: Implement both a sampling method and an [ImageDataGenerator](https://keras.io/preprocessing/image/) on the given data set.

- Lesson 4.3 - Evaluating Performance

  - A learning objective: Evaluate the improvement of performance using the F1-Score and a confusion matrix.


