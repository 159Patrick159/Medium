# Testing the effects of class balancing in cross-validated random forests
The following directory contains the Jupyter Notebook for examining the effect of class balancing in cross-validated random forest (CVRF) classifiers. The notebook title ETL.ipynb cleans, reduce and extracts features from the datasets located in the `taxi_fare` directory. Once the data is ready, the notebook creates, trains, and validates two CVRF's, one with a balanced class dataset and one with the original class imbalance. I use `GridSearchCV` with k-folds=5 to cross validate each machine learning model. I used parallel computing techinques to run 4 jobs on parallel to reduced run times during the training procedure.

I evaluated the model with two testing sets, the first one was the oringial split done in the notebook, and the second set was obtained from `taxi_fare/test.csv` which was orignially provided by Kaggle. Both evaluations indicate that class balancing reduced the the number of false negative prediction and increased the number of false positive predictions. Check out the medium post for further discussion: <https://medium.com/@patricksandovalromero/a-brief-introduction-cross-validated-random-forest-21423d3378d5>

### Reducing, cleaning and feature enginerring
I will create another Medium post where I go into detail on how I am using the `pandas` library along with the `numpy` library to reduce my data. However, we can see from the feature corner plot that the 'misc_fees' feature has the clearest patter of distinction for rides that applied and didn't applied surges.

<p align='center'>
    <img src="Figures/Feature Corner Plot.png" title="Feature Corner Plot" height="80%" width="80%">
</p>

We also see how the 'total_fare' feature shows some clear pattern for discriminating trips with applied surges.

### Cross-validating random forests using GridSearchCV
We utilized the K-fold method for cross-validating both models using 5-fold validation meaning we use a 80-20 ratio between the training and validation sets. The tuned hyperparameters for each model are very similar to each other with the exeption of the `number_of_estimators`, where the balanced CVRF has 75 trees and the imbalanced CVRG has 100.

### Evaluating the models and final results
Using the test set that composed 25% of the original `taxi_fare/train.csv` dataset we created the following confusion matrix to visualize the propotions of tru positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).

<p align='center'>
    <img src="Figures/Confusion Matrix.png" title="Confusion Matrix" height="80%" width="80%">
</p>

### Feature importance
Additionally, we use the mean decrease in impurity as a metric to measure the individual importance and predictive power of features. This allowed us to confirm that `misc_fees` and `total_fare` are the most predictive features for surges being applied.

<p align='center'>
    <img src="Figures/Feature Importance.png" title="Feature Importance" height="80%" width="80%">
</p>

For a more detailed discussion of this project please refer to the medium post reference at the top.