# Machine Learning Practical Exercise {#ml-pe}

We will now work on more examples to learn how to fit and evaluate multiple regression and classification models. 

## Modeling Workflow

Below is a general workflow provided by @brownlee2017mlwithR. It is important to remember that before you begin to work on a problem, that you determine if it's even appropriate to use machine learning. Many analyses get derailed by not working on the particular question(s) that need answering. 

1. Defined the problem and determined your study's objectives

2. Prepare Problem
   + Load libraries
   + Load dataset
   + Split-out validation dataset

3. Summarize Data
   + Descriptive statistics
   + Data visualizations

4. Prepare Data
   + Data Cleaning
   + Feature Selection
   + Data Transforms

5. Evaluate Algorithms
   + Test options and evaluation metric
   + Spot Check Algorithms
   + Compare Algorithms

6. Improve Accuracy
   + Algorithm Tuning
   + Ensembles

7. Finalize Model
   + Predictions on validation dataset
   + Create standalone model on entire training dataset
   + Save model for later use

## Performance Metrics

Regression models use root mean squared error (RMSE) and the $R^2$ evaluation metrics. In general, we want a model that has the lowest RMSE and highest $R^2$. However, we need to bear in mind that we are also evaluating a tradeoff between bias and variance as well. The general statistics used for classification problems are accuracy and [Cohen's Kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa) ($K$). We can also use the [receiver-operator characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) (ROC) curve to evaluate binary classification problems. There are other performance characteristics, but these are the most common; see @kuhn2017caret for more information.

## Practice Projects

We will now work on a regression, binary classification, and multi-class classification problem set. 
