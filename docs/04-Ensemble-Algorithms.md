# Ensemble Algorithms {#ens-algs}

## Bagging

Bagging is short for Bootstrapped Aggregation. As you can guess from the name, the Bagging algorithm's basis is the [bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)). With bootstrapping, you can use resampling techniques to estimate an unknown parameter of the data. The bootstrap method computes this estimate by taking random samples, with replacement, of the data and calculating the estimated value a total of $B$ times. The final step is to calculate the average of the estimates over the $B$ bootstrap trials.  

With Bagging, the algorithm aggregates predictions from multiple machine learning models. Bagging provides an advantage for models that are high variance by reducing the variance of these models. In essence, Bagging mimics the phenomenon known as the ["wisdom of the crowd"](http://galton.org/search/essays/pages/galton-1907-vox-populi_1.png). In practice the algorithm is relatively simple

<div class="figure" style="text-align: center">
<img src="img/bagging-algorithm.png" alt="Bagging algorithm from Kuhn and Johnson (2013)." width="90%" />
<p class="caption">(\#fig:bagging-algorithm)Bagging algorithm from Kuhn and Johnson (2013).</p>
</div>

Since Bagging works well on models with high variance, it is most widely used on CART models. Fundamentally, the Bagging algorithm cast votes on multiple trees that are individually weak learners. However, the aggregated response or classification has an overall reduced error rate without a loss in bias. 

### Practical Exercise {-}

For this and the remaining PEs in this chapter, we will use the same solubility and Pima datasets from Chapter \@ref(non-lin-algs).


```r
library(AppliedPredictiveModeling)
library(rpart)
library(caret)
library(partykit)
library(mlbench)
library(kernlab)
library(ipred)
library(randomForest)
```


```r
data(solubility)
pima_train <- read.csv("data/pima-train.csv",header=TRUE)
pima_train <- pima_train[complete.cases(pima_train),]
```

#### Regression {-}


```r
set.seed(100)
train_control <- trainControl(method='cv', number=5, returnResamp='none')
bag_regress <- train(solTrainXtrans, solTrainY,
                   method = "treebag",
                   trControl= train_control
                   )
bag_regress$results
```

```
##   parameter      RMSE  Rsquared       MAE    RMSESD RsquaredSD      MAESD
## 1      none 0.9208562 0.8011265 0.7022581 0.0404631 0.03403587 0.03130357
```

#### Classification {-}


```r
set.seed(100)
bag_class <- train(as.factor(diabetes) ~., 
                   data = pima_train,
                   method = "treebag",
                   trControl= train_control
                   )
bag_class$results
```

```
##   parameter  Accuracy     Kappa AccuracySD    KappaSD
## 1      none 0.7715822 0.4847493 0.04241899 0.08441819
```

## Random Forest

While Bagging significantly reduced the variance compared to some other machine learning models, it has some drawbacks. In particular, Bagging creates trees on the entire feature space for each sample. Thus, most trees, especially at the top layers will look very similar and as a result most of the trees are not independent from each other. The Random Forest algorithm fixes this problem. Reviewing the algorithm in Figure \@ref(fig:random-forest-algorithm), you will notice that instead of building a tree on the entire feature space, Random Forest trees are built using a random sample of $k < P$ of the original predictors. For classification, a general default for the number of predictors at each split point is $k\ =\ \sqrt{P}$. For regression, the default number of predictors at each split point is $k\ =\ \frac{P}{3}$. A side benefit of this algorithm is that Random Forest is more computatinally efficient since trees are not built on the entire set of features. 

<div class="figure" style="text-align: center">
<img src="img/random-forest-algorithm.png" alt="Random Forest algorithm from Kuhn and Johnson (2013)." width="90%" />
<p class="caption">(\#fig:random-forest-algorithm)Random Forest algorithm from Kuhn and Johnson (2013).</p>
</div>

### Practical Exercise {-}

#### Regresssion {-}


```r
set.seed(41)
rf_regress_model <- randomForest(solTrainXtrans,solTrainY,
                                 importance=TRUE,
                                 ntrees=500)
rf_regress_model
```

```
## 
## Call:
##  randomForest(x = solTrainXtrans, y = solTrainY, importance = TRUE,      ntrees = 500) 
##                Type of random forest: regression
##                      Number of trees: 500
## No. of variables tried at each split: 76
## 
##           Mean of squared residuals: 0.4222374
##                     % Var explained: 89.91
```

```r
head(rf_regress_model$importance)
```

```
##           %IncMSE IncNodePurity
## FP001 0.006445326      1.786854
## FP002 0.010880283      2.427839
## FP003 0.007691509      2.469004
## FP004 0.025182610     17.044367
## FP005 0.004678159      1.405516
## FP006 0.009439474      4.459932
```

```r
#We won't run this, but this is how you would train this model using CARET
# mtry_min <- floor(ncol(solTrainXtrans)/3)
# mtry_max <- ncol(solTrainXtrans)
# mtry <- seq(mtry_min,mtry_max)
# train_control <- trainControl(method='cv', number=5, search='random')
# model_metric <- "RMSE"
# tune_grid <- expand.grid(.mtry=mtry)
# 
# rf_random_regress <- train(solTrainXtrans, solTrainY,
#                    method = "rf",
#                    trControl= train_control,
#                    metric = model_metric,
#                    tuneGrid = tune_grid
#                    )
```

#### Classification {-}


```r
#Random search
set.seed(41)
mtry_min <- floor(sqrt(ncol(pima_train)-1))
mtry_max <- ncol(pima_train)-1
mtry <- seq(mtry_min,mtry_max)
train_control <- trainControl(method='cv', number=5, search='random')
model_metric <- "Accuracy"
tune_grid <- expand.grid(.mtry=mtry)


rf_random_class <- train(as.factor(diabetes) ~., 
                   data = pima_train,
                   method = "rf",
                   trControl= train_control,
                   metric = model_metric,
                   tuneGrid = tune_grid
                   )
rf_random_class$results
```

```
##   mtry  Accuracy     Kappa AccuracySD    KappaSD
## 1    2 0.7940864 0.5291364 0.02769544 0.06039786
## 2    3 0.7780581 0.4914831 0.03170987 0.06987462
## 3    4 0.8005909 0.5439405 0.02218887 0.04789905
## 4    5 0.7846155 0.5031842 0.02364283 0.06340367
## 5    6 0.7813880 0.4970204 0.03093920 0.07522984
## 6    7 0.7846667 0.5053203 0.03236422 0.08151918
## 7    8 0.7814921 0.4994510 0.03575589 0.08559938
```



## AdaBoost

<div class="figure" style="text-align: center">
<img src="img/adaboost-algorithm.png" alt="Adaboost classification algorithm from Kuhn and Johnson (2013)." width="90%" />
<p class="caption">(\#fig:adaboost-algorithm)Adaboost classification algorithm from Kuhn and Johnson (2013).</p>
</div>

### Practical Exercise {-}

## Gradient Boosting

<div class="figure" style="text-align: center">
<img src="img/gradient-boosting-regression.png" alt="Gradient Boosting regression algorithm from Kuhn and Johnson (2013)." width="90%" />
<p class="caption">(\#fig:gradient-boosting-regression)Gradient Boosting regression algorithm from Kuhn and Johnson (2013).</p>
</div>

<div class="figure" style="text-align: center">
<img src="img/gradient-boosting-classification.png" alt="Gradient Boosting classification algorithm from Kuhn and Johnson (2013)." width="90%" />
<p class="caption">(\#fig:gradient-boosting-classification)Gradient Boosting classification algorithm from Kuhn and Johnson (2013).</p>
</div>

### Practical Exercise {-}

