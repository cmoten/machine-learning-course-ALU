# Non-linear Algorithms {#non-lin-algs}

We now focus our attention on non-linear machine learning algorithms. As we learn about these algorithms, you should notice that many of these are an extension of the linear algorithms we learned in Chapter \@ref(lin-algs).  

## Classification and Regression Trees (CART)

The first algorithm we will examine is the CART algorithm. This algorithm is crucial as it forms the basis for ensemble algorithms such as Random Forests and Bagged Decision Trees which we will learn in Chapter \@ref(ens-algs). CART models are also used for both regression and classification problems. 

### What are CART models? {-}

CART models are simply decision trees. That is to say; the CART algorithm searches for points in the data to split the data into rectangular sections that increase the prediction accuracy. The more splits that are made within the data produces smaller and smaller segments up to a designated stopping point to prevent overfitting. A simple example will illustrate the intuition behind the CART algorithm.  Figure \@ref(fig:cart-example) demonstrates a simple CART model. Reviewing this output, we can see the definition of the model being

```
if Predictor A >= 1.7 then
   if Predictor B >= 202.1 the Outcome = 1.3
   else Outcome = 5.6
else Outcome = 2.5
```

<div class="figure" style="text-align: center">
<img src="img/cart-example.png" alt="Example output and decision tree model adapted from Kuhn and Johnson (2013)." width="90%" />
<p class="caption">(\#fig:cart-example)Example output and decision tree model adapted from Kuhn and Johnson (2013).</p>
</div>

Using the above decision algorithm, we can make future predictions based on the split values of Predictor A and B. 

### How does a CART model learn from data? {-}

#### Regression Trees {-}

For regression trees, CART models search through all the data points for each predictor to determine the optimal split point that partitions the data into two groups and the sum of squared errors (SSE) is the lowest possible value for that split. In the previous example, that value was 1.7 for Predictor A. From that first split; the method is repeated within each new region until the model reaches a designated stopping point, for instance, $n < 20$ data points in any new region. 


\[
SSE\ =\ \sum_{i\in S_1}^{ }\left(y_i-\overline{y_1}\right)^2\ +\ \sum_{i\in S_2}^{ }\left(y_i-\overline{y_2}\right)^2
\]


#### Classification Trees {-}

A frequently used measure for classification trees is the GINI index and is computed by

\[
G\ =\ \sum_{k=1}^Kp_k\times\left(1-p_k\right)
\]

where $p_k$ is the classification probability of the $k$th class. The optimal split point search process is similar to the regression method, except now the algorithm searches for the best split point based on the lowest Gini index indicating the purest node for that split. In this case, purity refers to a node having more of one particular class than another. 

#### Two-class Example {-}
To illustrate how to compute the Gini index, we will walk through a simple two-class example. The first step is to sort the sample based on the predictor values and then find the midpoint of the optimal split point. This split would create a contingency table like the one below. For this table, $n_{11}$ is the proportion of sample observations that are in group 1(samples that are greater than the split value) class 1. The same logic follows for the other three split values. The bold-faced values are the sub-totals of the split groups and the classifications. 


                        Class1                      Class2
--------------  -----------------------  -----------------------  ---------------------------
$>$ split          $n_{11}$               $n_{12}$                 $\mathbf{n_{>split}}$ 
$\leq$ split       $n_{21}$               $n_{22}$                 $\mathbf{n_{\leq split}}$
                 $\mathbf{n_{class1}}$    $\mathbf{n_{class2}}$    $\mathbf{n}$
---------------------------------------------------------------------------------------------

Before the split, the initial Gini index is 

\[G = 2\left(\frac{n_{class1}}{n}\right)\left(\frac{n_{class2}}{n}\right)\]. 

After the split, the Gini index changes to

\[
\begin{align}
G &=\ 2\left[\left(\frac{n_{11}}{n_{>split}}\right)\left(\frac{n_{12}}{n_{>split}}\right)\left(\frac{n_{>split}}{n}\right)\ +\ \left(\frac{n_{21}}{n_{\leq split}}\right)\left(\frac{n_{22}}{n_{\leq split}}\right)\left(\frac{n_{\leq split}}{n}\right)\right]\\
&=\ 2\left[\left(\frac{n_{11}}{n}\right)\left(\frac{n_{12}}{n_{>split}}\right)\ +\ \left(\frac{n_{21}}{n}\right)\left(\frac{n_{22}}{n_{\leq split}}\right)\right]
\end{align}
\]

We can see from the above equation that the Gini index now depends upon the proportion of samples of each class within a region that is weighted by the proportion of sample points in each split group. We compare the new Gini index value to the previous value of the Gini index.  If the new value is smaller than the previous value, the model makes the split the proposed split otherwise. 

Another frequently used method is the Information (Entropy) index and is calculated by

\[
I =\ \sum_{k=1}^{K}-p_klog_2\left(p_k\right)
\]

Similar to the Gini index for K = 2 classes, the information before a split is

\[
I(\text{before split}) = -\left[\frac{n_{class1}}{n}\ \times log_2\left(\frac{n_{class1}}{n}\right)\right]\ - \left[\frac{n_{class2}}{n}\ \times log_2\left(\frac{n_{class2}}{n}\right)\right]
\]

To determine how well a split improved the model, we will compute the information gain statistic. An increase in gain is an advantage, and a decrease in gain is a disadvantage. The calculation of gain is

\[
gain(\text{split}) =\ I(\text{before split})\ -\ I(\text{after split})
\]

To calculate the information index after the split, do the following

\[
\begin{align}
I(>split) &=\ -\left[\frac{n_{11}}{n_{>split}}\ \times\ log_2\left(\frac{n_{11}}{n_{>split}}\right)\right]\ - \left[\frac{n_{12}}{n_{>split}}\ \times\ log_2\left(\frac{n_{12}}{n_{>split}}\right)\right]\\
I(\leq split) &=\ -\left[\frac{n_{21}}{n_{\leq split}}\ \times\ log_2\left(\frac{n_{21}}{n_{\leq split}}\right)\right]\ - \left[\frac{n_{22}}{n_{\leq split}}\ \times\ log_2\left(\frac{n_{22}}{n_{\leq split}}\right)\right]\\ 
I(\text{after split}) &=\ \frac{n_{>split}}{n}\ I(>split)\ +\ \frac{n_{\leq split}}{n}\ I(\leq split)
\end{align}
\]

##### Gini Example {-}
We will now work through an example problem using the Gini index. Figure \@ref(fig:gini-example) shows the results of predicted classes with regions for a two-class model. There are a total of 208 observations: 111 observations for Class 1 and 97 observations for Class 2. Using this information, we can compute the Gini index before any splits.


```r
n_obs <- 208
n_class_one <- 111
n_class_two <- 97
gini_before <- 2 * (n_class_one/n_obs) * (n_class_two/n_obs)
```

Based on the above calculation the pre-split Gini index is 0.498. 

<div class="figure" style="text-align: center">
<img src="img/applied-pred-Ch14Fig01.png" alt="Example classification model results." width="90%" />
<p class="caption">(\#fig:gini-example)Example classification model results.</p>
</div>


The contingency table for Predictor B of the above figure is below. Using this information, we can compute the post-split Gini index 

                  Class1     Class2
--------------  ---------- ----------
$B > 0.197$         91         30
$B \leq 0.197$      20         67
-------------------------------------


```r
n11 <- 91; n12 <- 30; n21 <- 20; n22 <- 67;
n_group_one <- 121; n_group_two <- 87;
group_one_prop <- (n11/n_obs)*(n12/n_group_one)
group_two_prop <- (n21/n_obs)*(n22/n_group_two)
gini_after <- 2 * sum(group_one_prop,group_two_prop)
```

The final Gini index post-split is now 0.365 which indicates an improvement in classification purity. We can also observe that any value $\leq 0.197$ will receive a classification of 2 and a classification of 1 otherwise with regards to this particular split point. 

### Pre-processing requirements? {-}

CART models do not require any special pre-processing of the data, but you can center and scale values based on skewness and other factors.

### Practical Exerecise

#### Libraries{-}
This exercise will use the ``AppliedPredictiveModeling``, ``rpart``, ``caret``, and ``partykit`` packages.


```r
library(AppliedPredictiveModeling)
library(rpart)
library(caret)
library(partykit)
library(mlbench)
library(kernlab)
```

#### Regression Tree {-}

##### Data {-}
For this exercise, we will use the solubility dataset described in @kuhn2013applied. In short, the features of this dataset are:

* 208 binary "fingerprints" that indicate the presence or absence of a particular chemical sub-structure; 
* 16 count descriptors (such as the number of bonds or the number of Bromine atoms);
* 4 continuous descriptors (such as molecular weight or surface area) [@kuhn2014package]. 

The authors centered and scaled the data to account for skewness. The target variable is a vector of log10 solubility values. The goal of this exercise is to predict the solubility value based on the set of features. Below is a view of some of the features and target values


```r
data(solubility)
str(solTrainXtrans[,c(1:10,209:228)])
```

```
## 'data.frame':	951 obs. of  30 variables:
##  $ FP001            : int  0 0 1 0 0 1 0 1 1 1 ...
##  $ FP002            : int  1 1 1 0 0 0 1 0 0 1 ...
##  $ FP003            : int  0 0 1 1 1 1 0 1 1 1 ...
##  $ FP004            : int  0 1 1 0 1 1 1 1 1 1 ...
##  $ FP005            : int  1 1 1 0 1 0 1 0 0 1 ...
##  $ FP006            : int  0 1 0 0 1 0 0 0 1 1 ...
##  $ FP007            : int  0 1 0 1 0 0 0 1 1 1 ...
##  $ FP008            : int  1 1 1 0 0 0 1 0 0 0 ...
##  $ FP009            : int  0 0 0 0 1 1 1 0 1 0 ...
##  $ FP010            : int  0 0 1 0 0 0 0 0 0 0 ...
##  $ MolWeight        : num  5.34 5.9 5.33 4.92 5.44 ...
##  $ NumAtoms         : num  3.37 3.91 3.53 3.3 3.47 ...
##  $ NumNonHAtoms     : num  2.83 3.3 2.77 2.4 2.77 ...
##  $ NumBonds         : num  3.43 3.97 3.53 3.3 3.47 ...
##  $ NumNonHBonds     : num  4.01 4.87 3.71 3.08 3.71 ...
##  $ NumMultBonds     : num  5.26 4.68 3.24 1.38 2.94 ...
##  $ NumRotBonds      : num  0 1.609 1.609 0.693 1.792 ...
##  $ NumDblBonds      : num  0 0 0.567 0.805 0 ...
##  $ NumAromaticBonds : num  2.83 2.56 1.95 0 1.95 ...
##  $ NumHydrogen      : num  3.86 5.32 4.73 4.47 4.47 ...
##  $ NumCarbon        : num  4.18 5.09 4.02 3.51 3.32 ...
##  $ NumNitrogen      : num  0.585 0.642 0 0 0.694 ...
##  $ NumOxygen        : num  0 0.693 1.099 0 0 ...
##  $ NumSulfer        : num  0 0.375 0 0 0 0.375 0 0 0 0 ...
##  $ NumChlorine      : num  0 0 0 0 0.375 ...
##  $ NumHalogen       : num  0 0 0 0 0.375 ...
##  $ NumRings         : num  1.386 1.609 0.693 0.693 0.693 ...
##  $ HydrophilicFactor: num  -1.607 -0.441 -0.385 -2.373 -0.071 ...
##  $ SurfaceArea1     : num  6.81 9.75 8.25 0 9.91 ...
##  $ SurfaceArea2     : num  6.81 12.03 8.25 0 9.91 ...
```

```r
str(solTrainY)
```

```
##  num [1:951] -3.97 -3.98 -3.99 -4 -4.06 -4.08 -4.08 -4.1 -4.1 -4.11 ...
```

##### Create and Analyze Regression Tree {-}

The ``rpart()`` function in `R` is a widely used method for computing trees using the CART method, and we will use this function. Another package ``party`` uses the [conditional inference framework](https://stats.stackexchange.com/questions/12140/conditional-inference-trees-vs-traditional-decision-trees) to form its trees.


```r
set.seed(100)
rpartTune <- train(solTrainXtrans, solTrainY,
                   method = "rpart2",
                   tuneLength = 10,
                   trControl= trainControl(method = "cv")
                   )
rpartTune$results
```

```
##    maxdepth     RMSE  Rsquared       MAE     RMSESD RsquaredSD      MAESD
## 1         1 1.617667 0.3745252 1.2657915 0.11511437 0.05777279 0.08191460
## 2         2 1.433114 0.5067404 1.1326186 0.07599686 0.04909341 0.04940391
## 3         3 1.357672 0.5568291 1.0657348 0.07354389 0.05231774 0.06091190
## 4         4 1.263596 0.6166997 0.9974476 0.10201869 0.05547696 0.07947602
## 5         5 1.192831 0.6581800 0.9429124 0.11324197 0.05669830 0.08594278
## 6         6 1.142654 0.6853056 0.9009065 0.10585813 0.05990671 0.08607556
## 7         7 1.111858 0.7020728 0.8707216 0.10580483 0.06389863 0.08126706
## 8         8 1.094535 0.7110088 0.8545809 0.11400541 0.06474333 0.09512021
## 9         9 1.091880 0.7116190 0.8465921 0.11938842 0.06737339 0.10068304
## 10       10 1.068799 0.7236716 0.8232469 0.12842861 0.07102897 0.10641491
```

```r
#Build the initial model

training_data <- data.frame(cbind(solTrainXtrans,solTrainY))
training_model <- rpart(solTrainY ~., data = training_data,
                    control = rpart.control(maxdepth = 10))

#Examine the tree complexity
plotcp(training_model)
```

<img src="03-Non-Linear-Algorithms_files/figure-html/cart-tree-calc-1.png" width="672" />

```r
training_model$cptable
```

```
##            CP nsplit rel error    xerror       xstd
## 1  0.37300506      0 1.0000000 1.0010223 0.05357024
## 2  0.13770014      1 0.6269949 0.6314019 0.03143820
## 3  0.06971510      2 0.4892948 0.4945930 0.02321245
## 4  0.06180269      3 0.4195797 0.4434574 0.02133679
## 5  0.04729111      4 0.3577770 0.3838988 0.01904376
## 6  0.02650301      5 0.3104859 0.3514391 0.01837681
## 7  0.01789274      6 0.2839829 0.3062709 0.01570413
## 8  0.01553523      7 0.2660901 0.2989517 0.01566820
## 9  0.01178134      8 0.2505549 0.2910022 0.01551438
## 10 0.01150195      9 0.2387736 0.2879867 0.01543448
## 11 0.01000000     10 0.2272716 0.2761810 0.01525549
```

```r
#Add min(xerror+xstd) and find the smallest tree w/xerror < min(xerror+xstd)
which(training_model$cptable[,4] < min(training_model$cptable[,4]+training_model$cptable[,5]))
```

```
##  9 10 11 
##  9 10 11
```

```r
#Prune the tree
training_model <- rpart(solTrainY ~., data = training_data,
                    cp = .014)
model_tree <- as.party
```

Figure \@ref(fig:cart-plot) displays the final results that we can use for interpretation of the model. To create the plot just use the code ``plot(model_tree)``. You could also use the ``prp`` function from the ``rpart.plot`` package. Using that package the ``prp`` plot would be 

``prp(training_model,type=4,extra=106,box.col = c("#deebf7","#fff7bc")[training_model$frame$yval],cex = 0.6)``

<div class="figure" style="text-align: center">
<img src="img/final-cart-plot.png" alt="Final CART model regression results." width="90%" />
<p class="caption">(\#fig:cart-plot)Final CART model regression results.</p>
</div>

#### Classification Tree {-}

For this exercise, we will use the ``PimaIndianDiabetes2`` data from the ``mlbench`` package based on the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes). Click the link for a description of the dataset. In this exercise, we will build a classification tree that will classify a person as "pos" or "neg" for diabetes from a set of input features based on personal characteristics. 

##### Data {-}

I already created a training and test dataset from the original data. There are some missing values, so we will only use the complete cases for this example since the algorithms won't work with missing data. 


```r
pima_train <- read.csv("data/pima-train.csv",header=TRUE)
pima_train <- pima_train[complete.cases(pima_train),]
str(pima_train)
```

```
## 'data.frame':	311 obs. of  9 variables:
##  $ pregnant: int  1 0 2 5 0 1 1 3 11 10 ...
##  $ glucose : int  89 137 197 166 118 103 115 126 143 125 ...
##  $ pressure: int  66 40 70 72 84 30 70 88 94 70 ...
##  $ triceps : int  23 35 45 19 47 38 30 41 33 26 ...
##  $ insulin : int  94 168 543 175 230 83 96 235 146 115 ...
##  $ mass    : num  28.1 43.1 30.5 25.8 45.8 43.3 34.6 39.3 36.6 31.1 ...
##  $ pedigree: num  0.167 2.288 0.158 0.587 0.551 ...
##  $ age     : int  21 33 53 51 31 33 32 27 51 41 ...
##  $ diabetes: Factor w/ 2 levels "neg","pos": 1 2 2 2 2 1 2 1 2 2 ...
```

##### Create and Analyze Classification Tree {-}


```r
#Gini Index
set.seed(33)
train_ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
gini_tune <- train(as.factor(diabetes) ~., data = pima_train, method = "rpart",
                   trControl=train_ctrl,
                   tuneLength = 10, 
                   parms=list(split='gini'))
gini_tune
```

```
## CART 
## 
## 311 samples
##   8 predictor
##   2 classes: 'neg', 'pos' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 3 times) 
## Summary of sample sizes: 280, 280, 280, 280, 280, 280, ... 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa    
##   0.00000000  0.7790703  0.4994923
##   0.03219107  0.7740995  0.4937550
##   0.06438214  0.7699731  0.4670725
##   0.09657321  0.7580735  0.4369132
##   0.12876428  0.7591151  0.4401497
##   0.16095535  0.7410013  0.4188441
##   0.19314642  0.7420430  0.4219648
##   0.22533749  0.7420430  0.4219648
##   0.25752856  0.7420430  0.4219648
##   0.28971963  0.7150448  0.3111903
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was cp = 0.
```

```r
pima_gini_model <- rpart(as.factor(diabetes) ~., data = pima_train,
                    cp = .004)
pima_gini_tree <- as.party(pima_gini_model)

#Information Index
set.seed(33)
info_tune <- train(as.factor(diabetes) ~., data = pima_train, method = "rpart",
                   trControl=train_ctrl,
                   tuneLength = 10, 
                   parms=list(split='information'))
info_tune
```

```
## CART 
## 
## 311 samples
##   8 predictor
##   2 classes: 'neg', 'pos' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 3 times) 
## Summary of sample sizes: 280, 280, 280, 280, 280, 280, ... 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa    
##   0.00000000  0.7768907  0.5019724
##   0.03219107  0.7720542  0.4896788
##   0.06438214  0.7731340  0.4906999
##   0.09657321  0.7430847  0.4352926
##   0.12876428  0.7441263  0.4385291
##   0.16095535  0.7366644  0.4268724
##   0.19314642  0.7377061  0.4299930
##   0.22533749  0.7377061  0.4299930
##   0.25752856  0.7377061  0.4299930
##   0.28971963  0.6912097  0.2398796
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was cp = 0.
```

```r
pima_info_model <- rpart(as.factor(diabetes)~., data = pima_train,
                         cp = .004)
pima_info_tree <- as.party(pima_info_model)
```

 

<div class="figure" style="text-align: center">
<img src="img/cart-gini-example.png" alt="Gini Index CART model classification results." width="90%" />
<p class="caption">(\#fig:cart-gini-plot)Gini Index CART model classification results.</p>
</div>

<div class="figure" style="text-align: center">
<img src="img/cart-info-example.png" alt="Information Index CART model classification results." width="90%" />
<p class="caption">(\#fig:cart-info-plot)Information Index CART model classification results.</p>
</div>

## Naive Bayes

Recall Baye's Theorem from Chapter \@ref(lin-algs)

\[
\Pr\left(Y\ =\ k\ |X\right)\ =\ \frac{P\left(X|Y\ =\ k\right)P\left(Y\right)}{P\left(X\right)}
\]

where we want to answer the question "what is the probability of a particular target classification given the observed features?"

Upon calculating the posterior probability for each classification, you can then select the classification with the highest probability. In the literature, this calculation is the maximum a posteriori (MAP), and we find it by

\[
\begin{align}
MAP(Y) &=\ max\left(P(Y \vert X\right)\\
&=\ max\left(\frac{P\left(X|Y\ =\ k\right)P\left(Y\right)}{P\left(X\right)}\right)\\
&=\ max\left(P\left(X|Y\ =\ k\right)P\left(Y\right)\right)
\end{align}
\]

We can ignore the denominator of the original equation because the $P(X)$ is a constant for terms. Also, the reason why this method is called Naive Bayes is that the features are assumed to be independent. To put it another way, instead of computing $P(x_1,x_2,\dots,x_p\ \vert Y)$, the independence assumption simplifies this calculation to
 
\[
P\left(X\vert Y\ =\ k\right) = \prod_{j=1}^{P}P\left(X \vert Y\ = k\right)
\]

Another aspect of the Naive Bayes method is the distribution of the features. A Gaussian distribution will be used for continuous features, and kern density estimates for discrete features. 

### Practical Exerecise

We will use the Pima data for this exercise.

#### Naive Bayes Model {-}
We will use the ``naiveBayes`` function from the ``klaR`` package along with the ``caret`` package. 


```r
set.seed(33)
library(klaR)
nb_tune <- train(as.factor(diabetes) ~ ., 
         data=pima_train,
         method = "nb",
         trControl = trainControl(method="none"),
         tuneGrid = data.frame(fL=0, usekernel=FALSE, adjust=1))
nb_preds <- predict(nb_tune,pima_train,type = "raw")
confusionMatrix(nb_preds,as.factor(pima_train$diabetes))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction neg pos
##        neg 169  35
##        pos  35  72
##                                           
##                Accuracy : 0.7749          
##                  95% CI : (0.7244, 0.8201)
##     No Information Rate : 0.6559          
##     P-Value [Acc > NIR] : 3.352e-06       
##                                           
##                   Kappa : 0.5013          
##  Mcnemar's Test P-Value : 1               
##                                           
##             Sensitivity : 0.8284          
##             Specificity : 0.6729          
##          Pos Pred Value : 0.8284          
##          Neg Pred Value : 0.6729          
##              Prevalence : 0.6559          
##          Detection Rate : 0.5434          
##    Detection Prevalence : 0.6559          
##       Balanced Accuracy : 0.7507          
##                                           
##        'Positive' Class : neg             
## 
```

## k-Nearest Neigbors

The basic idea of the $k$-nearest neighbors (KNN) algorithm is to create a distance matrix of the all the feature variables and choose the $k$ most adjacent data points closest to an evaluated point. Since KNN uses the entire dataset, no learning is necessary from the algorithm. The primary choice of the modeler is what decision metric to use. The primary parameter used is a variation of the Minkowski distance metric. You can compute this metric by

\[
\left(\sum_{i=1}^{P}\vert x_{ai} - x_{bi} \vert^q\right)^\frac{1}{q}
\]

where $\mathbf{x_a}$ and $\mathbf{x_b}$ are two sample points in the dataset. When $q\ =\ 1$ this distance metric is the Manhattan (city-block) distance, and when $q\ =\ 2$ this distance is the Euclidean distance. You will use Euclidean distance for continuous predictors and Manhattan distance for categorical or binary predictors. 

### Curse of Dimensionality {-}

Just like other machine learning methods the KNN method has its disadvantages. One disadvantage deals with high dimensional data. In essence, distances in higher dimensions are larger which ultimately mean that similar points are not necessarily local to each other. Figure \@ref(fig:knn-curse) demonstrates this problem. The figure on the left shows a unit hypercube with a sub-cube that captures a fraction of the data of the original hypercube. The sub-figure on the right shows how much of the range of each coordinate you need to capture within the sub-cube. For instance, if you want to capture 10% of the data, you will need to capture 80% of the range of coordinates for a 10-dimension dataset. This percentage increases exponentially with additional dimensions. 

<div class="figure" style="text-align: center">
<img src="img/knn-curse.png" alt="Illustration of dimensionality curse adapted from Hastie, Tibshirani, and Friedman (2009." width="90%" />
<p class="caption">(\#fig:knn-curse)Illustration of dimensionality curse adapted from Hastie, Tibshirani, and Friedman (2009.</p>
</div>

### Practical Exerecise

#### Regression {-}

##### Data {-}
We will use the solubility data for this exercise.


```r
knn_data <- solTrainXtrans[,-nearZeroVar(solTrainXtrans)]
```

##### Create the model {-}

```r
set.seed(100)
knn_reg_model <- train(knn_data,
                       solTrainY,
                       method = "knn",
                       #Center and scaling will occur for new predictors
                       preProc = c("center", "scale"),
                       tuneGrid = data.frame(.k = 1:20),
                       trControl = trainControl(method = "cv")
                       )
knn_reg_model$finalModel
```

```
## 4-nearest neighbor regression model
```

The final model selected was a model based on a value of $k\ =\ 4$. Figure \@ref(fig:knn-reg-plot) shows graphically why this model was the best of the 20 analyzed. 

<div class="figure" style="text-align: center">
<img src="img/knn-reg-plot.png" alt="Plots of RMSE and RSquared for values of k." width="90%" />
<p class="caption">(\#fig:knn-reg-plot)Plots of RMSE and RSquared for values of k.</p>
</div>

#### Classification {-}
We will again use the Pima data.


```r
set.seed(202)
pima_knn <- train(as.factor(diabetes)~.,
                  data = pima_train,
                  method = "knn",
                  metric = "ROC",
                  preProc = c("center", "scale"),
                  tuneGrid = data.frame(.k=1:50),
                  trControl = trainControl(method = "cv",
                                           classProbs = TRUE,
                                           summaryFunction = twoClassSummary))
pima_knn
```

```
## k-Nearest Neighbors 
## 
## 311 samples
##   8 predictor
##   2 classes: 'neg', 'pos' 
## 
## Pre-processing: centered (8), scaled (8) 
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 280, 281, 280, 279, 280, 281, ... 
## Resampling results across tuning parameters:
## 
##   k   ROC        Sens       Spec     
##    1  0.6758874  0.7990476  0.5527273
##    2  0.7472781  0.7840476  0.5318182
##    3  0.7854405  0.8530952  0.5327273
##    4  0.7962013  0.8435714  0.5709091
##    5  0.7963950  0.8585714  0.5700000
##    6  0.7987013  0.8733333  0.5800000
##    7  0.8115682  0.8785714  0.5681818
##    8  0.8184437  0.8978571  0.5218182
##    9  0.8219935  0.8926190  0.5045455
##   10  0.8193268  0.8926190  0.5127273
##   11  0.8269091  0.8780952  0.5318182
##   12  0.8344470  0.8876190  0.4945455
##   13  0.8369232  0.8973810  0.4936364
##   14  0.8384621  0.8971429  0.4745455
##   15  0.8404080  0.8976190  0.4936364
##   16  0.8378268  0.8973810  0.5036364
##   17  0.8421396  0.9121429  0.4663636
##   18  0.8384405  0.9119048  0.4745455
##   19  0.8382392  0.9019048  0.4745455
##   20  0.8325335  0.9019048  0.4936364
##   21  0.8388268  0.9119048  0.5018182
##   22  0.8414740  0.9119048  0.4845455
##   23  0.8419794  0.9166667  0.4836364
##   24  0.8454058  0.9169048  0.4836364
##   25  0.8452814  0.9169048  0.4636364
##   26  0.8434935  0.9169048  0.4563636
##   27  0.8491331  0.9216667  0.4554545
##   28  0.8474957  0.9314286  0.4645455
##   29  0.8446385  0.9216667  0.4363636
##   30  0.8437154  0.9266667  0.4363636
##   31  0.8392987  0.9266667  0.4263636
##   32  0.8399816  0.9216667  0.4272727
##   33  0.8390368  0.9266667  0.4272727
##   34  0.8381483  0.9266667  0.4190909
##   35  0.8404892  0.9316667  0.4281818
##   36  0.8392532  0.9316667  0.4381818
##   37  0.8412446  0.9269048  0.4281818
##   38  0.8404448  0.9319048  0.4281818
##   39  0.8436937  0.9466667  0.4281818
##   40  0.8449589  0.9416667  0.4281818
##   41  0.8423431  0.9366667  0.4381818
##   42  0.8405693  0.9366667  0.4372727
##   43  0.8415855  0.9464286  0.4372727
##   44  0.8449123  0.9561905  0.4463636
##   45  0.8451190  0.9511905  0.4281818
##   46  0.8455054  0.9416667  0.4372727
##   47  0.8452846  0.9464286  0.4281818
##   48  0.8459361  0.9464286  0.4372727
##   49  0.8466537  0.9414286  0.4181818
##   50  0.8445725  0.9514286  0.4090909
## 
## ROC was used to select the optimal model using the largest value.
## The final value used for the model was k = 27.
```

```r
pima_knn$finalModel
```

```
## 27-nearest neighbor model
## Training set outcome distribution:
## 
## neg pos 
## 204 107
```


The final model was computed from a total of $k\ =\ 27$ neighbors and a comparison of the ROC metric for each neighbor is shown in Figure \@ref(fig:knn-class-plot).

<div class="figure" style="text-align: center">
<img src="img/knn-class-plot.png" alt="Receiver-Operator Characteristic (ROC) curve results." width="90%" />
<p class="caption">(\#fig:knn-class-plot)Receiver-Operator Characteristic (ROC) curve results.</p>
</div>

#### Exercise {-}
Create a new KNN model using only three predictor variables for the Pima data and compare those results to the current KNN model. Use the plot from the regression tree as a guide to determine which variables to choose. 

## Support Vector Machines

Support vector machines (SVM) started as a method for classification, but has extensions for regression as well. The goal of using SVM is to compute a regression line or decision boundary. 

### Regression {-}

For regression, the SVM seeks to minimize the cost function

\[
Cost\sum_{i=1}^nL_\epsilon \left(y_i - \hat{y_i}\right)\ +\ \sum_{j=1}^{P}\beta_{j}^2 
\]

where $Cost$ is the residual cost penalty and $L_\epsilon (\cdot)$ is the margin threshold.

Furthermore, we transform the prediction equation for $\hat{y}$ to

\[
\begin{align}
\hat{y} &=\ \beta_0\ +\ \beta_1u_1\ +\ \dots\ +\ \beta_pu_p\\
&=\ \beta_0\ +\ \sum_{j=1}^{P}\beta_ju_j\\
&=\ \beta_0\ +\ \sum_{j=1}^{P}\sum_{i=1}^{n}\alpha_ix_{ij}u_j\\
&=\ \beta_0\ +\ \sum_{i=1}^{n}\alpha_i\left(\sum_{j=1}^{P}x_{ij}u_j\right)
\end{align}
\]

where $\alpha_i$ is an unkown parameter estimated by the SVM algorithm. We can generalize the above equation for $\hat{y}$ even further to matrix form with

\[
f(u) = \beta_0\ +\ \sum_{i=1}^{n}\alpha_iK\left(x_i,\mathbf{u}\right)
\]

where $K\left(\cdot\right)$ is a kernel function. For SVM, there are a few models to choose for the kernel function to encompass linear and nonlinear functions of the predictors. The most common kernel functions are

\[
\begin{align}
\text{linear} &=\ x_i^{T}\mathbf{u}\\
\text{polynomial} &=\ \left(\phi(x^T\mathbf{u})\ +\ 1\right)^{degree}\\
\text{radial basis function} &=\ e^{\left(-\sigma\vert\vert x\ -\ \mathbf{u} \vert\vert^2\right)}\\
\text{hyperbolic tangent} &=\ tanh\left(\phi(x^T\mathbf{u})\ +\ 1\right)
\end{align}
\]


To demonstrate why you would use SVM for some regression situations, examine Figure \@ref(fig:svm-regress) below. The top of the figure shows a comparison of a simple linear regression model using least squares and SVM. The item to notice in this plot is the outlier at the top left of the plot. You will notice that the least squares model is more sensitive to this value, while SVM is not. The middle plot shows the relationship of the residuals to the predictive values. What is important to note with this plot is that the grey points are values that contributed to the regression line estimation, which the values shown by the red crosses did not contribute. This is because these non-contributing values are within the $\pm\ \epsilon\ =\ 0.01$ threshold set by the modeler. We explain why this happens in a little bit. The bottom plot shows a nonlinear application of SVM to a sine wave model. Again, there are outlier values in the bottom left of the plot and the least squqres method is more sensitive to these values than SVM. The reason for this is because least squares and other regression models like logistic regression are more global in their behavior, while SVM uses more local contributions from the data for its estimations. 

<div class="figure" style="text-align: center">
<img src="img/applied-pred-Ch7Fig07.png" alt="Example SVM regressions compared to least squares adapted from Kuhn and Johnson (2013)" width="90%" />
<p class="caption">(\#fig:svm-regress)Example SVM regressions compared to least squares adapted from Kuhn and Johnson (2013)</p>
</div>


As we stated previously, the samples that contribute to the calcluation of the regression curve, also called suppprt vectors, are the samples  outside of the $\epsilon$ threshold. At first, this finding may seem counterintutive, so examining Figure \@ref(fig:svm-res-models) could help with building an intuition for this. Specifically looking at the bottom right model, we can see that the residuals that are within the $\epsilon$ threshold are in fact zero and all other residuals contribute linearly to the regression equation.  

<div class="figure" style="text-align: center">
<img src="img/applied-pred-Ch7Fig06.png" alt="Plots of various contributions to the regression line of model residuals adapted from Kuhn and Johnson (2013)" width="90%" />
<p class="caption">(\#fig:svm-res-models)Plots of various contributions to the regression line of model residuals adapted from Kuhn and Johnson (2013)</p>
</div>

### Classification {-}

For the classification version of SVM, we want to compute an optimal decision boundary between seperable and non-seperable classes or categories. Figure \@ref(fig:svm-example) shows how SVM attempts to solve this problem. On the left, you will notice two seperable classes with infinite classification boundaries. On the right, is how SVM solves this problem. Specifically a boundary called the maximum margin classifier is computed. What's special about this boundary is that the boundary itself(solid black line) has margins that are set to the closets points for each class. The result of this is that now, unlike the regression model, samples that are closest to the boundary contribute to the classification model and those that are furtheest away from the boundary do not contribute at all. The equations for the classification version of SVM are similar to the regression equations. 

The general decision boundary equation is 

\[
D(u) = \beta_0\ +\ \sum_{i=1}^{n}y_i\alpha_iK(x_i,\mathbf{u})
\]

The key thing to notice is how the classification equation now includes the actual class of $y_i$ which is usally a value of -1 or 1. 

The kernel functions are

\[
\begin{align}
\text{linear} &=\ x_i^{T}\mathbf{u}\\
\text{polynomial} &=\ \left(scale(x^T\mathbf{u})\ +\ 1\right)^{degree}\\
\text{radial basis function} &=\ e^{\left(-\sigma\vert\vert x\ -\ \mathbf{u} \vert\vert^2\right)}\\
\text{hyperbolic tangent} &=\ tanh\left(scale(x^T\mathbf{u})\ +\ 1\right)
\end{align}
\]
\]

<div class="figure" style="text-align: center">
<img src="img/applied-pred-Ch13Fig09.png" alt="Datasets with separable classes adapted from Kuhn and Johnson (2013)" width="90%" />
<p class="caption">(\#fig:svm-example)Datasets with separable classes adapted from Kuhn and Johnson (2013)</p>
</div>

### Optimizaton Forumulation {-}

Ultimately, both the regression and classification equations are reformulated into a quadratic programming problem. While it is beyond the scope of this lesson to detail this formulation, the reader should review @friedman2001elements, specifically the section discussing SVMs. What's key to understanding this formulation is that the tuning of the SVM depends primarily on the cost parameter when estimating other parameters.  

### Practical Exerecise

#### Regression {-}

##### Data {-}
We will use the solubility data for this portion.

##### Model {-}


```r
set.seed(33)
solubility_svm <- train(solTrainXtrans, solTrainY,
                        method = "svmRadial",
                        preProc = c("center", "scale"),
                        tuneLength = 14,
                        trControl = trainControl(method = "cv"))
solubility_svm
```

```
## Support Vector Machines with Radial Basis Function Kernel 
## 
## 951 samples
## 228 predictors
## 
## Pre-processing: centered (228), scaled (228) 
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 855, 856, 855, 855, 858, 857, ... 
## Resampling results across tuning parameters:
## 
##   C        RMSE       Rsquared   MAE      
##      0.25  0.8014848  0.8681906  0.6039660
##      0.50  0.7102286  0.8890663  0.5327464
##      1.00  0.6613592  0.9000814  0.4934836
##      2.00  0.6314286  0.9066489  0.4684522
##      4.00  0.6186151  0.9091987  0.4541152
##      8.00  0.6064508  0.9127569  0.4457087
##     16.00  0.6018292  0.9140239  0.4427055
##     32.00  0.6018878  0.9138726  0.4429424
##     64.00  0.6007705  0.9141895  0.4423807
##    128.00  0.5998743  0.9145279  0.4443426
##    256.00  0.6025174  0.9136982  0.4463239
##    512.00  0.6039626  0.9132093  0.4469188
##   1024.00  0.6042640  0.9131080  0.4474508
##   2048.00  0.6062670  0.9125208  0.4497784
## 
## Tuning parameter 'sigma' was held constant at a value of 0.002740343
## RMSE was used to select the optimal model using the smallest value.
## The final values used for the model were sigma = 0.002740343 and C = 128.
```

```r
solubility_svm$finalModel
```

```
## Support Vector Machine object of class "ksvm" 
## 
## SV type: eps-svr  (regression) 
##  parameter : epsilon = 0.1  cost C = 128 
## 
## Gaussian Radial Basis kernel function. 
##  Hyperparameter : sigma =  0.0027403433192481 
## 
## Number of Support Vectors : 638 
## 
## Objective Function Value : -726.094 
## Training error : 0.009299
```

#### Classification {-}

##### Data {-}

We will us the Pima dataset

##### Model {-}


```r
set.seed(202)
sigmaRange <- sigest(as.factor(diabetes) ~.,data=pima_train)
svmGrid <- expand.grid(.sigma = sigmaRange[1],
                       .C = 2^(seq(-4,4))) 
set.seed(386)
pima_svm <- train(as.factor(diabetes)~.,
                  data = pima_train,
                  method = "svmRadial",
                  metric = "ROC",
                  preProc = c("center","scale"),
                  tuneGrid = svmGrid,
                  fit = FALSE,
                  trControl = trainControl(method = "cv",
                                           classProbs = TRUE,
                                           summaryFunction = twoClassSummary))
pima_svm
```

```
## Support Vector Machines with Radial Basis Function Kernel 
## 
## 311 samples
##   8 predictor
##   2 classes: 'neg', 'pos' 
## 
## Pre-processing: centered (8), scaled (8) 
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 280, 279, 279, 279, 280, 280, ... 
## Resampling results across tuning parameters:
## 
##   C        ROC        Sens       Spec     
##    0.0625  0.8434134  0.7795238  0.6945455
##    0.1250  0.8434134  0.7842857  0.6945455
##    0.2500  0.8503506  0.8426190  0.6090909
##    0.5000  0.8476623  0.8573810  0.5909091
##    1.0000  0.8478203  0.8678571  0.5727273
##    2.0000  0.8427922  0.8528571  0.5736364
##    4.0000  0.8400390  0.8626190  0.5154545
##    8.0000  0.8388355  0.8828571  0.4872727
##   16.0000  0.8367100  0.9066667  0.4972727
## 
## Tuning parameter 'sigma' was held constant at a value of 0.03176525
## ROC was used to select the optimal model using the largest value.
## The final values used for the model were sigma = 0.03176525 and C = 0.25.
```

```r
pima_svm$finalModel
```

```
## Support Vector Machine object of class "ksvm" 
## 
## SV type: C-svc  (classification) 
##  parameter : cost C = 0.25 
## 
## Gaussian Radial Basis kernel function. 
##  Hyperparameter : sigma =  0.0317652466552473 
## 
## Number of Support Vectors : 197 
## 
## Objective Function Value : -43.388 
## Probability model included.
```


