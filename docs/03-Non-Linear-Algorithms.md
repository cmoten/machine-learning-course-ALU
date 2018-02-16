# Non-linear Algorithms {#non-lin-algs}

We now focus our attention to non-linear machine learning algorithms. As we learn about these algorithms you should notice that many of these are an extension of the linear algorithms we learned in Chapter \@ref(lin-algs).  

## Classification and Regression Trees (CART)

Our first algorithm we will examine is the CART algorithm. This algoritm is important as it forms the basis for ensemble algorithms such as Random Forests and Bagged Decision Trees which we will learn in Chaper \@ref(ens-algs). CART models are also used for both regression and classification problems. 

### What are CART models? {-}

CART models are simply decision trees. That is to say, the CART algorithm searches for points in the data to split the data into rectangular sections that increase the prediction accuracy. The more splits that are made within the data produces smaller and smaller segments up until a designated stopping point to prevent overfitting. A simple example will illustate the intuition behind CART.  Figure \@ref(fig:cart-example) demonstrates a simple CART model. Reviewing this output we can see the definition of the model being

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

Using the above decision algorithm, we can make future predictions based of the split values of Predictor A and B. 

### How does a CART model learn from data? {-}

#### Regression Trees {-}

For regresdsion trees CART models search through all the data points for each predictor to determine the optimal split point that partitions the data into two groups and the sum of squred errors (SSE) is the lowest possible value for that split. In the previous example, that value was 1.7 for Predictor A. From that first split, the method is repeated within each new region until the model reaches a designated stopping point, for instance $n < 20$ data points in any new region. 


\[
SSE\ =\ \sum_{i\in S_1}^{ }\left(y_i-\overline{y_1}\right)^2\ +\ \sum_{i\in S_2}^{ }\left(y_i-\overline{y_2}\right)^2
\]


#### Classification Trees {-}

A frequently used measure for classification trees is the GINI index and is computed by

\[
G\ =\ \sum_{k=1}^np_k\times\left(1-p_k\right)
\]

where $p_k$ is the classification probability of the $k$th class. Using a process similar to the regression method, the algorithms searches for the best split point based on the lowest Gini index indicating the purest node for that split. In this case, purity refers to a node having more of one particular class than another. 

#### Two-class Example {-}
To illustrate how to compute the Gini indext, we will walk through a simple two-class example. The first step is sort the sample based on the predictor values and then find the midpoint of the optimal split point. This would create a contengincy table like the one below. For this table $n_{11}$ is the proportion of sample observations that are in group 1(samples that are greater than the split value) class 1. The same logic follows for the other three split values. The bold faced values are the sub-totals of the split groups and the classifications. 

                        Class1                      Class2
--------------  -----------------------  -----------------------  ---------------------------
$>$ split          $n_{11}$               $n_{12}$                 $\mathbf{n_{>split}}$ 
$\leq$ split       $n_{21}$               $n_{22}$                 $\mathbf{n_{\leq split}}$
                 $\mathbf{n_{class1}}$    $\mathbf{n_{class2}}$    $\mathbf{n}$
---------------------------------------------------------------------------------------------

Before the split, the initial Gini index is 

\[G = 2\left(\frac{n_{class1}}{n}\right)\left(\frac{n_{class2}}{n}\right)\]. 

After the split the Gini index changes to

\[
\begin{align}
G &=\ 2\left[\left(\frac{n_{11}}{n_{>split}}\right)\left(\frac{n_{12}}{n_{>split}}\right)\left(\frac{n_{>split}}{n}\right)\ +\ \left(\frac{n_{21}}{n_{\leq split}}\right)\left(\frac{n_{22}}{n_{\leq split}}\right)\left(\frac{n_{\leq split}}{n}\right)\right]\\
&=\ 2\left[\left(\frac{n_{11}}{n}\right)\left(\frac{n_{12}}{n_{>split}}\right)\ +\ \left(\frac{n_{21}}{n}\right)\left(\frac{n_{22}}{n_{\leq split}}\right)\right]
\end{align}
\]

We can see from the above equation we see that the Gini index now depends upon the proportion of samples of each class within a region that is weighted by the proportion of sample points in each split group. This new value is compared to the previous value and if the new value is smaller, the split is made, and ignored otherwise. 

We will now work through an example problem. Figure \@ref(fig:gini-example) shows the results of predicted classes with regions for a two-class model. There are a total of 208 observations: 111 observations for Class 1 and 97 observations for Class 2. Using this information, we can compute the Gini index before any splits.


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


The contingency table for Predictor B of the above figure is below. Using this information we can compute the post-split Gini index 

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
This exercise will use the ``AppliedPredictiveModeling``, ``rpart``, and ``caret``, packages.


```r
library(AppliedPredictiveModeling)
library(rpart)
library(caret)
library(partykit)
```

#### Data {-}
For this exercise, we will use the solubility data set described in @kuhn2013applied. In short the features for this data set are:

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

The ``rpart()`` function in `R` is a widely used method for computing trees using CART, and we will use this function. Another package ``party`` uses the [conditional inference framework](https://stats.stackexchange.com/questions/12140/conditional-inference-trees-vs-traditional-decision-trees) to form its trees.


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
#To match the figure in the Applied Predictive Modeling book, we will choose a max depth of 5.

training_data <- data.frame(cbind(solTrainXtrans,solTrainY))
training_model <- rpart(solTrainY ~., data = training_data,
                    control = rpart.control(maxdepth = 5))
model_tree <- as.party(training_model)
```

<div class="figure" style="text-align: center">
<img src="img/final-cart-plot.png" alt="Final CART model regression results." width="90%" />
<p class="caption">(\#fig:cart-plot)Final CART model regression results.</p>
</div>

## Naive Bayes

### Practical Exerecise

## k-Nearest Neigbors

### Practical Exerecise

## Support Vector Machines

### Practical Exerecise


