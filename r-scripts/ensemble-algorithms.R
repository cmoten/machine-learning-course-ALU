library(AppliedPredictiveModeling)
library(rpart)
library(caret)
library(partykit)
library(mlbench)
library(kernlab)
library(ipred)
library(randomForest)
library(gbm)

## Data
data(solubility)
pima_train <- read.csv("data/pima-train.csv",header=TRUE)
pima_train <- pima_train[complete.cases(pima_train),]

##### Bagging #####

## Regression 
set.seed(100)
train_control <- trainControl(method='cv', number=5, returnResamp='none')
bag_regress <- train(solTrainXtrans, solTrainY,
                     method = "treebag",
                     trControl= train_control
)
bag_regress$results


## Classification
set.seed(100)
bag_class <- train(as.factor(diabetes) ~., 
                   data = pima_train,
                   method = "treebag",
                   trControl= train_control
)
bag_class$results
###############################

##### Random Forest #####

## Regression

set.seed(41)
rf_regress_model <- randomForest(solTrainXtrans,solTrainY,
                                 importance=TRUE,
                                 ntrees=500)
rf_regress_model
head(rf_regress_model$importance)


#We won't run this, but this is how you would train this model using CARET
# mtry_min <- floor(ncol(solTrainXtrans)/3)
# mtry_max <- ncol(solTrainXtrans)
# mtry <- seq(mtry_min,mtry_max)
# train_control <- trainControl(method='cv', number=5, search='grid')
# model_metric <- "RMSE"
# tune_grid <- expand.grid(.mtry=mtry)
# 
# rf_random_regress <- train(solTrainXtrans, solTrainY,
#                    method = "rf",
#                    trControl= train_control,
#                    metric = model_metric,
#                    tuneGrid = tune_grid
#                    )

## Classification
#Random search
set.seed(41)
mtry_min <- floor(sqrt(ncol(pima_train)-1))
mtry_max <- ncol(pima_train)-1
mtry <- mtry_min
train_control <- trainControl(method='cv', number=5, search='random')
model_metric <- "Accuracy"
tune_grid <- expand.grid(.mtry=mtry)


rf_random_class <- train(as.factor(diabetes) ~., 
                         data = pima_train,
                         method = "rf",
                         trControl= train_control,
                         metric = model_metric
)
rf_random_class$results
###############################

##### Adaboost #####

set.seed(100)
gbm_grid <- expand.grid(n.trees=c(100, 500),
                        shrinkage=c(0.01, 0.001),
                        interaction.depth=c(1,5),
                        n.minobsinnode=10)
gbm_control <- trainControl(method='cv', 
                            number=5,
                            classProbs = TRUE,
                            summaryFunction = twoClassSummary
)
gbm_metric <- "ROC"
ada_class <- train(as.factor(diabetes) ~., 
                   data = pima_train,
                   method = "gbm",
                   distribution="adaboost",
                   verbose=FALSE,
                   trControl= gbm_control,
                   metric = gbm_metric,
                   tuneGrid = gbm_grid
)
ada_class$results
###############################

##### Gradient Boosting #####
set.seed(100)
gbm_class <- train(as.factor(diabetes) ~., 
                   data = pima_train,
                   method = "gbm",
                   distribution="bernoulli",
                   verbose=FALSE,
                   trControl= gbm_control,
                   metric = gbm_metric,
                   tuneGrid = gbm_grid
)
gbm_class$results
