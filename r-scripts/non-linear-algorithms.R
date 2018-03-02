library(AppliedPredictiveModeling)
library(rpart)
library(caret)
library(partykit)
library(mlbench)
library(kernlab)

##### Regression Tree #####

data(solubility)
str(solTrainXtrans[,c(1:10,209:228)])
str(solTrainY)

set.seed(100)
rpartTune <- train(solTrainXtrans, solTrainY,
                   method = "rpart2",
                   tuneLength = 10,
                   trControl= trainControl(method = "cv")
)
rpartTune$results

#Build the initial model

training_data <- data.frame(cbind(solTrainXtrans,solTrainY))
training_model <- rpart(solTrainY ~., data = training_data,
                        control = rpart.control(maxdepth = 10))

#Examine the tree complexity
plotcp(training_model)

training_model$cptable

#Add min(xerror+xstd) and find the smallest tree w/xerror < min(xerror+xstd)
which(training_model$cptable[,4] < min(training_model$cptable[,4]+training_model$cptable[,5]))

#Prune the tree
training_model <- rpart(solTrainY ~., data = training_data,
                        cp = .014)
model_tree <- as.party
#################################################

##### Classification Tree #####

pima_train <- read.csv("data/pima-train.csv",header=TRUE)
pima_train <- pima_train[complete.cases(pima_train),]
str(pima_train)

#Gini Index
set.seed(33)
train_ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
gini_tune <- train(as.factor(diabetes) ~., data = pima_train, method = "rpart",
                   trControl=train_ctrl,
                   tuneLength = 10, 
                   parms=list(split='gini'))
gini_tune

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

pima_info_model <- rpart(as.factor(diabetes)~., data = pima_train,
                         cp = .004)
pima_info_tree <- as.party(pima_info_model)
#################################################

##### Naive Bayes #####
set.seed(33)
library(klaR)
nb_tune <- train(as.factor(diabetes) ~ ., 
                 data=pima_train,
                 method = "nb",
                 trControl = trainControl(method="none"),
                 tuneGrid = data.frame(fL=0, usekernel=FALSE, adjust=1))
nb_preds <- predict(nb_tune,pima_train,type = "raw")
confusionMatrix(nb_preds,as.factor(pima_train$diabetes))
#################################################


##### K Nearest Neighbors #####

## Regression Data

knn_data <- solTrainXtrans[,-nearZeroVar(solTrainXtrans)]

## KNN Regression Model

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

## KNN Classification Model

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
pima_knn$finalModel
#################################################


##### Support Vector Machines #####

## Regression Model
set.seed(33)
solubility_svm <- train(solTrainXtrans, solTrainY,
                        method = "svmRadial",
                        preProc = c("center", "scale"),
                        tuneLength = 14,
                        trControl = trainControl(method = "cv"))
solubility_svm
solubility_svm$finalModel

## Classification Model
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
pima_svm$finalModel
