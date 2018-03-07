##### Libraries #####
library(mlbench)
library(caret)
library(caretEnsemble)
library(randomForest)
#####################

##### Data #####
data("Sonar")
?Sonar
sonar_data <- Sonar
rm(Sonar)
str(sonar_data)
my_seed <- 7

# split input and output
x <- sonar_data[,1:60]
y <- sonar_data[,61]
# class distribution
cbind(freq=table(y), percentage=prop.table(table(y))*100)

correlations <- cor(x)
corrplot(correlations, method="circle")

# box and whisker plots for each attribute
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="box", scales=scales)
# density plots for each attribute by class value
featurePlot(x=x, y=y, plot="density", scales=scales)

#Center and Scale features
sonar_data[,1:60] <- scale(sonar_data[,1:60])

control <- trainControl(method="repeatedcv", number=10, repeats=3)
# GLM
set.seed(my_seed)
fit.glm <- train(Class~., data=sonar_data, method="glm", metric="Accuracy", trControl=control)
# LDA
set.seed(my_seed)
fit.lda <- train(Class~., data=sonar_data, method="lda", metric="Accuracy", trControl=control)
# SVM
set.seed(my_seed)
grid <- expand.grid(.sigma=c(0.01,0.05,0.1), .C=c(1))
fit.svm <- train(Class~., data=sonar_data, method="svmRadial", metric="Accuracy", tuneGrid=grid, trControl=control)
# CART
set.seed(my_seed)
grid <- expand.grid(.cp=c(0.01,0.05,0.1))
fit.cart <- train(Class~., data=sonar_data, method="rpart", metric="Accuracy", tuneGrid=grid, trControl=control)
# kNN
set.seed(my_seed)
grid <- expand.grid(.k=c(1,3,5,7))
fit.knn <- train(Class~., data=sonar_data, method="knn", metric="Accuracy", tuneGrid=grid, trControl=control)
# Compare algorithms
results <- resamples(list(SVM=fit.svm, CART=fit.cart, kNN=fit.knn, glm=fit.glm, lda=fit.lda))
summary(results)
dotplot(results,main="Classification Algorithms")

# alternative evaluation method using caretEnsemble
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
models <- caretList(Class~., data=sonar_data, trControl=control, metric='Accuracy', methodList=c('glm', 'lda'),
                    tuneList=list(
                      svmRadial=caretModelSpec(method='svmRadial', tuneGrid=expand.grid(.sigma=c(0.05), .C=c(1))),
                      rpart=caretModelSpec(method='rpart', tuneGrid=expand.grid(.cp=c(0.1))),
                      knn=caretModelSpec(method='knn', tuneGrid=expand.grid(.k=c(1)))
                    )
)
ensemble <- caretEnsemble(models)
summary(ensemble)

## Tuning a Random Forest Model
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "Accuracy"
set.seed(my_seed)
mtry <- sqrt(ncol(x))
tunegrid <- expand.grid(.mtry=mtry)
rf_default <- train(Class~., data=sonar_data, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_default)

# Random Search
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
set.seed(my_seed)
mtry <- sqrt(ncol(x))
rf_random <- train(Class~., data=sonar_data, method="rf", metric=metric, tuneLength=15, trControl=control)
print(rf_random)
plot(rf_random, main="Random Search")

# Grid Search
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
set.seed(my_seed)
tunegrid <- expand.grid(.mtry=c(1:15))
rf_gridsearch <- train(Class~., data=sonar_data, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch,main="Grid Search")

# Algorithm Tune (tuneRF)
set.seed(my_seed)
bestmtry <- tuneRF(x, y, stepFactor=1.5, improve=1e-5, ntree=500)
print(bestmtry)

# Manual Search
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
tunegrid <- expand.grid(.mtry=c(sqrt(ncol(x))))
modellist <- list()
for (ntree in c(1000, 1500, 2000, 2500)) {
  set.seed(my_seed)
  fit <- train(Class~., data=sonar_data, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control, ntree=ntree)
  key <- toString(ntree)
  modellist[[key]] <- fit
}
# compare results
results <- resamples(modellist)
summary(results)
dotplot(results,main="RF NTrees")

# Custom Caret Algorithm
# customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
# customRF$parameters <- data.frame(parameter = c("mtry", "ntree"), class = rep("numeric", 2), label = c("mtry", "ntree"))
# customRF$grid <- function(x, y, len = NULL, search = "grid") {}
# customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
#   randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
# }
# customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
#   predict(modelFit, newdata)
# customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
#   predict(modelFit, newdata, type = "prob")
# customRF$sort <- function(x) x[order(x[,1]),]
# customRF$levels <- function(x) x$classes
# # train model
# control <- trainControl(method="repeatedcv", number=10, repeats=3)
# tunegrid <- expand.grid(.mtry=c(1:15), .ntree=c(1000, 1500, 2000, 2500))
# set.seed(my_seed)
# custom <- train(Class~., data=sonar_data, method=customRF, metric=metric, tuneGrid=tunegrid, trControl=control)
# print(custom)
# max(custom$results$Accuracy)
# plot(custom)

