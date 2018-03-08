##### Libraries #####
library(mlbench)
library(caret)
library(corrplot)
library(Cubist)
#####################

##### Data #####

data("BostonHousing")
?BostonHousing #To read a general description of the data
str(BostonHousing)
my_seed <- 7

## Split the data
set.seed(my_seed)
training_index <- createDataPartition(BostonHousing$medv, p=0.8, list = FALSE)
boston_training <- BostonHousing[training_index,]
boston_test <- BostonHousing[-training_index,]

## Explore the dataset
str(boston_training)

#Scatterplot of continuous variables
features <- colnames(BostonHousing)[1:13]
theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
featurePlot(x = boston_training[, features[-4]], 
            y = boston_training$medv, 
            plot = "scatter",
            type = c('p','smooth'),
            layout = c(3, 4))

#Boxplot of Charles River Variable
featurePlot(x = boston_training$medv, 
            y = boston_training$chas, 
            plot = "box", 
            ## Pass in options to bwplot() 
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),  
            layout = c(1,1 ), 
            auto.key = list(columns = 1))

#Density plot of Charles River variable
transparentTheme(trans = .9)
featurePlot(x = boston_training$medv, 
            y = boston_training$chas,
            plot = "density", 
            ## Pass in options to xyplot() to 
            ## make it prettier
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5, 
            pch = "|", 
            layout = c(1, 1), 
            auto.key = list(columns = 2))

correlations <- cor(boston_training[,c(1:3,5:13)])
corrplot(correlations, method="circle")

#Baseline Moeel Evaluation
# Run algorithms using 10-fold cross validation
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
# lm
set.seed(my_seed)
fit.lm <- train(medv~., data=boston_training, method="lm", metric=metric, preProc=c("center", "scale"), trControl=control)
# GLM
set.seed(my_seed)
fit.glm <- train(medv~., data=boston_training, method="glm", metric=metric, preProc=c("center", "scale"), trControl=control)
# GLMNET
set.seed(my_seed)
fit.glmnet <- train(medv~., data=boston_training, method="glmnet", metric=metric, preProc=c("center", "scale"), trControl=control)
# SVM
set.seed(my_seed)
fit.svm <- train(medv~., data=boston_training, method="svmRadial", metric=metric, preProc=c("center", "scale"), trControl=control)
# CART
set.seed(my_seed)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
fit.cart <- train(medv~., data=boston_training, method="rpart", metric=metric, tuneGrid=grid, preProc=c("center", "scale"), trControl=control)
# kNN
set.seed(my_seed)
fit.knn <- train(medv~., data=boston_training, method="knn", metric=metric, preProc=c("center", "scale"), trControl=control)
# Compare algorithms
results <- resamples(list(LM=fit.lm, GLM=fit.glm, GLMNET=fit.glmnet, SVM=fit.svm, CART=fit.cart, KNN=fit.knn))
summary(results)
dotplot(results,main="Baseline")

# Evaluate Algorithms: Feature Selection

# remove correlated attributes
# find attributes that are highly corrected
set.seed(my_seed)
cutoff <- 0.70
correlations <- cor(boston_training[,c(1:3,5:13)])
highlyCorrelated <- findCorrelation(correlations, cutoff=cutoff,names = TRUE)
highlyCorrelated

# create a new dataset without highly corrected features
remove_cols <- which(colnames(boston_training) %in% highlyCorrelated)
new_boston_features <- boston_training[,-remove_cols]
dim(new_boston_features)


# Run algorithms using 10-fold cross validation
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
# lm
set.seed(my_seed)
fit.lm <- train(medv~., data=new_boston_features, method="lm", metric=metric, preProc=c("center", "scale"), trControl=control)
# GLM
set.seed(my_seed)
fit.glm <- train(medv~., data=new_boston_features, method="glm", metric=metric, preProc=c("center", "scale"), trControl=control)
# GLMNET
set.seed(my_seed)
fit.glmnet <- train(medv~., data=new_boston_features, method="glmnet", metric=metric, preProc=c("center", "scale"), trControl=control)
# SVM
set.seed(my_seed)
fit.svm <- train(medv~., data=new_boston_features, method="svmRadial", metric=metric, preProc=c("center", "scale"), trControl=control)
# CART
set.seed(my_seed)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
fit.cart <- train(medv~., data=new_boston_features, method="rpart", metric=metric, tuneGrid=grid, preProc=c("center", "scale"), trControl=control)
# kNN
set.seed(my_seed)
fit.knn <- train(medv~., data=new_boston_features, method="knn", metric=metric, preProc=c("center", "scale"), trControl=control)
# Compare algorithms
feature_results <- resamples(list(LM=fit.lm, GLM=fit.glm, GLMNET=fit.glmnet, SVM=fit.svm, CART=fit.cart, KNN=fit.knn))
summary(feature_results)
dotplot(feature_results,main="Low Correlation Features") 


# Evaluate Algorithnms: Box-Cox Transform

# See https://socialsciences.mcmaster.ca/jfox/Courses/Brazil-2009/slides-handout.pdf for explanation of Box-Cox

# Run algorithms using 10-fold cross validation
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
# lm
set.seed(my_seed)
fit.lm <- train(medv~., data=boston_training, method="lm", metric=metric, preProc=c("center", "scale", "BoxCox"), trControl=control)
# GLM
set.seed(my_seed)
fit.glm <- train(medv~., data=boston_training, method="glm", metric=metric, preProc=c("center", "scale", "BoxCox"), trControl=control)
# GLMNET
set.seed(my_seed)
fit.glmnet <- train(medv~., data=boston_training, method="glmnet", metric=metric, preProc=c("center", "scale", "BoxCox"), trControl=control)
# SVM
set.seed(my_seed)
fit.svm <- train(medv~., data=boston_training, method="svmRadial", metric=metric, preProc=c("center", "scale", "BoxCox"), trControl=control)
# CART
set.seed(my_seed)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
fit.cart <- train(medv~., data=boston_training, method="rpart", metric=metric, tuneGrid=grid, preProc=c("center", "scale", "BoxCox"), trControl=control)
# kNN
set.seed(my_seed)
fit.knn <- train(medv~., data=boston_training, method="knn", metric=metric, preProc=c("center", "scale", "BoxCox"), trControl=control)
# Compare algorithms
transform_results <- resamples(list(LM=fit.lm, GLM=fit.glm, GLMNET=fit.glmnet, SVM=fit.svm, CART=fit.cart, KNN=fit.knn))
summary(transform_results)
dotplot(transform_results,main = "Box Cox Transforms")

# Improve Results With Tuning

# look at parameters
print(fit.svm)

# tune SVM sigma and C parametres
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
set.seed(my_seed)
grid <- expand.grid(.sigma=c(0.025, 0.05, 0.1, 0.15), .C=seq(1, 10, by=1))

#This model will take a couple of minutes
fit.svm <- train(medv~., data=boston_training, method="svmRadial", metric=metric, tuneGrid=grid, preProc=c("BoxCox"), trControl=control)
print(fit.svm)
plot(fit.svm)
fit.svm$finalModel

# Ensemble Methods

# try ensembles
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
# Random Forest
set.seed(my_seed)
fit.rf <- train(medv~., data=boston_training, method="rf", metric=metric, preProc=c("BoxCox"), trControl=control)
# Stochastic Gradient Boosting
set.seed(my_seed)
fit.gbm <- train(medv~., data=boston_training, method="gbm", metric=metric, preProc=c("BoxCox"), trControl=control, verbose=FALSE)
# Cubist
set.seed(my_seed)
fit.cubist <- train(medv~., data=boston_training, method="cubist", metric=metric, preProc=c("BoxCox"), trControl=control)
# Compare algorithms
ensemble_results <- resamples(list(RF=fit.rf, GBM=fit.gbm, CUBIST=fit.cubist))
summary(ensemble_results)
dotplot(ensemble_results,main="Ensemble Models")

#Cubist seems to be the best model, so we will choose it's tuning parameters
fit.cubist$bestTune #20 committees and 5 neighbors


# Finalize Model


# prepare the data transform using training data
set.seed(my_seed)
x <- boston_training[,1:13]
y <- boston_training[,14]
preprocessParams <- preProcess(x, method=c("BoxCox"))
trans_x <- predict(preprocessParams, x)
# train the final model
finalModel <- cubist(x=trans_x, y=y, committees=20) #derived from fit.cubist$bestTune
summary(finalModel)

# transform the validation dataset
set.seed(my_seed)
val_x <- boston_test[,1:13]
trans_val_x <- predict(preprocessParams, val_x)
val_y <- boston_test[,14]
# use final model to make predictions on the validation dataset
predictions <- predict(finalModel, newdata=trans_val_x, neighbors=3) #adjusted from fit.cubist$bestTune
# calculate RMSE
rmse <- RMSE(predictions, val_y)
r2 <- R2(predictions, val_y)
print(rmse)
