#### Sigmoid function ####

sigmoid <- function(z){
  res <- 1 / (1 + exp(-z))
  res
}

x <- seq(-5,5,length.out = 1000)
y <- sigmoid(x)
plot(x,y, xlab = "x", ylab = "sigmoid(x)",type="l")
abline(h = 0.5,lty=2)
##########################

#### Simple Linear Regression ####

## Data
data <- read.csv("data/Advertising-Radio.csv",header=TRUE)
head(data)
plot(data$radio,data$sales,xlab= "Radio", ylab = "Sales", col="dodgerblue",pch=20)

## Functions
simple_regress <- function(features,weight,bias){
  return(weight*features + bias)
}

cost_function <- function(features,targets,weight,bias){
  num_items <- length(targets)
  total_error <- 0
  for(i in seq_along(1:num_items)){
    total_error <- total_error + (targets[i] - (weight * features[i] + bias))^2
  }
  return(total_error/num_items)
}

update_weight <- function(features,targets,weight,bias,learning_rate){
  delta_weight <- 0
  delta_bias <- 0
  num_items <- length(targets)
  
  for(i in seq_along(1:num_items)){
    #Calculate gradients
    error <- (targets[i] - (weight * features[i] + bias))
    delta_weight <- delta_weight + -2 * features[i] *  error
    delta_bias <- delta_bias + -2 * error
    
  }
  
  weight <- weight - learning_rate * (delta_weight/num_items)
  bias <- bias - learning_rate * (delta_bias/num_items)
  
  
  
  res <- c(weight,bias)
  res
}

## Model training
train <- function(features,targets,weight,bias,learning_rate,iters){
  cost_history <- numeric(iters)
  coef_history <- list()
  
  for(i in seq_along(1:iters)){
    tmp_coef <- update_weight(features,targets,weight,bias,learning_rate)
    weight <- tmp_coef[1]
    bias <- tmp_coef[2]
    coef_history[[i]] <- c(bias,weight)
    cost <- cost_function(features,targets,weight = weight, bias = bias)
    cost_history[i] <- cost
    
    if(i == 1 | i %% 10 == 0){
      cat("iter: ", i, "weight: ", weight, "bias: ", bias, "cost: ", cost, "\n")
    }
    
  }
  res <- list(Weight = weight, Bias = bias, Cost = cost_history,Coefs = coef_history)
  res
}

fit <- train(features = data$radio,targets = data$sales,weight = 0.03, bias = 0.0014, learning_rate = 0.001,iters = 30)

plot(data$radio,data$sales,xlab= "Radio", ylab = "Sales", col="dodgerblue",pch=20,main="Final Plot With Coefficient History")
for(i in 1:30){
  abline(coef=fit$Coefs[[i]], col = rgb(0.8,0,0,0.3))
}
abline(coef = c(fit$Bias,fit$Weight),col="red")
plot(fit$Cost,type="l",col="blue", xlab = "Iteration",ylab="Cost",main = "Error Rate Per Iteration")
#######################################

#### Multiple Variable Regression ####

## Data
multi_data <- read.csv("data/Advertising.csv", header = TRUE)
head(multi_data)


# Code for panel.cor found at https://www.r-bloggers.com/scatter-plot-matrices-in-r/

panel.cor <- function(x, y, digits = 2, cex.cor, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  # correlation coefficient
  r <- cor(x, y)
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste("r= ", txt, sep = "")
  text(0.5, 0.6, txt)
  
  # p-value calculation
  p <- cor.test(x, y)$p.value
  txt2 <- format(c(p, 0.123456789), digits = digits)[1]
  txt2 <- paste("p= ", txt2, sep = "")
  if(p<0.01) txt2 <- paste("p= ", "<0.01", sep = "")
  text(0.5, 0.4, txt2)
}

pairs(multi_data[,2:5],lower.panel = panel.smooth, upper.panel = panel.cor)

## Functions
multi_cost <- function(features,target,theta){
  sum((features %*% theta - target)^2) / (2*length(target))
}

multi_train <- function(features,target,theta,learn_rate,iters){
  cost_history <- double(iters)
  for(i in seq_along(cost_history)){
    error <- (features %*% theta - target)
    delta <- t(features) %*% error / length(target)
    theta <- theta  - learn_rate * delta
    cost_history[i] <- multi_cost(features,target,theta)
  }
  
  res <- list(Coefs = theta, Costs = cost_history)
  res
}

normalize_data <- function(data){
  cols <- ncol(data)
  for(i in 1:cols){
    tmp_mean <- mean(data[,i])
    tmp_range <- range(data[,i])[2] - range(data[,i])[1]
    res <- (data[,i] - tmp_mean) / tmp_range
    data[,i] <- res
  }
  data
}

## Model Training

multi_features <- multi_data[,2:4]
multi_features <- normalize_data(multi_features)
multi_target <- multi_data[,5]
features_matrix <- cbind(1,as.matrix(multi_features))
theta <- matrix(0,nrow=ncol(features_matrix))
rownames(theta) <- c("Intercept",names(multi_features))
num_iters <- 1000
learn_rate <- 0.0005
multi_fit <- multi_train(features_matrix,multi_target,theta,learn_rate,num_iters)
multi_fit$Coefs
plot(multi_fit$Costs,type="l",col="blue",xlab="Iteration",ylab="Cost",main="Error Rate Per Iteration")
test_fit <- lm(sales ~ TV + radio + newspaper, data = multi_data)
######################################

#### Logistic Regression ####

## Data
log_data <- read.csv("data/data_classification.csv",header=TRUE)
head(log_data)
color_vec <- ifelse(log_data$passed==1,"orange","blue")
plot(log_data$slept,log_data$studied,col=color_vec,xlab="Hours Slept",ylab="Hours Studied")
legend("topright",c("Pass","Fail"),col=c("orange","blue"),pch=c(1,1))

## Functions
log_predict <- function(features, theta){
  z <- features %*% theta
  res <- sigmoid(z)
  res
}

log_cost <- function(features, theta, targets){
  m <- length(targets)
  g <- log_predict(features,theta)
  res <- (1/m) * sum((-targets * log(g)) - ((1-targets) * log(1-g)))
  res
}

boundary <- function(prob){
  res <- ifelse(prob>=.5,1,0)
  res
}

log_accuracy <- function(preds,targets){
  diff <- preds - targets
  res <- 1 - sum(diff)/length(diff)
}

log_train <- function(features,theta,targets,learn_rate, iters){
  cost_history <- double(iters)
  for(i in seq_along(cost_history)){
    preds <- log_predict(features,theta)
    error <- (preds - targets)
    delta <- t(features) %*% error / length(targets)
    theta <- theta  - learn_rate * delta
    cost <- log_cost(features,theta,targets)
    cost_history[i] <- cost
    
    if(i == 1 | i %% 1000 == 0){
      cat("iter: ", i, "cost: ", cost, "\n")
    }
  }
  
  res <- list(Coefs = theta, Costs = cost_history)
  res
}

## Model Training

log_features <- log_data[,1:2]
log_targets <- log_data[,3]
log_design <- cbind(1,as.matrix(log_features))
log_theta <- matrix(0,nrow=ncol(log_design))
rownames(log_theta) <- c("Intercept",names(log_features))
learn_rate <- 0.02
num_iters <- 3000
log_fit <- log_train(log_design,log_theta,log_targets,learn_rate,num_iters)
log_fit$Coefs
plot(log_fit$Costs,type="l",col="blue")
predictions <- log_predict(log_design,log_fit$Coefs)
classifications <- boundary(predictions)
fit_accuracy <- log_accuracy(classifications,log_targets)
fit_accuracy

plot(predictions,col=color_vec)
abline(h=0.5,lty=2)
title("Actual Classification vs Predicted Probability")
legend("topright",c("Pass","Fail"),col=c("orange","blue"),pch=c(1,1),horiz = TRUE)
######################################

#### Linear Discriminant Analysis ####

## Data
lda_data <- read.csv("data/lda-toy.csv",header=TRUE)
head(lda_data)
col_vec <- ifelse(lda_data$y==0,"orange","blue")
plot(1:40,lda_data$x,xlab = "predictor",ylab = "value", col = col_vec)

## LDA Scoring
num_class <- 2
ldazero_index <- which(lda_data$y==0)
prob_zero <- length(lda_data$y[ldazero_index])/length(lda_data$y)
prob_zero
prob_one <- length(lda_data$y[-ldazero_index])/length(lda_data$y)
prob_one
mu_zero <- sum(lda_data$x[ldazero_index])/length(lda_data$x[ldazero_index])
mu_zero
mu_one <- sum(lda_data$x[-ldazero_index])/length(lda_data$x[-ldazero_index])
mu_one
squaredev_zero <- sum((lda_data$x[ldazero_index]-mu_zero)^2)
squaredev_one <- sum((lda_data$x[-ldazero_index]-mu_one)^2)
squaredev_zero
squaredev_one
lda_var <- 1/(length(lda_data$x) - num_class) * sum(squaredev_one,squaredev_zero)
lda_var


## Discriminant Scoring Function
disc_score <- function(x,mu,sigma,prob){
  res <- (x*(mu/sigma)) - (mu^2/(2*sigma)) + log(prob)
  res
}
disc_score(lda_data$x[1],mu_zero,lda_var,prob_zero)
disc_score(lda_data$x[1],mu_one,lda_var,prob_one)

## Final Predictions
score_zero <- disc_score(lda_data$x,mu_zero,lda_var,prob_zero)
score_one <- disc_score(lda_data$x,mu_one,lda_var,prob_one)
preds <- numeric(length(lda_data$x))
for(i in seq_along(preds)){
  if(score_zero[i] > score_one[i]){
    next
  }
  else{
    preds[i] <- 1
  }
}
table(lda_data$y,preds)



