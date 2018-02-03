# Linear Algorithms {#lin-algs}


## Gradient Descent

One of the most common concepts for all machine learning algorithms is optimization. Of the many optimization methods, the most widely used optimization method in machine learning is gradient descent. This extensive use of gradient descent is because gradient descent is straightforward to learn and compatible with any machine learning algorithm. For this section, we will use gradient descent with linear and logistic regression. 




### Linear Regression Model {-}

Before discussing the gradient descent algorithm, let's review the linear regression. Recall the general linear regression equation is:

$$y = h(x) =  \sum_{i=1}^{N}w_ix_i + b + \epsilon$$

where $y$ is the target variable, $x$ is the feature variables, $w_i$ are the weights of the $i$th feature variable, $b$ is the bias, and $\epsilon$ is the irreducible error. We use machine learning algorithms to estimate the bias and the weights of the feature variables. 

### Cost Function {-}

#### Simple Linear Regression {-}

In order to optimize the weight and the bias variable, we need to optimize the cost (loss) function. For linear regression this function, $J(w,b)$ is the Mean Squared Error (MSE) and we can calculate it by:

$$MSE = J(w,b) = \frac{1}{m}\sum_{i = 1}^{m}\left(y_i - \left(wx_i+b\right)\right)^2$$

Since we are adjusting the cost function by the weight and the bias parameters, we must take the partial derivative with respect to each of these to calculate the gradient.

$$
J'(w,b) = 
\begin{bmatrix}
\frac{\partial J}{\partial w}\\
\frac{\partial J}{\partial b}
\end{bmatrix}
=
\begin{bmatrix}
\frac{1}{m}\sum-2*x_i(y_i - (wx_i + b)) = \delta_w\\
\frac{1}{m}\sum-2*(y_i - (wx_i + b)) = \delta_b
\end{bmatrix}
$$

#### Multiple Linear Regression {-}

For multiple linear regression, we introduce a parameter matrix $\theta$ that contains the bias and weight parameters where:

$$
\begin{align}
h_\theta(x) &= \theta_0 + \theta_1x_1 + \dots + \theta_nx_n\\
h(x) &= \sum_{i=0}^{n} \theta_ix_i = \theta^Tx
\end{align}
$$

The cost function is:

$$
J(\theta) = \frac{1}{2m}\sum_{i=1}^{m} \left(h_\theta(x^{(i)})-y^{(i)}\right)^2
$$

In matrix form, the cost function becomes

$$
J(\theta) = \frac{1}{2m}(X\theta - y)^T(X\theta-y)
$$
where $X$ is a $m \times n$ design matrix, $\theta$ is a $n \times 1$ parameter matrix and $y$ is a $m \times 1$ vector of observed targets.

Taking the derivative with respect to $\theta$ yields:

$$
J'(\theta) = \frac{1}{m}X^T(X\theta - y)
$$

See the video below for an example derivation of the derivative of the cost matrix:

<iframe width="560" height="315" src="https://www.youtube.com/embed/tGkMr57ZvAk" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

### Gradient Descent Algorithm {-}

Finally, to solve for the optimal weight and bias, we will add a learning parameter, $\alpha$, to adjust the steps of the gradient. 

#### Simple Linear Regression {-}

The algorithm we will use is:

$$
\text{Repat until convergence } \{\\
w := w -  \alpha\delta_w\\
b := b - \alpha\delta_b\\
\}
$$

#### Multiple Linear Regression {-}

For multiple linear regression the algorithm changes to:

$$
\text{Repat until convergence } \{\\
\theta_j := \theta_j - \alpha \frac{1}{m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)}-y^{(i)}\right)x_{j}^{(i)}\\
\}
$$

In this algorithm we are simultaneously updating the weights, $\theta_j$, for all $j\in{(0,\dots,n)}$. Recall that $\theta_0$ is the bias term and $x_0^{1}=1$.

In matrix form, our algorithm will look like this:

$$
\text{Repat until convergence } \{\\
\delta = \frac{1}{m}X^T(X\theta - y)\\
\theta:=\theta-\alpha\delta\\
\}
$$

### Gradient Descent Intuition {-}


Figure \@ref(fig:grad-descent-intuition) demonstrates the basic intuition behind the gradient algorithm. Fundamentally, if we pick a point along the graph of the cost function, and the gradient is negative, the algorithm will update by moving the more to the right. Conversely, if the gradient is positive, the algorithm will move the cost value more to the left. 

<div class="figure" style="text-align: center">
<img src="img/gradient-descent-example.png" alt="A simple example of gradient descent." width="90%" />
<p class="caption">(\#fig:grad-descent-intuition)A simple example of gradient descent.</p>
</div>


Figure \@ref(fig:surface-plot) is a multi-dimensional view of the cost function, and the underlying concept is still the same.

<div class="figure" style="text-align: center">
<img src="img/surface-plot.png" alt="A surface plot of a quadratic cost function." width="90%" />
<p class="caption">(\#fig:surface-plot)A surface plot of a quadratic cost function.</p>
</div>

While the ideal cost function to minimize would be a convex function, this is not always practical and there are ways to deal with that as discussed in the following video.

<iframe width="560" height="315" src="https://www.youtube.com/embed/8zdo6cnCW2w" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

## Practical Exercises

This practical exercieses are based on code provided by @fortuner2017mlcheat and @brownlee2017mlmastery.

### Simple Linear Regression

#### Data {-}
Suppose we have the following dataset in which we have a unique Company ID, radio advertising expenses in dollars, and annual sales as a result of those expenses in dollars. 


```r
data <- read.csv("data/Advertising-Radio.csv",header=TRUE)
head(data)
```

```
##   company radio sales
## 1       1  37.8  22.1
## 2       2  39.3  10.4
## 3       3  45.9   9.3
## 4       4  41.3  18.5
## 5       5  10.8  12.9
## 6       6  48.9   7.2
```

A view of the data shows that there appears to be a positive correlation between radio advertising spending and sales. 


```r
plot(data$radio,data$sales,xlab= "Radio", ylab = "Sales", col="dodgerblue",pch=20)
```

<img src="02-Linear-Algorithms_files/figure-html/sales-plot-1.png" width="672" />

#### Making Predictions {-}
For this model, we want to predict sales based on the amount spent for radio advertising. Thus our formula will be

$$\text{Sales} = \text{Weight} \times \text{Radio} + \text{Bias}$$

The gradient descent algorithm will attempt to learn the optimal values for the Weight and Bias.

#### Simple Regression Function {-}

```r
simple_regress <- function(features,weight,bias){
  return(weight*features + bias)
}
```

#### Cost function Code {-}


```r
cost_function <- function(features,targets,weight,bias){
  num_items <- length(targets)
  total_error <- 0
  for(i in seq_along(1:num_items)){
    total_error <- total_error + (targets[i] - (weight * features[i] + bias))^2
  }
  return(total_error/num_items)
}
```



#### Gradient Descent Code {-}




```r
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
```

#### Training the model {-}

We are now ready to train the final model. To do this we will iterate over a set number of trials and update the weight and bias parameters at each iteration. We will also track the cost history.


```r
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
```

```
## iter:  1 weight:  0.7255664 bias:  0.02804636 cost:  86.42445 
## iter:  10 weight:  0.484637 bias:  0.06879067 cost:  42.72917 
## iter:  20 weight:  0.4837035 bias:  0.1219333 cost:  42.44643 
## iter:  30 weight:  0.4820883 bias:  0.1747496 cost:  42.1673
```


The plot below shows how the ``train()`` funtion iterated through the coefficient history.


```r
plot(data$radio,data$sales,xlab= "Radio", ylab = "Sales", col="dodgerblue",pch=20,main="Final Plot With Coefficient History")
for(i in 1:30){
  abline(coef=fit$Coefs[[i]], col = rgb(0.8,0,0,0.3))
}
abline(coef = c(fit$Bias,fit$Weight),col="red")
```

<img src="02-Linear-Algorithms_files/figure-html/final-plot-1.png" width="672" />

This is a plot of the cost history.

```r
plot(fit$Cost,type="l",col="blue", xlab = "Iteration",ylab="Cost",main = "Error Rate Per Iteration")
```

<img src="02-Linear-Algorithms_files/figure-html/cost-plot-1.png" width="672" />

#### Exercises {-}
1. Run the command `` res <- lm(data$sales ~ data$radio)`` and note the values for the weight and bias.
2. Plot the fitted line from ``res`` with the data and comapre that line to the trained model.
3. Adjust the fit object to obtain an estimate close to the noted parameters.

### Multiple Linear Regression

For this exercise, we will predict total sales based on TV, Radio, and Newspaper advertising costs.

#### Data {-}


```r
multi_data <- read.csv("data/Advertising.csv", header = TRUE)
head(multi_data)
```

```
##   company    TV radio newspaper sales
## 1       1 230.1  37.8      69.2  22.1
## 2       2  44.5  39.3      45.1  10.4
## 3       3  17.2  45.9      69.3   9.3
## 4       4 151.5  41.3      58.5  18.5
## 5       5 180.8  10.8      58.4  12.9
## 6       6   8.7  48.9      75.0   7.2
```

Since we are now dealing with multiple variables, we will need to view a pairs plot


```r
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
```

<img src="02-Linear-Algorithms_files/figure-html/pairs-1.png" width="672" />

#### Cost Function Code


```r
multi_cost <- function(features,target,theta){
  sum((features %*% theta - target)^2) / (2*length(target))
}
```

#### Training Function Code {-}


```r
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
```

#### Results {-}

To make computing the gradient easier, we will normalize the data to fall within the range of -1 to 1. 


```r
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
```



```r
multi_features <- multi_data[,2:4]
multi_features <- normalize_data(multi_features)
#names(multi_features) <- NULL
multi_target <- multi_data[,5]
#names(multi_target) <- NULL
features_matrix <- cbind(1,as.matrix(multi_features))
theta <- matrix(0,nrow=ncol(features_matrix))
rownames(theta) <- c("Intercept",names(multi_features))
num_iters <- 1000
learn_rate <- 0.0005
multi_fit <- multi_train(features_matrix,multi_target,theta,learn_rate,num_iters)
multi_fit$Coefs
```

```
##                [,1]
## Intercept 5.5184872
## TV        0.5767349
## radio     0.4366550
## newspaper 0.1098191
```

```r
plot(multi_fit$Costs,type="l",col="blue",xlab="Iteration",ylab="Cost",main="Error Rate Per Iteration")
```

<img src="02-Linear-Algorithms_files/figure-html/multi-reults-1.png" width="672" />

```r
test_fit <- lm(sales ~ TV + radio + newspaper, data = multi_data)
```


#### Exercises {-}
1. Tune the iterations and the learning rate and attempt to reduce the model cost.
2. Run the command ``test_fit <- lm(sales ~ TV + radio + sales, data = multi_data)``
3. Compute the cost of ``test_fit`` (Hint: use ``names(test_fit)`` to find out how to extract the coefficients of the model ) 

## Linear Discriminant Analysis







