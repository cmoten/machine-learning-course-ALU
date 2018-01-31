# Linear Algorithms {#lin-algs}


## Gradient Descent

One of the most common concepts for all machine learning algorithms is optimization. Of the many optimization methods, the most widely used optimization method is gradient descent. This is because, gradient descent is simple to learn and can be used in tandem with any machine learning algorithm. For this section, we will use gradient descent with linear and logistic regression. 




### Linear Regression Model {-}

Before discussing the gradient descent algorithm, let's review the linear regression. Recall the general linear regression equation is:

$$y = h(x) =  \sum_{i=1}^{N}w_ix_i + b + \epsilon$$

where $y$ is the target variable, $x$ is the feature variables, $w_i$ are the weights of the $i$th feature variable, $b$ is the bias, and $\epsilon$ is the irreducible error. We use machine learning algorithms to estimate the bias and the weights of the feature variables. 

### Cost Function {-}

In order to optimize the weight and the bias variable, we need to optimize the cost (loss) function. For linear regression this function, $J(w,b)$ is the Mean Squared Error (MSE) and we can calculate it by:

$$MSE = J(w,b) = \frac{1}{N}\sum_{i = 1}^{n}\left(y_i - \left(wx_i+b\right)\right)^2$$

Since we are adjusting the cost function by the weight and the bias parameters, we must take the parital derivative with respect to each of these to calculate the gradient.

$$
J'(w,b) = 
\begin{bmatrix}
\frac{\partial J}{\partial w}\\
\frac{\partial J}{\partial b}
\end{bmatrix}
=
\begin{bmatrix}
\frac{1}{N}\sum-2*x_i(y_i - (wx_i + b)) = \delta_w\\
\frac{1}{N}\sum-2*(y_i - (wx_i + b)) = \delta_b
\end{bmatrix}
$$

### Gradient Descent Algorithm {-}

Finally, to solve for the optimal weight and bias, we will add a learning parameter, $\alpha$, to adjust the steps of the gradient. Thus, the algorithm we will use is:

$$
\text{Repat until convergence} \{\\
w := w -  \alpha\delta_w\\
b := b - \alpha\delta_b\\
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
  
  for(i in seq_along(1:iters)){
    tmp_coef <- update_weight(features,targets,weight,bias,learning_rate)
    weight <- tmp_coef[1]
    bias <- tmp_coef[2]
    cost <- cost_function(features,targets,weight = weight, bias = bias)
    cost_history[i] <- cost
    
    if(i == 1 | i %% 10 == 0){
       cat("iter: ", i, "weight: ", weight, "bias: ", bias, "cost: ", cost, "\n")
    }
   
  }
  res <- list(Weight = weight, Bias = bias, Cost = cost_history)
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

```r
plot(fit$Cost,type="l",col="blue")
```

<img src="02-Linear-Algorithms_files/figure-html/train-simple-model-1.png" width="672" />

#### Exercise 1 {-}
1. Run the command ``lm(data$sales ~ data$radio)`` and note the values for the weight and bias.
2. Adjust the fit object to obtain an estimate close to the noted parameters.





