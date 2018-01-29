# Linear Algorithms {#lin-algs}


## Gradient Descent

One of the most common concepts for all machine learning algorithms is optimization. Of the many optimization methods, the most widely used optimization method is gradient descent. This is because, gradient descent is simple to learn and can be used in tandem with any machine learning algorithm. For this section, we will use gradient descent with linear and logistic regression. 

### Gradient Descent Intuition {-}

Before discussing the gradient descent algorithm, let's review the linear regression. Recall the general linear regression equation is:

$$y = h(x) =  w_0 + \sum_{i=1}^{N}w_ix_i + \epsilon$$

where $y$ is the target variable, $x$ is the feature variables, $w_0$ is the bias, $w_i$ are the weights of the $i$th feature variable, and $\epsilon$ is the irreducible error. We use machine learning algorithms to estimate the bias and the weights of the feature variables. This is done by creating a cost (loss) function, $J(w)$.

Figure \@ref(fig:grad-descent-intuition) demonstrates the basic intuition behind the gradient algorithm.

<div class="figure" style="text-align: center">
<img src="img/gradient-descent-example.png" alt="A simple example of gradient descent." width="90%" />
<p class="caption">(\#fig:grad-descent-intuition)A simple example of gradient descent.</p>
</div>


<iframe width="560" height="315" src="https://www.youtube.com/embed/8zdo6cnCW2w" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

<div class="figure" style="text-align: center">
<img src="img/surface-plot.png" alt="A surface plot of a quadratic cost function." width="90%" />
<p class="caption">(\#fig:surface-plot)A surface plot of a quadratic cost function.</p>
</div>


This practical exerciese is based on code provided by @fortuner2017mlcheat and @brownlee2017mlmastery.
