library(rgl)
library(pracma)
x <- c(2.4,2.8,3.2,3.6,4.0,4.2,5.0)
y <- c(1.70,1.85,1.79,1.95,2.10,2.00,2.70)

weight <- seq(-2,4, length.out = 7)
bias <- seq(-6,8,length.out = 7)

m<- meshgrid(slope,intercept)$X
b <- meshgrid(slope,intercept)$Y

cost <- matrix(rep(0,49),nrow=7)

for(i in 1:length(y)){
  cost <- r + (y[i] - (m*x[i] + b))^2 / length(y)
}

# contour(slope,intercept,log(r))
# plot3d(slope,intercept,r)
# surface3d(slope,intercept,r, color = cm.colors(32))
# persp(slope,intercept,r)
persp(weight,bias,cost, col=cm.colors(32), theta=-35, phi=10)
