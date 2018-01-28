x <- seq(-1,1,length.out = 1000)
y <- x^2
tangent <- 0.5
tangent_slope <- 2 * tangent
y0 <- tangent ^2
tangent_line <- tangent_slope * (x-tangent) + y0
plot(x,y,type="l",col="blue")
points(tangent,y0,col="red",pch=20)
lines(x,tangent_line,lty=3)
