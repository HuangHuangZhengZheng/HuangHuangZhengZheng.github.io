# DATA100-L11: Ordinary Least Squares

## linear in theta
linear combination of parameters $\theta$
![alt text](image.png)
## define multiple linear regression
![alt text](image-1.png)

# OLS problem formulation
ordinary least squares (OLS) 

用线性代数重写之
$$
\mathbb{\hat{Y}} = \mathbb{X}\theta
$$

## multiple linear regression model
![alt text](image-2.png)
## MSE
![alt text](image-3.png)
$$
R(\theta) = \frac{1}{n}||\mathbb{Y}-\hat{\mathbb{Y}}||_2^2
$$
# geometric derivation
## lin alg review: orthogonality, span
$$
span(\mathbb{A})是一个由列向量组成的space
$$
![alt text](image-4.png)
正交
![alt text](image-5.png)


## least squares estimate proof
![alt text](image-6.png)
# performance: residuals, multiple R-squared
lec11.ipynb

![alt text](image-7.png)
$$
R^2∈[0,1]
$$
越大拟合效果越好
# OLS properties
## residuals
![alt text](image-8.png)
## the bias/intercept term
![alt text](image-10.png)
## existence of a unique solution
![alt text](image-9.png)
