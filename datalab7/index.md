# DATA100-lab7: Gradient Descent and Feature Engineering


```python
# Initialize Otter
import otter
grader = otter.Notebook("lab07.ipynb")
```

# Lab 7: Gradient Descent and Feature Engineering

In this lab, we will work through the process of:
1. Defining loss functions
1. Feature engineering
1. Minimizing loss functions using numeric methods and analytical methods 
1. Understanding what happens if we use the analytical solution for OLS on a matrix with redundant features
1. Computing a gradient for a nonlinear model
1. Using gradient descent to optimize the nonline model

This lab will continue using the toy `tips` calculation dataset used in Labs 5 and 6.

<br/><br/>
<hr style="border: 5px solid #003262;" />
<hr style="border: 1px solid #fdb515;" />

# Loading the Tips Dataset

To begin, let's load the tips dataset from the `seaborn` library.  This dataset contains records of tips, total bill, and information about the person who paid the bill. As earlier, we'll be trying to predict tips from the other data.


```python
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
np.random.seed(42)
plt.style.use('fivethirtyeight')
sns.set()
sns.set_context("talk")
%matplotlib inline
```


```python
data = sns.load_dataset("tips")

print("Number of Records:", len(data))
data.head()
```

    Number of Records: 244
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_bill</th>
      <th>tip</th>
      <th>sex</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1.01</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1.66</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>3.50</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.68</td>
      <td>3.31</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.59</td>
      <td>3.61</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



---

## Intro to Feature Engineering

So far, we've only considered models of the form $\hat{y} = f_{\theta}(x) = \sum_{j=0}^d x_j\theta_j$, where $\hat{y}$ is quantitative continuous. 

We call this a linear model because it is a linear combination of the features (the $x_j$). However, our features don't need to be numbers: we could have categorical values such as names. Additionally, the true relationship doesn't have to be linear, as we could have a relationship that is quadratic, such as the relationship between the height of a projectile and time.

In these cases, we often apply **feature functions**, functions that take in some value and output another value. This might look like converting a string into a number, combining multiple numeric values, or creating a boolean value from some filter.

Then, if we call $\phi$ ("phi") our "phi"-ture function, our model takes the form $\hat{y} = f_{\theta}(x) = \sum_{j=0}^d \phi(x)_j\theta_j$.

### Example feature functions 编码一直是一个先验工程问题？ vs AutoEncoders？

1. One-hot encoding
    - converts a single categorical feature into many binary features, each of which represents one of the possible values in the original column
    - each of the binary feature columns produced contains a 1 for rows that had that column's label in the original column, and 0 elsewhere
1. Polynomial features
    - create polynomial combinations of features

<br/>
<hr style="border: 1px solid #fdb515;" />

## Question 1: Defining the Model and Feature Engineering

In Lab 6 we used the constant model. Now let's make a more complicated model that utilizes other features in our dataset. You can imagine that we might want to use the features with an equation that looks as shown below:

$$ \text{Tip} = \theta_1 \cdot \text{total}\_\text{bill} + \theta_2 \cdot \text{sex} + \theta_3 \cdot \text{smoker} + \theta_4 \cdot \text{day} + \theta_5 \cdot \text{time} + \theta_6 \cdot \text{size} $$

Unfortunately, that's not possible because some of these features like "day" are not numbers, so it doesn't make sense to multiply by a numerical parameter.

Let's start by converting some of these non-numerical values into numerical values. Before we do this, let's separate out the tips and the features into two separate variables.


```python
tips = data['tip']
X = data.drop(columns='tip')
```

---
### Question 1a: Feature Engineering

First, let's convert our features to numerical values. A straightforward approach is to map some of these non-numerical features into numerical ones. 

For example, we can treat the day as a value from 1-7. However, one of the disadvantages in directly translating to a numeric value is that we unintentionally assign certain features disproportionate weight. Consider assigning Sunday to the numeric value of 7, and Monday to the numeric value of 1. In our linear model, Sunday will have 7 times the influence of Monday, which can lower the accuracy of our model.

Instead, let's use one-hot encoding to better represent these features! 

As you will learn in lecture, one-hot encoding is a way that we can produce a binary vector to indicate non-numeric features. 

In the `tips` dataset for example, we encode Sunday as the vector `[0 0 0 1]` because our dataset only contains bills from Thursday through Sunday. This assigns a more even weight across each category in non-numeric features. Complete the code below to one-hot encode our dataset. This dataframe holds our "featurized" data, which is also often denoted by $\phi$.

**Hint:** You may find the [pd.get_dummies method](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html) or the [DictVectorizer class](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html) useful when doing your one-hot encoding.

<!--
BEGIN QUESTION
name: q1a
points: 2
-->


```python
def one_hot_encode(data):
    """
    Return the one-hot encoded dataframe of our input data.
    
    Parameters
    -----------
    data: a dataframe that may include non-numerical features
    
    Returns
    -----------
    A one-hot encoded dataframe that only contains numeric features
    
    """
    return pd.get_dummies(data, dtype=float)
    
    
one_hot_X = one_hot_encode(X)
one_hot_X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_bill</th>
      <th>size</th>
      <th>sex_Male</th>
      <th>sex_Female</th>
      <th>smoker_Yes</th>
      <th>smoker_No</th>
      <th>day_Thur</th>
      <th>day_Fri</th>
      <th>day_Sat</th>
      <th>day_Sun</th>
      <th>time_Lunch</th>
      <th>time_Dinner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>2</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>3</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>3</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.68</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.59</td>
      <td>4</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
grader.check("q1a")
```

---
### Question 1b: Defining the Model

Now that all of our data is numeric, we can begin to define our model function. Notice that after one-hot encoding our data, we now have 12 features instead of 6. Therefore, our linear model now looks like:

$$ \text{Tip} = \theta_1 \cdot \text{size} + \theta_2 \cdot \text{total}\_\text{bill} + \theta_3 \cdot \text{day}\_\text{Thur} + \theta_4 \cdot \text{day}\_\text{Fri} + ... + \theta_{11} \cdot \text{time}\_\text{Lunch} + \theta_{12} \cdot \text{time}\_\text{Dinner} $$

We can represent the linear combination above as a matrix-vector product. Implement the `linear_model` function to evaluate this product.

Below, we create a `MyLinearModel` class with two methods, `predict` and `fit`. When fitted, this model fails to do anything useful, setting all of its 12 parameters to zero.


```python
class MyLinearModel():    
    def predict(self, X):
        return X @ self._thetas
    
    def fit(self, X, y):
        number_of_features = X.shape[1]
        self._thetas = np.zeros(shape = (number_of_features, 1))
```


```python
model = MyLinearModel()
model.fit(one_hot_X, tips)
model._thetas
```




    array([[0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.]])



<br/>
<hr style="border: 1px solid #fdb515;" />

## Question 2: Fitting a Linear Model using scipy.optimize.minimize Methods

Recall in Lab 5 and in lecture 12 we defined multiple loss functions and found the optimal theta using the `scipy.optimize.minimize` function. Adapt the code below to implement the fit method of the linear model.

Note that we've added a `loss_function` parameter where the model is fit using the desired loss function, i.e. not necssarily the L2 loss. Example loss function are given as `l1` and `l2`.
    

<!--
BEGIN QUESTION
name: q2
points: 2
-->


```python
from scipy.optimize import minimize

def l1(y, y_hat):
    return np.abs(y - y_hat)

def l2(y, y_hat):
    return (y - y_hat)**2

class MyLinearModel():    
    def predict(self, X):
        return X @ self._thetas
    
    def fit(self, loss_function, X, y):
        """
        Produce the estimated optimal _thetas for the given loss function, 
        feature matrix X, and observations y.

        Parameters
        -----------
        loss_function: either the squared or absolute loss functions defined above
        X: a 2D dataframe (or numpy array) of numeric features (one-hot encoded)
        y: a 1D vector of tip amounts

        Returns
        -----------
        The estimate for the optimal theta vector that minimizes our loss
        """
        
        number_of_features = X.shape[1]

        ## Notes on the following function call which you need to finish:
        # 
        # 0. The starting guess should be some arbitrary array of the correct length.
        #    Note the "number of features" variable above."
        # 1. The ... in "lambda theta: ..." should be replaced by the average loss if we
        #    compute X @ theta. The loss is measured using the given loss function,
        #    relative to the observations in the variable y.
        
        starting_guess = np.random.rand(number_of_features)
        self._thetas = minimize(lambda theta: 
                                loss_function(y, X @ theta).mean()
                                , x0 = starting_guess)['x']
        # Notice above that we extract the 'x' entry in the dictionary returned by `minimize`. 
        # This entry corresponds to the optimal theta estimated by the function. Sorry
        # we know it's a little confusing, but 'x' is hard coded into the minimize function
        # because of the fact that in the optimization universe "x" is what you optimize over.
        # It'd be less confusing for DS100 students if they used "theta".
        
# When you run the code below, you should get back some non zero thetas.
        
model = MyLinearModel()
model.fit(l2, one_hot_X, tips)
model._thetas
```




    array([ 0.09448702,  0.17599315,  0.31373886,  0.34618029, -0.22256393,
           -0.13615575,  0.30569628,  0.46797868,  0.34653012,  0.44250887,
            0.1939605 ,  0.1258012 ])




```python
grader.check("q2")
```




<p><strong><pre style='display: inline;'>q2</pre></strong> passed! 💯</p>



The MSE for your model above should be just slightly larger than 1:


```python
from sklearn.metrics import mean_squared_error
mean_squared_error(model.predict(one_hot_X), tips)
```




    np.float64(1.0103535612506567)



<br/>
<hr style="border: 1px solid #fdb515;" />

## Question 3: Fitting the Model using Analytic Methods

Let's also fit our model analytically for the L2 loss function. Recall from lecture that with a linear model, we are solving the following optimization problem for least squares:

$$\min_{\theta} ||\Bbb{X}\theta - \Bbb{y}||^2$$

We showed in [Lecture 11](https://docs.google.com/presentation/d/15eEbroVt2r36TXh28C2wm6wgUHlCBCsODR09kLHhDJ8/edit#slide=id.g113dfce000f_0_2682) that the optimal $\hat{\theta}$ when $X^TX$ is invertible is given by the equation: $(X^TX)^{-1}X^TY$

---
### Question 3a: Analytic Solution Using Explicit Inverses

For this problem, implement the analytic solution above using `np.linalg.inv` to compute the inverse of $X^TX$.

Reminder: To compute the transpose of a matrix, you can use `X.T` or `X.transpose()`


```python
class MyAnalyticallyFitOLSModel():    
    def predict(self, X):
        return X @ self._thetas
    
    def fit(self, X, y):
        """
        Sets _thetas using the analytical solution to the ordinary least squares problem

        Parameters
        -----------
        X: a 2D dataframe (or numpy array) of numeric features (one-hot encoded)
        y: a 1D vector of tip amounts

        Returns
        -----------
        The estimate for theta computed using the equation mentioned above
        """
        
        xTx = X.T @ X
        xTy = X.T @ y
        self._thetas = np.linalg.inv(xTx) @ xTy

        
```

Now, run the cell below to find the analytical solution for the `tips` dataset. Depending on the machine that you run your code on, you should either see a singular matrix error or end up with thetas that are nonsensical (magnitudes greater than 10^15). This is not good!


```python
# When you run the code below, you should get back some non zero thetas.
        
model = MyAnalyticallyFitOLSModel()
model.fit(one_hot_X, tips)
analytical_thetas = model._thetas
analytical_thetas
```




    array([ 9.66544413e+00, -1.89677732e+02, -8.30149679e+17, -8.30149679e+17,
            8.30149679e+17,  8.30149679e+17, -2.56000000e+02,  0.00000000e+00,
           -3.20000000e+01,  3.20000000e+01, -8.00000000e+00,  0.00000000e+00])



In the cell below, explain why we got the error（指参数不对？） above when trying to calculate the analytical solution for our one-hot encoded `tips` dataset.

<!--
BEGIN QUESTION
name: q3a
-->

_本质上是因为矩阵 **不可逆**，独热编码某些线性组合之后可以轻易看出矩阵$X^TX$不是满秩的_


---
### Question 3b: Fixing our One-Hot Encoding

Now, let's fix our one-hot encoding approach from question 1 so we don't get the error we saw in question 3a. Complete the code below to one-hot-encode our dataset such that `one_hot_X_revised` has no redundant features.

<!--
BEGIN QUESTION
name: q3b
-->


```python
def one_hot_encode_revised(data):
    """
    Return the one-hot encoded dataframe of our input data, removing redundancies.
    
    Parameters
    -----------
    data: a dataframe that may include non-numerical features
    
    Returns
    -----------
    A one-hot encoded dataframe that only contains numeric features without any redundancies.
    
    """
    columns = ['sex', 'smoker', 'day', 'time']
    for column in columns:
        values = data[column].unique()
        for value in values[:-1]: # 这是用[]切片的技巧，从values中取除了最后一个元素的所有元素
            data[column + '=' + value] = (data[column] == value).astype(int)
        data = data.drop(column, axis=1) # 删除原始的列
    return data

one_hot_X_revised = one_hot_encode_revised(X)    
    
numerical_model = MyLinearModel()
numerical_model.fit(l2, one_hot_X_revised, tips)
    
analytical_model = MyAnalyticallyFitOLSModel()
analytical_model.fit(one_hot_X_revised, tips)


print("Our numerical model's loss is: ", mean_squared_error(numerical_model.predict(one_hot_X_revised), tips))
print("Our analytical model's loss is: ", mean_squared_error(analytical_model.predict(one_hot_X_revised), tips))
```

    Our numerical model's loss is:  1.0255082437778105
    Our analytical model's loss is:  1.0255082436053506
    


```python
grader.check("q3b")
```

---
### Question 3c: Analyzing our new One-Hot Encoding

Why did removing redundancies in our one-hot encoding fix the problem we had in 3a?
<!--
BEGIN QUESTION
name: q3c
-->

_不是全部进行独热编码操作，避免线性相关性_

---

Note: An alternate approach is to use `np.linalg.solve` instead of `np.linalg.inv`. For the example above, even with the redundant features, `np.linalg.solve` will work well. Though in general, it's best to drop redundant features anyway.

In case you want to learn more, here is a relevant Stack Overflow post: https://stackoverflow.com/questions/31256252/why-does-numpy-linalg-solve-offer-more-precise-matrix-inversions-than-numpy-li

<br/>
<hr style="border: 1px solid #fdb515;" />

## Question 4: Gradient Descent


```python
# Run this cell to load the data for this problem
df = pd.read_csv("lab7_data.csv", index_col=0)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-5.000000</td>
      <td>-7.672309</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-4.966555</td>
      <td>-7.779735</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-4.933110</td>
      <td>-7.995938</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-4.899666</td>
      <td>-8.197059</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-4.866221</td>
      <td>-8.183883</td>
    </tr>
  </tbody>
</table>
</div>



If we plot this data, we see that there is a clear sinusoidal relationship between x and y.


```python
import plotly.express as px
px.scatter(df, x = "x", y = "y")
```



In this exercise, we'll show gradient descent is so powerful it can even optimize a nonlinear model. Specifically, we're going to model the relationship of our data by:

$$\Large{
f_{\boldsymbol{\theta(x)}} = \theta_1x + sin(\theta_2x)
}$$

Our model is parameterized by both $\theta_1$ and $\theta_2$, which we can represent in the vector, $\boldsymbol{\theta}$.

Note that a general sine function $a\sin(bx+c)$ has three parameters: amplitude scaling parameter $a$, frequency parameter $b$ and phase shifting parameter $c$. 

Here, we're assuming the amplitude $a$ is around 1, and the phase shifting parameter $c$ is around zero. We do not attempt to justify this assumption and you're welcome to see what happens if you ignore this assumption at the end of this lab.

You might ask why we don't just create a linear model like we did earlier with a sinusoidal feature. The issue is that the theta is INSIDE the sin function. In other words, linear models use their parameters to adjust the scale of each feature, but $\theta_2$ in this model adjusts the frequency of the feature. There are tricks we could play to use our linear model framework here, but we won't attempt this in our lab.

We define the `sin_model` function below that predicts $\textbf{y}$ (the $y$-values) using $\textbf{x}$ (the $x$-values) based on our new equation.


```python
def sin_model(x, theta):
    """
    Predict the estimate of y given x, theta_1, theta_2

    Keyword arguments:
    x -- the vector of values x
    theta -- a vector of length 2, where theta[0] = theta_1 and theta[1] = theta_2
    """
    theta_1 = theta[0]
    theta_2 = theta[1]
    return theta_1 * x + np.sin(theta_2 * x)
```

---
### Question 4a: Computing the Gradient of the MSE With Respect to Theta on the Sin Model

Recall $\hat{\theta}$ is the value of $\theta$ that minimizes our loss function. One way of solving for $\hat{\theta}$ is by computing the gradient of our loss function with respect to $\theta$, like we did in lecture: https://docs.google.com/presentation/d/1j9ESgjn-aeZSOX5ON1wjkF5WBZHc4IN7XvTpYnX1pFs/edit#slide=id.gfc76b62ec3_0_27. Recall that the gradient is a column vector of two partial derivatives.

Write/derive the expressions for following values and use them to fill in the functions below.

* $L(\textbf{x}, \textbf{y}, \theta_1, \theta_2)$: our loss function, the mean squared error
* $\frac{\partial L }{\partial \theta_1}$: the partial derivative of $L$ with respect to $\theta_1$
* $\frac{\partial L }{\partial \theta_2}$: the partial derivative of $L$ with respect to $\theta_2$

Recall that $L(\textbf{x}, \textbf{y}, \theta_1, \theta_2) = \frac{1}{n} \sum_{i=1}^{n} (\textbf{y}_i - \hat{\textbf{y}}_i)^2$

Specifically, the functions `sin_MSE`, `sin_MSE_dt1` and `sin_MSE_dt2` should compute $R$, $\frac{\partial R }{\partial \theta_1}$ and $\frac{\partial R }{\partial \theta_2}$ respectively. Use the expressions you wrote for $\frac{\partial R }{\partial \theta_1}$ and $\frac{\partial R }{\partial \theta_2}$ to implement these functions. In the functions below, the parameter `theta` is a vector that looks like $\begin{bmatrix} \theta_1 \\ \theta_2 \end{bmatrix}$. We have completed `sin_MSE_gradient`, which calls `dt1` and `dt2` and returns the gradient `dt` for you.

Notes: 
* Keep in mind that we are still working with our original set of data, `df`.
* To keep your code a bit more concise, be aware that `np.mean` does the same thing as `np.sum` divided by the length of the numpy array. *注意mean的层级*
* Another way to keep your code more concise is to use the function `sin_model` we defined which computes the output of the model.

<!--
BEGIN QUESTION
name: q4a
points: 3
-->


```python
def sin_MSE(theta, x, y):
    """
    Compute the numerical value of the l2 loss of our sinusoidal model given theta

    Keyword arguments:
    theta -- the vector of values theta
    x     -- the vector of x values
    y     -- the vector of y values
    """
    return np.mean((y - sin_model(x, theta))**2)

def sin_MSE_dt1(theta, x, y):
    """
    Compute the numerical value of the partial of l2 loss with respect to theta_1

    Keyword arguments:
    theta -- the vector of values theta
    x     -- the vector of x values
    y     -- the vector of y values
    """
    return np.mean(-2 * (y - sin_model(x, theta)) * x)
    
def sin_MSE_dt2(theta, x, y):
    """
    Compute the numerical value of the partial of l2 loss with respect to theta_2

    Keyword arguments:
    theta -- the vector of values theta
    x     -- the vector of x values
    y     -- the vector of y values
    """
    return np.mean(-2*(y-sin_model(x, theta))*x*np.cos(theta[1]*x))
    
# This function calls dt1 and dt2 and returns the gradient dt. It is already implemented for you.
def sin_MSE_gradient(theta, x, y):
    """
    Returns the gradient of l2 loss with respect to vector theta

    Keyword arguments:
    theta -- the vector of values theta
    x     -- the vector of x values
    y     -- the vector of y values
    """
    return np.array([sin_MSE_dt1(theta, x, y), sin_MSE_dt2(theta, x, y)])
```


```python
grader.check("q4a")
```

---
### Question 4b: Implementing Gradient Descent and Using It to Optimize the Sin Model

Let's now implement gradient descent. 

Note that the function you're implementing here is somewhat different than the gradient descent function we created in lecture. The version in lecture was `gradient_descent(df, initial_guess, alpha, n)`, where `df` was the gradient of the function we are minimizing and `initial_guess` are the starting parameters for that function. Here our signature is a bit different (described below) than the `gradient_descent` [implementation from lecture](https://ds100.org/sp22/resources/assets/lectures/lec12/lec12.html).

<!--
BEGIN QUESTION
name: q4b
points: 3
-->


```python
def init_theta():
    """Creates an initial theta [0, 0] of shape (2,) as a starting point for gradient descent"""
    return np.array([0, 0])

def grad_desc(loss_f, gradient_loss_f, theta, data, num_iter=20, alpha=0.1):
    """
    Run gradient descent update for a finite number of iterations and static learning rate

    Keyword arguments:
    loss_f -- the loss function to be minimized (used for computing loss_history)
    gradient_loss_f -- the gradient of the loss function to be minimized
    theta -- the vector of values theta to use at first iteration
    data -- the data used in the model 
    num_iter -- the max number of iterations
    alpha -- the learning rate (also called the step size)
    
    Return:
    theta -- the optimal value of theta after num_iter of gradient descent
    theta_history -- the series of theta values over each iteration of gradient descent
    loss_history -- the series of loss values over each iteration of gradient descent
    """
    theta_history = []
    loss_history = []
    for i in range(num_iter):
        theta_history.append(theta) # 先append比较好
        loss_history.append(loss_f(theta, data['x'], data['y']))
        d_b = gradient_loss_f(theta, data['x'], data['y'])
        theta = theta - alpha * d_b

    return theta, theta_history, loss_history

theta_start = init_theta()
theta_hat, thetas_used, losses_calculated = grad_desc(
    sin_MSE, sin_MSE_gradient, theta_start, df, num_iter=20, alpha=0.1
)
for b, l in zip(thetas_used, losses_calculated):
    print(f"theta: {b}, Loss: {l}")
```

    theta: [0 0], Loss: 20.859191416422235
    theta: [2.60105745 2.60105745], Loss: 9.285008173048666
    theta: [0.90342728 2.59100602], Loss: 4.680169273815357
    theta: [2.05633644 2.9631291 ], Loss: 2.6242517936325833
    theta: [1.15892347 2.86687431], Loss: 1.4765157174727774
    theta: [1.79388042 3.07275573], Loss: 0.9073271435862448
    theta: [1.32157494 3.00146569], Loss: 0.541531643291128
    theta: [1.64954491 3.02910866], Loss: 0.3775841142469479
    theta: [1.42325294 2.98820303], Loss: 0.2969750688130759
    theta: [1.58295041 3.01033846], Loss: 0.2590425421375732
    theta: [1.47097255 2.98926519], Loss: 0.23973439443291833
    theta: [1.55040965 3.0017442 ], Loss: 0.23034782416254634
    theta: [1.49439132 2.99135194], Loss: 0.2255775832667724
    theta: [1.5341564  2.99797824], Loss: 0.22321772191904068
    theta: [1.50603995 2.99286671], Loss: 0.22202363967204045
    theta: [1.52598919 2.99628665], Loss: 0.22142811500262397
    theta: [1.51186655 2.99375531], Loss: 0.22112776381775168
    theta: [1.52188208 2.99549617], Loss: 0.22097741373654575
    theta: [1.51478773 2.99423497], Loss: 0.22090173185683032
    theta: [1.51981739 2.99511516], Loss: 0.2208637810584589
    


```python
grader.check("q4b")
```

If you pass the tests above, you're done coding for this lab, though there are some cool visualizations below we'd like you to think about.

Let's visually inspect our results of running gradient descent to optimize $\boldsymbol\theta$. The code below plots our $x$-values with our model's predicted $\hat{y}$-values over the original scatter plot. You should notice that gradient descent successfully optimized $\boldsymbol\theta$.


```python
theta_init = init_theta()

theta_est, thetas, loss = grad_desc(sin_MSE, sin_MSE_gradient, theta_init, df)
```

Plotting our model output over our observaitons shows that gradient descent did  a great job finding both the overall increase (slope) of the data, as well as the oscillation frequency.


```python
x, y = df['x'], df['y']
y_pred = sin_model(x, theta_est)

plt.plot(x, y_pred, label='Model ($\hat{y}$)')
plt.scatter(x, y, alpha=0.5, label='Observation ($y$)', color='gold')
plt.legend();
```

    <>:4: SyntaxWarning:
    
    invalid escape sequence '\h'
    
    <>:4: SyntaxWarning:
    
    invalid escape sequence '\h'
    
    C:\Users\86135\AppData\Local\Temp\ipykernel_10128\2413075366.py:4: SyntaxWarning:
    
    invalid escape sequence '\h'
    
    


    
![png](lab07_files/lab07_54_1.png)
    


<br/>
<hr style="border: 1px solid #fdb515;" />

## Visualizing Loss (Extra)

Let's visualize our loss functions and gain some insight as to how gradient descent optimizes our model parameters.

In the previous plot we saw the loss decrease with each iteration. In this part, we'll see the trajectory of the algorithm as it travels the loss surface? Run the following cells to see visualization of this trajectory. 


```python
thetas = np.array(thetas).squeeze()
loss = np.array(loss)
thetas
```




    array([[0.        , 0.        ],
           [2.60105745, 2.60105745],
           [0.90342728, 2.59100602],
           [2.05633644, 2.9631291 ],
           [1.15892347, 2.86687431],
           [1.79388042, 3.07275573],
           [1.32157494, 3.00146569],
           [1.64954491, 3.02910866],
           [1.42325294, 2.98820303],
           [1.58295041, 3.01033846],
           [1.47097255, 2.98926519],
           [1.55040965, 3.0017442 ],
           [1.49439132, 2.99135194],
           [1.5341564 , 2.99797824],
           [1.50603995, 2.99286671],
           [1.52598919, 2.99628665],
           [1.51186655, 2.99375531],
           [1.52188208, 2.99549617],
           [1.51478773, 2.99423497],
           [1.51981739, 2.99511516]])




```python
# Run me to see a 3D plot (gradient descent with static alpha)
from lab7_utils import plot_3d
plot_3d(thetas[:, 0], thetas[:, 1], loss, mean_squared_error, sin_model, x, y)
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-2.34.0.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




<div>                            <div id="876062a6-7410-420f-b994-0db1f1acd2ce" class="plotly-graph-div" style="height:700px; width:800px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("876062a6-7410-420f-b994-0db1f1acd2ce")) {                    Plotly.newPlot(                        "876062a6-7410-420f-b994-0db1f1acd2ce",                        [{"line":{"color":"rgb(50,170, 140)","width":3},"marker":{"color":[-20.859191416422235,-9.285008173048666,-4.680169273815357,-2.6242517936325833,-1.4765157174727774,-0.9073271435862448,-0.541531643291128,-0.3775841142469479,-0.2969750688130759,-0.2590425421375732,-0.23973439443291833,-0.23034782416254634,-0.2255775832667724,-0.22321772191904068,-0.22202363967204045,-0.22142811500262397,-0.22112776381775168,-0.22097741373654575,-0.22090173185683032,-0.2208637810584589],"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"size":4},"x":[0.0,2.601057454779614,0.9034272767622247,2.0563364416095036,1.1589234696036672,1.7938804194525289,1.3215749439533506,1.6495449106923081,1.4232529432779681,1.5829504072241183,1.4709725522541734,1.5504096528840599,1.4943913185184219,1.5341564032256916,1.506039954902419,1.525989188155241,1.5118665534078317,1.521882079272916,1.5147877319328085,1.519817390738254],"y":[0.0,2.601057454779614,2.5910060231049896,2.963129099991904,2.8668743087666506,3.072755727714994,3.001465693449906,3.0291086593677257,2.988203025298647,3.010338461886587,2.9892651852224326,3.001744204213725,2.9913519372340183,2.997978235955032,2.99286670937898,2.99628664791577,2.993755312996005,2.9954961669353444,2.994234974327257,2.9951151604637363],"z":[20.859191416422235,9.285008173048666,4.680169273815357,2.6242517936325833,1.4765157174727774,0.9073271435862448,0.541531643291128,0.3775841142469479,0.2969750688130759,0.2590425421375732,0.23973439443291833,0.23034782416254634,0.2255775832667724,0.22321772191904068,0.22202363967204045,0.22142811500262397,0.22112776381775168,0.22097741373654575,0.22090173185683032,0.2208637810584589],"type":"scatter3d"},{"x":[-0.1,-0.042835562147354816,0.014328875705290373,0.07149331355793556,0.12865775141058075,0.18582218926322594,0.24298662711587113,0.3001510649685163,0.35731550282116153,0.4144799406738068,0.4716443785264519,0.528808816379097,0.5859732542317423,0.6431376920843875,0.7003021299370327,0.7574665677896778,0.814631005642323,0.8717954434949683,0.9289598813476135,0.9861243192002586,1.0432887570529037,1.100453194905549,1.157617632758194,1.2147820706108392,1.2719465084634844,1.3291109463161297,1.386275384168775,1.44343982202142,1.5006042598740652,1.5577686977267104,1.6149331355793555,1.6720975734320007,1.729262011284646,1.7864264491372912,1.8435908869899365,1.9007553248425815,1.957919762695227,2.015084200547872,2.072248638400517,2.1294130762531625,2.1865775141058075,2.2437419519584525,2.300906389811098,2.358070827663743,2.415235265516388,2.4723997033690335,2.5295641412216785,2.586728579074324,2.643893016926969,2.701057454779614],"y":[-0.1,-0.03320906678132665,0.0335818664373467,0.10037279965602006,0.1671637328746934,0.23395466609336676,0.30074559931204015,0.3675365325307135,0.43432746574938685,0.5011183989680602,0.5679093321867336,0.6347002654054069,0.7014911986240803,0.7682821318427536,0.835073065061427,0.9018639982801003,0.9686549314987737,1.035445864717447,1.1022367979361203,1.1690277311547936,1.235818664373467,1.3026095975921403,1.3694005308108137,1.436191464029487,1.5029823972481604,1.5697733304668338,1.6365642636855071,1.7033551969041805,1.7701461301228538,1.8369370633415272,1.9037279965602005,1.970518929778874,2.0373098629975472,2.1041007962162204,2.170891729434894,2.2376826626535675,2.3044735958722407,2.371264529090914,2.4380554623095874,2.504846395528261,2.571637328746934,2.638428261965607,2.705219195184281,2.7720101284029544,2.8388010616216275,2.9055919948403006,2.972382928058974,3.0391738612776478,3.105964794496321,3.172755727714994],"z":[[26.32282044860889,24.673923664437837,23.079854063068225,21.54061164450004,20.05619640873328,18.62660835576796,17.251847485604074,15.931913798241615,14.66680729368059,13.456527971920991,12.30107583296283,11.200450876806102,10.154653103450801,9.163682512896935,8.227539105144503,7.3462228801934994,6.519733838043929,5.748071978695791,5.0312373021490835,4.36922980840381,3.762049497459968,3.209696369317557,2.7121704239765796,2.2694716614370325,1.881600081698917,1.5485556847622342,1.2703384706269831,1.0469484392931647,0.8783855907607775,0.7646499250298221,0.7057414421002989,0.7016601419722076,0.7524060246455484,0.8579790901203213,1.0183793383965258,1.233606769474162,1.5036613833532317,1.8285431800337313,2.2082521595156623,2.6427883217990287,3.132151666883824,3.6763421947700508,4.275359905457713,4.929204798946804,5.637876875237326,6.401376134329285,7.219702576222671,8.092856200917495,9.020837008413745,10.003644998711424],[24.470263957582933,22.883123381205607,21.350809987629717,19.873323776855255,18.45066474888223,17.082832903710635,15.769828241340468,14.511650761771737,13.308300465004434,12.159777351038565,11.066081419874132,10.02721267151113,9.043171105949552,8.113956723189412,7.239569523230705,6.420009506073431,5.655276671717585,4.945371020163171,4.2902925514101895,3.690041265458643,3.144617162308527,2.6540202419598415,2.2182505044125898,1.8373079496667681,1.5111925777223787,1.2399043885794216,1.023443382237896,0.8618095586978033,0.755002917959142,0.7030234600219125,0.7058711848861152,0.7635460925517495,0.8760481830188163,1.0433774562873148,1.2655339123572453,1.5425175512286073,1.874328372901403,2.260966377375628,2.7024315646512846,3.1987239347283776,3.7498434876068982,4.355790223286849,5.016564141768239,5.732165243051056,6.5025935271353035,7.327848994020988,8.207931643708099,9.142841476196649,10.132578491486623,11.177142689578032],[22.626388681683842,21.103128552230952,19.63469560557949,18.221089841729462,16.862311260680865,15.558359862433706,14.30923564698797,13.114938614343673,11.975468764500803,10.890826097459364,9.861010613219362,8.88602231178079,7.965861193143651,7.10052725730794,6.290020504273665,5.534340934040821,4.8334885466094075,4.187463341979425,3.5962653201508767,3.0598944811237616,2.578350824898077,2.151634351473824,1.7797450608510035,1.4626829530296142,1.2004480280096566,0.9930402857911311,0.8404597263740378,0.7427063497583766,0.6997801559441471,0.7116811449313496,0.7784093167199839,0.8999646713100503,1.0763472087015489,1.3075569288944793,1.5935938318888418,1.9344579176846348,2.3301491862818633,2.78066763768052,3.2860132718806088,3.8461860888821335,4.4611860886850865,5.131013271289469,5.855667636695291,6.635149184902539,7.469457915911216,8.358593829721336,9.302556926332878,10.301347205745861,11.354964667960267,12.463409312976104],[20.916018343290457,19.454490735594117,18.04779031069921,16.695917068605727,15.398871009313677,14.156652132823062,12.969260439133878,11.836695928246126,10.758958600159806,9.736048454874915,8.767965492391461,7.854709712709438,6.996281115828845,6.192679701749685,5.443905470471957,4.749958421995661,4.110838556320796,3.526545873447363,2.9970803733753626,2.5224420561047953,2.1026309216356593,1.7376469699679542,1.4274902011016821,1.172160615036841,0.9716582117734318,0.8259829913114548,0.7351349536509099,0.6991140987917969,0.7179204267341159,0.7915539374778668,0.9200146310230491,1.103302507369664,1.341417566517711,1.63435980846719,1.9821292332181009,2.3847258407704417,2.8421496311242187,3.354400604279424,3.921478760236061,4.543384098994133,5.220116620553635,5.951676324914566,6.738063212076936,7.5792772820407315,8.47531853480596,9.426186970372624,10.431882588740715,11.492405389910248,12.6077553738812,13.77793254065359],[19.440563686353922,18.03451935465846,16.68330220576443,15.386912239671826,14.145349456380657,12.958613855890922,11.82670543820262,10.749624203315745,9.727370151230305,8.759943281946295,7.8473435954637205,6.989571091782576,6.186625770902863,5.438507632824581,4.7452166775477345,4.106752905072317,3.523116315398332,2.994306908525778,2.520324684454657,2.101169643184969,1.7368417847167121,1.4273411090498866,1.172667616184494,0.9728213061205324,0.8278021788580028,0.7376102343969054,0.7022454727372398,0.7217078938790062,0.7959974978222045,0.925114284566835,1.1090582541128968,1.347829406460391,1.6414277416093175,1.989853259559676,2.3931059603114666,2.851185843864687,3.3640929102193438,3.931827159375428,4.554388591332945,5.231777206091897,5.963993003652276,6.751035984014088,7.592906147177336,8.489603493142013,9.441128021908119,10.447479733475664,11.508658627844637,12.624664705015048,13.79549796498688,15.021158407760145],[18.274084928922743,16.913669165164208,15.608080584207103,14.357319186051432,13.161384970697194,12.020277938144387,10.933998088393015,9.902545421443074,8.92591993729456,8.004121635947481,7.137150517401836,6.325006581657622,5.567689828714837,4.865200258573485,4.217537871233568,3.624702666695081,3.0866946449580253,2.6035138060224017,2.1751601498882103,1.8016336765554521,1.4829343860241249,1.2190622782942295,1.0100173533657661,0.8557996112387347,0.7564090519131348,0.7118456753889669,0.7221094816662315,0.7872004707449275,0.9071186426250555,1.081863997306616,1.3114365347896075,1.5958362550740315,1.9350631581598876,2.329117244047176,2.7779985127358966,3.2817069642260464,3.8402425985176336,4.453605415610647,5.121795415505094,5.844812598200976,6.622656963698285,7.455328511997025,8.342827243097206,9.285153156998808,10.282306253701847,11.334286533206324,12.441093995512222,13.602728640619564,14.819190468528328,16.090479479238525],[17.464395964087338,16.136926610201055,14.864284439116206,13.646469450832791,12.483481645350805,11.375321022670253,10.321987582791134,9.323481325713447,8.379802251437189,7.490950359962362,6.656925651288971,5.877728125417012,5.153357782346482,4.483814622077384,3.8690986446097213,3.3092098499434885,2.804148238078687,2.3539138090153178,1.9585065627533804,1.6179264992928764,1.3321736186338036,1.101247920776162,0.925149405719953,0.8038780734651757,0.73743392401183,0.7258169573599166,0.7690271735094353,0.8670645724603854,1.019929154212768,1.2276209187665823,1.490139866121828,1.8074859962785066,2.1796593092366168,2.606659804996159,3.088487483557134,3.6251423449195377,4.216624389083379,4.862933616048647,5.564070025815347,6.320033618383484,7.1308243937530476,7.996442351924043,8.916887492896477,9.892159816670336,10.922259323245626,12.007186012622357,13.146939884800512,14.341520939780105,15.590929177561126,16.89516459814357],[17.036790822248832,15.727751403507343,14.47353916756729,13.274154114428672,12.129596244091482,11.039865556555723,10.0049620518214,9.024885729888508,8.099636590757042,7.229214634427014,6.413619860898417,5.652852270171253,4.946911862245519,4.295798637121216,3.699512594798347,3.1580537352769102,2.671422058556904,2.2396175646383294,1.8626402535211875,1.5404901252054783,1.2731671796912007,1.0606714169783544,0.9030028370669405,0.8001614399569582,0.7521472256484079,0.7589601941412896,0.8206003454356032,0.9370676795313485,1.1083621964285264,1.334483896127136,1.6154327786271763,1.95120884392865,2.3418120920315557,2.7872425229358933,3.2875001366416625,3.842584933148863,4.452496912457499,5.117236074567562,5.836802419479056,6.611195947191989,7.440416657706348,8.324464551022137,9.263339627139366,10.25704188605802,11.305571327778107,12.408927952299633,13.56711175962258,14.78012274974697,16.047960922672786,17.370626278400028],[16.997662509909667,15.691825366734296,14.440815406360361,13.244632628787862,12.103277034016788,11.016748622047153,9.985047392878945,9.00817334651217,8.086126482946826,7.218906802182916,6.406514304220436,5.648948989059391,4.946210856699775,4.29829990714159,3.70521614038484,3.166959556429521,2.683530155275634,2.254927936923177,1.881152901372154,1.5622050486225634,1.298084378674404,1.0887908915276763,0.9343245871823808,0.834685465638517,0.7898735268960848,0.799888770955085,0.8647311978155172,0.9844008074773808,1.158897599940677,1.388221575205405,1.672372733271564,2.0113510741391556,2.40515659780818,2.853789304278636,3.357249193550524,3.915536265623843,4.528650520498597,5.196591958174778,5.919360578652392,6.696956381931442,7.52937936801192,8.416629536893828,9.358706888577178,10.35561142306195,11.407343140348154,12.513902040435797,13.675288123324863,14.891501389015374,16.162541837507302,17.48840946880067],[17.336054611229248,16.018638513807375,14.756049599186936,13.548287867367925,12.39535331835035,11.297245952134205,10.253965768719494,9.265512768106213,8.331886950294365,7.453088315283946,6.629116863074963,5.85997259366741,5.1456555070612895,4.486165603256599,3.8815028822533426,3.3316673440515183,2.836658988651125,2.3964778160521636,2.011123826254634,1.6805970192585376,1.4048973950638728,1.1840249536706393,1.0179796950788378,0.9067616192884683,0.8503707262995306,0.8488070161120248,0.9020704887259512,1.010161144141309,1.1730789823580996,1.3908240033763217,1.663396207195975,1.9907955938170616,2.37302216323958,2.8100759154635297,3.3019568504889127,3.848664968315725,4.4502002689439735,5.106562752373649,5.817752418604757,6.583769267637302,7.404613299471272,8.280284514106675,9.210782911543516,10.196108491781784,11.236261254821484,12.33124120066262,13.481048329305183,14.685682640749185,15.94514413499461,17.259432812041467],[18.022678326192622,16.68044325223466,15.393035361078134,14.160454652723041,12.982701127169381,11.859774784417148,10.791675624466349,9.778403647316981,8.819958852969044,7.9163412414225425,7.06755081267747,6.273587566733833,5.534451503591623,4.850142623250847,4.220660925711504,3.646006410973594,3.126179079037114,2.661178929902065,2.251005963568449,1.8956601800362662,1.5951415793055148,1.3494501613761944,1.1585859262483067,1.0225488739218505,0.9413390043968264,0.914956317673234,0.9434008137510737,1.0266724926303448,1.1647713543110485,1.3576973987931844,1.605450626076751,1.9080310361617505,2.265438629048182,2.677673404736046,3.1447353632253416,3.6666245045160677,4.243340828608229,4.874884335501818,5.56125502519684,6.302452897693299,7.098477952991183,7.9493301910905,8.855009611991255,9.815516215693433,10.830850002197046,11.901010971502098,13.025999123608573,14.205814458516489,15.440456976225827,16.7299266767366],[19.00752568169107,17.629714314180863,16.306730129472093,15.038573127564748,13.825243308458836,12.666740672154356,11.56306521865131,10.514216947949695,9.52019586004951,8.581001954950759,7.69663523265344,6.867095693157556,6.092383336463099,5.372498162570074,4.707440171478484,4.097209363188325,3.5418057376995966,3.0412292950123003,2.595480035126436,2.204557958042005,1.8684630637590058,1.5871953522774378,1.3607548235973022,1.189141477718598,1.0723553146413258,1.0103963343654854,1.0032645368910773,1.0509599222181008,1.1534824903465566,1.3108322412764442,1.523009175007763,1.7900132915405147,2.111844590874699,2.4885030730103144,2.919988737947362,3.40630158568584,3.9474416162257544,4.543408829567095,5.194203225709869,5.899824804654079,6.660273566399716,7.475549510946784,8.345652638295292,9.270582948445224,10.250340441396586,11.28492511714939,12.374336975703619,13.518576017059285,14.717642241216378,15.971535648174898],[20.21827677269866,18.797327118011786,17.431204646126343,16.119909357042335,14.863441250759758,13.661800327278616,12.514986586598898,11.42300002872062,10.385840653643767,9.403508461368348,8.476003451894364,7.603325625221813,6.78547498135069,6.022451520280999,5.3142552420127425,4.660886146545917,4.062344233880524,3.5186295040165607,3.029741956954031,2.595681592692934,2.2164484112332685,1.892042412575034,1.622463596718233,1.4077119636628626,1.2477875134089238,1.1426902459564174,1.0924201613053433,1.0969772594557008,1.1563615404074903,1.2705730041607117,1.4396116507153647,1.66347748007145,1.9421704922289678,2.2756906871879177,2.664038064948299,3.1072126255101105,3.605214368873358,4.158043295038034,4.765699404004141,5.428182695771685,6.145493170340656,6.917630827711058,7.744595667882899,8.626387690856165,9.563006896630862,10.554453285207,11.600726856584561,12.701827610763564,13.857755547743988,15.068510667525846],[21.561817449574487,20.093792341471406,18.680594416169747,17.32222367366953,16.01868011397074,14.769963737073375,13.576074542977448,12.437012531682951,11.352777703189888,10.323370057498256,9.348789594608059,8.429036314519289,7.564110217231954,6.75401130274605,5.998739571061578,5.298295022178539,4.652677656096931,4.061887472816754,3.52592447233801,3.0447886546606995,2.61848001978482,2.2469985677103708,1.9303442984373556,1.668517211965771,1.461517308295618,1.3093445874268976,1.2119990493596091,1.1694806940937528,1.1817895216293282,1.2489255319663353,1.3708887251047743,1.5476791010446456,1.779296659785949,2.0657414013286846,2.407013325672852,2.803112432818449,3.2540387227654826,3.7597921955139433,4.320372851063838,4.935780689415168,5.606015710567923,6.331077914522112,7.110967301277739,7.945683870834792,8.835227623193274,9.779598558353197,10.778796676314544,11.832821977077332,12.941674460641543,14.105354127007185],[22.930307122646543,21.41501222941707,19.954544518989017,18.548903991362405,17.19809064653722,15.902104484513472,14.660945505291153,13.474613708870267,12.343109095250812,11.266431664432787,10.244581416416198,9.277558351201042,8.365362468787312,7.507993769175017,6.705452252364154,5.957737918354724,5.264850767146724,4.626790798740156,4.043558013135021,3.5151524103313188,3.0415739903290486,2.622822753128209,2.2588986987288022,1.9498018271308266,1.6955321383342827,1.4960896323391708,1.351474309145491,1.2616861687532437,1.2267252111624278,1.246591436373044,1.321284844385092,1.450805435198572,1.6351532088134844,1.8743281652298285,2.1683303044476046,2.5171596264668117,2.920816131287453,3.379299818909524,3.8926106893330266,4.460748742557964,5.0837139785843295,5.761506397412128,6.494125999041362,7.281572783472023,8.123846750704118,9.020947900737646,9.972876233572604,10.979631749209002,12.041214447646823,13.157624328886074],[24.211697189994467,22.652487053248425,21.14810409930382,19.698548328160648,18.303819739818902,16.963918334278592,15.678844111539718,14.448597071602272,13.273177214466257,12.152584540131675,11.086819048598525,10.075880739866808,9.119769613936521,8.218485670807665,7.372028910480244,6.580399332954254,5.843596938229695,5.161621726306566,4.534473697184873,3.962152850864611,3.4446591873457812,2.9819927066283825,2.5741534087124167,2.2211412935978814,1.9229563612847778,1.679598611773107,1.491068045062868,1.3573646611540608,1.278488460046686,1.2544394417407425,1.285217606236231,1.370822953533152,1.5112554836315046,1.7065151965312895,1.956602092232506,2.2615161707351534,2.6212574320392363,3.0358258761447474,3.5052215030516907,4.029444312760069,4.6084943052698755,5.242371480581114,5.93107583869379,6.674607379607892,7.472966103323424,8.326152009840397,9.234165099158794,10.19700537127863,11.214672826199891,12.287167463922584],[25.30297739595644,23.706276413833972,22.16440261451294,20.677355997993335,19.245136564275168,17.867744313358426,16.545179245243123,15.277441359929243,14.0645306574168,12.906447137705788,11.80319080079621,10.754761646688063,9.761159675381347,8.822384886876062,7.938437281172213,7.109316858269793,6.335023618168805,5.615557560869249,4.950918686371124,4.341106994674434,3.786122485779175,3.2859651596853467,2.8406350163929517,2.4501320559019866,2.114456278212454,1.833607683324354,1.6075862712376856,1.4363920419524496,1.3200249954686452,1.2584851317862726,1.2517724509053318,1.2998869528258237,1.402828637547747,1.5605975050711025,1.77319355539589,2.040616788522108,2.3628672044497616,2.7399448031788434,3.1718495847093577,3.6585815490413065,4.200140696174684,4.796527026109493,5.447740538845739,6.153781234383413,6.9146491127225165,7.730344173863059,8.600866417805028,9.526215844548434,10.506392454093266,11.541396246439529],[26.12332146325116,24.49791108847396,22.92732789649819,21.411571887323856,19.950643060950945,18.54454141737947,17.193266956609428,15.896819678640822,14.655199583473644,13.468406671107896,12.336440941543582,11.2593023947807,10.236991030819247,9.269506849659232,8.356849851300646,7.499020035743492,6.696017402987769,5.947841953033477,5.254493685880617,4.615972601529193,4.032278699979199,3.503411981230635,3.029372445283505,2.6101600921378063,2.2457749217935383,1.9362169342507032,1.6814861295092998,1.4815825075693294,1.33650606843079,1.2462568120936823,1.210834738558007,1.2302398478237635,1.3044721398909522,1.4335316147595727,1.6174182724296253,1.856132112901109,2.149673136174027,2.4980413422483743,2.9012367311241536,3.3592593028013678,3.87210905728001,4.439785994560084,5.062290114641595,5.739621417524533,6.471779903208902,7.258765571694711,8.100578422981943,8.997218457070616,9.948685673960714,10.95498007365224],[26.62411417180046,24.980256747936462,23.3912265068739,21.857023448612765,20.37764757315307,18.9530988804948,17.58337737063796,16.26848304358256,15.008415899328586,13.803175937876043,12.652763159224934,11.55717756337526,10.516419150327016,9.5304879200802,8.599383872634819,7.7231070079908735,6.901657326148355,6.13503482710727,5.423239510867617,4.766271377429398,4.164130426792609,3.616816658957252,3.124330073923328,2.686670671690834,2.3038384522597726,1.975833415630143,1.7026555618019452,1.4843048907751801,1.3207814025498468,1.212085097125945,1.1582159745034755,1.159174034682438,1.2149592776628322,1.3255717034446586,1.4910113120279167,1.7112781034126063,1.9863720775987297,2.316293234586283,2.701041574375268,3.140617096965688,3.635019802357536,4.184249690550816,4.788306761545533,5.447191015341677,6.160902451939252,6.929441071338265,7.752806873538705,8.630999858540582,9.564020026343886,10.551867376948621],[26.793625812944743,25.142107765461404,23.545416900779497,22.00355321889902,20.516516719819975,19.084307403542365,17.706925270066186,16.384370319391444,15.116642551518122,13.903741966446239,12.745668564175785,11.642422344706768,10.594003308039179,9.60041145417302,8.661646783108296,7.777709294845007,6.948598989383145,6.174315866722716,5.45485992686372,4.790231169806157,4.180429595550025,3.625455204095324,3.1253079954420566,2.6799879695902193,2.289495126539814,1.9538294662908409,1.6729909888432999,1.4469796941971913,1.2757955823525144,1.1594386533092693,1.0979089070674561,1.0912063436270747,1.1393309629881256,1.2422827651506085,1.4000617501145234,1.6126679178798695,1.8801012684466494,2.202361801814859,2.5794495179845,3.011364416955576,3.498106498728082,4.039675763302019,4.636072210677391,5.287295840854191,5.993346653832423,6.754224649612093,7.56992982819319,8.440462189575722,9.365821733759683,10.346008460745074],[26.655561161480144,25.00674271032824,23.41275144197777,21.873587356428725,20.389250453681118,18.959740733734943,17.585058196590197,16.265202842246882,15.000174670705,13.78997368196455,12.634599876025534,11.534053252887949,10.488333812551797,9.497441555017073,8.561376480283785,7.680138588351928,6.853727879221503,6.0821443528925085,5.365388009364946,4.703458848638818,4.096356870714121,3.5440820755908553,3.0466344632690228,2.60401403374862,2.21622078702965,1.8832547231121115,1.6051158419960052,1.3818041436813318,1.2133196281680894,1.0996622954562791,1.0408321455459009,1.0368291784369543,1.08765339412944,1.1933047926233575,1.3537833739187075,1.569089138015488,1.839222084913703,2.1641822146133474,2.5439695271144234,2.978584022416935,3.468025700520875,4.012294561426246,4.611390605133054,5.265313831641289,5.9740642409509555,6.73764183306206,7.55604660797459,8.42927856568856,9.357337706203953,10.340224029520781],[26.262325271507777,24.625280209924533,23.043062331142725,21.51567163516235,20.043108121983405,18.62537179160589,17.26246264402981,15.954380679255161,14.701125897281942,13.502698298110158,12.359097881739805,11.270324648170885,10.236378597403395,9.257259729437335,8.33296804427271,7.4635035419095175,6.648866222347756,5.889056085587425,5.1840731316285265,4.5339173604710625,3.938588772115029,3.3980873665604268,2.912413143807258,2.481566103855519,2.1055462467052126,1.7843535723563377,1.5179880808088955,1.3064497720628858,1.149738646118307,1.0478547029751606,1.000797942633446,1.0085683650931634,1.0711659703543128,1.1885907584168944,1.360842729280908,1.5879218829463522,1.869828219413231,2.2065617386815393,2.5981224407512795,3.044510325622454,3.545725393295058,4.101767643769093,4.7126370770445645,5.378333693121463,6.098857491999794,6.874208473679561,7.704386638160756,8.58939198544339,9.529224515527446,10.523884228412937],[25.685069265600674,24.06688718925221,22.503532295705178,20.995004584959577,19.541304057015406,18.142430711872674,16.798384549531367,15.509165569991497,14.274773773253054,13.09520915931604,11.970471728180465,10.900561479846322,9.885478414313608,8.925222531582326,8.01979383165248,7.169192314524062,6.373417980197076,5.632470828671522,4.9463508599474,4.315058074024713,3.738592470903456,3.2169540505836296,2.7501428130652363,2.338158758348275,1.981001886432745,1.6786721973186467,1.4311696910059806,1.2384943674947473,1.100646226784945,1.0176252688765748,0.9894314937696368,1.0160649014641305,1.0975254919600566,1.2338132652574143,1.424928221356204,1.6708703602564252,1.9716396819580801,2.327236186461165,2.7376598737656814,3.2029107438716324,3.722988796779012,4.297894032487824,4.927626450998071,5.612186052309747,6.3515728364228545,7.1457868033373995,7.9948279530533695,8.898696285570779,9.857391800889612,10.870914499009881],[25.003035452082532,23.408341355529174,21.868474441777245,20.38343471082675,18.95322216267769,17.577836797330054,16.257278614783857,14.991547615039089,13.780643798095756,12.624567163953852,11.52331771261338,10.476895444074344,9.485300358336735,8.54853245540056,7.666591735265817,6.8394781979325066,6.067191843400627,5.349732671670179,4.687100682741162,4.079295876613581,3.52631825328743,3.0281678127627107,2.584844555039424,2.1963484801175674,1.8626795879971436,1.5838378786781513,1.3598233521605911,1.1906360084444636,1.0762758475297676,1.0167428694165033,1.0120370741046711,1.062158461594271,1.1671070318853027,1.3268827849777665,1.5414857208716621,1.8109158395669889,2.1351731410637504,2.514257625361941,2.948169292461563,3.4369081423626207,3.9804741750651065,4.578867390569024,5.232087788874378,5.940135369981159,6.703010133889372,7.520712080599023,8.3932412101101,9.320597522422615,10.302781017536555,11.339791695451925],[24.294344999359076,22.725069182540366,21.2106205485231,19.750999097307254,18.346204828892848,16.99623774327987,15.701097840468325,14.460785120458215,13.275299583249534,12.144641228842284,11.068810057236465,10.047806068432083,9.081629262429127,8.170279639227607,7.313757198827519,6.512061941228862,5.765193866431636,5.073152974435842,4.43593926524148,3.8535527388485518,3.3259933952570546,2.853261234466989,2.4353562564783564,2.0722784612911544,1.764027848905384,1.510604419321046,1.3120081725381398,1.1682391085566664,1.0792972273766244,1.0451825289980141,1.0658950134208358,1.1414346806450897,1.2718015306707757,1.4569955634978935,1.6970167791264434,1.991865177556424,2.3415407587878394,2.746043522820684,3.20537346965496,3.719530599290673,4.2885149117278125,4.912326406966383,5.590965085006392,6.324430945847827,7.112723989490693,7.955844215934997,8.853791625180728,9.806566217227898,10.814167992076493,11.876596949726517],[23.62940179987157,22.08480872953498,20.59504284199983,19.16010413726611,17.77999261533382,16.45470827620297,15.184251119873545,13.968621146345553,12.807818355618991,11.701842747693863,10.650694322570168,9.654373080247906,8.71287902072707,7.826212144007671,6.994372450089704,6.217359938973169,5.495174610658062,4.82781646514439,4.21528550243215,3.6575817225213427,3.1547051254119665,2.7066557111040215,2.3134334795975096,1.9750384308924287,1.6914705649887793,1.4627298818865622,1.2888163815857772,1.1697300640864248,1.1054709293885034,1.0960389774920143,1.141434208396957,1.2416566221033318,1.396706218611139,1.6065829979203776,1.8712869600310484,2.1908181049431503,2.565176432656687,2.9943619431716524,3.4783746364880495,4.017214512605881,4.610881571525142,5.259375813245835,5.962697237767965,6.720845845091521,7.5338216352165075,8.401624608142935,9.324254763870785,10.301712102400076,11.33399662373079,12.421108327862937],[23.066955201384488,21.543915701999705,20.07570338541635,18.662318251634435,17.303760300653952,16.000029532474894,14.751125947097274,13.557049544521083,12.417800324746326,11.333378287772994,10.303783433601103,9.329015762230641,8.40907527366161,7.543961967894011,6.733675844927845,5.97821690476311,5.277585147399806,4.631780572837935,4.0408031810774965,3.504652972118491,3.0233299459609166,2.5968341026047734,2.225165442050063,1.9083239642967835,1.646309669344936,1.4391225571945203,1.2867626278455366,1.1892298812979858,1.1465243175518662,1.1586459366071786,1.2255947384639228,1.347370723122099,1.5239738905817077,1.755404240842748,2.0416617739052207,2.3827464897691235,2.778658388434462,3.2293974699012282,3.734963734169426,4.295357181239061,4.910577811110124,5.580625623782617,6.305500619256549,7.085202797531906,7.919732158608696,8.809088702486925,9.753272429166575,10.752283338647667,11.806121430930185,12.914786706014132],[22.65201766856868,21.145486863207406,19.693783240647555,18.29690680088915,16.954857543932167,15.667635469776615,14.435240578422503,13.25767286986982,12.134932344118564,11.067019001168742,10.053932841020353,9.095673863673397,8.192242069127872,7.343637457383778,6.549860028441117,5.810909782299888,5.12678671896009,4.497490838421725,3.92302214068479,3.403380625749291,2.9385662936152217,2.528579144282584,2.173419177751379,1.8730863940216045,1.6275807930932624,1.4369023749663523,1.3010511396408742,1.2200270871168288,1.1938302173942148,1.2224605304730327,1.305918026353282,1.4442027050349642,1.6373145665180777,1.8852536108026239,2.1880198378886013,2.5456132477760103,2.958033840464854,3.425281615955126,3.9473565742468306,4.52425871533997,5.155988039234537,5.8425445459305365,6.583928235427973,7.380139107726836,8.231177162827132,9.137042400728863,10.097734821432024,11.113254424936619,12.18360121124264,13.308775180350096],[22.414557170714676,20.918202962894807,19.476675937876365,18.089976095659356,16.758103436243783,15.481057959629634,14.258839665816922,13.091448554805641,11.978884626595791,10.921147881187373,9.918238318580388,8.970155938774836,8.076900741770713,7.238472727568021,6.4548718961667655,5.726098247566941,5.052151781768546,4.433032498771584,3.8687403985760525,3.3592754811819563,2.904637746589291,2.504827194798057,2.1598438258082555,1.8696876396198852,1.6343586362329465,1.4538568156474398,1.3281821778633653,1.2573347228807235,1.2413144506995126,1.2801213613197342,1.3737554547413875,1.522216730964473,1.7255051899889902,1.9836208318149393,2.2965636564423213,2.664333663871133,3.0869308541013805,3.5643552271330563,4.096606782966164,4.683685521600707,5.325591443036679,6.0223245472740805,6.773884834312922,7.58027230415319,8.441486956794886,9.357528792238023,10.328397810482583,11.354094011528586,12.43461739537601,13.569967962024867],[22.368204957411404,20.875120080500857,19.43686238639173,18.053431875084048,16.724828546577793,15.451052400872967,14.232103437969577,13.06798165786762,11.958687060567092,10.904219646067993,9.904579414370334,8.959766365474103,8.069780499379304,7.2346218160859355,6.454290315594002,5.728785997903498,5.058108863014426,4.442258910926786,3.881236141640578,3.375040555155804,2.9236721514724606,2.5271309305905487,2.1854168925100694,1.8985300372310216,1.666470364753405,1.489237875077221,1.3668325682024687,1.299254444129149,1.2865035028572607,1.3285797443868044,1.42548316871778,1.5772137758501874,1.7837715657840272,2.0451565385192993,2.361368694056003,2.732408032394137,3.158274553533707,3.6389682574747044,4.174489144217135,4.764837213761001,5.410012466106294,6.1100149012530185,6.864844519201181,7.674501319950768,8.538985303501791,9.458296469854249,10.432434819008135,11.461400350963459,12.545193065720206,13.683812963278385],[22.508914136742206,21.01233983319443,19.570592712448093,18.183672774503187,16.851580019359712,15.574314447017672,14.351876057477057,13.184264850737879,12.071480826800133,11.013523985663815,10.010394327328934,9.062091851795484,8.168616559063464,7.329968449132875,6.546147522003721,5.817153777675998,5.142987216149707,4.523647837424846,3.9591356415014176,3.4494506283794233,2.9945927980588603,2.5945621505397285,2.24935868582203,1.9589824039057613,1.7234333047909252,1.542711388477521,1.4168166549655485,1.345749104255009,1.3295087363459008,1.3680955512382247,1.4615095489319798,1.6097507294271676,1.8128190927237877,2.070714638821839,2.3834373677213225,2.7509872794222376,3.173364373924587,3.6505686512283653,4.182600111333575,4.769458754240221,5.411144579948294,6.1076575884577995,6.8589977797687425,7.6651651538811105,8.526159710794913,9.441981450510148,10.412630373026813,11.438106478344919,12.518409766464446,13.653540237385405],[22.814208178516424,21.308199660691837,19.857018325668687,18.460664173446965,17.119137204026668,15.83243741740781,14.600564813590381,13.423519392574386,12.301301154359823,11.233910098946692,10.221346226334992,9.263609536524724,8.36070002951589,7.5126177053084815,6.7193625639025125,5.980934605297973,5.297333829494864,4.668560236493187,4.094613826292943,3.575494598894132,3.111202554296752,2.701737692500804,2.3471000135062883,2.0472895173132035,1.8023062039215505,1.6121500733313299,1.476821125542541,1.3963193605551847,1.3706447783692597,1.3997973789847673,1.483777162401706,1.6225841286200768,1.8162182776398799,2.0646796094611157,2.3679681240837827,2.7260838215078804,3.139026701733414,3.606796764760375,4.1293940105887685,4.706818439218598,5.3390700506498545,6.026148844882543,6.768054821916668,7.564787981752222,8.416348324389203,9.322735849827627,10.283950558067476,11.299992449108762,12.370861522951472,13.496557779595618],[23.24402365239605,21.723996204871376,20.258795940148136,18.84842285822633,17.492876959105956,16.19215824278701,14.946266709269494,13.755202358553415,12.618965190638765,11.537555205525548,10.510972403213763,9.539216783703411,8.622288346994491,7.7601870930870005,6.952913021980944,6.20046613367632,5.502846428173126,4.860053905471365,4.2720885655710354,3.738950408472139,3.2606394341746747,2.8371556426786406,2.4684990339840396,2.15466960809087,1.8956673649991318,1.6914923047088262,1.5421444272199523,1.4476237325325108,1.407930220646501,1.4230638915619231,1.493024745278777,1.6178127817970627,1.7974280011167814,2.031870403237931,2.3211399881605135,2.6652367558845262,3.064160706409974,3.51791183973685,4.026490155865158,4.589895654794902,5.208128336526074,5.881188201058677,6.609075248392719,7.391789478528187,8.229330891465084,9.12169948720342,10.068895265743182,11.070918227084384,12.127768371227011,13.23944569817107],[23.743991833744676,22.20710067982911,20.725036708714967,19.297799920402262,17.925390314890986,16.607807892181143,15.345052652272727,14.137124595165748,12.984023720860202,11.885750029356085,10.842303520653404,9.853684194752152,8.919892051652331,8.040927091353943,7.216789313856989,6.447478719161464,5.732995307267373,5.073339078174712,4.468510031883483,3.9185081683936875,3.4233334877053245,2.982985989818392,2.5974656747328924,2.266772542448823,1.9909065929661862,1.7698678262849816,1.603656242405209,1.492271841326869,1.43571462304996,1.433984587574483,1.4870817349004382,1.5950060650278253,1.7577575779566443,1.9753362736868962,2.2477421522185788,2.574975213551693,2.957035457686241,3.39392288462222,3.885637494359629,4.4321792868984735,5.033548262238747,5.689744420380451,6.400767761323594,7.166618285068162,7.987295991614161,8.8628008809616,9.793132953110462,10.778292208060765,11.818278645812494,12.913092266365654],[24.25136527598889,22.696690576114268,21.196843059041072,19.751822724769312,18.361629573298984,17.02626360463009,15.74572481876262,14.52001321569659,13.349128795431985,12.233071557968815,11.17184150330708,10.165438631446778,9.213862942387902,8.31711443613046,7.475193112674451,6.688098972019875,5.955832014166729,5.278392239115014,4.655779646864733,4.087994237415885,3.575036010768468,3.1169049669224815,2.7136011058779292,2.365124427634807,2.0714749321931163,1.832652619552858,1.6486574897140318,1.5194895426766384,1.4451487784406762,1.4256351970061458,1.4609487983730474,1.5510895825413815,1.696057549511147,1.8958526992823446,2.1504750318549744,2.4599245472290354,2.82420124540453,3.243305126381455,3.7172361901598108,4.245994436739602,4.829579866120823,5.467992478303473,6.161232273287562,6.909299251073077,7.712193411660022,8.569914755048407,9.482463281238218,10.449838990229466,11.472041882022141,12.549071956616247],[24.70292979967692,23.131458462827222,21.614814308778957,20.152997337532124,18.746007549086723,17.393844943442755,16.09650952060022,14.854001280559114,13.666320223319442,12.533466348881198,11.455439657244389,10.432240148409015,9.463867822375068,8.550322679142553,7.691604718711475,6.887713941081826,6.138650346253608,5.444413934226823,4.805004705001468,4.220422658577549,3.69066779495506,3.215740114134002,2.7956396161143773,2.4303663008961838,2.1199201684794216,1.8643012188640917,1.6635094520501938,1.5175448680377284,1.4264074668266944,1.3900972484170924,1.4086142128089223,1.4819583600021844,1.6101296899968782,1.7931282027930042,2.0309538983905626,2.323606776789551,2.6710868379899746,3.073394081991827,3.5305285087951113,4.042490118399832,4.609278910805979,5.230894886013559,5.907338044022576,6.638608384833019,7.424705908444894,8.265630614858205,9.161382504072947,10.111961576089122,11.117367830906726,12.177601268525759],[25.043497650274603,23.45791396413864,21.9271574608041,20.451228140270995,19.03012600253933,17.663851047609086,16.35240327548028,15.095782686152907,13.893989279626961,12.747023055902446,11.654884014979368,10.61757215685772,9.635087481537504,8.707429989018719,7.83459967930137,7.016596552385449,6.253420608270959,5.545071846957903,4.891550268446277,4.292855872736087,3.748988659827327,3.2599486297199984,2.8257357824141027,2.446350117909638,2.121791636206605,1.852060337305004,1.6371562212048352,1.4770792879060988,1.3718295374087939,1.3214069697129205,1.3258115848184795,1.3850433827254705,1.4991023634338934,1.6679885269437484,1.8917018732550355,2.1702424023677533,2.5036101142819054,2.891805008997488,3.3348270865145007,3.83267634683295,4.385352789952827,4.992856415874135,5.6551872245968795,6.372345216121053,7.1443303904466555,7.971142747573698,8.852782287502166,9.789249010232073,10.780542915763403,11.826664004096168],[25.233253710969752,23.637569137093166,22.096711746018016,20.610681537744295,19.179478512272002,17.803102669601145,16.481554009731724,15.214832532663726,14.002938238397164,12.84587112693203,11.743631198268334,10.696218452406068,9.703632889345233,8.765874509085828,7.8829433116278596,7.054839296971321,6.281562465116214,5.563112816062539,4.899490349810295,4.290695066359485,3.7367269657101074,3.23758604786216,2.7932723128156454,2.403785760570562,2.0691263911269098,1.7892942044846905,1.5642892006439029,1.3941113796045483,1.2787607413666247,1.2182372859301331,1.2125410132950734,1.2616719234614457,1.3656300164292499,1.5244152921984864,1.7380277507691546,2.006467392141255,2.329734216314788,2.7078282232897513,3.1407494130661457,3.6284977856439764,4.171073341023234,4.768476079203925,5.420706000186051,6.127763103969605,6.889647390554589,7.706358859941013,8.577897512128862,9.504263347118153,10.485456364908863,11.521476565501008],[25.25245553702754,23.651522636772825,22.105416919319534,20.614138384667676,19.177687032817254,17.796062863768263,16.469265877520705,15.197296074074577,13.980153453429885,12.817838015586618,11.710349760544789,10.657688688304392,9.659854798865423,8.716848092227886,7.828668568391785,6.9953162273571134,6.216791069123872,5.493093093692064,4.824222301061687,4.210178691232745,3.650962264205234,3.1465730199791535,2.697010958554507,2.30227607993129,1.9623683841095052,1.6772878710891526,1.4470345408702325,1.2716083934527445,1.1510094288366879,1.0852376470220637,1.074293048008871,1.1181756317971105,1.2168853983867818,1.3704223477778852,1.5787864799704205,1.8419777949643872,2.159996292759788,2.532841973356618,2.9605148367548795,3.4430148829545777,3.980342111955703,4.572496523758259,5.219478118362254,5.921286895767674,6.677922855974526,7.489385998982816,8.355676324792533,9.276793833403689,10.252738524816266,11.283510399030279],[25.1026858117931,23.50165085283186,21.95544307667205,20.464062483313676,19.02750907275673,17.645782845001218,16.31888380004714,15.046811937894487,13.82956725854327,12.667149761993482,11.55955944824513,10.50679631729821,9.50886036915272,8.565751603808662,7.677470021266037,6.844015621524844,6.065388404585082,5.341588370446752,4.672615519109854,4.058469850574389,3.4991513648403556,2.9946600619077537,2.544995941776585,2.150159004446846,1.8101492499185399,1.5249666781916655,1.294611289266223,1.1190830831422136,0.9983820598196348,0.9325082192984883,0.9214615615787739,0.9652420866604913,1.063849794543641,1.2172846852282222,1.425546758714236,1.6886360150016804,2.0065524540905595,2.3792960759808675,2.806866880672607,3.289264868165783,3.8264900384603866,4.4185423915564215,5.065421927453893,5.767128646152792,6.523662547653123,7.335023631954892,8.201211899058086,9.122227348962719,10.098069981668775,11.128739797176266],[24.804763684332737,23.20851625226584,21.66709600300037,20.180502936536335,18.74873705287373,17.37179835201256,16.04968683395282,14.782402498694513,13.569945346237636,12.412315376582193,11.309512589728179,10.261536985675601,9.268388564424452,8.330067325974737,7.44657327032645,6.617906397479599,5.844066707434178,5.125054200190188,4.4608688757476305,3.8515107341065087,3.296979775266817,2.7972759992285554,2.3523994059917275,1.96234999555633,1.6271277679223648,1.3467327230898316,1.1211648610587304,0.9504241818290616,0.8345106854008244,0.773424371774019,0.7671652409486457,0.8157332929247042,0.919128527702195,1.0773509452811176,1.290400545661472,1.558277328843258,1.8809812948264781,2.258512443611127,2.690870775197208,3.178056289584725,3.720068986773669,4.316908866764046,4.968575929555859,5.675070175149099,6.436391603543771,7.252540214739881,8.123516008737415,9.04931898553639,10.029949145136786,11.065406487538619],[24.394214070810655,22.806889805600505,21.274392723191795,19.796722823584513,18.373880106778664,17.005864572774247,15.692676221571261,14.434315053169707,13.230781067569584,12.082074264770894,10.988194644773635,9.94914220757781,8.964916953183417,8.035518881590454,7.160947992798926,6.341204286808826,5.57628776362016,4.866198423232925,4.210936265647122,3.610501290862753,3.0648934988798153,2.5741128896983088,2.1381594633182353,1.7570332197395924,1.4307341589623812,1.159262280986602,0.942617585812255,0.7808000734393408,0.6738097438678577,0.6216465970978068,0.6243106331291877,0.6818018519620006,0.7941202535962455,0.9612658380319226,1.1832386052690314,1.4600385553075717,1.7916656881475455,2.1781200037889494,2.6194015022317845,3.1155101834760552,3.6664460475217537,4.272209094368884,4.932799324017452,5.648216736467447,6.418461331718873,7.243533109771737,8.123432070626025,9.058158214281756,10.047711540738907,11.092092049997492],[23.91561000288471,22.340194826777108,20.819606833470935,19.35384602296619,17.942912395262884,16.586805950361008,15.285526688260564,14.039074608961553,12.84744971246397,11.710651998767819,10.628681467873104,9.601538119779821,8.629221954487967,7.711732971997544,6.849071172308555,6.0412365554209995,5.288229121334874,4.590048870050181,3.946695801566918,3.35816991588509,2.824471213004693,2.3455996929257275,1.921555355648195,1.5523382011720928,1.2379482294974227,0.9783854406241846,0.7736498345523787,0.6237414112820052,0.5286601708130632,0.48840611314555304,0.5029792382794749,0.5723795462148288,0.6966070369516146,0.8756617104898325,1.1095435668294826,1.3982526059705633,1.7417888279130784,2.140152232657023,2.5933428202023996,3.1013605905492114,3.66420554369745,4.281877679647122,4.954376998398231,5.681703499950766,6.463857184304732,7.300838051460136,8.192646101416967,9.139281334175235,10.140743749734932,11.197033348096058],[23.417043177671644,21.85511505368358,20.34801411249694,18.89574035411173,17.498293778527955,16.155674385745616,14.867882175764704,13.634917148585227,12.45677930420718,11.333468642630562,10.26498516385538,9.251328867881632,8.292499754709313,7.388497824338425,6.539323076768971,5.744975512000949,5.005455130034358,4.320761930869199,3.6908959145054707,3.1158570809431776,2.595645430182315,2.130260962222884,1.7197036770648848,1.3639735747083175,1.0630706551531819,0.816994918399478,0.6257463644472064,0.48932499329636725,0.4077308049469596,0.38096379939898395,0.40902397665244006,0.49191133670732823,0.6296258795636486,0.822167605221401,1.0695365136805852,1.3717326049412002,1.7287558790032502,2.140606335866729,2.6072839755316393,3.128788797997986,3.70512080326576,4.3362799913349654,5.022266362205608,5.763079915877677,6.5587206523511785,7.409188571626117,8.314483673702483,9.274605958580286,10.289555426259515,11.359332076740175],[22.945532821072575,21.39716158082269,19.903617523374244,18.464900648727227,17.081010956881645,15.751948447837494,14.477713121594777,13.25830497815349,12.093724017513631,10.983970239675205,9.929043644638217,8.928944232402658,7.983672002968529,7.0932269563358314,6.257609092504568,5.476818411474737,4.750854913246337,4.079718597819368,3.4634094651938305,2.9019275153697275,2.3952727483470553,1.9434451641258148,1.546444762706007,1.2042715440876302,0.9169255082706851,0.684406655255172,0.506714985041091,0.3838504976284426,0.31581319301722566,0.30260307120744057,0.34422013219908737,0.4406643759921663,0.5919358025866773,0.7980344119826204,1.0589602041799955,1.3747131791788012,1.7452933369790415,2.170700677580711,2.6509352009838123,3.185996907188349,3.775885796194314,4.42060186800171,5.120145122610544,5.8745155600208045,6.683713180232496,7.547737983245626,8.46658996906018,9.440269137676175,10.468775489093595,11.552109023312445],[22.54358374214121,21.007392525546965,19.526028491754158,18.09949164076278,16.727781972572835,15.410899487184324,14.148844184597241,12.941616064811594,11.789215127827374,10.691641373644586,9.648894802263236,8.660975413683312,7.727883207904824,6.849618184927764,6.02618034475214,5.257569687377947,4.543786212805185,3.884829921033854,3.2807008120639556,2.731398885895491,2.236924142528457,1.797276581962855,1.4124562041986852,1.0824630092359464,0.8072969970746396,0.586958167714765,0.4214465211563225,0.31076205739931223,0.2549047764437335,0.25387467828958676,0.30767176293687193,0.4162960303855891,0.5797474806357384,0.7980261136873198,1.071131929540333,1.3990649281947771,1.781825109650656,2.219412473907964,2.7118270209667035,3.259068750826878,3.8611376634884818,4.518033758951517,5.229757037215988,5.996307498281887,6.817685142149217,7.6938899688179845,8.624921978288178,9.610781170559811,10.65146754563287,11.746981103507359],[22.24661820515059,20.71999333671768,19.248195651086203,17.831225148256156,16.469081828227537,15.161765691000358,13.909276736574606,12.711614964950291,11.5687803761274,10.480772970105946,9.447592746885924,8.469239706467333,7.545713848850175,6.677015174034447,5.863143682020153,5.1040993728072905,4.399882246395859,3.7504923027858594,3.1559295419772924,2.616193963970159,2.131285568764456,1.701204356360184,1.3259503267573456,1.005523479955938,0.7399238159559621,0.5291513347574185,0.37320603636030675,0.2720879207646274,0.2257969879703796,0.23433323797756386,0.29769667078617984,0.415887286396228,0.5889050848077081,0.8167500660206203,1.0994222300349648,1.4369215768507395,1.8292481064679493,2.2764018188865878,2.778382714106658,3.3351907921281647,3.9468260529510983,4.613288496575465,5.334578123001267,6.110694932228497,6.941638924257159,7.827410099087257,8.76800845671878,9.763433997151745,10.813686720386134,11.918766626421952],[22.08082386038941,20.560245237200768,19.094493796813563,17.683569539227786,16.327472464443442,15.026202572460528,13.77975986327905,12.588144336899001,11.451355993320385,10.3693948325432,9.34226085456745,8.36995405939313,7.452474447020242,6.589822017448784,5.781996770678761,5.02899870671017,4.330827825543009,3.68748412717728,3.0989676116129825,2.5652782788501196,2.0864161288886875,1.662381161728687,1.293173377370119,0.978792775812982,0.7192393570572768,0.5145131211030036,0.36461406795016266,0.269542197598754,0.22929751004877688,0.24388000530023182,0.31328968335311835,0.4375265442074372,0.616590587863188,0.8504818143203712,1.1392002235789858,1.4827458156390316,1.881118590500512,2.334318548163421,2.8423456886277623,3.4052000118935393,4.022881517960744,4.695390206829379,5.422726078499452,6.204889132970954,7.041879370243884,7.933696790318254,8.880341393194051,9.881813178871285,10.938112147349944,12.049238298630035],[22.06112151621621,20.54257423744866,19.078854141482548,17.66996122831786,16.315895497954607,15.016656950392784,13.772245585632396,12.58266140367344,11.447904404515915,10.36797458815982,9.342871954605158,8.37259650385193,7.457148235900132,6.5965271507497665,5.790733248400833,5.039766528853333,4.3436269921072626,3.702314638162623,3.115829467019417,2.5841714786776446,2.107340673137303,1.6853370503983924,1.3181606104609151,1.0058113533248687,0.7482892789902541,0.5455943874570716,0.397726678725321,0.30468615279500283,0.26647280966611625,0.2830866493386618,0.35452767181263883,0.48079587708804816,0.6618912651648895,0.897813836043163,1.1885635897228686,1.5341405262040049,1.934544645486576,2.3897759475705755,2.8998344324560072,3.4647201001428742,4.08443295063117,4.758972983920896,5.488340200012061,6.272534598904651,7.111556180598672,8.005404945094133,8.954080892391017,9.957584022489344,11.015914335389093,12.129071831090274],[22.189343746406795,20.66876481724732,19.203013070889273,17.792088507332664,16.435991126577484,15.134720928623738,13.888277913471423,12.696662081120538,11.559873431571086,10.477911964823063,9.450777680876477,8.478470579731322,7.5609906613875975,6.698337925845306,5.890512373104446,5.137514003165018,4.439342816027021,3.7959988116904566,3.207481990155324,2.6737923514216253,2.1949298954893575,1.7708946223585205,1.401686532029117,1.0873056245011443,0.8277518997746033,0.6230253578494944,0.47312599872581756,0.3780538224035731,0.3378088288827602,0.3523910181633793,0.42180039024543003,0.5460369451289131,0.7251006828138281,0.9589916033001755,1.247709706587954,1.5912549926771642,1.989627461567809,2.442827113259882,2.9508539477533873,3.5137079650483285,4.1313891651446975,4.803897548042499,5.531233113741736,6.3133958622424,7.150385793544495,8.042202907648031,8.98884720455299,9.990318684259389,11.046617346767212,12.157743192076468]],"type":"surface"}],                        {"autosize":true,"height":700,"scene":{"aspectmode":"manual","aspectratio":{"x":1,"y":1,"z":0.7},"camera":{"eye":{"x":-1.7428,"y":1.0707,"z":0.71},"up":{"x":0,"y":0,"z":1}},"xaxis":{"gridcolor":"rgb(255, 255, 255)","showbackground":true,"zerolinecolor":"rgb(255, 255, 255)"},"yaxis":{"gridcolor":"rgb(255, 255, 255)","showbackground":true,"zerolinecolor":"rgb(255, 255, 255)"},"zaxis":{"gridcolor":"rgb(255, 255, 255)","showbackground":true,"zerolinecolor":"rgb(255, 255, 255)"}},"template":{"data":{"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"Gradient Descent"},"width":800},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('876062a6-7410-420f-b994-0db1f1acd2ce');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
import plotly
import plotly.graph_objs as go
```


```python
def contour_plot(title, theta_history, loss_function, model, x, y):
    """
    The function takes the following as argument:
        theta_history: a (N, 2) array of theta history
        loss: a list or array of loss value
        loss_function: for example, l2_loss
        model: for example, sin_model
        x: the original x input
        y: the original y output
    """
    theta_1_series = theta_history[:,0] # a list or array of theta_1 value
    theta_2_series = theta_history[:,1] # a list or array of theta_2 value

    ## In the following block of code, we generate the z value
    ## across a 2D grid
    theta1_s = np.linspace(np.min(theta_1_series) - 0.1, np.max(theta_1_series) + 0.1)
    theta2_s = np.linspace(np.min(theta_2_series) - 0.1, np.max(theta_2_series) + 0.1)

    x_s, y_s = np.meshgrid(theta1_s, theta2_s)
    data = np.stack([x_s.flatten(), y_s.flatten()]).T
    ls = []
    for theta1, theta2 in data:
        l = loss_function(model(x, np.array([theta1, theta2])), y)
        ls.append(l)
    z = np.array(ls).reshape(50, 50)
    
    # Create trace of theta point
    # Create the contour 
    theta_points = go.Scatter(name="theta Values", 
                              x=theta_1_series, 
                              y=theta_2_series,
                              mode="lines+markers")
    lr_loss_contours = go.Contour(x=theta1_s, 
                                  y=theta2_s, 
                                  z=z, 
                                  colorscale='Viridis', reversescale=True)

    plotly.offline.iplot(go.Figure(data=[lr_loss_contours, theta_points], layout={'title': title}))
```


```python
contour_plot('Gradient Descent with Static Learning Rate', thetas, mean_squared_error, sin_model, df["x"], df["y"])
```


<div>                            <div id="dd255593-17d3-45f1-b259-69fc15746238" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("dd255593-17d3-45f1-b259-69fc15746238")) {                    Plotly.newPlot(                        "dd255593-17d3-45f1-b259-69fc15746238",                        [{"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"reversescale":true,"x":[-0.1,-0.042835562147354816,0.014328875705290373,0.07149331355793556,0.12865775141058075,0.18582218926322594,0.24298662711587113,0.3001510649685163,0.35731550282116153,0.4144799406738068,0.4716443785264519,0.528808816379097,0.5859732542317423,0.6431376920843875,0.7003021299370327,0.7574665677896778,0.814631005642323,0.8717954434949683,0.9289598813476135,0.9861243192002586,1.0432887570529037,1.100453194905549,1.157617632758194,1.2147820706108392,1.2719465084634844,1.3291109463161297,1.386275384168775,1.44343982202142,1.5006042598740652,1.5577686977267104,1.6149331355793555,1.6720975734320007,1.729262011284646,1.7864264491372912,1.8435908869899365,1.9007553248425815,1.957919762695227,2.015084200547872,2.072248638400517,2.1294130762531625,2.1865775141058075,2.2437419519584525,2.300906389811098,2.358070827663743,2.415235265516388,2.4723997033690335,2.5295641412216785,2.586728579074324,2.643893016926969,2.701057454779614],"y":[-0.1,-0.03320906678132665,0.0335818664373467,0.10037279965602006,0.1671637328746934,0.23395466609336676,0.30074559931204015,0.3675365325307135,0.43432746574938685,0.5011183989680602,0.5679093321867336,0.6347002654054069,0.7014911986240803,0.7682821318427536,0.835073065061427,0.9018639982801003,0.9686549314987737,1.035445864717447,1.1022367979361203,1.1690277311547936,1.235818664373467,1.3026095975921403,1.3694005308108137,1.436191464029487,1.5029823972481604,1.5697733304668338,1.6365642636855071,1.7033551969041805,1.7701461301228538,1.8369370633415272,1.9037279965602005,1.970518929778874,2.0373098629975472,2.1041007962162204,2.170891729434894,2.2376826626535675,2.3044735958722407,2.371264529090914,2.4380554623095874,2.504846395528261,2.571637328746934,2.638428261965607,2.705219195184281,2.7720101284029544,2.8388010616216275,2.9055919948403006,2.972382928058974,3.0391738612776478,3.105964794496321,3.172755727714994],"z":[[26.32282044860889,24.673923664437837,23.079854063068225,21.54061164450004,20.05619640873328,18.62660835576796,17.251847485604074,15.931913798241615,14.66680729368059,13.456527971920991,12.30107583296283,11.200450876806102,10.154653103450801,9.163682512896935,8.227539105144503,7.3462228801934994,6.519733838043929,5.748071978695791,5.0312373021490835,4.36922980840381,3.762049497459968,3.209696369317557,2.7121704239765796,2.2694716614370325,1.881600081698917,1.5485556847622342,1.2703384706269831,1.0469484392931647,0.8783855907607775,0.7646499250298221,0.7057414421002989,0.7016601419722076,0.7524060246455484,0.8579790901203213,1.0183793383965258,1.233606769474162,1.5036613833532317,1.8285431800337313,2.2082521595156623,2.6427883217990287,3.132151666883824,3.6763421947700508,4.275359905457713,4.929204798946804,5.637876875237326,6.401376134329285,7.219702576222671,8.092856200917495,9.020837008413745,10.003644998711424],[24.470263957582933,22.883123381205607,21.350809987629717,19.873323776855255,18.45066474888223,17.082832903710635,15.769828241340468,14.511650761771737,13.308300465004434,12.159777351038565,11.066081419874132,10.02721267151113,9.043171105949552,8.113956723189412,7.239569523230705,6.420009506073431,5.655276671717585,4.945371020163171,4.2902925514101895,3.690041265458643,3.144617162308527,2.6540202419598415,2.2182505044125898,1.8373079496667681,1.5111925777223787,1.2399043885794216,1.023443382237896,0.8618095586978033,0.755002917959142,0.7030234600219125,0.7058711848861152,0.7635460925517495,0.8760481830188163,1.0433774562873148,1.2655339123572453,1.5425175512286073,1.874328372901403,2.260966377375628,2.7024315646512846,3.1987239347283776,3.7498434876068982,4.355790223286849,5.016564141768239,5.732165243051056,6.5025935271353035,7.327848994020988,8.207931643708099,9.142841476196649,10.132578491486623,11.177142689578032],[22.626388681683842,21.103128552230952,19.63469560557949,18.221089841729462,16.862311260680865,15.558359862433706,14.30923564698797,13.114938614343673,11.975468764500803,10.890826097459364,9.861010613219362,8.88602231178079,7.965861193143651,7.10052725730794,6.290020504273665,5.534340934040821,4.8334885466094075,4.187463341979425,3.5962653201508767,3.0598944811237616,2.578350824898077,2.151634351473824,1.7797450608510035,1.4626829530296142,1.2004480280096566,0.9930402857911311,0.8404597263740378,0.7427063497583766,0.6997801559441471,0.7116811449313496,0.7784093167199839,0.8999646713100503,1.0763472087015489,1.3075569288944793,1.5935938318888418,1.9344579176846348,2.3301491862818633,2.78066763768052,3.2860132718806088,3.8461860888821335,4.4611860886850865,5.131013271289469,5.855667636695291,6.635149184902539,7.469457915911216,8.358593829721336,9.302556926332878,10.301347205745861,11.354964667960267,12.463409312976104],[20.916018343290457,19.454490735594117,18.04779031069921,16.695917068605727,15.398871009313677,14.156652132823062,12.969260439133878,11.836695928246126,10.758958600159806,9.736048454874915,8.767965492391461,7.854709712709438,6.996281115828845,6.192679701749685,5.443905470471957,4.749958421995661,4.110838556320796,3.526545873447363,2.9970803733753626,2.5224420561047953,2.1026309216356593,1.7376469699679542,1.4274902011016821,1.172160615036841,0.9716582117734318,0.8259829913114548,0.7351349536509099,0.6991140987917969,0.7179204267341159,0.7915539374778668,0.9200146310230491,1.103302507369664,1.341417566517711,1.63435980846719,1.9821292332181009,2.3847258407704417,2.8421496311242187,3.354400604279424,3.921478760236061,4.543384098994133,5.220116620553635,5.951676324914566,6.738063212076936,7.5792772820407315,8.47531853480596,9.426186970372624,10.431882588740715,11.492405389910248,12.6077553738812,13.77793254065359],[19.440563686353922,18.03451935465846,16.68330220576443,15.386912239671826,14.145349456380657,12.958613855890922,11.82670543820262,10.749624203315745,9.727370151230305,8.759943281946295,7.8473435954637205,6.989571091782576,6.186625770902863,5.438507632824581,4.7452166775477345,4.106752905072317,3.523116315398332,2.994306908525778,2.520324684454657,2.101169643184969,1.7368417847167121,1.4273411090498866,1.172667616184494,0.9728213061205324,0.8278021788580028,0.7376102343969054,0.7022454727372398,0.7217078938790062,0.7959974978222045,0.925114284566835,1.1090582541128968,1.347829406460391,1.6414277416093175,1.989853259559676,2.3931059603114666,2.851185843864687,3.3640929102193438,3.931827159375428,4.554388591332945,5.231777206091897,5.963993003652276,6.751035984014088,7.592906147177336,8.489603493142013,9.441128021908119,10.447479733475664,11.508658627844637,12.624664705015048,13.79549796498688,15.021158407760145],[18.274084928922743,16.913669165164208,15.608080584207103,14.357319186051432,13.161384970697194,12.020277938144387,10.933998088393015,9.902545421443074,8.92591993729456,8.004121635947481,7.137150517401836,6.325006581657622,5.567689828714837,4.865200258573485,4.217537871233568,3.624702666695081,3.0866946449580253,2.6035138060224017,2.1751601498882103,1.8016336765554521,1.4829343860241249,1.2190622782942295,1.0100173533657661,0.8557996112387347,0.7564090519131348,0.7118456753889669,0.7221094816662315,0.7872004707449275,0.9071186426250555,1.081863997306616,1.3114365347896075,1.5958362550740315,1.9350631581598876,2.329117244047176,2.7779985127358966,3.2817069642260464,3.8402425985176336,4.453605415610647,5.121795415505094,5.844812598200976,6.622656963698285,7.455328511997025,8.342827243097206,9.285153156998808,10.282306253701847,11.334286533206324,12.441093995512222,13.602728640619564,14.819190468528328,16.090479479238525],[17.464395964087338,16.136926610201055,14.864284439116206,13.646469450832791,12.483481645350805,11.375321022670253,10.321987582791134,9.323481325713447,8.379802251437189,7.490950359962362,6.656925651288971,5.877728125417012,5.153357782346482,4.483814622077384,3.8690986446097213,3.3092098499434885,2.804148238078687,2.3539138090153178,1.9585065627533804,1.6179264992928764,1.3321736186338036,1.101247920776162,0.925149405719953,0.8038780734651757,0.73743392401183,0.7258169573599166,0.7690271735094353,0.8670645724603854,1.019929154212768,1.2276209187665823,1.490139866121828,1.8074859962785066,2.1796593092366168,2.606659804996159,3.088487483557134,3.6251423449195377,4.216624389083379,4.862933616048647,5.564070025815347,6.320033618383484,7.1308243937530476,7.996442351924043,8.916887492896477,9.892159816670336,10.922259323245626,12.007186012622357,13.146939884800512,14.341520939780105,15.590929177561126,16.89516459814357],[17.036790822248832,15.727751403507343,14.47353916756729,13.274154114428672,12.129596244091482,11.039865556555723,10.0049620518214,9.024885729888508,8.099636590757042,7.229214634427014,6.413619860898417,5.652852270171253,4.946911862245519,4.295798637121216,3.699512594798347,3.1580537352769102,2.671422058556904,2.2396175646383294,1.8626402535211875,1.5404901252054783,1.2731671796912007,1.0606714169783544,0.9030028370669405,0.8001614399569582,0.7521472256484079,0.7589601941412896,0.8206003454356032,0.9370676795313485,1.1083621964285264,1.334483896127136,1.6154327786271763,1.95120884392865,2.3418120920315557,2.7872425229358933,3.2875001366416625,3.842584933148863,4.452496912457499,5.117236074567562,5.836802419479056,6.611195947191989,7.440416657706348,8.324464551022137,9.263339627139366,10.25704188605802,11.305571327778107,12.408927952299633,13.56711175962258,14.78012274974697,16.047960922672786,17.370626278400028],[16.997662509909667,15.691825366734296,14.440815406360361,13.244632628787862,12.103277034016788,11.016748622047153,9.985047392878945,9.00817334651217,8.086126482946826,7.218906802182916,6.406514304220436,5.648948989059391,4.946210856699775,4.29829990714159,3.70521614038484,3.166959556429521,2.683530155275634,2.254927936923177,1.881152901372154,1.5622050486225634,1.298084378674404,1.0887908915276763,0.9343245871823808,0.834685465638517,0.7898735268960848,0.799888770955085,0.8647311978155172,0.9844008074773808,1.158897599940677,1.388221575205405,1.672372733271564,2.0113510741391556,2.40515659780818,2.853789304278636,3.357249193550524,3.915536265623843,4.528650520498597,5.196591958174778,5.919360578652392,6.696956381931442,7.52937936801192,8.416629536893828,9.358706888577178,10.35561142306195,11.407343140348154,12.513902040435797,13.675288123324863,14.891501389015374,16.162541837507302,17.48840946880067],[17.336054611229248,16.018638513807375,14.756049599186936,13.548287867367925,12.39535331835035,11.297245952134205,10.253965768719494,9.265512768106213,8.331886950294365,7.453088315283946,6.629116863074963,5.85997259366741,5.1456555070612895,4.486165603256599,3.8815028822533426,3.3316673440515183,2.836658988651125,2.3964778160521636,2.011123826254634,1.6805970192585376,1.4048973950638728,1.1840249536706393,1.0179796950788378,0.9067616192884683,0.8503707262995306,0.8488070161120248,0.9020704887259512,1.010161144141309,1.1730789823580996,1.3908240033763217,1.663396207195975,1.9907955938170616,2.37302216323958,2.8100759154635297,3.3019568504889127,3.848664968315725,4.4502002689439735,5.106562752373649,5.817752418604757,6.583769267637302,7.404613299471272,8.280284514106675,9.210782911543516,10.196108491781784,11.236261254821484,12.33124120066262,13.481048329305183,14.685682640749185,15.94514413499461,17.259432812041467],[18.022678326192622,16.68044325223466,15.393035361078134,14.160454652723041,12.982701127169381,11.859774784417148,10.791675624466349,9.778403647316981,8.819958852969044,7.9163412414225425,7.06755081267747,6.273587566733833,5.534451503591623,4.850142623250847,4.220660925711504,3.646006410973594,3.126179079037114,2.661178929902065,2.251005963568449,1.8956601800362662,1.5951415793055148,1.3494501613761944,1.1585859262483067,1.0225488739218505,0.9413390043968264,0.914956317673234,0.9434008137510737,1.0266724926303448,1.1647713543110485,1.3576973987931844,1.605450626076751,1.9080310361617505,2.265438629048182,2.677673404736046,3.1447353632253416,3.6666245045160677,4.243340828608229,4.874884335501818,5.56125502519684,6.302452897693299,7.098477952991183,7.9493301910905,8.855009611991255,9.815516215693433,10.830850002197046,11.901010971502098,13.025999123608573,14.205814458516489,15.440456976225827,16.7299266767366],[19.00752568169107,17.629714314180863,16.306730129472093,15.038573127564748,13.825243308458836,12.666740672154356,11.56306521865131,10.514216947949695,9.52019586004951,8.581001954950759,7.69663523265344,6.867095693157556,6.092383336463099,5.372498162570074,4.707440171478484,4.097209363188325,3.5418057376995966,3.0412292950123003,2.595480035126436,2.204557958042005,1.8684630637590058,1.5871953522774378,1.3607548235973022,1.189141477718598,1.0723553146413258,1.0103963343654854,1.0032645368910773,1.0509599222181008,1.1534824903465566,1.3108322412764442,1.523009175007763,1.7900132915405147,2.111844590874699,2.4885030730103144,2.919988737947362,3.40630158568584,3.9474416162257544,4.543408829567095,5.194203225709869,5.899824804654079,6.660273566399716,7.475549510946784,8.345652638295292,9.270582948445224,10.250340441396586,11.28492511714939,12.374336975703619,13.518576017059285,14.717642241216378,15.971535648174898],[20.21827677269866,18.797327118011786,17.431204646126343,16.119909357042335,14.863441250759758,13.661800327278616,12.514986586598898,11.42300002872062,10.385840653643767,9.403508461368348,8.476003451894364,7.603325625221813,6.78547498135069,6.022451520280999,5.3142552420127425,4.660886146545917,4.062344233880524,3.5186295040165607,3.029741956954031,2.595681592692934,2.2164484112332685,1.892042412575034,1.622463596718233,1.4077119636628626,1.2477875134089238,1.1426902459564174,1.0924201613053433,1.0969772594557008,1.1563615404074903,1.2705730041607117,1.4396116507153647,1.66347748007145,1.9421704922289678,2.2756906871879177,2.664038064948299,3.1072126255101105,3.605214368873358,4.158043295038034,4.765699404004141,5.428182695771685,6.145493170340656,6.917630827711058,7.744595667882899,8.626387690856165,9.563006896630862,10.554453285207,11.600726856584561,12.701827610763564,13.857755547743988,15.068510667525846],[21.561817449574487,20.093792341471406,18.680594416169747,17.32222367366953,16.01868011397074,14.769963737073375,13.576074542977448,12.437012531682951,11.352777703189888,10.323370057498256,9.348789594608059,8.429036314519289,7.564110217231954,6.75401130274605,5.998739571061578,5.298295022178539,4.652677656096931,4.061887472816754,3.52592447233801,3.0447886546606995,2.61848001978482,2.2469985677103708,1.9303442984373556,1.668517211965771,1.461517308295618,1.3093445874268976,1.2119990493596091,1.1694806940937528,1.1817895216293282,1.2489255319663353,1.3708887251047743,1.5476791010446456,1.779296659785949,2.0657414013286846,2.407013325672852,2.803112432818449,3.2540387227654826,3.7597921955139433,4.320372851063838,4.935780689415168,5.606015710567923,6.331077914522112,7.110967301277739,7.945683870834792,8.835227623193274,9.779598558353197,10.778796676314544,11.832821977077332,12.941674460641543,14.105354127007185],[22.930307122646543,21.41501222941707,19.954544518989017,18.548903991362405,17.19809064653722,15.902104484513472,14.660945505291153,13.474613708870267,12.343109095250812,11.266431664432787,10.244581416416198,9.277558351201042,8.365362468787312,7.507993769175017,6.705452252364154,5.957737918354724,5.264850767146724,4.626790798740156,4.043558013135021,3.5151524103313188,3.0415739903290486,2.622822753128209,2.2588986987288022,1.9498018271308266,1.6955321383342827,1.4960896323391708,1.351474309145491,1.2616861687532437,1.2267252111624278,1.246591436373044,1.321284844385092,1.450805435198572,1.6351532088134844,1.8743281652298285,2.1683303044476046,2.5171596264668117,2.920816131287453,3.379299818909524,3.8926106893330266,4.460748742557964,5.0837139785843295,5.761506397412128,6.494125999041362,7.281572783472023,8.123846750704118,9.020947900737646,9.972876233572604,10.979631749209002,12.041214447646823,13.157624328886074],[24.211697189994467,22.652487053248425,21.14810409930382,19.698548328160648,18.303819739818902,16.963918334278592,15.678844111539718,14.448597071602272,13.273177214466257,12.152584540131675,11.086819048598525,10.075880739866808,9.119769613936521,8.218485670807665,7.372028910480244,6.580399332954254,5.843596938229695,5.161621726306566,4.534473697184873,3.962152850864611,3.4446591873457812,2.9819927066283825,2.5741534087124167,2.2211412935978814,1.9229563612847778,1.679598611773107,1.491068045062868,1.3573646611540608,1.278488460046686,1.2544394417407425,1.285217606236231,1.370822953533152,1.5112554836315046,1.7065151965312895,1.956602092232506,2.2615161707351534,2.6212574320392363,3.0358258761447474,3.5052215030516907,4.029444312760069,4.6084943052698755,5.242371480581114,5.93107583869379,6.674607379607892,7.472966103323424,8.326152009840397,9.234165099158794,10.19700537127863,11.214672826199891,12.287167463922584],[25.30297739595644,23.706276413833972,22.16440261451294,20.677355997993335,19.245136564275168,17.867744313358426,16.545179245243123,15.277441359929243,14.0645306574168,12.906447137705788,11.80319080079621,10.754761646688063,9.761159675381347,8.822384886876062,7.938437281172213,7.109316858269793,6.335023618168805,5.615557560869249,4.950918686371124,4.341106994674434,3.786122485779175,3.2859651596853467,2.8406350163929517,2.4501320559019866,2.114456278212454,1.833607683324354,1.6075862712376856,1.4363920419524496,1.3200249954686452,1.2584851317862726,1.2517724509053318,1.2998869528258237,1.402828637547747,1.5605975050711025,1.77319355539589,2.040616788522108,2.3628672044497616,2.7399448031788434,3.1718495847093577,3.6585815490413065,4.200140696174684,4.796527026109493,5.447740538845739,6.153781234383413,6.9146491127225165,7.730344173863059,8.600866417805028,9.526215844548434,10.506392454093266,11.541396246439529],[26.12332146325116,24.49791108847396,22.92732789649819,21.411571887323856,19.950643060950945,18.54454141737947,17.193266956609428,15.896819678640822,14.655199583473644,13.468406671107896,12.336440941543582,11.2593023947807,10.236991030819247,9.269506849659232,8.356849851300646,7.499020035743492,6.696017402987769,5.947841953033477,5.254493685880617,4.615972601529193,4.032278699979199,3.503411981230635,3.029372445283505,2.6101600921378063,2.2457749217935383,1.9362169342507032,1.6814861295092998,1.4815825075693294,1.33650606843079,1.2462568120936823,1.210834738558007,1.2302398478237635,1.3044721398909522,1.4335316147595727,1.6174182724296253,1.856132112901109,2.149673136174027,2.4980413422483743,2.9012367311241536,3.3592593028013678,3.87210905728001,4.439785994560084,5.062290114641595,5.739621417524533,6.471779903208902,7.258765571694711,8.100578422981943,8.997218457070616,9.948685673960714,10.95498007365224],[26.62411417180046,24.980256747936462,23.3912265068739,21.857023448612765,20.37764757315307,18.9530988804948,17.58337737063796,16.26848304358256,15.008415899328586,13.803175937876043,12.652763159224934,11.55717756337526,10.516419150327016,9.5304879200802,8.599383872634819,7.7231070079908735,6.901657326148355,6.13503482710727,5.423239510867617,4.766271377429398,4.164130426792609,3.616816658957252,3.124330073923328,2.686670671690834,2.3038384522597726,1.975833415630143,1.7026555618019452,1.4843048907751801,1.3207814025498468,1.212085097125945,1.1582159745034755,1.159174034682438,1.2149592776628322,1.3255717034446586,1.4910113120279167,1.7112781034126063,1.9863720775987297,2.316293234586283,2.701041574375268,3.140617096965688,3.635019802357536,4.184249690550816,4.788306761545533,5.447191015341677,6.160902451939252,6.929441071338265,7.752806873538705,8.630999858540582,9.564020026343886,10.551867376948621],[26.793625812944743,25.142107765461404,23.545416900779497,22.00355321889902,20.516516719819975,19.084307403542365,17.706925270066186,16.384370319391444,15.116642551518122,13.903741966446239,12.745668564175785,11.642422344706768,10.594003308039179,9.60041145417302,8.661646783108296,7.777709294845007,6.948598989383145,6.174315866722716,5.45485992686372,4.790231169806157,4.180429595550025,3.625455204095324,3.1253079954420566,2.6799879695902193,2.289495126539814,1.9538294662908409,1.6729909888432999,1.4469796941971913,1.2757955823525144,1.1594386533092693,1.0979089070674561,1.0912063436270747,1.1393309629881256,1.2422827651506085,1.4000617501145234,1.6126679178798695,1.8801012684466494,2.202361801814859,2.5794495179845,3.011364416955576,3.498106498728082,4.039675763302019,4.636072210677391,5.287295840854191,5.993346653832423,6.754224649612093,7.56992982819319,8.440462189575722,9.365821733759683,10.346008460745074],[26.655561161480144,25.00674271032824,23.41275144197777,21.873587356428725,20.389250453681118,18.959740733734943,17.585058196590197,16.265202842246882,15.000174670705,13.78997368196455,12.634599876025534,11.534053252887949,10.488333812551797,9.497441555017073,8.561376480283785,7.680138588351928,6.853727879221503,6.0821443528925085,5.365388009364946,4.703458848638818,4.096356870714121,3.5440820755908553,3.0466344632690228,2.60401403374862,2.21622078702965,1.8832547231121115,1.6051158419960052,1.3818041436813318,1.2133196281680894,1.0996622954562791,1.0408321455459009,1.0368291784369543,1.08765339412944,1.1933047926233575,1.3537833739187075,1.569089138015488,1.839222084913703,2.1641822146133474,2.5439695271144234,2.978584022416935,3.468025700520875,4.012294561426246,4.611390605133054,5.265313831641289,5.9740642409509555,6.73764183306206,7.55604660797459,8.42927856568856,9.357337706203953,10.340224029520781],[26.262325271507777,24.625280209924533,23.043062331142725,21.51567163516235,20.043108121983405,18.62537179160589,17.26246264402981,15.954380679255161,14.701125897281942,13.502698298110158,12.359097881739805,11.270324648170885,10.236378597403395,9.257259729437335,8.33296804427271,7.4635035419095175,6.648866222347756,5.889056085587425,5.1840731316285265,4.5339173604710625,3.938588772115029,3.3980873665604268,2.912413143807258,2.481566103855519,2.1055462467052126,1.7843535723563377,1.5179880808088955,1.3064497720628858,1.149738646118307,1.0478547029751606,1.000797942633446,1.0085683650931634,1.0711659703543128,1.1885907584168944,1.360842729280908,1.5879218829463522,1.869828219413231,2.2065617386815393,2.5981224407512795,3.044510325622454,3.545725393295058,4.101767643769093,4.7126370770445645,5.378333693121463,6.098857491999794,6.874208473679561,7.704386638160756,8.58939198544339,9.529224515527446,10.523884228412937],[25.685069265600674,24.06688718925221,22.503532295705178,20.995004584959577,19.541304057015406,18.142430711872674,16.798384549531367,15.509165569991497,14.274773773253054,13.09520915931604,11.970471728180465,10.900561479846322,9.885478414313608,8.925222531582326,8.01979383165248,7.169192314524062,6.373417980197076,5.632470828671522,4.9463508599474,4.315058074024713,3.738592470903456,3.2169540505836296,2.7501428130652363,2.338158758348275,1.981001886432745,1.6786721973186467,1.4311696910059806,1.2384943674947473,1.100646226784945,1.0176252688765748,0.9894314937696368,1.0160649014641305,1.0975254919600566,1.2338132652574143,1.424928221356204,1.6708703602564252,1.9716396819580801,2.327236186461165,2.7376598737656814,3.2029107438716324,3.722988796779012,4.297894032487824,4.927626450998071,5.612186052309747,6.3515728364228545,7.1457868033373995,7.9948279530533695,8.898696285570779,9.857391800889612,10.870914499009881],[25.003035452082532,23.408341355529174,21.868474441777245,20.38343471082675,18.95322216267769,17.577836797330054,16.257278614783857,14.991547615039089,13.780643798095756,12.624567163953852,11.52331771261338,10.476895444074344,9.485300358336735,8.54853245540056,7.666591735265817,6.8394781979325066,6.067191843400627,5.349732671670179,4.687100682741162,4.079295876613581,3.52631825328743,3.0281678127627107,2.584844555039424,2.1963484801175674,1.8626795879971436,1.5838378786781513,1.3598233521605911,1.1906360084444636,1.0762758475297676,1.0167428694165033,1.0120370741046711,1.062158461594271,1.1671070318853027,1.3268827849777665,1.5414857208716621,1.8109158395669889,2.1351731410637504,2.514257625361941,2.948169292461563,3.4369081423626207,3.9804741750651065,4.578867390569024,5.232087788874378,5.940135369981159,6.703010133889372,7.520712080599023,8.3932412101101,9.320597522422615,10.302781017536555,11.339791695451925],[24.294344999359076,22.725069182540366,21.2106205485231,19.750999097307254,18.346204828892848,16.99623774327987,15.701097840468325,14.460785120458215,13.275299583249534,12.144641228842284,11.068810057236465,10.047806068432083,9.081629262429127,8.170279639227607,7.313757198827519,6.512061941228862,5.765193866431636,5.073152974435842,4.43593926524148,3.8535527388485518,3.3259933952570546,2.853261234466989,2.4353562564783564,2.0722784612911544,1.764027848905384,1.510604419321046,1.3120081725381398,1.1682391085566664,1.0792972273766244,1.0451825289980141,1.0658950134208358,1.1414346806450897,1.2718015306707757,1.4569955634978935,1.6970167791264434,1.991865177556424,2.3415407587878394,2.746043522820684,3.20537346965496,3.719530599290673,4.2885149117278125,4.912326406966383,5.590965085006392,6.324430945847827,7.112723989490693,7.955844215934997,8.853791625180728,9.806566217227898,10.814167992076493,11.876596949726517],[23.62940179987157,22.08480872953498,20.59504284199983,19.16010413726611,17.77999261533382,16.45470827620297,15.184251119873545,13.968621146345553,12.807818355618991,11.701842747693863,10.650694322570168,9.654373080247906,8.71287902072707,7.826212144007671,6.994372450089704,6.217359938973169,5.495174610658062,4.82781646514439,4.21528550243215,3.6575817225213427,3.1547051254119665,2.7066557111040215,2.3134334795975096,1.9750384308924287,1.6914705649887793,1.4627298818865622,1.2888163815857772,1.1697300640864248,1.1054709293885034,1.0960389774920143,1.141434208396957,1.2416566221033318,1.396706218611139,1.6065829979203776,1.8712869600310484,2.1908181049431503,2.565176432656687,2.9943619431716524,3.4783746364880495,4.017214512605881,4.610881571525142,5.259375813245835,5.962697237767965,6.720845845091521,7.5338216352165075,8.401624608142935,9.324254763870785,10.301712102400076,11.33399662373079,12.421108327862937],[23.066955201384488,21.543915701999705,20.07570338541635,18.662318251634435,17.303760300653952,16.000029532474894,14.751125947097274,13.557049544521083,12.417800324746326,11.333378287772994,10.303783433601103,9.329015762230641,8.40907527366161,7.543961967894011,6.733675844927845,5.97821690476311,5.277585147399806,4.631780572837935,4.0408031810774965,3.504652972118491,3.0233299459609166,2.5968341026047734,2.225165442050063,1.9083239642967835,1.646309669344936,1.4391225571945203,1.2867626278455366,1.1892298812979858,1.1465243175518662,1.1586459366071786,1.2255947384639228,1.347370723122099,1.5239738905817077,1.755404240842748,2.0416617739052207,2.3827464897691235,2.778658388434462,3.2293974699012282,3.734963734169426,4.295357181239061,4.910577811110124,5.580625623782617,6.305500619256549,7.085202797531906,7.919732158608696,8.809088702486925,9.753272429166575,10.752283338647667,11.806121430930185,12.914786706014132],[22.65201766856868,21.145486863207406,19.693783240647555,18.29690680088915,16.954857543932167,15.667635469776615,14.435240578422503,13.25767286986982,12.134932344118564,11.067019001168742,10.053932841020353,9.095673863673397,8.192242069127872,7.343637457383778,6.549860028441117,5.810909782299888,5.12678671896009,4.497490838421725,3.92302214068479,3.403380625749291,2.9385662936152217,2.528579144282584,2.173419177751379,1.8730863940216045,1.6275807930932624,1.4369023749663523,1.3010511396408742,1.2200270871168288,1.1938302173942148,1.2224605304730327,1.305918026353282,1.4442027050349642,1.6373145665180777,1.8852536108026239,2.1880198378886013,2.5456132477760103,2.958033840464854,3.425281615955126,3.9473565742468306,4.52425871533997,5.155988039234537,5.8425445459305365,6.583928235427973,7.380139107726836,8.231177162827132,9.137042400728863,10.097734821432024,11.113254424936619,12.18360121124264,13.308775180350096],[22.414557170714676,20.918202962894807,19.476675937876365,18.089976095659356,16.758103436243783,15.481057959629634,14.258839665816922,13.091448554805641,11.978884626595791,10.921147881187373,9.918238318580388,8.970155938774836,8.076900741770713,7.238472727568021,6.4548718961667655,5.726098247566941,5.052151781768546,4.433032498771584,3.8687403985760525,3.3592754811819563,2.904637746589291,2.504827194798057,2.1598438258082555,1.8696876396198852,1.6343586362329465,1.4538568156474398,1.3281821778633653,1.2573347228807235,1.2413144506995126,1.2801213613197342,1.3737554547413875,1.522216730964473,1.7255051899889902,1.9836208318149393,2.2965636564423213,2.664333663871133,3.0869308541013805,3.5643552271330563,4.096606782966164,4.683685521600707,5.325591443036679,6.0223245472740805,6.773884834312922,7.58027230415319,8.441486956794886,9.357528792238023,10.328397810482583,11.354094011528586,12.43461739537601,13.569967962024867],[22.368204957411404,20.875120080500857,19.43686238639173,18.053431875084048,16.724828546577793,15.451052400872967,14.232103437969577,13.06798165786762,11.958687060567092,10.904219646067993,9.904579414370334,8.959766365474103,8.069780499379304,7.2346218160859355,6.454290315594002,5.728785997903498,5.058108863014426,4.442258910926786,3.881236141640578,3.375040555155804,2.9236721514724606,2.5271309305905487,2.1854168925100694,1.8985300372310216,1.666470364753405,1.489237875077221,1.3668325682024687,1.299254444129149,1.2865035028572607,1.3285797443868044,1.42548316871778,1.5772137758501874,1.7837715657840272,2.0451565385192993,2.361368694056003,2.732408032394137,3.158274553533707,3.6389682574747044,4.174489144217135,4.764837213761001,5.410012466106294,6.1100149012530185,6.864844519201181,7.674501319950768,8.538985303501791,9.458296469854249,10.432434819008135,11.461400350963459,12.545193065720206,13.683812963278385],[22.508914136742206,21.01233983319443,19.570592712448093,18.183672774503187,16.851580019359712,15.574314447017672,14.351876057477057,13.184264850737879,12.071480826800133,11.013523985663815,10.010394327328934,9.062091851795484,8.168616559063464,7.329968449132875,6.546147522003721,5.817153777675998,5.142987216149707,4.523647837424846,3.9591356415014176,3.4494506283794233,2.9945927980588603,2.5945621505397285,2.24935868582203,1.9589824039057613,1.7234333047909252,1.542711388477521,1.4168166549655485,1.345749104255009,1.3295087363459008,1.3680955512382247,1.4615095489319798,1.6097507294271676,1.8128190927237877,2.070714638821839,2.3834373677213225,2.7509872794222376,3.173364373924587,3.6505686512283653,4.182600111333575,4.769458754240221,5.411144579948294,6.1076575884577995,6.8589977797687425,7.6651651538811105,8.526159710794913,9.441981450510148,10.412630373026813,11.438106478344919,12.518409766464446,13.653540237385405],[22.814208178516424,21.308199660691837,19.857018325668687,18.460664173446965,17.119137204026668,15.83243741740781,14.600564813590381,13.423519392574386,12.301301154359823,11.233910098946692,10.221346226334992,9.263609536524724,8.36070002951589,7.5126177053084815,6.7193625639025125,5.980934605297973,5.297333829494864,4.668560236493187,4.094613826292943,3.575494598894132,3.111202554296752,2.701737692500804,2.3471000135062883,2.0472895173132035,1.8023062039215505,1.6121500733313299,1.476821125542541,1.3963193605551847,1.3706447783692597,1.3997973789847673,1.483777162401706,1.6225841286200768,1.8162182776398799,2.0646796094611157,2.3679681240837827,2.7260838215078804,3.139026701733414,3.606796764760375,4.1293940105887685,4.706818439218598,5.3390700506498545,6.026148844882543,6.768054821916668,7.564787981752222,8.416348324389203,9.322735849827627,10.283950558067476,11.299992449108762,12.370861522951472,13.496557779595618],[23.24402365239605,21.723996204871376,20.258795940148136,18.84842285822633,17.492876959105956,16.19215824278701,14.946266709269494,13.755202358553415,12.618965190638765,11.537555205525548,10.510972403213763,9.539216783703411,8.622288346994491,7.7601870930870005,6.952913021980944,6.20046613367632,5.502846428173126,4.860053905471365,4.2720885655710354,3.738950408472139,3.2606394341746747,2.8371556426786406,2.4684990339840396,2.15466960809087,1.8956673649991318,1.6914923047088262,1.5421444272199523,1.4476237325325108,1.407930220646501,1.4230638915619231,1.493024745278777,1.6178127817970627,1.7974280011167814,2.031870403237931,2.3211399881605135,2.6652367558845262,3.064160706409974,3.51791183973685,4.026490155865158,4.589895654794902,5.208128336526074,5.881188201058677,6.609075248392719,7.391789478528187,8.229330891465084,9.12169948720342,10.068895265743182,11.070918227084384,12.127768371227011,13.23944569817107],[23.743991833744676,22.20710067982911,20.725036708714967,19.297799920402262,17.925390314890986,16.607807892181143,15.345052652272727,14.137124595165748,12.984023720860202,11.885750029356085,10.842303520653404,9.853684194752152,8.919892051652331,8.040927091353943,7.216789313856989,6.447478719161464,5.732995307267373,5.073339078174712,4.468510031883483,3.9185081683936875,3.4233334877053245,2.982985989818392,2.5974656747328924,2.266772542448823,1.9909065929661862,1.7698678262849816,1.603656242405209,1.492271841326869,1.43571462304996,1.433984587574483,1.4870817349004382,1.5950060650278253,1.7577575779566443,1.9753362736868962,2.2477421522185788,2.574975213551693,2.957035457686241,3.39392288462222,3.885637494359629,4.4321792868984735,5.033548262238747,5.689744420380451,6.400767761323594,7.166618285068162,7.987295991614161,8.8628008809616,9.793132953110462,10.778292208060765,11.818278645812494,12.913092266365654],[24.25136527598889,22.696690576114268,21.196843059041072,19.751822724769312,18.361629573298984,17.02626360463009,15.74572481876262,14.52001321569659,13.349128795431985,12.233071557968815,11.17184150330708,10.165438631446778,9.213862942387902,8.31711443613046,7.475193112674451,6.688098972019875,5.955832014166729,5.278392239115014,4.655779646864733,4.087994237415885,3.575036010768468,3.1169049669224815,2.7136011058779292,2.365124427634807,2.0714749321931163,1.832652619552858,1.6486574897140318,1.5194895426766384,1.4451487784406762,1.4256351970061458,1.4609487983730474,1.5510895825413815,1.696057549511147,1.8958526992823446,2.1504750318549744,2.4599245472290354,2.82420124540453,3.243305126381455,3.7172361901598108,4.245994436739602,4.829579866120823,5.467992478303473,6.161232273287562,6.909299251073077,7.712193411660022,8.569914755048407,9.482463281238218,10.449838990229466,11.472041882022141,12.549071956616247],[24.70292979967692,23.131458462827222,21.614814308778957,20.152997337532124,18.746007549086723,17.393844943442755,16.09650952060022,14.854001280559114,13.666320223319442,12.533466348881198,11.455439657244389,10.432240148409015,9.463867822375068,8.550322679142553,7.691604718711475,6.887713941081826,6.138650346253608,5.444413934226823,4.805004705001468,4.220422658577549,3.69066779495506,3.215740114134002,2.7956396161143773,2.4303663008961838,2.1199201684794216,1.8643012188640917,1.6635094520501938,1.5175448680377284,1.4264074668266944,1.3900972484170924,1.4086142128089223,1.4819583600021844,1.6101296899968782,1.7931282027930042,2.0309538983905626,2.323606776789551,2.6710868379899746,3.073394081991827,3.5305285087951113,4.042490118399832,4.609278910805979,5.230894886013559,5.907338044022576,6.638608384833019,7.424705908444894,8.265630614858205,9.161382504072947,10.111961576089122,11.117367830906726,12.177601268525759],[25.043497650274603,23.45791396413864,21.9271574608041,20.451228140270995,19.03012600253933,17.663851047609086,16.35240327548028,15.095782686152907,13.893989279626961,12.747023055902446,11.654884014979368,10.61757215685772,9.635087481537504,8.707429989018719,7.83459967930137,7.016596552385449,6.253420608270959,5.545071846957903,4.891550268446277,4.292855872736087,3.748988659827327,3.2599486297199984,2.8257357824141027,2.446350117909638,2.121791636206605,1.852060337305004,1.6371562212048352,1.4770792879060988,1.3718295374087939,1.3214069697129205,1.3258115848184795,1.3850433827254705,1.4991023634338934,1.6679885269437484,1.8917018732550355,2.1702424023677533,2.5036101142819054,2.891805008997488,3.3348270865145007,3.83267634683295,4.385352789952827,4.992856415874135,5.6551872245968795,6.372345216121053,7.1443303904466555,7.971142747573698,8.852782287502166,9.789249010232073,10.780542915763403,11.826664004096168],[25.233253710969752,23.637569137093166,22.096711746018016,20.610681537744295,19.179478512272002,17.803102669601145,16.481554009731724,15.214832532663726,14.002938238397164,12.84587112693203,11.743631198268334,10.696218452406068,9.703632889345233,8.765874509085828,7.8829433116278596,7.054839296971321,6.281562465116214,5.563112816062539,4.899490349810295,4.290695066359485,3.7367269657101074,3.23758604786216,2.7932723128156454,2.403785760570562,2.0691263911269098,1.7892942044846905,1.5642892006439029,1.3941113796045483,1.2787607413666247,1.2182372859301331,1.2125410132950734,1.2616719234614457,1.3656300164292499,1.5244152921984864,1.7380277507691546,2.006467392141255,2.329734216314788,2.7078282232897513,3.1407494130661457,3.6284977856439764,4.171073341023234,4.768476079203925,5.420706000186051,6.127763103969605,6.889647390554589,7.706358859941013,8.577897512128862,9.504263347118153,10.485456364908863,11.521476565501008],[25.25245553702754,23.651522636772825,22.105416919319534,20.614138384667676,19.177687032817254,17.796062863768263,16.469265877520705,15.197296074074577,13.980153453429885,12.817838015586618,11.710349760544789,10.657688688304392,9.659854798865423,8.716848092227886,7.828668568391785,6.9953162273571134,6.216791069123872,5.493093093692064,4.824222301061687,4.210178691232745,3.650962264205234,3.1465730199791535,2.697010958554507,2.30227607993129,1.9623683841095052,1.6772878710891526,1.4470345408702325,1.2716083934527445,1.1510094288366879,1.0852376470220637,1.074293048008871,1.1181756317971105,1.2168853983867818,1.3704223477778852,1.5787864799704205,1.8419777949643872,2.159996292759788,2.532841973356618,2.9605148367548795,3.4430148829545777,3.980342111955703,4.572496523758259,5.219478118362254,5.921286895767674,6.677922855974526,7.489385998982816,8.355676324792533,9.276793833403689,10.252738524816266,11.283510399030279],[25.1026858117931,23.50165085283186,21.95544307667205,20.464062483313676,19.02750907275673,17.645782845001218,16.31888380004714,15.046811937894487,13.82956725854327,12.667149761993482,11.55955944824513,10.50679631729821,9.50886036915272,8.565751603808662,7.677470021266037,6.844015621524844,6.065388404585082,5.341588370446752,4.672615519109854,4.058469850574389,3.4991513648403556,2.9946600619077537,2.544995941776585,2.150159004446846,1.8101492499185399,1.5249666781916655,1.294611289266223,1.1190830831422136,0.9983820598196348,0.9325082192984883,0.9214615615787739,0.9652420866604913,1.063849794543641,1.2172846852282222,1.425546758714236,1.6886360150016804,2.0065524540905595,2.3792960759808675,2.806866880672607,3.289264868165783,3.8264900384603866,4.4185423915564215,5.065421927453893,5.767128646152792,6.523662547653123,7.335023631954892,8.201211899058086,9.122227348962719,10.098069981668775,11.128739797176266],[24.804763684332737,23.20851625226584,21.66709600300037,20.180502936536335,18.74873705287373,17.37179835201256,16.04968683395282,14.782402498694513,13.569945346237636,12.412315376582193,11.309512589728179,10.261536985675601,9.268388564424452,8.330067325974737,7.44657327032645,6.617906397479599,5.844066707434178,5.125054200190188,4.4608688757476305,3.8515107341065087,3.296979775266817,2.7972759992285554,2.3523994059917275,1.96234999555633,1.6271277679223648,1.3467327230898316,1.1211648610587304,0.9504241818290616,0.8345106854008244,0.773424371774019,0.7671652409486457,0.8157332929247042,0.919128527702195,1.0773509452811176,1.290400545661472,1.558277328843258,1.8809812948264781,2.258512443611127,2.690870775197208,3.178056289584725,3.720068986773669,4.316908866764046,4.968575929555859,5.675070175149099,6.436391603543771,7.252540214739881,8.123516008737415,9.04931898553639,10.029949145136786,11.065406487538619],[24.394214070810655,22.806889805600505,21.274392723191795,19.796722823584513,18.373880106778664,17.005864572774247,15.692676221571261,14.434315053169707,13.230781067569584,12.082074264770894,10.988194644773635,9.94914220757781,8.964916953183417,8.035518881590454,7.160947992798926,6.341204286808826,5.57628776362016,4.866198423232925,4.210936265647122,3.610501290862753,3.0648934988798153,2.5741128896983088,2.1381594633182353,1.7570332197395924,1.4307341589623812,1.159262280986602,0.942617585812255,0.7808000734393408,0.6738097438678577,0.6216465970978068,0.6243106331291877,0.6818018519620006,0.7941202535962455,0.9612658380319226,1.1832386052690314,1.4600385553075717,1.7916656881475455,2.1781200037889494,2.6194015022317845,3.1155101834760552,3.6664460475217537,4.272209094368884,4.932799324017452,5.648216736467447,6.418461331718873,7.243533109771737,8.123432070626025,9.058158214281756,10.047711540738907,11.092092049997492],[23.91561000288471,22.340194826777108,20.819606833470935,19.35384602296619,17.942912395262884,16.586805950361008,15.285526688260564,14.039074608961553,12.84744971246397,11.710651998767819,10.628681467873104,9.601538119779821,8.629221954487967,7.711732971997544,6.849071172308555,6.0412365554209995,5.288229121334874,4.590048870050181,3.946695801566918,3.35816991588509,2.824471213004693,2.3455996929257275,1.921555355648195,1.5523382011720928,1.2379482294974227,0.9783854406241846,0.7736498345523787,0.6237414112820052,0.5286601708130632,0.48840611314555304,0.5029792382794749,0.5723795462148288,0.6966070369516146,0.8756617104898325,1.1095435668294826,1.3982526059705633,1.7417888279130784,2.140152232657023,2.5933428202023996,3.1013605905492114,3.66420554369745,4.281877679647122,4.954376998398231,5.681703499950766,6.463857184304732,7.300838051460136,8.192646101416967,9.139281334175235,10.140743749734932,11.197033348096058],[23.417043177671644,21.85511505368358,20.34801411249694,18.89574035411173,17.498293778527955,16.155674385745616,14.867882175764704,13.634917148585227,12.45677930420718,11.333468642630562,10.26498516385538,9.251328867881632,8.292499754709313,7.388497824338425,6.539323076768971,5.744975512000949,5.005455130034358,4.320761930869199,3.6908959145054707,3.1158570809431776,2.595645430182315,2.130260962222884,1.7197036770648848,1.3639735747083175,1.0630706551531819,0.816994918399478,0.6257463644472064,0.48932499329636725,0.4077308049469596,0.38096379939898395,0.40902397665244006,0.49191133670732823,0.6296258795636486,0.822167605221401,1.0695365136805852,1.3717326049412002,1.7287558790032502,2.140606335866729,2.6072839755316393,3.128788797997986,3.70512080326576,4.3362799913349654,5.022266362205608,5.763079915877677,6.5587206523511785,7.409188571626117,8.314483673702483,9.274605958580286,10.289555426259515,11.359332076740175],[22.945532821072575,21.39716158082269,19.903617523374244,18.464900648727227,17.081010956881645,15.751948447837494,14.477713121594777,13.25830497815349,12.093724017513631,10.983970239675205,9.929043644638217,8.928944232402658,7.983672002968529,7.0932269563358314,6.257609092504568,5.476818411474737,4.750854913246337,4.079718597819368,3.4634094651938305,2.9019275153697275,2.3952727483470553,1.9434451641258148,1.546444762706007,1.2042715440876302,0.9169255082706851,0.684406655255172,0.506714985041091,0.3838504976284426,0.31581319301722566,0.30260307120744057,0.34422013219908737,0.4406643759921663,0.5919358025866773,0.7980344119826204,1.0589602041799955,1.3747131791788012,1.7452933369790415,2.170700677580711,2.6509352009838123,3.185996907188349,3.775885796194314,4.42060186800171,5.120145122610544,5.8745155600208045,6.683713180232496,7.547737983245626,8.46658996906018,9.440269137676175,10.468775489093595,11.552109023312445],[22.54358374214121,21.007392525546965,19.526028491754158,18.09949164076278,16.727781972572835,15.410899487184324,14.148844184597241,12.941616064811594,11.789215127827374,10.691641373644586,9.648894802263236,8.660975413683312,7.727883207904824,6.849618184927764,6.02618034475214,5.257569687377947,4.543786212805185,3.884829921033854,3.2807008120639556,2.731398885895491,2.236924142528457,1.797276581962855,1.4124562041986852,1.0824630092359464,0.8072969970746396,0.586958167714765,0.4214465211563225,0.31076205739931223,0.2549047764437335,0.25387467828958676,0.30767176293687193,0.4162960303855891,0.5797474806357384,0.7980261136873198,1.071131929540333,1.3990649281947771,1.781825109650656,2.219412473907964,2.7118270209667035,3.259068750826878,3.8611376634884818,4.518033758951517,5.229757037215988,5.996307498281887,6.817685142149217,7.6938899688179845,8.624921978288178,9.610781170559811,10.65146754563287,11.746981103507359],[22.24661820515059,20.71999333671768,19.248195651086203,17.831225148256156,16.469081828227537,15.161765691000358,13.909276736574606,12.711614964950291,11.5687803761274,10.480772970105946,9.447592746885924,8.469239706467333,7.545713848850175,6.677015174034447,5.863143682020153,5.1040993728072905,4.399882246395859,3.7504923027858594,3.1559295419772924,2.616193963970159,2.131285568764456,1.701204356360184,1.3259503267573456,1.005523479955938,0.7399238159559621,0.5291513347574185,0.37320603636030675,0.2720879207646274,0.2257969879703796,0.23433323797756386,0.29769667078617984,0.415887286396228,0.5889050848077081,0.8167500660206203,1.0994222300349648,1.4369215768507395,1.8292481064679493,2.2764018188865878,2.778382714106658,3.3351907921281647,3.9468260529510983,4.613288496575465,5.334578123001267,6.110694932228497,6.941638924257159,7.827410099087257,8.76800845671878,9.763433997151745,10.813686720386134,11.918766626421952],[22.08082386038941,20.560245237200768,19.094493796813563,17.683569539227786,16.327472464443442,15.026202572460528,13.77975986327905,12.588144336899001,11.451355993320385,10.3693948325432,9.34226085456745,8.36995405939313,7.452474447020242,6.589822017448784,5.781996770678761,5.02899870671017,4.330827825543009,3.68748412717728,3.0989676116129825,2.5652782788501196,2.0864161288886875,1.662381161728687,1.293173377370119,0.978792775812982,0.7192393570572768,0.5145131211030036,0.36461406795016266,0.269542197598754,0.22929751004877688,0.24388000530023182,0.31328968335311835,0.4375265442074372,0.616590587863188,0.8504818143203712,1.1392002235789858,1.4827458156390316,1.881118590500512,2.334318548163421,2.8423456886277623,3.4052000118935393,4.022881517960744,4.695390206829379,5.422726078499452,6.204889132970954,7.041879370243884,7.933696790318254,8.880341393194051,9.881813178871285,10.938112147349944,12.049238298630035],[22.06112151621621,20.54257423744866,19.078854141482548,17.66996122831786,16.315895497954607,15.016656950392784,13.772245585632396,12.58266140367344,11.447904404515915,10.36797458815982,9.342871954605158,8.37259650385193,7.457148235900132,6.5965271507497665,5.790733248400833,5.039766528853333,4.3436269921072626,3.702314638162623,3.115829467019417,2.5841714786776446,2.107340673137303,1.6853370503983924,1.3181606104609151,1.0058113533248687,0.7482892789902541,0.5455943874570716,0.397726678725321,0.30468615279500283,0.26647280966611625,0.2830866493386618,0.35452767181263883,0.48079587708804816,0.6618912651648895,0.897813836043163,1.1885635897228686,1.5341405262040049,1.934544645486576,2.3897759475705755,2.8998344324560072,3.4647201001428742,4.08443295063117,4.758972983920896,5.488340200012061,6.272534598904651,7.111556180598672,8.005404945094133,8.954080892391017,9.957584022489344,11.015914335389093,12.129071831090274],[22.189343746406795,20.66876481724732,19.203013070889273,17.792088507332664,16.435991126577484,15.134720928623738,13.888277913471423,12.696662081120538,11.559873431571086,10.477911964823063,9.450777680876477,8.478470579731322,7.5609906613875975,6.698337925845306,5.890512373104446,5.137514003165018,4.439342816027021,3.7959988116904566,3.207481990155324,2.6737923514216253,2.1949298954893575,1.7708946223585205,1.401686532029117,1.0873056245011443,0.8277518997746033,0.6230253578494944,0.47312599872581756,0.3780538224035731,0.3378088288827602,0.3523910181633793,0.42180039024543003,0.5460369451289131,0.7251006828138281,0.9589916033001755,1.247709706587954,1.5912549926771642,1.989627461567809,2.442827113259882,2.9508539477533873,3.5137079650483285,4.1313891651446975,4.803897548042499,5.531233113741736,6.3133958622424,7.150385793544495,8.042202907648031,8.98884720455299,9.990318684259389,11.046617346767212,12.157743192076468]],"type":"contour"},{"mode":"lines+markers","name":"theta Values","x":[0.0,2.601057454779614,0.9034272767622247,2.0563364416095036,1.1589234696036672,1.7938804194525289,1.3215749439533506,1.6495449106923081,1.4232529432779681,1.5829504072241183,1.4709725522541734,1.5504096528840599,1.4943913185184219,1.5341564032256916,1.506039954902419,1.525989188155241,1.5118665534078317,1.521882079272916,1.5147877319328085,1.519817390738254],"y":[0.0,2.601057454779614,2.5910060231049896,2.963129099991904,2.8668743087666506,3.072755727714994,3.001465693449906,3.0291086593677257,2.988203025298647,3.010338461886587,2.9892651852224326,3.001744204213725,2.9913519372340183,2.997978235955032,2.99286670937898,2.99628664791577,2.993755312996005,2.9954961669353444,2.994234974327257,2.9951151604637363],"type":"scatter"}],                        {"template":{"data":{"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"Gradient Descent with Static Learning Rate"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('dd255593-17d3-45f1-b259-69fc15746238');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


As we can see, gradient descent is able to navigate even this fairly complex loss space and find a nice minimum.

# Congratulations! You finished the lab!

---



