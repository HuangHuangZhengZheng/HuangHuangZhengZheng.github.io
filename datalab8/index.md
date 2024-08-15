# DATA100-lab8: Model Selection, Regularization, and Cross-Validation


```python
# Initialize Otter
import otter
grader = otter.Notebook("lab08.ipynb")
```

# Lab 8: Model Selection, Regularization, and Cross-Validation
In this lab, you will practice using `scikit-learn` to generate models of various complexity. You'll then use the holdout method and K-fold cross-validation to select the models that generalize best.



```python
# Run this cell to set up your notebook
import seaborn as sns
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
sns.set()
sns.set_context("talk")

from IPython.display import display, Latex, Markdown
```

### Introduction

For this lab, we will use a toy dataset to predict the house prices in Boston with data provided by the `sklearn.datasets` package. There are more interesting datasets in the package if you want to explore them during your free time!

Run the following cell to load the data. `load_boston()` will return a dictionary object which includes keys for:
- `data` : the covariates (X)
- `target` : the response vector (Y)
- `feature_names`: the column names
- `DESCR` : a full description of the data
- `filename`: name of the csv file



```python
import pickle
boston_data = pickle.load(open("boston_data.pickle", "rb")) 


print(boston_data.keys())
sum(boston_data.data)
```

    dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename', 'data_module'])
    




    array([1.82844292e+03, 5.75000000e+03, 5.63521000e+03, 3.50000000e+01,
           2.80675700e+02, 3.18002500e+03, 3.46989000e+04, 1.92029160e+03,
           4.83200000e+03, 2.06568000e+05, 9.33850000e+03, 6.40245000e+03])




```python
print(boston_data['DESCR'])
```

    .. _boston_dataset:
    
    Boston house prices dataset
    ---------------------------
    
    **Data Set Characteristics:**  
    
        :Number of Instances: 506 
    
        :Number of Attributes: 12 numeric/categorical predictive. Median Value (attribute 13) is usually the target.
    
        :Attribute Information (in order):
            - CRIM     per capita crime rate by town
            - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
            - INDUS    proportion of non-retail business acres per town
            - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            - NOX      nitric oxides concentration (parts per 10 million)
            - RM       average number of rooms per dwelling
            - AGE      proportion of owner-occupied units built prior to 1940
            - DIS      weighted distances to five Boston employment centres
            - RAD      index of accessibility to radial highways
            - TAX      full-value property-tax rate per $10,000
            - PTRATIO  pupil-teacher ratio by town
            - LSTAT    % lower status of the population
            - MEDV     Median value of owner-occupied homes in $1000's
    
        :Missing Attribute Values: None
    
        :Creator: Harrison, D. and Rubinfeld, D.L.
    
    This is a copy of UCI ML housing dataset.
    https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
    
    
    This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
    
    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.
    
    The Boston house-price data has been used in many machine learning papers that address regression
    problems.   
         
    .. topic:: References
    
       - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
       - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
    
    

A look at the `DESCR` attribute tells us the data contains these features:

    1. CRIM      per capita crime rate by town
    2. ZN        proportion of residential land zoned for lots over 
                 25,000 sq.ft.
    3. INDUS     proportion of non-retail business acres per town
    4. CHAS      Charles River dummy variable (= 1 if tract bounds 
                 river; 0 otherwise)
    5. NOX       nitric oxides concentration (parts per 10 million)
    6. RM        average number of rooms per dwelling
    7. AGE       proportion of owner-occupied units built prior to 1940
    8. DIS       weighted distances to five Boston employment centres
    9. RAD       index of accessibility to radial highways
    10. TAX      full-value property-tax rate per 10,000 USD
    11. PTRATIO  pupil-teacher ratio by town
    12. LSTAT    % lower status of the population
    
Let's now convert this data into a pandas DataFrame. 


```python
boston = pd.DataFrame(boston_data['data'], columns=boston_data['feature_names'])
boston.head()
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
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>4.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>9.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>4.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>2.94</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>5.33</td>
    </tr>
  </tbody>
</table>
</div>



### Question 1

Let's model this housing price data! Before we can do this, however, we need to split the data into training and test sets. Remember that the response vector (housing prices) lives in the `target` attribute. A random seed is set here so that we can deterministically generate the same splitting in the future if we want to test our result again and find potential bugs.

Use the [`train_test_split`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function to split out 10% of the data for the test set. Call the resulting splits `X_train`, `X_holdout`, `Y_train`, `Y_holdout`. Here "holdout" refers to the fact that we're going to hold this data our when training our model.

<!--
BEGIN QUESTION
name: q1
-->


```python
from sklearn.model_selection import train_test_split
np.random.seed(45)

X = boston
Y = pd.Series(boston_data['target'])

X_train, X_holdout, Y_train, Y_holdout = train_test_split(X, Y, test_size=0.1)
```


```python
grader.check("q1")
```




<p><strong><pre style='display: inline;'>q1</pre></strong> passed! 🚀</p>



### Question 2

As a warmup, fit a linear model to describe the relationship between the housing price and all available covariates. We've imported `sklearn.linear_model` as `lm`, so you can use that instead of typing out the whole module name. Fill in the cells below to fit a linear regression model to the covariates and create a scatter plot for our predictions vs. the true prices.

<!--
BEGIN QUESTION
name: q2
-->


```python
import sklearn.linear_model as lm

linear_model = lm.LinearRegression()

# Fit your linear model
linear_model.fit(X_train, Y_train)

# Predict housing prices on the test set
Y_pred = linear_model.predict(X_holdout)

# Plot predicted vs true prices
plt.scatter(Y_holdout, Y_pred, alpha=0.5)
plt.xlabel("Prices $(y)$")
plt.ylabel("Predicted Prices $(\hat{y})$")
plt.title("Prices vs Predicted Prices");
```

    <>:14: SyntaxWarning: invalid escape sequence '\h'
    <>:14: SyntaxWarning: invalid escape sequence '\h'
    C:\Users\86135\AppData\Local\Temp\ipykernel_6688\1494534656.py:14: SyntaxWarning: invalid escape sequence '\h'
      plt.ylabel("Predicted Prices $(\hat{y})$")
    


    
![png](lab08_files/lab08_12_1.png)
    


Briefly analyze the scatter plot above. Do you notice any outliers? Write your answer in the cell below.

_理想情况应该是均匀？且较窄分布于y=x直线上_

Alternately, we can plot the residuals vs. our model predictions. Ideally they'd all be zero. Given the inevitably of noise, we'd at least like them to be scatter randomly across the line where the residual is zero. By contrast, there appears to be a possible pattern, with our model consistently underestimating prices for both very low and very high values, and possibly consistently overestimating prices towards the middle range.


```python
plt.scatter(Y_pred, Y_holdout - Y_pred, alpha=0.5)
plt.ylabel("Residual $(y - \hat{y})$")
plt.xlabel("Predicted Prices $(\hat{y})$")
plt.title("Residuals vs Predicted Prices")
plt.title("Residual of prediction for i'th house")
plt.axhline(y = 0, color='r');
```

    <>:2: SyntaxWarning: invalid escape sequence '\h'
    <>:3: SyntaxWarning: invalid escape sequence '\h'
    <>:2: SyntaxWarning: invalid escape sequence '\h'
    <>:3: SyntaxWarning: invalid escape sequence '\h'
    C:\Users\86135\AppData\Local\Temp\ipykernel_6688\2491234216.py:2: SyntaxWarning: invalid escape sequence '\h'
      plt.ylabel("Residual $(y - \hat{y})$")
    C:\Users\86135\AppData\Local\Temp\ipykernel_6688\2491234216.py:3: SyntaxWarning: invalid escape sequence '\h'
      plt.xlabel("Predicted Prices $(\hat{y})$")
    


    
![png](lab08_files/lab08_16_1.png)
    


### Question 3

As we find from the scatter plot, our model is not perfect. If it were perfect, we would see the identity line (i.e. a line of slope 1). Compute the root mean squared error (RMSE) of the predicted responses: 

$$
\textbf{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n \left( y_i - \hat{y}_i \right)^2 }
$$

Fill out the function below and compute the RMSE for our predictions on both the training data `X_train` and the held out set `X_holdout`.  Your implementation **should not** use for loops.

<!--
BEGIN QUESTION
name: q3
-->


```python
def rmse(actual_y, predicted_y):
    """
    Args:
        predicted_y: an array of the prediction from the model
        actual_y: an array of the groudtruth label
        
    Returns:
        The root mean square error between the prediction and the groudtruth
    """
    return np.sqrt(np.mean((predicted_y - actual_y)**2))

train_error = rmse(Y_train, linear_model.predict(X_train))
holdout_error = rmse(Y_holdout, Y_pred)

print("Training RMSE:", train_error)
print("Holdout RMSE:", holdout_error)
```

    Training RMSE: 4.633297105625516
    Holdout RMSE: 5.685160866583937
    


```python
grader.check("q3")
```




<p><strong><pre style='display: inline;'>q3</pre></strong> passed! 🙌</p>



Is your training error lower than the error on the data the model never got to see? If so, why could this be happening? Answer in the cell below.

_稍微过拟合？_

## Overfitting

Sometimes we can get even higher accuracy by adding more features. For example, the code below adds the square, square root, and hyperbolic tangent of every feature to the design matrix. We've chosen these bizarre features specifically to highlight overfitting.


```python
boston_with_extra_features = boston.copy()
for feature_name in boston.columns:
    boston_with_extra_features[feature_name + "^2"] = boston_with_extra_features[feature_name] ** 2
    boston_with_extra_features["sqrt" + feature_name] = np.sqrt(boston_with_extra_features[feature_name])
    boston_with_extra_features["tanh" + feature_name] = np.tanh(boston_with_extra_features[feature_name])
    
boston_with_extra_features.head(5)
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
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>...</th>
      <th>tanhRAD</th>
      <th>TAX^2</th>
      <th>sqrtTAX</th>
      <th>tanhTAX</th>
      <th>PTRATIO^2</th>
      <th>sqrtPTRATIO</th>
      <th>tanhPTRATIO</th>
      <th>LSTAT^2</th>
      <th>sqrtLSTAT</th>
      <th>tanhLSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>...</td>
      <td>0.761594</td>
      <td>87616.0</td>
      <td>17.204651</td>
      <td>1.0</td>
      <td>234.09</td>
      <td>3.911521</td>
      <td>1.0</td>
      <td>24.8004</td>
      <td>2.231591</td>
      <td>0.999905</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>...</td>
      <td>0.964028</td>
      <td>58564.0</td>
      <td>15.556349</td>
      <td>1.0</td>
      <td>316.84</td>
      <td>4.219005</td>
      <td>1.0</td>
      <td>83.5396</td>
      <td>3.023243</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>...</td>
      <td>0.964028</td>
      <td>58564.0</td>
      <td>15.556349</td>
      <td>1.0</td>
      <td>316.84</td>
      <td>4.219005</td>
      <td>1.0</td>
      <td>16.2409</td>
      <td>2.007486</td>
      <td>0.999368</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>...</td>
      <td>0.995055</td>
      <td>49284.0</td>
      <td>14.899664</td>
      <td>1.0</td>
      <td>349.69</td>
      <td>4.324350</td>
      <td>1.0</td>
      <td>8.6436</td>
      <td>1.714643</td>
      <td>0.994426</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>...</td>
      <td>0.995055</td>
      <td>49284.0</td>
      <td>14.899664</td>
      <td>1.0</td>
      <td>349.69</td>
      <td>4.324350</td>
      <td>1.0</td>
      <td>28.4089</td>
      <td>2.308679</td>
      <td>0.999953</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 48 columns</p>
</div>



We split up our data again and refit the model. From this cell forward, we append `2` to the variable names `X_train, X_holdout, Y_train, Y_holdout, train_error, holdout_error` in order to maintain our original data. **Make sure you use these variable names from this cell forward**, at least until we get to the part where we create version 3 of each of these.


```python
np.random.seed(25)
X = boston_with_extra_features
X_train2, X_holdout2, Y_train2, Y_holdout2 = train_test_split(X, Y, test_size = 0.10)
linear_model.fit(X_train2, Y_train2);
```

Looking at our training and test RMSE, we see that they are lower than you computed earlier. This strange model is seemingly better, even though it includes seemingly useless features like the hyperbolic tangent of the average number of rooms per dwelling.


```python
train_error2 = rmse(Y_train2, linear_model.predict(X_train2)) 
holdout_error2 = rmse(Y_holdout2, linear_model.predict(X_holdout2))

print("Training RMSE:", train_error2)
print("Holdout RMSE:", holdout_error2)
```

    Training RMSE: 3.3514483036916287
    Holdout RMSE: 5.410120414381265
    

The code below generates the training and holdout RMSE for 49 different models stores the results in a DataFrame. The first model uses only the first feature "CRIM". The second model uses the first two features "CRIM" and "ZN", and so forth.


```python
errors_vs_N = pd.DataFrame(columns = ["N", "Training Error", "Holdout Error"])
range_of_num_features = range(1, X_train2.shape[1] + 1)

for N in range_of_num_features:
    X_train_first_N_features = X_train2.iloc[:, :N]    
    
    linear_model.fit(X_train_first_N_features, Y_train2)
    train_error_overfit = rmse(Y_train2, linear_model.predict(X_train_first_N_features))
    
    X_holdout_first_N_features = X_holdout2.iloc[:, :N]
    holdout_error_overfit = rmse(Y_holdout2, linear_model.predict(X_holdout_first_N_features))    
    errors_vs_N.loc[len(errors_vs_N)] = [N, train_error_overfit, holdout_error_overfit]
    
errors_vs_N
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
      <th>N</th>
      <th>Training Error</th>
      <th>Holdout Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>8.536340</td>
      <td>7.825177</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>8.085693</td>
      <td>7.637465</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>7.776942</td>
      <td>7.213870</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>7.643897</td>
      <td>6.391482</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>7.634894</td>
      <td>6.372166</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.0</td>
      <td>5.698878</td>
      <td>7.635694</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.0</td>
      <td>5.689554</td>
      <td>7.585860</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.0</td>
      <td>5.399034</td>
      <td>7.158563</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9.0</td>
      <td>5.379679</td>
      <td>7.281769</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10.0</td>
      <td>5.318218</td>
      <td>7.231629</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11.0</td>
      <td>5.088829</td>
      <td>6.922974</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12.0</td>
      <td>4.680294</td>
      <td>5.437528</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13.0</td>
      <td>4.679671</td>
      <td>5.443388</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14.0</td>
      <td>4.664717</td>
      <td>5.448438</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15.0</td>
      <td>4.627661</td>
      <td>5.479720</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16.0</td>
      <td>4.613226</td>
      <td>5.488425</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17.0</td>
      <td>4.580971</td>
      <td>5.389309</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18.0</td>
      <td>4.580622</td>
      <td>5.391183</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19.0</td>
      <td>4.507301</td>
      <td>5.185114</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20.0</td>
      <td>4.482925</td>
      <td>5.194924</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21.0</td>
      <td>4.482412</td>
      <td>5.188007</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22.0</td>
      <td>4.482412</td>
      <td>5.188007</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23.0</td>
      <td>4.482412</td>
      <td>5.188007</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24.0</td>
      <td>4.482412</td>
      <td>5.188007</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25.0</td>
      <td>4.482224</td>
      <td>5.191621</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26.0</td>
      <td>4.471079</td>
      <td>5.256722</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27.0</td>
      <td>4.460457</td>
      <td>5.308239</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28.0</td>
      <td>3.909139</td>
      <td>4.582940</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29.0</td>
      <td>3.889483</td>
      <td>4.951462</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30.0</td>
      <td>3.728687</td>
      <td>6.963946</td>
    </tr>
    <tr>
      <th>30</th>
      <td>31.0</td>
      <td>3.697400</td>
      <td>6.986045</td>
    </tr>
    <tr>
      <th>31</th>
      <td>32.0</td>
      <td>3.672688</td>
      <td>7.081944</td>
    </tr>
    <tr>
      <th>32</th>
      <td>33.0</td>
      <td>3.672674</td>
      <td>7.082046</td>
    </tr>
    <tr>
      <th>33</th>
      <td>34.0</td>
      <td>3.638518</td>
      <td>6.916599</td>
    </tr>
    <tr>
      <th>34</th>
      <td>35.0</td>
      <td>3.570632</td>
      <td>6.535278</td>
    </tr>
    <tr>
      <th>35</th>
      <td>36.0</td>
      <td>3.515723</td>
      <td>6.417958</td>
    </tr>
    <tr>
      <th>36</th>
      <td>37.0</td>
      <td>3.513539</td>
      <td>6.420029</td>
    </tr>
    <tr>
      <th>37</th>
      <td>38.0</td>
      <td>3.502764</td>
      <td>6.453024</td>
    </tr>
    <tr>
      <th>38</th>
      <td>39.0</td>
      <td>3.502689</td>
      <td>6.452923</td>
    </tr>
    <tr>
      <th>39</th>
      <td>40.0</td>
      <td>3.498222</td>
      <td>6.373783</td>
    </tr>
    <tr>
      <th>40</th>
      <td>41.0</td>
      <td>3.450329</td>
      <td>6.493727</td>
    </tr>
    <tr>
      <th>41</th>
      <td>42.0</td>
      <td>3.450329</td>
      <td>6.493727</td>
    </tr>
    <tr>
      <th>42</th>
      <td>43.0</td>
      <td>3.448549</td>
      <td>6.486777</td>
    </tr>
    <tr>
      <th>43</th>
      <td>44.0</td>
      <td>3.448549</td>
      <td>6.486974</td>
    </tr>
    <tr>
      <th>44</th>
      <td>45.0</td>
      <td>3.448549</td>
      <td>6.486974</td>
    </tr>
    <tr>
      <th>45</th>
      <td>46.0</td>
      <td>3.420689</td>
      <td>5.823307</td>
    </tr>
    <tr>
      <th>46</th>
      <td>47.0</td>
      <td>3.353809</td>
      <td>5.398784</td>
    </tr>
    <tr>
      <th>47</th>
      <td>48.0</td>
      <td>3.351448</td>
      <td>5.410120</td>
    </tr>
  </tbody>
</table>
</div>



If we plot the training and holdout error as we add each additional feature, our training error gets lower and lower (since our model bias is increasing), and in fact it's possible to prove with linear algebra that the training error will decrease monotonically.

By contrast, the error on unseen held out data is higher for the models with more parameters, since the lessons learned from these last 20+ features aren't actually useful when applied to unseen data. That is, these models aren't generalizable.


```python
import plotly.express as px
px.line(errors_vs_N, x = "N", y = ["Training Error", "Holdout Error"])
```



Note that this diagram resembles are cartoon from [Lecture 15](https://docs.google.com/presentation/d/1-Cga_fOn0dTMt1ss7vNmManX-NUYPXwXDQDAsaInuQM/edit#slide=id.g119768bc0e3_0_516).

This plot is a useful tool for **model selection**: the best model is the one the lowest error on the holdout set, i.e. the one that includes parameters 1 through 28.

## Regularization

As an alternative and more realistic example, instead of using only the first N features, we can use various different regularization strengths. For example, for really low regularization strengths (e.g. $\alpha = 10^{-3}$), we get a model that is very identical to our linear regression model.


```python
from sklearn.linear_model import Ridge
regularized_model = Ridge(alpha = 10**-5)
regularized_model.fit(X_train2, Y_train2)
regularized_model.coef_
```

    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_ridge.py:216: LinAlgWarning:
    
    Ill-conditioned matrix (rcond=6.11696e-19): result may not be accurate.
    
    




    array([ 4.44044277e-01, -3.00268517e-02,  2.03776925e+00,  3.54247206e-01,
           -1.19704083e+02,  1.63780073e+01, -3.10555372e-01, -1.31182539e+01,
            2.87010751e+00,  7.68411439e-01,  2.43201974e+01,  2.09160420e+00,
           -1.17012738e-03, -5.60565882e+00,  6.79680723e+00,  1.02949752e-03,
           -1.31223400e+00,  6.99621340e+00, -3.55165065e-02, -7.66339676e+00,
           -2.53950130e+00,  3.54247186e-01,  3.54247186e-01,  2.69792455e-01,
            1.91778126e+00,  3.11293526e+02, -1.53815298e+02,  8.03364965e-01,
           -1.17792246e+02,  3.25883430e+02,  1.08476149e-03,  2.42998443e+00,
            2.52462516e+02,  3.55080093e-01,  3.78504405e+01, -8.11283072e+01,
           -5.18073808e-02, -8.51699934e+00,  1.14213610e+01, -2.86248788e-04,
           -2.10606164e+01,  0.00000000e+00, -1.85988225e-01, -1.54605184e+02,
            5.73422430e-06, -1.79546600e-02, -1.53342390e+01, -4.25637232e+01])




```python
linear_model.fit(X_train2, Y_train2)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "�?;
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "�?;
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;LinearRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LinearRegression.html">?<span>Documentation for LinearRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>LinearRegression()</pre></div> </div></div></div></div>




```python
linear_model.coef_
```




    array([ 3.65647144e-01,  7.96329260e-02,  1.50196461e+00,  3.72759210e-01,
           -1.82281287e+03,  6.19862020e+02, -2.86690023e-01, -1.29491141e+01,
            1.68693762e+00,  7.86841735e-01,  1.62893036e+01,  1.95113824e+00,
           -9.11835586e-04, -5.02513063e+00,  5.90016774e+00,  6.12889765e-04,
           -2.21247181e+00,  8.90275845e+00, -2.73913970e-02, -5.40098561e+00,
           -4.23462112e+00,  3.72978675e-01,  3.72978861e-01,  2.84060205e-01,
            5.41748851e+02,  4.88274463e+02,  1.16998609e+03, -1.36350124e+01,
           -2.23299632e+03,  5.18647024e+04,  1.04162650e-03,  2.14549424e+00,
            4.31003519e+02,  3.51263646e-01,  3.77337190e+01, -8.06896603e+01,
           -2.88295129e-02, -4.52779826e+00,  8.15771554e+00, -2.99443268e-04,
           -2.14061912e+01,  3.63797881e-12, -1.15683673e-01, -1.07968511e+02,
            1.52846060e-03, -2.03166630e-02, -1.38532349e+01, -4.22894414e+01])



However, if we pick a large regularization strength, e.g. $\alpha = 10^4$, we see that the resulting parameters are much smaller in magnitude. 


```python
from sklearn.linear_model import Ridge
regularized_model = Ridge(alpha = 10**4)
regularized_model.fit(X_train2, Y_train2)
regularized_model.coef_
```




    array([-2.64236947e-02, -9.32767913e-03, -2.42925745e-02,  5.47079848e-03,
           -2.54276859e-03,  1.92843599e-02, -5.85037883e-02, -2.06397155e-02,
            2.62611572e-02, -4.16712719e-02, -1.95840395e-03, -1.91841765e-01,
           -1.08846586e-03, -4.28805626e-03,  1.70791430e-03,  6.51767238e-04,
            1.71133790e-03,  1.07486010e-03, -1.19407955e-03, -7.15970642e-03,
           -7.29287455e-04,  5.47079848e-03,  5.47079848e-03,  4.16652815e-03,
           -3.60910235e-03, -1.50954020e-03, -1.59681172e-03,  3.35928833e-01,
            3.11186224e-03, -2.79750628e-06,  4.48782500e-04, -5.71759051e-03,
            2.22943575e-06, -6.59740404e-02, -7.01191670e-03, -1.58200606e-03,
            1.32454447e-03,  8.15878522e-03,  1.17645581e-03,  3.59660322e-05,
           -2.54207413e-03,  0.00000000e+00, -2.57499245e-02, -3.15683513e-04,
           -8.10128212e-15, -6.45893053e-03, -4.20286900e-02, -2.29035441e-04])



### Standard Scaling

### 归一�?
Recall from lecture that in order to properly regularize a model, the features should be at the same scale. Otherwise the model has to spend more of its parameter budget to use "small" features (e.g. lengths in inches) compared to "large" features (e.g. lengths in kilometers).

To do this we can use a Standard Scaler to create a new version of the DataFrame where every column has zero mean and a standard deviation of 1.


```python
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(boston_with_extra_features)
boston_with_extra_features_scaled = pd.DataFrame(ss.transform(boston_with_extra_features), columns = boston_with_extra_features.columns)
boston_with_extra_features_scaled
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
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>...</th>
      <th>tanhRAD</th>
      <th>TAX^2</th>
      <th>sqrtTAX</th>
      <th>tanhTAX</th>
      <th>PTRATIO^2</th>
      <th>sqrtPTRATIO</th>
      <th>tanhPTRATIO</th>
      <th>LSTAT^2</th>
      <th>sqrtLSTAT</th>
      <th>tanhLSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.419782</td>
      <td>0.284830</td>
      <td>-1.287909</td>
      <td>-0.272599</td>
      <td>-0.144217</td>
      <td>0.413672</td>
      <td>-0.120013</td>
      <td>0.140214</td>
      <td>-0.982843</td>
      <td>-0.666608</td>
      <td>...</td>
      <td>-4.863216</td>
      <td>-0.682024</td>
      <td>-0.644166</td>
      <td>0.0</td>
      <td>-1.458429</td>
      <td>-1.453573</td>
      <td>0.135095</td>
      <td>-0.789529</td>
      <td>-1.202689</td>
      <td>0.103530</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.417339</td>
      <td>-0.487722</td>
      <td>-0.593381</td>
      <td>-0.272599</td>
      <td>-0.740262</td>
      <td>0.194274</td>
      <td>0.367166</td>
      <td>0.557160</td>
      <td>-0.867883</td>
      <td>-0.987329</td>
      <td>...</td>
      <td>-0.521299</td>
      <td>-0.866530</td>
      <td>-1.053383</td>
      <td>0.0</td>
      <td>-0.373078</td>
      <td>-0.266921</td>
      <td>0.179012</td>
      <td>-0.540454</td>
      <td>-0.399953</td>
      <td>0.128396</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.417342</td>
      <td>-0.487722</td>
      <td>-0.593381</td>
      <td>-0.272599</td>
      <td>-0.740262</td>
      <td>1.282714</td>
      <td>-0.265812</td>
      <td>0.557160</td>
      <td>-0.867883</td>
      <td>-0.987329</td>
      <td>...</td>
      <td>-0.521299</td>
      <td>-0.866530</td>
      <td>-1.053383</td>
      <td>0.0</td>
      <td>-0.373078</td>
      <td>-0.266921</td>
      <td>0.179012</td>
      <td>-0.825825</td>
      <td>-1.429933</td>
      <td>-0.037847</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.416750</td>
      <td>-0.487722</td>
      <td>-1.306878</td>
      <td>-0.272599</td>
      <td>-0.835284</td>
      <td>1.016303</td>
      <td>-0.809889</td>
      <td>1.077737</td>
      <td>-0.752922</td>
      <td>-1.106115</td>
      <td>...</td>
      <td>0.144191</td>
      <td>-0.925467</td>
      <td>-1.216415</td>
      <td>0.0</td>
      <td>0.057783</td>
      <td>0.139631</td>
      <td>0.179251</td>
      <td>-0.858040</td>
      <td>-1.726876</td>
      <td>-1.338649</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.412482</td>
      <td>-0.487722</td>
      <td>-1.306878</td>
      <td>-0.272599</td>
      <td>-0.835284</td>
      <td>1.228577</td>
      <td>-0.511180</td>
      <td>1.077737</td>
      <td>-0.752922</td>
      <td>-1.106115</td>
      <td>...</td>
      <td>0.144191</td>
      <td>-0.925467</td>
      <td>-1.216415</td>
      <td>0.0</td>
      <td>0.057783</td>
      <td>0.139631</td>
      <td>0.179251</td>
      <td>-0.774228</td>
      <td>-1.124522</td>
      <td>0.116050</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>501</th>
      <td>-0.413229</td>
      <td>-0.487722</td>
      <td>0.115738</td>
      <td>-0.272599</td>
      <td>0.158124</td>
      <td>0.439316</td>
      <td>0.018673</td>
      <td>-0.625796</td>
      <td>-0.982843</td>
      <td>-0.803212</td>
      <td>...</td>
      <td>-4.863216</td>
      <td>-0.765138</td>
      <td>-0.813468</td>
      <td>0.0</td>
      <td>1.255407</td>
      <td>1.136187</td>
      <td>0.179299</td>
      <td>-0.498180</td>
      <td>-0.312324</td>
      <td>0.128400</td>
    </tr>
    <tr>
      <th>502</th>
      <td>-0.415249</td>
      <td>-0.487722</td>
      <td>0.115738</td>
      <td>-0.272599</td>
      <td>0.158124</td>
      <td>-0.234548</td>
      <td>0.288933</td>
      <td>-0.716639</td>
      <td>-0.982843</td>
      <td>-0.803212</td>
      <td>...</td>
      <td>-4.863216</td>
      <td>-0.765138</td>
      <td>-0.813468</td>
      <td>0.0</td>
      <td>1.255407</td>
      <td>1.136187</td>
      <td>0.179299</td>
      <td>-0.545089</td>
      <td>-0.410031</td>
      <td>0.128395</td>
    </tr>
    <tr>
      <th>503</th>
      <td>-0.413447</td>
      <td>-0.487722</td>
      <td>0.115738</td>
      <td>-0.272599</td>
      <td>0.158124</td>
      <td>0.984960</td>
      <td>0.797449</td>
      <td>-0.773684</td>
      <td>-0.982843</td>
      <td>-0.803212</td>
      <td>...</td>
      <td>-4.863216</td>
      <td>-0.765138</td>
      <td>-0.813468</td>
      <td>0.0</td>
      <td>1.255407</td>
      <td>1.136187</td>
      <td>0.179299</td>
      <td>-0.759808</td>
      <td>-1.057406</td>
      <td>0.121757</td>
    </tr>
    <tr>
      <th>504</th>
      <td>-0.407764</td>
      <td>-0.487722</td>
      <td>0.115738</td>
      <td>-0.272599</td>
      <td>0.158124</td>
      <td>0.725672</td>
      <td>0.736996</td>
      <td>-0.668437</td>
      <td>-0.982843</td>
      <td>-0.803212</td>
      <td>...</td>
      <td>-4.863216</td>
      <td>-0.765138</td>
      <td>-0.813468</td>
      <td>0.0</td>
      <td>1.255407</td>
      <td>1.136187</td>
      <td>0.179299</td>
      <td>-0.716638</td>
      <td>-0.884300</td>
      <td>0.127164</td>
    </tr>
    <tr>
      <th>505</th>
      <td>-0.415000</td>
      <td>-0.487722</td>
      <td>0.115738</td>
      <td>-0.272599</td>
      <td>0.158124</td>
      <td>-0.362767</td>
      <td>0.434732</td>
      <td>-0.613246</td>
      <td>-0.982843</td>
      <td>-0.803212</td>
      <td>...</td>
      <td>-4.863216</td>
      <td>-0.765138</td>
      <td>-0.813468</td>
      <td>0.0</td>
      <td>1.255407</td>
      <td>1.136187</td>
      <td>0.179299</td>
      <td>-0.631389</td>
      <td>-0.619088</td>
      <td>0.128327</td>
    </tr>
  </tbody>
</table>
<p>506 rows × 48 columns</p>
</div>



Let's now regenerate the training and holdout sets using this new rescaled dataset.


```python
np.random.seed(25)
X = boston_with_extra_features_scaled
X_train3, X_holdout3, Y_train3, Y_holdout3 = train_test_split(X, Y, test_size = 0.10)
```

Fitting our regularized model with $\alpha = 10^4$ on this scaled data, we now see that our coefficients are of about the same magnitude. This is because all of our features are of around the same magnitude, whereas in the unscaled data, some of the features like TAX^2 were much larger than others.


```python
from sklearn.linear_model import Ridge
regularized_model = Ridge(alpha = 10**2)
regularized_model.fit(X_train3, Y_train3)
regularized_model.coef_
```




    array([-0.61501301, -0.04142115, -0.13765546,  0.11847529, -0.48559141,
            1.08393358, -0.11193453, -0.6446524 ,  0.25956768, -0.41922265,
           -0.48366805, -1.23850023, -0.22227015, -0.51281683,  0.40952134,
            0.2537374 , -0.07390569,  0.06674777,  0.11386252, -0.32684806,
           -0.39658025,  0.11847529,  0.11847529,  0.11847529, -0.67728184,
           -0.385382  , -0.36114118,  1.652695  ,  0.78959095, -1.09450355,
           -0.02430294, -0.14153645,  0.11511136, -0.41673303, -0.72747143,
           -1.36478486,  0.21308676,  0.30241207,  0.45131889, -0.16799052,
           -0.59340155,  0.        , -0.43637213, -0.50878723, -0.16529828,
           -0.04194842, -1.94295189, -0.70807685])



### Finding an Optimum Alpha

In the cell below, write code that generates a DataFrame with the training and holdout error for the range of alphas given. Make sure you're using the 3rd training and holdout sets, which have been rescaled!

**Note: You should use all 48 features for every single model that you fit, i.e. you're not going to be keeping only the first N features.**


```python
error_vs_alpha = pd.DataFrame(columns = ["alpha", "Training Error", "Holdout Error"])
range_of_alphas = 10**np.linspace(-5, 4, 40)

# for N in range_of_num_features:
#     X_train_first_N_features = X_train2.iloc[:, :N]    
    
#     linear_model.fit(X_train_first_N_features, Y_train2)
#     train_error_overfit = rmse(Y_train2, linear_model.predict(X_train_first_N_features))
    
#     X_holdout_first_N_features = X_holdout2.iloc[:, :N]
#     holdout_error_overfit = rmse(Y_holdout2, linear_model.predict(X_holdout_first_N_features))    
#     errors_vs_N.loc[len(errors_vs_N)] = [N, train_error_overfit, holdout_error_overfit]
for alpha in range_of_alphas:
    linear_model = Ridge(alpha=alpha)
    linear_model.fit(X_train3, Y_train3)
    training_error = rmse(Y_train3, linear_model.predict(X_train3))
    holdout_error = rmse(Y_holdout3, linear_model.predict(X_holdout3))
    error_vs_alpha.loc[len(error_vs_alpha)] = [alpha, training_error, holdout_error]
error_vs_alpha
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
      <th>alpha</th>
      <th>Training Error</th>
      <th>Holdout Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000010</td>
      <td>3.344803</td>
      <td>5.389722</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000017</td>
      <td>3.344885</td>
      <td>5.362696</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000029</td>
      <td>3.345093</td>
      <td>5.318839</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000049</td>
      <td>3.345588</td>
      <td>5.249551</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000084</td>
      <td>3.346672</td>
      <td>5.144906</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.000143</td>
      <td>3.348827</td>
      <td>4.997596</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.000242</td>
      <td>3.352670</td>
      <td>4.810448</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.000412</td>
      <td>3.358709</td>
      <td>4.603154</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.000702</td>
      <td>3.366898</td>
      <td>4.408047</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.001194</td>
      <td>3.376490</td>
      <td>4.252523</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.002031</td>
      <td>3.386611</td>
      <td>4.144918</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.003455</td>
      <td>3.396946</td>
      <td>4.077740</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.005878</td>
      <td>3.407582</td>
      <td>4.038919</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.010000</td>
      <td>3.418347</td>
      <td>4.018141</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.017013</td>
      <td>3.428713</td>
      <td>4.007542</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.028943</td>
      <td>3.438401</td>
      <td>4.001021</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.049239</td>
      <td>3.447793</td>
      <td>3.994133</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.083768</td>
      <td>3.457708</td>
      <td>3.984607</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.142510</td>
      <td>3.468839</td>
      <td>3.972858</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.242446</td>
      <td>3.481455</td>
      <td>3.962098</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.412463</td>
      <td>3.495804</td>
      <td>3.958457</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.701704</td>
      <td>3.512882</td>
      <td>3.971376</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1.193777</td>
      <td>3.534575</td>
      <td>4.011992</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2.030918</td>
      <td>3.562638</td>
      <td>4.086328</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3.455107</td>
      <td>3.597518</td>
      <td>4.187414</td>
    </tr>
    <tr>
      <th>25</th>
      <td>5.878016</td>
      <td>3.638674</td>
      <td>4.296469</td>
    </tr>
    <tr>
      <th>26</th>
      <td>10.000000</td>
      <td>3.686303</td>
      <td>4.392487</td>
    </tr>
    <tr>
      <th>27</th>
      <td>17.012543</td>
      <td>3.742258</td>
      <td>4.458995</td>
    </tr>
    <tr>
      <th>28</th>
      <td>28.942661</td>
      <td>3.809021</td>
      <td>4.486227</td>
    </tr>
    <tr>
      <th>29</th>
      <td>49.238826</td>
      <td>3.889335</td>
      <td>4.478730</td>
    </tr>
    <tr>
      <th>30</th>
      <td>83.767764</td>
      <td>3.989339</td>
      <td>4.470314</td>
    </tr>
    <tr>
      <th>31</th>
      <td>142.510267</td>
      <td>4.121409</td>
      <td>4.524381</td>
    </tr>
    <tr>
      <th>32</th>
      <td>242.446202</td>
      <td>4.300992</td>
      <td>4.693465</td>
    </tr>
    <tr>
      <th>33</th>
      <td>412.462638</td>
      <td>4.541284</td>
      <td>4.968124</td>
    </tr>
    <tr>
      <th>34</th>
      <td>701.703829</td>
      <td>4.854189</td>
      <td>5.289802</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1193.776642</td>
      <td>5.251478</td>
      <td>5.615333</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2030.917621</td>
      <td>5.733147</td>
      <td>5.946439</td>
    </tr>
    <tr>
      <th>37</th>
      <td>3455.107295</td>
      <td>6.275742</td>
      <td>6.304280</td>
    </tr>
    <tr>
      <th>38</th>
      <td>5878.016072</td>
      <td>6.841884</td>
      <td>6.698886</td>
    </tr>
    <tr>
      <th>39</th>
      <td>10000.000000</td>
      <td>7.394722</td>
      <td>7.119279</td>
    </tr>
  </tbody>
</table>
</div>



Below we plot your training and holdout set error for the range of alphas given. You should see a figure similar to [this one from lecture](https://docs.google.com/presentation/d/1-Cga_fOn0dTMt1ss7vNmManX-NUYPXwXDQDAsaInuQM/edit#slide=id.g11981b6c024_154_1068), where training error goes down as model complexity increases, but the error on the held out set is large for extreme values of alpha, and minimized for some intermediate value.

Note that on your plot, the **x-axis is in the inverse of complexity**! In other words, small alpha models (on the left) are complex, because there is no regularization. That's why the training error is lowest on the left side of the plot, as this is where overfitting occurs.


```python
px.line(error_vs_alpha, x = "alpha", y = ["Training Error", "Holdout Error"], log_x=True)
```



From the plot above, what is the best alpha to use?

_training error尽可能小，同时hold-out error尽可能小 ==> 0.01~1左右_

## REMINDER: Test Set vs. Validation Set (a.k.a. Development Set)

In the plots above, we trained our models on a training set, and plotted the resulting RMSE on the training set in blue. We also held out a set of data, and plotted the error on this holdout set in red, calling it the "holdout set error". 

For the example above, since we used the holdout set to pick a hyperparameter, we'd call the holdout set a "validation set" or "development set". These terms are exactly synonomous.

It would not be accurate to call this line the "test set error", because we did not use this dataset as a test set. While it is true that your code never supplied X_test3 or Y_test3 to the fit function of the ridge regression models, ***once you decide to use the holdout set to select*** between different models, different hyperparameters, or different sets of features, then we are not using that dataset as a "test set".

That is, since we've used this holdout set for picking alpha, the resulting errors are no longer unbiased predictors of our performance on unseen models -- the true error on an unseen dataset is likely to be somewhat higher than the validation set. After all, we trained 40 models and picked the best one!

In many real world contexts, model builders will split their data into three sets: training, validation, and test sets, where ***the test set is only ever used once***. That is, there are two holdout sets: One used as a development set (for model selection), and one used a test set (for providing an unbiased estimate of error).

## An Alternate Strategy for Hyper Parameter Selection: K-Fold Cross Validation

Earlier we used the holdout method for model selection (the holdout method is also sometimes called "simple cross validation"). Another approach is K-fold cross validation. This allows us to use more data for training instead of having to set aside some specifically for hyperparameter selection. However, doing so requires more computation resources as we'll have to fit K models per hyperparameter choice.

In our course Data 100, there's really no reason not to use cross validation. However, in environments where models are very expensive to train (e.g. deep learning), you'll typically prefer using a holdout set (simple cross validation) rather than K-fold cross validation.

To emphasize what K-fold cross validation actually means, we're going to manually carry out the procedure. Recall the approach looks something like the figure below for 4-fold cross validation:

<img src="cv.png" width=500px>

When we use K-fold cross validation, rather than using a held out set for model selection, we instead use the training set for model selection. To select between various features, various models, or various hyperparameters, we split the training set further into multiple temporary train and validation sets (each split is called a "fold", hence k-fold cross validation). We will use the average validation error across all k folds to make our optimal feature, model, and hyperparameter choices. In this example, we'll only use this procedure for hyperparameter selection, specifically to choose the best alpha.

### Question 4 重点在于怎么切分数据集！

Scikit-learn has built-in support for cross validation.  However, to better understand how cross validation works complete the following function which cross validates a given model.

1. Use the [`KFold.split`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) function to get 4 splits on the training data. Note that `split` returns the indices of the data for that split.
2. For **each** split:
    1. Select out the training and validation rows and columns based on the split indices and features.
    2. Compute the RMSE on the validation split.
    3. Return the average error across all cross validation splits.

<!--
BEGIN QUESTION
name: q4
-->


```python
from sklearn.model_selection import KFold

def compute_CV_error(model, X_train, Y_train):
    '''
    Split the training data into 4 subsets.
    For each subset, 
        fit a model holding out that subset
        compute the MSE on that subset (the validation set)
    You should be fitting 4 models total.
    Return the average MSE of these 4 folds.

    Args:
        model: an sklearn model with fit and predict functions 
        X_train (data_frame): Training data
        Y_train (data_frame): Label 

    Return:
        the average validation MSE for the 4 splits.
    '''
    kf = KFold(n_splits=4)
    validation_errors = []
    
    for train_idx, valid_idx in kf.split(X_train):
        # split the data
        split_X_train, split_X_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        split_Y_train, split_Y_valid = Y_train.iloc[train_idx], Y_train.iloc[valid_idx]

        # Fit the model on the training split
        model.fit(split_X_train, split_Y_train)
        
        # Compute the RMSE on the validation split
        error = rmse(model.predict(split_X_valid), split_Y_valid)


        validation_errors.append(error)
        
    return np.mean(validation_errors)
```


```python
grader.check("q4")
```

### Question 5

Use `compute_CV_error` to add a new column to `error_vs_alpha` which gives the 4-fold cross validation error for the given choice of alpha.
<!--
BEGIN QUESTION
name: q5
-->


```python
cv_errors = []
range_of_alphas = 10**np.linspace(-5, 4, 40)

for alpha in range_of_alphas:
    cv_error = compute_CV_error(Ridge(alpha=alpha), X_train3, Y_train3)
    cv_errors.append(cv_error)

error_vs_alpha["CV Error"] = cv_errors
```


```python
error_vs_alpha
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
      <th>alpha</th>
      <th>Training Error</th>
      <th>Holdout Error</th>
      <th>CV Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000010</td>
      <td>3.344803</td>
      <td>5.389722</td>
      <td>10.763338</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000017</td>
      <td>3.344885</td>
      <td>5.362696</td>
      <td>10.578003</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000029</td>
      <td>3.345093</td>
      <td>5.318839</td>
      <td>10.254709</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000049</td>
      <td>3.345588</td>
      <td>5.249551</td>
      <td>9.756308</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000084</td>
      <td>3.346672</td>
      <td>5.144906</td>
      <td>9.054988</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.000143</td>
      <td>3.348827</td>
      <td>4.997596</td>
      <td>8.147759</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.000242</td>
      <td>3.352670</td>
      <td>4.810448</td>
      <td>7.069916</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.000412</td>
      <td>3.358709</td>
      <td>4.603154</td>
      <td>5.905299</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.000702</td>
      <td>3.366898</td>
      <td>4.408047</td>
      <td>4.810950</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.001194</td>
      <td>3.376490</td>
      <td>4.252523</td>
      <td>4.104387</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.002031</td>
      <td>3.386611</td>
      <td>4.144918</td>
      <td>4.080071</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.003455</td>
      <td>3.396946</td>
      <td>4.077740</td>
      <td>4.240810</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.005878</td>
      <td>3.407582</td>
      <td>4.038919</td>
      <td>4.224883</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.010000</td>
      <td>3.418347</td>
      <td>4.018141</td>
      <td>4.086858</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.017013</td>
      <td>3.428713</td>
      <td>4.007542</td>
      <td>3.956585</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.028943</td>
      <td>3.438401</td>
      <td>4.001021</td>
      <td>3.889772</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.049239</td>
      <td>3.447793</td>
      <td>3.994133</td>
      <td>3.867618</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.083768</td>
      <td>3.457708</td>
      <td>3.984607</td>
      <td>3.858856</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.142510</td>
      <td>3.468839</td>
      <td>3.972858</td>
      <td>3.850327</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.242446</td>
      <td>3.481455</td>
      <td>3.962098</td>
      <td>3.842001</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.412463</td>
      <td>3.495804</td>
      <td>3.958457</td>
      <td>3.837080</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.701704</td>
      <td>3.512882</td>
      <td>3.971376</td>
      <td>3.838459</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1.193777</td>
      <td>3.534575</td>
      <td>4.011992</td>
      <td>3.848340</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2.030918</td>
      <td>3.562638</td>
      <td>4.086328</td>
      <td>3.867120</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3.455107</td>
      <td>3.597518</td>
      <td>4.187414</td>
      <td>3.893089</td>
    </tr>
    <tr>
      <th>25</th>
      <td>5.878016</td>
      <td>3.638674</td>
      <td>4.296469</td>
      <td>3.924624</td>
    </tr>
    <tr>
      <th>26</th>
      <td>10.000000</td>
      <td>3.686303</td>
      <td>4.392487</td>
      <td>3.962520</td>
    </tr>
    <tr>
      <th>27</th>
      <td>17.012543</td>
      <td>3.742258</td>
      <td>4.458995</td>
      <td>4.009721</td>
    </tr>
    <tr>
      <th>28</th>
      <td>28.942661</td>
      <td>3.809021</td>
      <td>4.486227</td>
      <td>4.070020</td>
    </tr>
    <tr>
      <th>29</th>
      <td>49.238826</td>
      <td>3.889335</td>
      <td>4.478730</td>
      <td>4.149246</td>
    </tr>
    <tr>
      <th>30</th>
      <td>83.767764</td>
      <td>3.989339</td>
      <td>4.470314</td>
      <td>4.257353</td>
    </tr>
    <tr>
      <th>31</th>
      <td>142.510267</td>
      <td>4.121409</td>
      <td>4.524381</td>
      <td>4.406670</td>
    </tr>
    <tr>
      <th>32</th>
      <td>242.446202</td>
      <td>4.300992</td>
      <td>4.693465</td>
      <td>4.607861</td>
    </tr>
    <tr>
      <th>33</th>
      <td>412.462638</td>
      <td>4.541284</td>
      <td>4.968124</td>
      <td>4.870040</td>
    </tr>
    <tr>
      <th>34</th>
      <td>701.703829</td>
      <td>4.854189</td>
      <td>5.289802</td>
      <td>5.203950</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1193.776642</td>
      <td>5.251478</td>
      <td>5.615333</td>
      <td>5.617020</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2030.917621</td>
      <td>5.733147</td>
      <td>5.946439</td>
      <td>6.099994</td>
    </tr>
    <tr>
      <th>37</th>
      <td>3455.107295</td>
      <td>6.275742</td>
      <td>6.304280</td>
      <td>6.625052</td>
    </tr>
    <tr>
      <th>38</th>
      <td>5878.016072</td>
      <td>6.841884</td>
      <td>6.698886</td>
      <td>7.158474</td>
    </tr>
    <tr>
      <th>39</th>
      <td>10000.000000</td>
      <td>7.394722</td>
      <td>7.119279</td>
      <td>7.665518</td>
    </tr>
  </tbody>
</table>
</div>



The code below shows the holdout error that we computed in the previous problem as well as the 4-fold cross validation error. Note that the cross validation error shows a similar dependency on alpha relative to the holdout error. This is because they are both doing the same thing, namely trying to estimate the expected error on unseen data drawn from distribution from which the training set was drawn. 

In other words, this figure compares the holdout method with 4-fold cross validation.

Note: I don't know why the CV error is so much higher for very ***small (i.e. very complex)*** models. Let me know if you figure out why. I suspec ti'ts just random noise.


```python
px.line(error_vs_alpha, x = "alpha", y = ["Holdout Error", "CV Error"], log_x=True)
```



### Extra: Using GridSearchCV 自主找到最佳超参数

Above, we manually performed a search of the space of possible hyperparameters. In this section we'll discuss how to use sklearn to automatically perform such a search. The code below automatically tries out all alpha values in the given range.


```python
from sklearn.model_selection import GridSearchCV
params = {'alpha': 10**np.linspace(-5, 4, 40)}

grid_search = GridSearchCV(Ridge(), params, cv = 4, scoring = "neg_root_mean_squared_error")
grid_search.fit(X_train3, Y_train3)
```




<style>#sk-container-id-3 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-3 {
  color: var(--sklearn-color-text);
}

#sk-container-id-3 pre {
  padding: 0;
}

#sk-container-id-3 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-3 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-3 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-3 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-3 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-3 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-3 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-3 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-3 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-3 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-3 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-3 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "�?;
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-3 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "�?;
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-3 div.sk-label label.sk-toggleable__label,
#sk-container-id-3 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-3 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-3 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-3 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-3 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-3 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-3 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-3 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-3 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-3 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=4, estimator=Ridge(),
             param_grid={&#x27;alpha&#x27;: array([1.00000000e-05, 1.70125428e-05, 2.89426612e-05, 4.92388263e-05,
       8.37677640e-05, 1.42510267e-04, 2.42446202e-04, 4.12462638e-04,
       7.01703829e-04, 1.19377664e-03, 2.03091762e-03, 3.45510729e-03,
       5.87801607e-03, 1.00000000e-02, 1.70125428e-02, 2.89426612e-02,
       4.92388263e-02, 8.37677640e-02, 1.42510267e-01, 2....6202e-01,
       4.12462638e-01, 7.01703829e-01, 1.19377664e+00, 2.03091762e+00,
       3.45510729e+00, 5.87801607e+00, 1.00000000e+01, 1.70125428e+01,
       2.89426612e+01, 4.92388263e+01, 8.37677640e+01, 1.42510267e+02,
       2.42446202e+02, 4.12462638e+02, 7.01703829e+02, 1.19377664e+03,
       2.03091762e+03, 3.45510729e+03, 5.87801607e+03, 1.00000000e+04])},
             scoring=&#x27;neg_root_mean_squared_error&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GridSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=4, estimator=Ridge(),
             param_grid={&#x27;alpha&#x27;: array([1.00000000e-05, 1.70125428e-05, 2.89426612e-05, 4.92388263e-05,
       8.37677640e-05, 1.42510267e-04, 2.42446202e-04, 4.12462638e-04,
       7.01703829e-04, 1.19377664e-03, 2.03091762e-03, 3.45510729e-03,
       5.87801607e-03, 1.00000000e-02, 1.70125428e-02, 2.89426612e-02,
       4.92388263e-02, 8.37677640e-02, 1.42510267e-01, 2....6202e-01,
       4.12462638e-01, 7.01703829e-01, 1.19377664e+00, 2.03091762e+00,
       3.45510729e+00, 5.87801607e+00, 1.00000000e+01, 1.70125428e+01,
       2.89426612e+01, 4.92388263e+01, 8.37677640e+01, 1.42510267e+02,
       2.42446202e+02, 4.12462638e+02, 7.01703829e+02, 1.19377664e+03,
       2.03091762e+03, 3.45510729e+03, 5.87801607e+03, 1.00000000e+04])},
             scoring=&#x27;neg_root_mean_squared_error&#x27;)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">best_estimator_: Ridge</label><div class="sk-toggleable__content fitted"><pre>Ridge(alpha=np.float64(0.41246263829013563))</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;Ridge<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.Ridge.html">?<span>Documentation for Ridge</span></a></label><div class="sk-toggleable__content fitted"><pre>Ridge(alpha=np.float64(0.41246263829013563))</pre></div> </div></div></div></div></div></div></div></div></div>



We can get the average RMSE for the four folds for each of the values of alpha with the code below. In other words, this array is the same as the one you computed earlier when you created the "CV Error" column.


```python
grid_search.cv_results_['mean_test_score']
```




    array([-10.7633381 , -10.57800314, -10.25470921,  -9.75630755,
            -9.05498816,  -8.14775947,  -7.06991566,  -5.90529929,
            -4.8109505 ,  -4.10438693,  -4.08007128,  -4.24080956,
            -4.22488284,  -4.08685828,  -3.95658497,  -3.88977241,
            -3.86761841,  -3.85885628,  -3.85032722,  -3.8420014 ,
            -3.83707965,  -3.83845914,  -3.84833967,  -3.86711956,
            -3.89308871,  -3.92462404,  -3.96251959,  -4.00972106,
            -4.07002011,  -4.14924607,  -4.25735297,  -4.4066697 ,
            -4.60786131,  -4.87004045,  -5.20394987,  -5.61702004,
            -6.09999442,  -6.62505185,  -7.15847442,  -7.66551837])



We can specifically see the lowest RMSE with `best_score_`:


```python
grid_search.best_score_
```




    np.float64(-3.8370796510062055)



And we can get the best model with `best_estimator_`, which you'll note is a Ridge regression model with alpha = 0.412.


```python
grid_search.best_estimator_
```




<style>#sk-container-id-4 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-4 {
  color: var(--sklearn-color-text);
}

#sk-container-id-4 pre {
  padding: 0;
}

#sk-container-id-4 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-4 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-4 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-4 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-4 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-4 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-4 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-4 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-4 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-4 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-4 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-4 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-4 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-4 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "�?;
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-4 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "�?;
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-4 div.sk-label label.sk-toggleable__label,
#sk-container-id-4 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-4 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-4 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-4 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-4 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-4 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-4 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-4 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-4 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-4 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Ridge(alpha=np.float64(0.41246263829013563))</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" checked><label for="sk-estimator-id-8" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Ridge<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.Ridge.html">?<span>Documentation for Ridge</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Ridge(alpha=np.float64(0.41246263829013563))</pre></div> </div></div></div></div>



We can even add the errors from `GridSearchCV` to our `error_vs_alpha` DataFrame and compare the results of our manual 4-fold cross validation with sklearn's implementation:


```python
error_vs_alpha["sklearn CV Score"] = grid_search.cv_results_['mean_test_score']
```


```python
px.line(error_vs_alpha, x = "alpha", y = ["CV Error", "sklearn CV Score"], log_x=True)
```



You'll notice they are exactly the same except that the sklearn CV score is the negative of the error. This is because GridSearchCV is conceptualized as a "maximizer", where the goal is to get the highest possible score, whereas our code was a "minimizer", where the goal was to get the lowest possible error. In other words, the error is just the negative of the score. 镜像由来�?

### Extra: Examining the Residuals of our Optimal Alpha Model

The code below plots the residuals of our best model (Ridge with alpha = 0.412) on the test set. Note that they now seem to be better distributed on either size of the line and are generally closer the line, though with a few more extreme outliers. 


```python
plt.figure(figsize=(10, 6))
predicted_values_on_holdout3 = grid_search.best_estimator_.predict(X_holdout3)
plt.scatter(predicted_values_on_holdout3, Y_holdout3 - predicted_values_on_holdout3, alpha = 0.5)
plt.ylabel("Residual $(y - \hat{y})$")
plt.xlabel("Predicted Prices $(\hat{y})$")
plt.title("Residuals vs Predicted Prices")
plt.title("Residual of prediction for i'th house")
plt.axhline(y = 0, color='r');
```

    <>:4: SyntaxWarning:
    
    invalid escape sequence '\h'
    
    <>:5: SyntaxWarning:
    
    invalid escape sequence '\h'
    
    <>:4: SyntaxWarning:
    
    invalid escape sequence '\h'
    
    <>:5: SyntaxWarning:
    
    invalid escape sequence '\h'
    
    C:\Users\86135\AppData\Local\Temp\ipykernel_6688\3088448444.py:4: SyntaxWarning:
    
    invalid escape sequence '\h'
    
    C:\Users\86135\AppData\Local\Temp\ipykernel_6688\3088448444.py:5: SyntaxWarning:
    
    invalid escape sequence '\h'
    
    


    
![png](lab08_files/lab08_81_1.png)
    


Lastly we can compute the RMSE on the test set. This gives the expected squared error on a new unseen data point that may come to us in the future from the same distribution as our training set.


```python
test_rmse = rmse(grid_search.best_estimator_.predict(X_holdout3), Y_holdout3)
test_rmse
```




    np.float64(3.9584573514348387)



### Extra: LASSO Regression

The code below finds an optimal Lasso model. Note that Lasso regression generalize behaves more poorly numerically, so you'll probably get a bunch of warnings.


```python
from sklearn.linear_model import Lasso
params = {'alpha': 10**np.linspace(-5, 4, 40)}

grid_search_lasso = GridSearchCV(Lasso(), params, cv = 4, scoring = "neg_root_mean_squared_error")
grid_search_lasso.fit(X_train3, Y_train3)
```

    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.262e+03, tolerance: 3.090e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.704e+03, tolerance: 2.493e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.126e+03, tolerance: 3.055e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.923e+03, tolerance: 3.005e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.259e+03, tolerance: 3.090e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.700e+03, tolerance: 2.493e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.122e+03, tolerance: 3.055e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.921e+03, tolerance: 3.005e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.254e+03, tolerance: 3.090e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.693e+03, tolerance: 2.493e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.116e+03, tolerance: 3.055e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.918e+03, tolerance: 3.005e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.244e+03, tolerance: 3.090e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.682e+03, tolerance: 2.493e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.104e+03, tolerance: 3.055e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.912e+03, tolerance: 3.005e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.228e+03, tolerance: 3.090e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.661e+03, tolerance: 2.493e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.085e+03, tolerance: 3.055e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.902e+03, tolerance: 3.005e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.201e+03, tolerance: 3.090e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.626e+03, tolerance: 2.493e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.051e+03, tolerance: 3.055e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.885e+03, tolerance: 3.005e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.153e+03, tolerance: 3.090e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.566e+03, tolerance: 2.493e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.994e+03, tolerance: 3.055e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.856e+03, tolerance: 3.005e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.072e+03, tolerance: 3.090e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.461e+03, tolerance: 2.493e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.895e+03, tolerance: 3.055e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.805e+03, tolerance: 3.005e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.930e+03, tolerance: 3.090e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.273e+03, tolerance: 2.493e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.722e+03, tolerance: 3.055e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.715e+03, tolerance: 3.005e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.681e+03, tolerance: 3.090e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 9.323e+02, tolerance: 2.493e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.433e+03, tolerance: 3.055e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.557e+03, tolerance: 3.005e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.209e+03, tolerance: 3.090e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 5.179e+02, tolerance: 2.493e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 9.328e+02, tolerance: 3.055e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.277e+03, tolerance: 3.005e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 8.499e+02, tolerance: 3.090e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 4.102e+02, tolerance: 2.493e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.951e+02, tolerance: 3.055e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 7.364e+02, tolerance: 3.005e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.381e+02, tolerance: 3.090e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.408e+02, tolerance: 2.493e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 5.715e+02, tolerance: 3.055e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 5.798e+02, tolerance: 3.005e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 9.441e+01, tolerance: 3.090e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.633e+01, tolerance: 2.493e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.300e+02, tolerance: 3.055e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.202e+02, tolerance: 3.005e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.157e+01, tolerance: 3.090e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.022e+01, tolerance: 2.493e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.556e+01, tolerance: 3.055e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.748e+01, tolerance: 3.005e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.546e+01, tolerance: 3.090e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 5.365e+00, tolerance: 2.493e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 9.596e+00, tolerance: 3.005e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.016e+01, tolerance: 3.090e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.053e+01, tolerance: 3.005e+00
    
    d:\miniconda3\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:697: ConvergenceWarning:
    
    Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.123e+01, tolerance: 3.882e+00
    
    




<style>#sk-container-id-5 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-5 {
  color: var(--sklearn-color-text);
}

#sk-container-id-5 pre {
  padding: 0;
}

#sk-container-id-5 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-5 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-5 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-5 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-5 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-5 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-5 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-5 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-5 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-5 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-5 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-5 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-5 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-5 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "�?;
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-5 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-5 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-5 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-5 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "�?;
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-5 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-5 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-5 div.sk-label label.sk-toggleable__label,
#sk-container-id-5 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-5 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-5 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-5 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-5 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-5 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-5 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-5 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-5 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-5 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-5 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-5 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=4, estimator=Lasso(),
             param_grid={&#x27;alpha&#x27;: array([1.00000000e-05, 1.70125428e-05, 2.89426612e-05, 4.92388263e-05,
       8.37677640e-05, 1.42510267e-04, 2.42446202e-04, 4.12462638e-04,
       7.01703829e-04, 1.19377664e-03, 2.03091762e-03, 3.45510729e-03,
       5.87801607e-03, 1.00000000e-02, 1.70125428e-02, 2.89426612e-02,
       4.92388263e-02, 8.37677640e-02, 1.42510267e-01, 2....6202e-01,
       4.12462638e-01, 7.01703829e-01, 1.19377664e+00, 2.03091762e+00,
       3.45510729e+00, 5.87801607e+00, 1.00000000e+01, 1.70125428e+01,
       2.89426612e+01, 4.92388263e+01, 8.37677640e+01, 1.42510267e+02,
       2.42446202e+02, 4.12462638e+02, 7.01703829e+02, 1.19377664e+03,
       2.03091762e+03, 3.45510729e+03, 5.87801607e+03, 1.00000000e+04])},
             scoring=&#x27;neg_root_mean_squared_error&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GridSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=4, estimator=Lasso(),
             param_grid={&#x27;alpha&#x27;: array([1.00000000e-05, 1.70125428e-05, 2.89426612e-05, 4.92388263e-05,
       8.37677640e-05, 1.42510267e-04, 2.42446202e-04, 4.12462638e-04,
       7.01703829e-04, 1.19377664e-03, 2.03091762e-03, 3.45510729e-03,
       5.87801607e-03, 1.00000000e-02, 1.70125428e-02, 2.89426612e-02,
       4.92388263e-02, 8.37677640e-02, 1.42510267e-01, 2....6202e-01,
       4.12462638e-01, 7.01703829e-01, 1.19377664e+00, 2.03091762e+00,
       3.45510729e+00, 5.87801607e+00, 1.00000000e+01, 1.70125428e+01,
       2.89426612e+01, 4.92388263e+01, 8.37677640e+01, 1.42510267e+02,
       2.42446202e+02, 4.12462638e+02, 7.01703829e+02, 1.19377664e+03,
       2.03091762e+03, 3.45510729e+03, 5.87801607e+03, 1.00000000e+04])},
             scoring=&#x27;neg_root_mean_squared_error&#x27;)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">best_estimator_: Lasso</label><div class="sk-toggleable__content fitted"><pre>Lasso(alpha=np.float64(0.017012542798525893))</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;Lasso<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.Lasso.html">?<span>Documentation for Lasso</span></a></label><div class="sk-toggleable__content fitted"><pre>Lasso(alpha=np.float64(0.017012542798525893))</pre></div> </div></div></div></div></div></div></div></div></div>



The best lasso model is below:


```python
grid_search_lasso.best_estimator_
```




<style>#sk-container-id-6 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-6 {
  color: var(--sklearn-color-text);
}

#sk-container-id-6 pre {
  padding: 0;
}

#sk-container-id-6 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-6 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-6 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-6 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-6 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-6 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-6 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-6 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-6 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-6 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-6 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-6 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-6 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-6 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "�?;
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-6 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-6 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-6 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-6 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "�?;
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-6 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-6 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-6 div.sk-label label.sk-toggleable__label,
#sk-container-id-6 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-6 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-6 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-6 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-6 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-6 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-6 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-6 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-6 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-6 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-6 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-6 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-6" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Lasso(alpha=np.float64(0.017012542798525893))</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" checked><label for="sk-estimator-id-12" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Lasso<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.Lasso.html">?<span>Documentation for Lasso</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Lasso(alpha=np.float64(0.017012542798525893))</pre></div> </div></div></div></div>



It's error on the same test set as our best Ridge model is shown below:


```python
test_rmse_lasso = rmse(grid_search_lasso.best_estimator_.predict(X_holdout3), Y_holdout3)
test_rmse_lasso
```




    np.float64(4.054830916690993)



Note that if we tried to use this test error to decide between Ridge and LASSO, then our holdout set is now being used as a validation set, not a test set!! In other words, you get to either use the holdout set to decide between models, or to provide an unbiased estimate of error, but not both!

If we look at the best estimator's parameters, we'll see that many of the parameters are zero, due to the inherent feature selecting nature of a LASSO model.


```python
grid_search_lasso.best_estimator_.coef_
```




    array([-0.00000000e+00, -6.85384379e-01,  0.00000000e+00,  0.00000000e+00,
           -0.00000000e+00, -0.00000000e+00, -7.38599400e-02, -5.29374425e-02,
            5.54295757e-01, -0.00000000e+00, -0.00000000e+00,  0.00000000e+00,
            4.37063521e-01, -3.80592597e+00,  1.61080715e+00,  6.37366884e-01,
           -0.00000000e+00,  2.22834586e-01,  6.01812381e-03, -9.40700489e-02,
           -4.02630887e-01,  3.07990173e-01,  3.72360525e-14,  0.00000000e+00,
           -2.51811102e+00,  0.00000000e+00,  0.00000000e+00,  9.85248689e+00,
           -7.21033868e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
            1.36076846e-01, -0.00000000e+00, -2.00907909e+00, -1.68923341e+00,
            1.77833261e+00,  2.55936962e-01,  5.04076324e-01,  0.00000000e+00,
           -1.82804827e+00,  0.00000000e+00, -0.00000000e+00, -1.53173793e+00,
           -7.84893470e-03,  1.09000336e+00, -5.21734363e+00, -3.87962203e-01])



*We can also stick these parameters in a Series showing us both the weights and the names:* 可解释性强一点点


```python
lasso_weights = pd.Series(grid_search_lasso.best_estimator_.coef_, 
             index = boston_with_extra_features_scaled.columns)
lasso_weights
```




    CRIM          -0.000000e+00
    ZN            -6.853844e-01
    INDUS          0.000000e+00
    CHAS           0.000000e+00
    NOX           -0.000000e+00
    RM            -0.000000e+00
    AGE           -7.385994e-02
    DIS           -5.293744e-02
    RAD            5.542958e-01
    TAX           -0.000000e+00
    PTRATIO       -0.000000e+00
    LSTAT          0.000000e+00
    CRIM^2         4.370635e-01
    sqrtCRIM      -3.805926e+00
    tanhCRIM       1.610807e+00
    ZN^2           6.373669e-01
    sqrtZN        -0.000000e+00
    tanhZN         2.228346e-01
    INDUS^2        6.018124e-03
    sqrtINDUS     -9.407005e-02
    tanhINDUS     -4.026309e-01
    CHAS^2         3.079902e-01
    sqrtCHAS       3.723605e-14
    tanhCHAS       0.000000e+00
    NOX^2         -2.518111e+00
    sqrtNOX        0.000000e+00
    tanhNOX        0.000000e+00
    RM^2           9.852487e+00
    sqrtRM        -7.210339e+00
    tanhRM        -0.000000e+00
    AGE^2         -0.000000e+00
    sqrtAGE       -0.000000e+00
    tanhAGE        1.360768e-01
    DIS^2         -0.000000e+00
    sqrtDIS       -2.009079e+00
    tanhDIS       -1.689233e+00
    RAD^2          1.778333e+00
    sqrtRAD        2.559370e-01
    tanhRAD        5.040763e-01
    TAX^2          0.000000e+00
    sqrtTAX       -1.828048e+00
    tanhTAX        0.000000e+00
    PTRATIO^2     -0.000000e+00
    sqrtPTRATIO   -1.531738e+00
    tanhPTRATIO   -7.848935e-03
    LSTAT^2        1.090003e+00
    sqrtLSTAT     -5.217344e+00
    tanhLSTAT     -3.879622e-01
    dtype: float64



Or sorting by the relative importance of each feature, we see that about a third of the parmaeters didn't end up getting used at all by the LASSO model.


```python
lasso_weights.sort_values(key = abs, ascending = False)
```




    RM^2           9.852487e+00
    sqrtRM        -7.210339e+00
    sqrtLSTAT     -5.217344e+00
    sqrtCRIM      -3.805926e+00
    NOX^2         -2.518111e+00
    sqrtDIS       -2.009079e+00
    sqrtTAX       -1.828048e+00
    RAD^2          1.778333e+00
    tanhDIS       -1.689233e+00
    tanhCRIM       1.610807e+00
    sqrtPTRATIO   -1.531738e+00
    LSTAT^2        1.090003e+00
    ZN            -6.853844e-01
    ZN^2           6.373669e-01
    RAD            5.542958e-01
    tanhRAD        5.040763e-01
    CRIM^2         4.370635e-01
    tanhINDUS     -4.026309e-01
    tanhLSTAT     -3.879622e-01
    CHAS^2         3.079902e-01
    sqrtRAD        2.559370e-01
    tanhZN         2.228346e-01
    tanhAGE        1.360768e-01
    sqrtINDUS     -9.407005e-02
    AGE           -7.385994e-02
    DIS           -5.293744e-02
    tanhPTRATIO   -7.848935e-03
    INDUS^2        6.018124e-03
    sqrtCHAS       3.723605e-14
    INDUS          0.000000e+00
    RM            -0.000000e+00
    NOX           -0.000000e+00
    PTRATIO       -0.000000e+00
    TAX           -0.000000e+00
    LSTAT          0.000000e+00
    CRIM          -0.000000e+00
    CHAS           0.000000e+00
    sqrtNOX        0.000000e+00
    sqrtZN        -0.000000e+00
    tanhCHAS       0.000000e+00
    sqrtAGE       -0.000000e+00
    AGE^2         -0.000000e+00
    tanhRM        -0.000000e+00
    tanhNOX        0.000000e+00
    TAX^2          0.000000e+00
    DIS^2         -0.000000e+00
    PTRATIO^2     -0.000000e+00
    tanhTAX        0.000000e+00
    dtype: float64



## Submission

Congratulations! You are finished with this assignment. 

---



