# DATA100-lab4: Visualization, Transformations, and KDEs


```python
# Initialize Otter
import otter
grader = otter.Notebook("lab04.ipynb")
```

# Lab 4: Visualization, Transformations, and KDEs

### Objective
In this lab you will get some practice plotting, applying data transformations, and working with kernel density estimators (KDEs).  We will be working with data from the World Bank containing various statistics for countries and territories around the world. 


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ds100_utils

plt.style.use('fivethirtyeight') # Use plt.style.available to see more styles
sns.set()
sns.set_context("talk")
%matplotlib inline
```

## Loading Data

Let us load some World Bank data into a `pd.DataFrame` object named ```wb```.


```python
wb = pd.read_csv("data/world_bank_misc.csv", index_col=0)
wb.head()
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
      <th>Primary completion rate: Male: % of relevant age group: 2015</th>
      <th>Primary completion rate: Female: % of relevant age group: 2015</th>
      <th>Lower secondary completion rate: Male: % of relevant age group: 2015</th>
      <th>Lower secondary completion rate: Female: % of relevant age group: 2015</th>
      <th>Youth literacy rate: Male: % of ages 15-24: 2005-14</th>
      <th>Youth literacy rate: Female: % of ages 15-24: 2005-14</th>
      <th>Adult literacy rate: Male: % ages 15 and older: 2005-14</th>
      <th>Adult literacy rate: Female: % ages 15 and older: 2005-14</th>
      <th>Students at lowest proficiency on PISA: Mathematics: % of 15 year-olds: 2015</th>
      <th>Students at lowest proficiency on PISA: Reading: % of 15 year-olds: 2015</th>
      <th>...</th>
      <th>Access to improved sanitation facilities: % of population: 1990</th>
      <th>Access to improved sanitation facilities: % of population: 2015</th>
      <th>Child immunization rate: Measles: % of children ages 12-23 months: 2015</th>
      <th>Child immunization rate: DTP3: % of children ages 12-23 months: 2015</th>
      <th>Children with acute respiratory infection taken to health provider: % of children under age 5 with ARI: 2009-2016</th>
      <th>Children with diarrhea who received oral rehydration and continuous feeding: % of children under age 5 with diarrhea: 2009-2016</th>
      <th>Children sleeping under treated bed nets: % of children under age 5: 2009-2016</th>
      <th>Children with fever receiving antimalarial drugs: % of children under age 5 with fever: 2009-2016</th>
      <th>Tuberculosis: Treatment success rate: % of new cases: 2014</th>
      <th>Tuberculosis: Cases detection rate: % of new estimated cases: 2015</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>62.0</td>
      <td>32.0</td>
      <td>45.0</td>
      <td>18.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>21.0</td>
      <td>32.0</td>
      <td>68.0</td>
      <td>78.0</td>
      <td>62.0</td>
      <td>41.0</td>
      <td>4.6</td>
      <td>11.8</td>
      <td>87.0</td>
      <td>58.0</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>108.0</td>
      <td>105.0</td>
      <td>97.0</td>
      <td>97.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>98.0</td>
      <td>96.0</td>
      <td>26.0</td>
      <td>7.0</td>
      <td>...</td>
      <td>78.0</td>
      <td>93.0</td>
      <td>98.0</td>
      <td>98.0</td>
      <td>70.0</td>
      <td>63.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>88.0</td>
      <td>76.0</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>106.0</td>
      <td>105.0</td>
      <td>68.0</td>
      <td>85.0</td>
      <td>96.0</td>
      <td>92.0</td>
      <td>83.0</td>
      <td>68.0</td>
      <td>51.0</td>
      <td>11.0</td>
      <td>...</td>
      <td>80.0</td>
      <td>88.0</td>
      <td>95.0</td>
      <td>95.0</td>
      <td>66.0</td>
      <td>42.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>88.0</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>American Samoa</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>61.0</td>
      <td>63.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>87.0</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>96.0</td>
      <td>97.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>83.0</td>
      <td>87.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 45 columns</p>
</div>



This table contains some interesting columns.  Take a look:


```python
list(wb.columns)
```




    ['Primary completion rate: Male: % of relevant age group: 2015',
     'Primary completion rate: Female: % of relevant age group: 2015',
     'Lower secondary completion rate: Male: % of relevant age group: 2015',
     'Lower secondary completion rate: Female: % of relevant age group: 2015',
     'Youth literacy rate: Male: % of ages 15-24: 2005-14',
     'Youth literacy rate: Female: % of ages 15-24: 2005-14',
     'Adult literacy rate: Male: % ages 15 and older: 2005-14',
     'Adult literacy rate: Female: % ages 15 and older: 2005-14',
     'Students at lowest proficiency on PISA: Mathematics: % of 15 year-olds: 2015',
     'Students at lowest proficiency on PISA: Reading: % of 15 year-olds: 2015',
     'Students at lowest proficiency on PISA: Science: % of 15 year-olds: 2015',
     'Population: millions: 2016',
     'Surface area: sq. km thousands: 2016',
     'Population density: people per sq. km: 2016',
     'Gross national income, Atlas method: $ billions: 2016',
     'Gross national income per capita, Atlas method: $: 2016',
     'Purchasing power parity gross national income: $ billions: 2016',
     'per capita: $: 2016',
     'Gross domestic product: % growth : 2016',
     'per capita: % growth: 2016',
     'Prevalence of smoking: Male: % of adults: 2015',
     'Prevalence of smoking: Female: % of adults: 2015',
     'Incidence of tuberculosis: per 100,000 people: 2015',
     'Prevalence of diabetes: % of population ages 20 to 79: 2015',
     'Incidence of HIV: Total: % of uninfected population ages 15-49: 2015',
     'Prevalence of HIV: Total: % of population ages 15-49: 2015',
     "Prevalence of HIV: Women's share of population ages 15+ living with HIV: %: 2015",
     'Prevalence of HIV: Youth, Male: % of population ages 15-24: 2015',
     'Prevalence of HIV: Youth, Female: % of population ages 15-24: 2015',
     'Antiretroviral therapy coverage: % of people living with HIV: 2015',
     'Cause of death: Communicable diseases and maternal, prenatal, and nutrition conditions: % of population: 2015',
     'Cause of death: Non-communicable diseases: % of population: 2015',
     'Cause of death: Injuries: % of population: 2015',
     'Access to an improved water source: % of population: 1990',
     'Access to an improved water source: % of population: 2015',
     'Access to improved sanitation facilities: % of population: 1990',
     'Access to improved sanitation facilities: % of population: 2015',
     'Child immunization rate: Measles: % of children ages 12-23 months: 2015',
     'Child immunization rate: DTP3: % of children ages 12-23 months: 2015',
     'Children with acute respiratory infection taken to health provider: % of children under age 5 with ARI: 2009-2016',
     'Children with diarrhea who received oral rehydration and continuous feeding: % of children under age 5 with diarrhea: 2009-2016',
     'Children sleeping under treated bed nets: % of children under age 5: 2009-2016',
     'Children with fever receiving antimalarial drugs: % of children under age 5 with fever: 2009-2016',
     'Tuberculosis: Treatment success rate: % of new cases: 2014',
     'Tuberculosis: Cases detection rate: % of new estimated cases: 2015']



# Part 1: Scaling

In the first part of this assignment we will look at the distribution of values for combined adult literacy rate as well as the gross national income per capita. The code below creates a copy of the DataFrame that contains only the two Series we want, and then drops all rows that contain null values in either column.

**Note:** *For this lab we are dropping null values without investigating them further. However, this is generally not the best practice and can severely affect our analyses.*

Here the combined literacy rate is the sum of the female and male literacy rates as reported by the World Bank. 0 represents no literacy, and 200 would represent total literacy by both genders that are included in the World Bank's dataset.

In this lab, we will be using the `sns.histplot`, `sns.rugplot`, and `sns.displot` function to visualize distributions. You may find it useful to consult the seaborn documentation on [distributions](https://seaborn.pydata.org/tutorial/distributions.html) and [functions](https://seaborn.pydata.org/tutorial/function_overview.html) for more details.


```python
#creates a DataFrame with the appropriate index
df = pd.DataFrame(index=wb.index)

#copies the Series we want
df['lit'] = wb['Adult literacy rate: Female: % ages 15 and older: 2005-14'] + wb["Adult literacy rate: Male: % ages 15 and older: 2005-14"]
df['inc'] = wb['Gross national income per capita, Atlas method: $: 2016']

#the line below drops all records that have a NaN value in either column
df.dropna(inplace=True)
print("Original records:", len(wb))
print("Final records:", len(df))
```

    Original records: 216
    Final records: 147
    


```python
df.head(5)
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
      <th>lit</th>
      <th>inc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>63.0</td>
      <td>580.0</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>194.0</td>
      <td>4250.0</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>151.0</td>
      <td>4270.0</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>142.0</td>
      <td>3440.0</td>
    </tr>
    <tr>
      <th>Antigua and Barbuda</th>
      <td>197.0</td>
      <td>13400.0</td>
    </tr>
  </tbody>
</table>
</div>



## Question 1a

Suppose we wanted to build a histogram of our data to understand the distribution of literacy rates and income per capita individually. We can use [`countplot`](https://seaborn.pydata.org/generated/seaborn.countplot.html) in seaborn to create bar charts from categorical data. 


```python
sns.countplot(x = "lit", data = df)
plt.xlabel("Combined literacy rate: % ages 15 and older: 2005-14")
plt.title('World Bank Combined Adult Literacy Rate')
```




    Text(0.5, 1.0, 'World Bank Combined Adult Literacy Rate')




    
![png](lab04_files/lab04_13_1.png)
    



```python
sns.countplot(x = "inc", data = df)
plt.xlabel('Gross national income per capita, Atlas method: $: 2016')
plt.title('World Bank Gross National Income Per Capita')
```




    Text(0.5, 1.0, 'World Bank Gross National Income Per Capita')




    
![png](lab04_files/lab04_14_1.png)
    


In the cell below, explain why `countplot` is NOT the right tool for visualizing the distribution of our data.

<!--
BEGIN QUESTION
name: q1a
-->

_It is so overwhelming! And ugly!_

## Question 1b

In the cell below, create a plot of **income per capita** (the second plot above) using the [`histplot`](https://seaborn.pydata.org/generated/seaborn.histplot.html) function. As above, you should have two subplots, where the left subplot is literacy, and the right subplot is income. 

Don't forget to title the plot and label axes!

**Hint:** *Copy and paste from above to start.*

<!--
BEGIN QUESTION
name: q1b1
-->


```python
sns.histplot(x = "inc", data = df)
plt.xlabel('Gross national income per capita, Atlas method: $: 2016')
plt.title('World Bank Gross National Income Per Capita');
```


    
![png](lab04_files/lab04_18_0.png)
    


You should see histograms that show the counts of how many data points appear in each bin. `distplot` uses a heuristic called the Freedman-Diaconis rule to automatically identify the best bin sizes, though it is possible to set the bins yourself (we won't).


In the cell below, we explore overlaying a rug plot on top of a histogram using `rugplot`. Note that the rug plot is hard to see.


```python
sns.histplot(x="inc", data = df)
sns.rugplot(x="inc", data = df)
plt.xlabel('Gross national income per capita, Atlas method: $: 2016')
plt.title('World Bank Gross National Income Per Capita')
```




    Text(0.5, 1.0, 'World Bank Gross National Income Per Capita')




    
![png](lab04_files/lab04_20_1.png)
    


One way to make it easier to see the difference between the rug plot and the bars is to set a different *color*, for example:


```python
sns.histplot(x="inc", data = df, color = "lightsteelblue")
sns.rugplot(x="inc", data = df)
plt.xlabel('Gross national income per capita, Atlas method: $: 2016')
plt.title('World Bank Gross National Income Per Capita')
```




    Text(0.5, 1.0, 'World Bank Gross National Income Per Capita')




    
![png](lab04_files/lab04_22_1.png)
    


There is also another function called `kdeplot` which plots a Kernel Density Estimate as described in class, and covered in more detail later in this lab.

Rather than manually calling `histplot`, `rugplot`, and `kdeplot` to plot histograms, rug plots, and KDE plots, respectively, we can instead use `displot`, which can simultaneously plot histogram bars, a rug plot, and a KDE plot, and adjust all the colors automatically for visbility. Using the documentation for [`displot`](https://seaborn.pydata.org/generated/seaborn.displot.html) ([Link](https://seaborn.pydata.org/generated/seaborn.displot.html)), make a plot of the income data that includes a histogram, rug plot, and KDE plot. 

**Hint**: _You'll need to set two parameters to `True`._


```python
sns.displot(x='inc', data=df, kde=True, rug=True)
plt.xlabel('Gross national income per capita, Atlas method: $: 2016') # 太长了，显示不全？
plt.title('World Bank Gross National Income Per Capita')
```




    Text(0.5, 1.0, 'World Bank Gross National Income Per Capita')




    
![png](lab04_files/lab04_24_1.png)
    


You should see roughly the same histogram as before. However, now you should see an overlaid smooth line. This is the kernel density estimate discussed in class. 

Above, the y-axis is labeled by the counts. We can also label the y-axis by the density. An example is given below, this time using the literacy data from the beginning of this lab.


```python
sns.displot(x="lit", data = df, rug = True, kde = True, stat = "density")
plt.xlabel("Adult literacy rate: Combined: % ages 15 and older: 2005-14")
plt.title('World Bank Combined Adult Literacy Rate')
```




    Text(0.5, 1.0, 'World Bank Combined Adult Literacy Rate')




    
![png](lab04_files/lab04_27_1.png)
    


Observations:
* You'll also see that the y-axis value is no longer the count. Instead it is a value such that the total **area** in the histogram is 1. For example, the area of the last bar is approximately 22.22 * 0.028 = 0.62

* The KDE is a smooth estimate of the distribution of the given variable. The area under the KDE is also 1. While it is not obvious from the figure, some of the area under the KDE is beyond the 100% literacy. In other words, the KDE is non-zero for values greater than 100%. This, of course, makes no physical sense. Nonetheless, it is a mathematical feature of the KDE.

We'll talk more about KDEs later in this lab.

## Question 1c

Looking at the income data, it is difficult to see the distribution among low income countries because they are all scrunched up at the left side of the plot. The KDE also has a problem where the density function has a lot of area below 0. 

Transforming the `inc` data logarithmically gives us a more symmetric distribution of values. This can make it easier to see patterns.

In addition, summary statistics like the mean and standard deviation (square-root of the variance) are more stable with symmetric distributions.

In the cell below, make a distribution plot of `inc` with the data transformed using `np.log10` and `kde=True`. If you want to see the exact counts, just set `kde=False`. If you don't specify the `kde` parameter, it is by default set to True. 

**Hint:** Unlike the examples above, you can pass a series to the `displot` function, i.e. rather than passing an entire DataFrame as `data` and a column as `x`, you can instead pass a series.

<!--
BEGIN QUESTION
name: q1c
-->


```python
ax = sns.displot(data=np.log10(df['inc']), kde=True, color='blue')
plt.title('World Bank Gross National Income Per Capita')
plt.ylabel('Density')
plt.xlabel('Log Gross national income per capita, Atlas method: $: 2016');
```


    
![png](lab04_files/lab04_30_0.png)
    


When a distribution has a long right tail, a ***log-transformation*** often does a good job of symmetrizing the distribution, as it did here.  Long right tails are common with variables that have a lower limit on the values. 

On the other hand, long left tails are common with distributions of variables that have an upper limit, such as percentages (can't be higher than 100%) and GPAs (can't be higher than 4).  That is the case for the literacy rate. Typically taking a ***power-transformation*** such 
as squaring or cubing the values can help symmetrize the left skew distribution.

In the cell below, we will make a distribution plot of `lit` with the data transformed using a power, i.e., raise `lit` to the 2nd, 3rd, and 4th power. We plot the transformation with the 4th power below.



```python
ax = sns.displot((df['lit']**4), kde = True) # 经典向量化numpy
plt.ylabel('Density')
plt.xlabel("Adult literacy rate: Combined: % ages 15 and older: 2005-14")
plt.title('World Bank Combined Adult Literacy Rate (4th power)', pad=30);
```


    
![png](lab04_files/lab04_32_0.png)
    


## Question 1d

If we want to examine the relationship between the female adult literacy rate and the gross national income per capita, we need to make a scatter plot. 

In the cell below, create a scatter plot of untransformed income per capita and literacy rate using the `sns.scatterplot` function. Make  sure to label both axes using `plt.xlabel` and `plt.ylabel`.

<!--
BEGIN QUESTION
name: q1d
-->


```python
sns.scatterplot(x=df['lit'], y=df['inc'])
plt.xlabel("Adult literacy rate: Combined: % ages 15 and older")
plt.ylabel('Gross national income per capita (non-log scale)')
plt.title('World Bank: Gross National Income Per Capita vs\n Combined Adult Literacy Rate');
```


    
![png](lab04_files/lab04_34_0.png)
    


We can better assess the relationship between two variables when they have been straightened because it is easier for us to recognize linearity.

In the cell below, we see a scatter plot of log-transformed income per capita against literacy rate.



```python
sns.scatterplot(x = df['lit'], y = np.log10(df['inc']))
plt.xlabel("Adult literacy rate: Combined: % ages 15 and older")
plt.ylabel('Gross national income per capita (log scale)')
plt.title('World Bank: Gross National Income Per Capita vs\n Combined Adult Literacy Rate');
```


    
![png](lab04_files/lab04_36_0.png)
    


## 双变换，思路打开
This scatter plot looks better. The relationship is closer to linear.

We can think of the log-linear relationship between x and y, as follows: a constant change in x corresponds to a percent (scaled) change in y.

We can also see that the long left tail of literacy is represented in this plot by a lot of the points being bunched up near 100. Try squaring literacy and taking the log of income. Does the plot look better? 

<!--
BEGIN QUESTION
name: q1d2
-->


```python
plt.figure(figsize=(10,5))
sns.scatterplot(x = (df['lit']**2), y = np.log10(df['inc']))
plt.xlabel("Adult literacy rate: Combined: % ages 15 and older")
plt.ylabel('Gross national income per capita (log vs. ^2)')
plt.title('World Bank: Gross National Income Per Capita vs\n Combined Adult Literacy Rate');
```


    
![png](lab04_files/lab04_38_0.png)
    


Choosing the best transformation for a relationship is often a balance between keeping the model simple and straightening the scatter plot.

# Part 2: Kernel Density Estimation

In this part of the lab you will develop a deeper understanding of how kernel density estimation works.
- Explain KDE briefly within the lab

### Overview

Kernel density estimation is used to estimate a probability density function (i.e. a density curve) from a set of data. Just like a histogram, a density function's total area must sum to 1.

KDE centrally revolves around this idea of a "kernel". A kernel is a function whose area sums to 1. The three steps involved in building a kernel density estimate are:
1. Placing a kernel at each observation
2. Normalizing kernels so that the sum of their areas is 1
3. Summing all kernels together

:yum:

The end result is a function, that takes in some value `x` and returns a density estimate at the point `x`.

When constructing a KDE, there are several choices to make regarding the kernel. Specifically, we need to choose the function we want to use as our kernel, as well as a bandwidth parameter, which tells us how wide or narrow each kernel should be. We will explore these ideas now.

Suppose we have 3 data points with values 2, 4, and 9. We can compute the (useless) histogram with a KDE as shown below.


```python
data3pts = np.array([2, 4, 9])
sns.displot(data3pts, kde = True, stat = "density");
```


    
![png](lab04_files/lab04_42_0.png)
    


To understand how KDEs are computed, we need to see the KDE outside the given range. The easiest way to do this is to use an old function called `distplot`. During the Spring 2022 offering of this course, `distplot` was still a working function in Seaborn, but it will be removed at a future date. If you get an error that says that `distplot` is not a valid function, sorry, you are too far in the future to do this lab exercise.


```python
sns.distplot(data3pts, kde = True);
```

    C:\Users\86135\AppData\Local\Temp\ipykernel_9696\4279347623.py:1: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(data3pts, kde = True);
    


    
![png](lab04_files/lab04_44_1.png)
    


#### 调整bandwidth
One question you might be wondering is how the kernel density estimator decides how "wide" each point should be. *It turns out this is a parameter you can set called `bw`, which stands for bandwith.* For example, the code below gives a bandwith value of 0.5 to each data point. You'll see the resulting KDE is quite different. Try experimenting with different values of bandwidth and see what happens.


```python
sns.distplot(data3pts, kde = True, kde_kws = {"bw": 0.5});
```

    C:\Users\86135\AppData\Local\Temp\ipykernel_9696\942060009.py:1: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(data3pts, kde = True, kde_kws = {"bw": 0.5});
    d:\miniconda3\envs\ds100\Lib\site-packages\seaborn\distributions.py:2496: UserWarning: 
    
    The `bw` parameter is deprecated in favor of `bw_method` and `bw_adjust`.
    Setting `bw_method=0.5`, but please see the docs for the new parameters
    and update your code. This will become an error in seaborn v0.14.0.
    
      kdeplot(**{axis: a}, ax=ax, color=kde_color, **kde_kws)
    


    
![png](lab04_files/lab04_46_1.png)
    


## Question 2a

As mentioned above, the kernel density estimate (KDE) is just the sum of a bunch of copies of the kernel, each centered on our data points. The default kernel used by the `distplot` function (as well as `kdeplot`) is the Gaussian kernel, given by:

$$\Large
K_\alpha(x, z) = \frac{1}{\sqrt{2 \pi \alpha^2}} \exp\left(-\frac{(x - z)^2}{2  \alpha ^2} \right)
$$

We've implemented the Gaussian kernel for you in Python below. Here, `alpha` is the smoothing or bandwidth parameter $\alpha$ for the KDE, `z` is the center of the Gaussian (i.e. a data point or an array of data points), and `x` is an array of values of the variable whose distribution we are plotting.


```python
def gaussian_kernel(alpha, x, z):
    return 1.0/np.sqrt(2. * np.pi * alpha**2) * np.exp(-(x - z) ** 2 / (2.0 * alpha**2))
```

For example, we can plot the Gaussian kernel centered at 9 with $\alpha$ = 0.5 as below: 


```python
xs = np.linspace(-2, 12, 200)
alpha=0.5
kde_curve = [gaussian_kernel(alpha, x, 9) for x in xs]
plt.plot(xs, kde_curve);
```


    
![png](lab04_files/lab04_52_0.png)
    


In the cell below, plot the 3 kernel density functions corresponding to our 3 data points on the same axis. Use an `alpha` value of 0.5. Recall that our three data points are 2, 4, and 9. 

**Note:** Make sure to normalize your kernels! This means that the area under each of your kernels should be $\frac{1}{3}$ since there are three data points.

You don't have to use the following hints, but they might be helpful in simplifying your code.

**Hint:** The `gaussian_kernel` function can also take a numpy array as an argument for `z`.

**Hint:** To plot multiple plots at once, you can use `plt.plot(xs, y)` with a two dimensional array as `y`.

<!--
BEGIN QUESTION
name: q2a
-->


```python
xs = np.linspace(-2, 12, 200)
alpha=0.5
kde_curve = [1/3*gaussian_kernel(alpha, x, data3pts) for x in xs] # 注意“正则化”！
plt.plot(xs, kde_curve);
```


    
![png](lab04_files/lab04_54_0.png)
    


In the cell below, we see a plot that shows the sum of all three of the kernels above. The plot resembles the kde shown when you called `distplot` function with bandwidth 0.5 earlier. The area under the final curve will be 1 since the area under each of the three normalized kernels is $\frac{1}{3}$.



```python
xs = np.linspace(-2, 12, 200)
alpha=0.5
kde_curve = np.array([1/3 * gaussian_kernel(alpha, x, data3pts) for x in xs])
plt.plot(xs, np.sum(kde_curve, axis = 1)); # 叠加曲线！
```


    
![png](lab04_files/lab04_56_0.png)
    


Recall that earlier we plotted the kernel density estimation for the logarithm of the income data, as shown again below.


```python
ax = sns.displot(np.log10(df['inc']), kind = "kde", rug = True)
plt.title('World Bank Gross National Income Per Capita')
plt.xlabel('Log Gross national income per capita, Atlas method: $: 2016');
```


    
![png](lab04_files/lab04_58_0.png)
    


In the cell below, a similar plot is shown using what was done in 2a. Try out different values of alpha in {0.1, 0.2, 0.3, 0.4, 0.5}. You will see that when alpha=0.2, the graph matches the previous graph well, except that the `displot` function hides the KDE values outside the range of the available data.


```python
xs = np.linspace(1, 6, 200)
alpha=0.2
kde_curve = np.array([1/len(df['inc']) * gaussian_kernel(alpha, x, np.log10(df['inc'])) for x in xs])
plt.title('World Bank Gross National Income Per Capita')
plt.xlabel('Log Gross national income per capita, Atlas method: $: 2016')
plt.plot(xs, np.sum(kde_curve, axis = 1));
```


    
![png](lab04_files/lab04_60_0.png)
    


## Question 2b

In your answers above, you hard-coded a lot of your work. In this problem, you'll build a more general kernel density estimator function.

Implement the KDE function which computes:

$$\Large
f_\alpha(x) = \frac{1}{n} \sum_{i=1}^n K_\alpha(x, z_i)
$$

Where $z_i$ are the data, $\alpha$ is a parameter to control the smoothness, and $K_\alpha$ is the kernel density function passed as `kernel`.

<!--
BEGIN QUESTION
name: q2b
-->


```python
def kde(kernel, alpha, x, data):
    """
    Compute the kernel density estimate for the single query point x.

    Args:
        kernel: a kernel function with 3 parameters: alpha, x, data
        alpha: the smoothing parameter to pass to the kernel
        x: a single query point (in one dimension)
        data: a numpy array of data points

    Returns:
        The smoothed estimate at the query point x
    """    
    return sum(kernel(alpha, x, zi) for zi in data) / len(data)
```


```python
grader.check("q2b")
```

Assuming you implemented `kde` correctly, the code below should generate the `kde` of the log of the income data as before.


```python
df['trans_inc'] = np.log10(df['inc'])
xs = np.linspace(df['trans_inc'].min(), df['trans_inc'].max(), 1000)
curve = [kde(gaussian_kernel, alpha, x, df['trans_inc']) for x in xs]
plt.hist(df['trans_inc'], density=True, color='orange')
plt.title('World Bank Gross National Income Per Capita')
plt.xlabel('Log Gross national income per capita, Atlas method: $: 2016');
plt.plot(xs, curve, 'k-');
```


    
![png](lab04_files/lab04_67_0.png)
    


And the code below should show a 3 x 3 set of plots showing the output of the kde for different `alpha` values. small to large


```python
plt.figure(figsize=(15,15))
alphas = np.arange(0.2, 2.0, 0.2)
for i, alpha in enumerate(alphas):
    plt.subplot(3, 3, i+1)
    xs = np.linspace(df['trans_inc'].min(), df['trans_inc'].max(), 1000)
    curve = [kde(gaussian_kernel, alpha, x, df['trans_inc']) for x in xs]
    plt.hist(df['trans_inc'], density=True, color='orange')
    plt.plot(xs, curve, 'k-')
plt.show()
```


    
![png](lab04_files/lab04_69_0.png)
    


Let's take a look at another kernel, the Boxcar kernel.


```python
def boxcar_kernel(alpha, x, z):
    return (((x-z)>=-alpha/2)&((x-z)<=alpha/2))/alpha
```

Run the cell below to enable interactive plots. It should give you a green 'OK' when it's finished.


```python
from ipywidgets import interact
!jupyter nbextension enable --py widgetsnbextension
# 这个是要notebook降级处理的插件
```

    Enabling notebook extension jupyter-js-widgets/extension...
          - Validating: ok
    

Now, we can plot the Boxcar and Gaussian kernel functions to see what they look like.


```python
x = np.linspace(-10,10,1000)
def f(alpha):
    plt.plot(x, boxcar_kernel(alpha,x,0), label='Boxcar')
    plt.plot(x, gaussian_kernel(alpha,x,0), label='Gaussian')
    plt.legend(title='Kernel Function')
    plt.show()
interact(f, alpha=(1,10,0.1));
```


    interactive(children=(FloatSlider(value=5.0, description='alpha', max=10.0, min=1.0), Output()), _dom_classes=…


Using the interactive plot below compare the the two kernel techniques:  (Generating the KDE plot is slow, so you may expect some latency after you move the slider)


```python
xs = np.linspace(df['trans_inc'].min(), df['trans_inc'].max(), 1000)
def f(alpha_g, alpha_b):
    plt.hist(df['trans_inc'], density=True, color='orange')
    g_curve = [kde(gaussian_kernel, alpha_g, x, df['trans_inc']) for x in xs]
    plt.plot(xs, g_curve, 'k-', label='Gaussian')
    b_curve = [kde(boxcar_kernel, alpha_b, x, df['trans_inc']) for x in xs]
    plt.plot(xs, b_curve, 'r-', label='Boxcar')
    plt.legend(title='Kernel Function')
    plt.show()
interact(f, alpha_g=(0.01,.5,0.01), alpha_b=(0.01,3,0.1));
```


    interactive(children=(FloatSlider(value=0.25, description='alpha_g', max=0.5, min=0.01, step=0.01), FloatSlide…


Briefly compare and contrast the Gaussian and Boxcar kernels in the cell below. How do the two kernels relate with each other for the same alpha value?

圆滑问题

**Congrats!** You are finished with this assignment.

## Optional--px用法讲解

Below are some examples using plotly. Recall that this is ***Josh's preferred*** plotting library, though it is not officially covered nor required in this class. This is purely for your future reference if you decide to use plotly on your own.


```python
import plotly.express as px
```


```python
px.histogram(df, x = "lit")
```



In my opinion, distribution plots are the one place where plotly falls short of seaborn. For example, if we want a rug, KDE, and histogram, the code below does this in plotly. I'm not personally a fan.


```python
import plotly.figure_factory as ff
ff.create_distplot([df["lit"]], ["lit"])
```



By contrast, I think many of plotly's other features are far superior to seaborn. For example, consider the interactive scatterplot below, where one can mouseover each datapoint in order to see the identity of each country.


```python
px.scatter(df, x = "lit", y = "inc", hover_name = df.index,
          labels={
                     "lit": "Adult literacy rate: Combined: % ages 15 and older",
                     "inc": "Gross national income per capita"
                 },
                title="World Bank: Gross National Income Per Capita vs\n Combined Adult Literacy Rate"
)
```



Naturally there are ways to adjust figure size, text size, marker, etc, but they are not covered here. I just wanted to give you a small taste of plotly.

---

