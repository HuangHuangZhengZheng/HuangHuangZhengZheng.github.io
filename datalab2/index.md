# DATA100-lab2

```python
# Initialize Otter
import otter
grader = otter.Notebook("lab02.ipynb")
```

## Lab 2: Pandas Overview 基础操作，注意numpy版本可能过高

To receive credit for a lab, answer all questions correctly and submit before the deadline.

**This lab is due  Tuesday, Feb 1st at 11:59 PM.**


### Collaboration Policy

Data science is a collaborative activity. While you may talk with others about the labs, we ask that you **write your solutions individually**. If you do discuss the assignments with others please **include their names** below. (That's a good way to learn your classmates' names.)

**Collaborators**: *list collaborators here*

---
[Pandas](https://pandas.pydata.org/) is one of the most widely used Python libraries in data science. In this lab, you will review commonly used data wrangling operations/tools in Pandas. We aim to give you familiarity with:

* Creating DataFrames
* Slicing DataFrames (i.e. selecting rows and columns)
* Filtering data (using boolean arrays and groupby.filter)
* Aggregating (using groupby.agg)

In this lab you are going to use several pandas methods. Reminder from lecture that you may press `shift+tab` on method parameters to see the documentation for that method. For example, if you were using the `drop` method in pandas, you couold press shift+tab to see what `drop` is expecting.

Pandas is very similar to the datascience library that you saw in Data 8. This [conversion notebook](https://github.com/data-8/materials-x19/blob/master/reference/Datascience%20to%20Pandas%20Conversion%20Notebook.ipynb) may serve as a useful guide!

This lab expects that you have watched the pandas lectures. If you have not, this lab will probably take a very long time.

**Note**: The Pandas interface is notoriously confusing for beginners, and the documentation is not consistently great. Throughout the semester, you will have to search through Pandas documentation and experiment, but remember it is part of the learning experience and will help shape you as a data scientist!


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
%matplotlib inline
```

## Creating DataFrames & Basic Manipulations

Recall that a [DataFrame](https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#dataframe) is a table in which each column has a specific data type; there is an index over the columns (typically string labels) and an index over the rows (typically ordinal numbers).

Usually you'll create DataFrames by using a function like `pd.read_csv`. However, in this section, we'll discuss how to create them from scratch.

The [documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) for the pandas `DataFrame` class provides several constructors for the DataFrame class.

**Syntax 1:** You can create a DataFrame by specifying the columns and values using a ***dictionary*** as shown below. 

The keys of the dictionary are the column names, and the values of the dictionary are lists containing the row entries.


```python
fruit_info = pd.DataFrame(
    data = {'fruit': ['apple', 'orange', 'banana', 'raspberry'],
          'color': ['red', 'orange', 'yellow', 'pink'],
          'price': [1.0, 0.75, 0.35, 0.05]
          })
fruit_info
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
      <th>fruit</th>
      <th>color</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>red</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>orange</td>
      <td>orange</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>banana</td>
      <td>yellow</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>raspberry</td>
      <td>pink</td>
      <td>0.05</td>
    </tr>
  </tbody>
</table>
</div>



**Syntax 2:** You can also define a DataFrame by specifying the rows as shown below. 

Each row corresponds to a distinct ***tuple***, and the columns are specified separately.


```python
fruit_info2 = pd.DataFrame(
    [("red", "apple", 1.0), ("orange", "orange", 0.75), ("yellow", "banana", 0.35),
     ("pink", "raspberry", 0.05)], 
    columns = ["color", "fruit", "price"])
fruit_info2
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
      <th>color</th>
      <th>fruit</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>red</td>
      <td>apple</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>orange</td>
      <td>orange</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>yellow</td>
      <td>banana</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pink</td>
      <td>raspberry</td>
      <td>0.05</td>
    </tr>
  </tbody>
</table>
</div>



You can obtain the dimensions of a DataFrame by using the shape attribute `DataFrame.shape`.


```python
fruit_info.shape
```




    (4, 3)



You can also **convert the entire DataFrame into a two-dimensional NumPy array.**


```python
fruit_info.values
```




    array([['apple', 'red', 1.0],
           ['orange', 'orange', 0.75],
           ['banana', 'yellow', 0.35],
           ['raspberry', 'pink', 0.05]], dtype=object)



There are other constructors but we do not discuss them here.

### REVIEW: Selecting Rows and Columns in Pandas

As you've seen in lecture and discussion, there are two verbose operators in Python for selecting rows: `loc` and `iloc`. Let's review them briefly.

#### Approach 1: `loc`

The first of the two verbose operators is `loc`, which takes two arguments. The first is one or more row **labels**, the second is one or more column **labels**.

The desired rows or columns can be provided individually, in slice notation, or as a list. Some examples are given below.

Note that **slicing in `loc` is inclusive** on the provided labels.


```python
#get rows 0 through 2 and columns fruit through price
fruit_info.loc[0:2, 'fruit':'price'] # 闭区间
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
      <th>fruit</th>
      <th>color</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>red</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>orange</td>
      <td>orange</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>banana</td>
      <td>yellow</td>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get rows 0 through 2 and columns fruit and price. 
# Note the difference in notation and result from the previous example.
fruit_info.loc[0:2, ['fruit', 'price']] # 离散的
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
      <th>fruit</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>orange</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>banana</td>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get rows 0 and 2 and columns fruit and price. 
fruit_info.loc[[0, 2], ['fruit', 'price']] # 更加离散的
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
      <th>fruit</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>banana</td>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get rows 0 and 2 and column fruit
fruit_info.loc[[0, 2], ['fruit']]
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
      <th>fruit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
    </tr>
    <tr>
      <th>2</th>
      <td>banana</td>
    </tr>
  </tbody>
</table>
</div>



Note that if we request a single column but don't enclose it in a list, the return type of the `loc` operator is a `Series` rather than a DataFrame. 

注意[ ]包裹问题


```python
# get rows 0 and 2 and column fruit, returning the result as a Series
fruit_info.loc[[0, 2], 'fruit'] 
```




    0     apple
    2    banana
    Name: fruit, dtype: object



If we provide only one argument to `loc`, it uses the provided argument to select rows, and returns all columns.


```python
fruit_info.loc[0:1] # 可以只给行， 不可以只给列
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
      <th>fruit</th>
      <th>color</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>red</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>orange</td>
      <td>orange</td>
      <td>0.75</td>
    </tr>
  </tbody>
</table>
</div>



Note that if you try to access columns without providing rows, `loc` will crash. 


```python
# uncomment, this code will crash
# fruit_info.loc[["fruit", "price"]]

# uncomment, this code works fine: 
fruit_info.loc[:, ["fruit", "price"]]
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
      <th>fruit</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>orange</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>banana</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>raspberry</td>
      <td>0.05</td>
    </tr>
  </tbody>
</table>
</div>



#### Approach 2: `iloc`

`iloc` is very similar to `loc` except that its arguments are row numbers and column numbers, rather than row labels and labels names. A usueful mnemonic is that the `i` stands for "integer".

In addition, **slicing for `iloc` is exclusive** on the provided integer indices. Some examples are given below:

考虑此时变成python经典索引


```python
# get rows 0 through 3 (exclusive) and columns 0 through 2 (exclusive)
fruit_info.iloc[0:3, 0:3]
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
      <th>fruit</th>
      <th>color</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>red</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>orange</td>
      <td>orange</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>banana</td>
      <td>yellow</td>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get rows 0 through 3 (exclusive) and columns 0 and 2.
fruit_info.iloc[0:3, [0, 2]]
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
      <th>fruit</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>orange</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>banana</td>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get rows 0 and 2 and columns 0 and 2.
fruit_info.iloc[[0, 2], [0, 2]]
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
      <th>fruit</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>banana</td>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




```python
#get rows 0 and 2 and column fruit
fruit_info.iloc[[0, 2], [0]]
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
      <th>fruit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
    </tr>
    <tr>
      <th>2</th>
      <td>banana</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get rows 0 and 2 and column fruit
fruit_info.iloc[[0, 2], 0] # return a Series!
```




    0     apple
    2    banana
    Name: fruit, dtype: object



Note that in these loc and iloc examples above, the row **label** and row **number** were always the same.

Let's see an example where they are different. If we sort our fruits by *price*, we get:


```python
fruit_info_sorted = fruit_info.sort_values("price")
fruit_info_sorted
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
      <th>fruit</th>
      <th>color</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>raspberry</td>
      <td>pink</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>banana</td>
      <td>yellow</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>1</th>
      <td>orange</td>
      <td>orange</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>red</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



Observe that the row number 0 now has index 3, row number 1 now has index 2, etc. These indices are the arbitrary numerical index generated when we created the DataFrame. For example, banana was originally in row 2, and so it has row label 2.

If we request the rows in positions 0 and 2 using `iloc`,  **we're indexing using the row NUMBERS, not labels.**

这里似乎并不是按照lab所说的那样？


```python
fruit_info_sorted.iloc[[0, 2], 0]
```




    3    raspberry
    1       orange
    Name: fruit, dtype: object



Lastly, similar to with `loc`, the second argument to `iloc` is optional. That is, if you provide only one argument to `iloc`, it treats the argument you provide as a set of desired row numbers, not column numbers.


```python
fruit_info.iloc[[0, 2]]
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
      <th>fruit</th>
      <th>color</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>red</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>banana</td>
      <td>yellow</td>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>



#### Approach 3: `[]` Notation for Accessing Rows and Columns

Pandas also supports a bare `[]` operator. It's similar to `loc` in that it lets you access rows and columns by their name.

However, unlike `loc`, which takes row names and also optionally column names, `[]` is more flexible. If you provde it only row names, it'll give you rows (same behavior as `loc`), and if you provide it with only column names, it'll give you columns (whereas `loc` will crash).

Some examples:


```python
fruit_info[0:2]
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
      <th>fruit</th>
      <th>color</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>red</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>orange</td>
      <td>orange</td>
      <td>0.75</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Here we're providing a list of fruits as single argument to []
fruit_info[["fruit", "color", "price"]]
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
      <th>fruit</th>
      <th>color</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>red</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>orange</td>
      <td>orange</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>banana</td>
      <td>yellow</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>raspberry</td>
      <td>pink</td>
      <td>0.05</td>
    </tr>
  </tbody>
</table>
</div>



Note that **slicing notation is not supported for columns if you use `[]` notation.** Use `loc` instead.


```python
# uncomment and this code crashes
# fruit_info["fruit":"price"]

# uncomment and this works fine
fruit_info.loc[:, "fruit":"price"]
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
      <th>fruit</th>
      <th>color</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>red</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>orange</td>
      <td>orange</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>banana</td>
      <td>yellow</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>raspberry</td>
      <td>pink</td>
      <td>0.05</td>
    </tr>
  </tbody>
</table>
</div>



`[]` and `loc` are quite similar. For example, the following two pieces of code are functionally equivalent for selecting the fruit and price columns.

1. `fruit_info[["fruit", "price"]]` 
2. `fruit_info.loc[:, ["fruit", "price"]]`.

Because it yields more concise code, you'll find that our code and your code both tend to feature `[]`. However, there are some subtle pitfalls of using `[]`. If you're ever having performance issues, weird behavior, or you see a `SettingWithCopyWarning` in pandas, switch from `[]` to `loc` and this may help.

To avoid getting too bogged down in indexing syntax, we'll avoid a more thorough discussion of `[]` and `loc`. We may return to this at a later point in the course.

For more on `[]` vs `loc`, you may optionally try reading:
1. https://stackoverflow.com/questions/48409128/what-is-the-difference-between-using-loc-and-using-just-square-brackets-to-filte
2. https://stackoverflow.com/questions/38886080/python-pandas-series-why-use-loc/65875826#65875826
3. https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas/53954986#53954986

Now that we've reviewed basic indexing, let's discuss how we can modify dataframes. We'll do this via a series of exercises. 

### Question 1(a)

For a DataFrame `d`, you can add a column with `d['new column name'] = ...` and assign a list or array of values to the column. Add a column of integers containing 1, 2, 3, and 4 called `rank1` to the `fruit_info` table which expresses your personal preference about the taste ordering for each fruit (1 is tastiest; 4 is least tasty). 

<!--
BEGIN QUESTION
name: q1a
-->


```python
fruit_info['rank1'] = [1,2,3,4]
fruit_info
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
      <th>fruit</th>
      <th>color</th>
      <th>price</th>
      <th>rank1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>red</td>
      <td>1.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>orange</td>
      <td>orange</td>
      <td>0.75</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>banana</td>
      <td>yellow</td>
      <td>0.35</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>raspberry</td>
      <td>pink</td>
      <td>0.05</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
grader.check("q1a")
```




<p><strong><pre style='display: inline;'>q1a</pre></strong> passed! 💯</p>



### Question 1(b)

You can also add a column to `d` with `d.loc[:, 'new column name'] = ...`. As above, the first parameter is for the rows and second is for columns. The `:` means change all rows and the `'new column name'` indicates the name of the column you are modifying (or in this case, adding). 

Add a column called `rank2` to the `fruit_info` table which contains the same values in the same order as the `rank1` column.

<!--
BEGIN QUESTION
name: q1b
-->


```python
fruit_info.loc[:, 'rank2'] = [1,2,3,4]
fruit_info
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
      <th>fruit</th>
      <th>color</th>
      <th>price</th>
      <th>rank1</th>
      <th>rank2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>red</td>
      <td>1.00</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>orange</td>
      <td>orange</td>
      <td>0.75</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>banana</td>
      <td>yellow</td>
      <td>0.35</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>raspberry</td>
      <td>pink</td>
      <td>0.05</td>
      <td>4</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
grader.check("q1b")
```




<p><strong><pre style='display: inline;'>q1b</pre></strong> passed! 🍀</p>



### Question 2

Use the `.drop()` method to [drop](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html) both the `rank1` and `rank2` columns you created. Make sure to use the `axis` parameter correctly. Note that `drop` does not change a table, but instead returns a new table with fewer columns or rows unless you set the optional `inplace` parameter.

*Hint*: Look through the [documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html) to see how you can drop multiple columns of a Pandas DataFrame at once using a list of column names.

<!--
BEGIN QUESTION
name: q2
-->


```python
fruit_info_original = fruit_info.drop(labels=['rank1','rank2'],axis=1)
fruit_info_original
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
      <th>fruit</th>
      <th>color</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>red</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>orange</td>
      <td>orange</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>banana</td>
      <td>yellow</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>raspberry</td>
      <td>pink</td>
      <td>0.05</td>
    </tr>
  </tbody>
</table>
</div>




```python
grader.check("q2")
```




<p><strong><pre style='display: inline;'>q2</pre></strong> passed! 💯</p>



### Question 3

Use the `.rename()` method to [rename](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html) the columns of `fruit_info_original` so they begin with capital letters. Set this new DataFrame to `fruit_info_caps`. For an example of how to use rename, see the linked documentation above.
<!--
BEGIN QUESTION
name: q3
-->


```python
fruit_info_caps = fruit_info_original.rename(columns={'fruit':'Fruit', 'color':'Color', 'price':'Price'})
fruit_info_caps
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
      <th>Fruit</th>
      <th>Color</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>red</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>orange</td>
      <td>orange</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>banana</td>
      <td>yellow</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>raspberry</td>
      <td>pink</td>
      <td>0.05</td>
    </tr>
  </tbody>
</table>
</div>




```python
grader.check("q3")
```




<p><strong><pre style='display: inline;'>q3</pre></strong> passed! 🎉</p>



### Babynames Dataset
For the new few questions of this lab, let's move on to a real world dataset. We'll be using the babynames dataset from Lecture 1. The babynames dataset contains a record of the given names of babies born in the United States each year.

First let's run the following cells to build the DataFrame `baby_names`.
The cells below download the data from the web and extract the data into a DataFrame. There should be a total of 6215834 records.

### `fetch_and_cache` Helper

The following function downloads and caches data in the `data/` directory and returns the `Path` to the downloaded file. The cell below the function describes how it works. You are not expected to understand this code, but you may find it useful as a reference as a practitioner of data science after the course. 


```python
import requests
from pathlib import Path

def fetch_and_cache(data_url, file, data_dir="data", force=False):
    """
    Download and cache a url and return the file object.
    
    data_url: the web address to download
    file: the file in which to save the results.
    data_dir: (default="data") the location to save the data
    force: if true the file is always re-downloaded 
    
    return: The pathlib.Path to the file.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir/Path(file)
    if force and file_path.exists():
        file_path.unlink()
    if force or not file_path.exists():
        print('Downloading...', end=' ')
        resp = requests.get(data_url)
        with file_path.open('wb') as f:
            f.write(resp.content)
        print('Done!')
    else:
        import time 
        created = time.ctime(file_path.stat().st_ctime)
        print("Using cached version downloaded at", created)
    return file_path
```

In Python, a `Path` object represents the filesystem paths to files (and other resources). The `pathlib` module is effective for writing code that works on different operating systems and filesystems. 

To check if a file exists at a path, use `.exists()`. To create a directory for a path, use `.mkdir()`. To remove a file that might be a [symbolic link](https://en.wikipedia.org/wiki/Symbolic_link), use `.unlink()`. 

This function creates a path to a directory that will contain data files. It ensures that the directory exists (which is required to write files in that directory), then proceeds to download the file based on its URL.

The benefit of this function is that not only can you force when you want a new file to be downloaded using the `force` parameter, but in cases when you don't need the file to be re-downloaded, you can use the cached version and save download time.

Below we use `fetch_and_cache` to download the `namesbystate.zip` zip file, which is a compressed directory of CSV files. 

**This might take a little while! Consider stretching.**


```python
data_url = 'https://www.ssa.gov/oact/babynames/state/namesbystate.zip'
namesbystate_path = fetch_and_cache(data_url, 'namesbystate.zip')
```

    Using cached version downloaded at Fri Jul 12 20:04:41 2024
    

The following cell builds the final full `baby_names` DataFrame. It first builds one DataFrame per state, because that's how the data are stored in the zip file. Here is documentation for [pd.concat](https://pandas.pydata.org/pandas-docs/version/1.2/reference/api/pandas.concat.html) if you want to know more about its functionality. As before, you are not expected to understand this code. 


```python
import zipfile
zf = zipfile.ZipFile(namesbystate_path, 'r')

column_labels = ['State', 'Sex', 'Year', 'Name', 'Count']

def load_dataframe_from_zip(zf, f):
    with zf.open(f) as fh: 
        return pd.read_csv(fh, header=None, names=column_labels)

states = [
    load_dataframe_from_zip(zf, f)
    for f in sorted(zf.filelist, key=lambda x:x.filename) 
    if f.filename.endswith('.TXT')
]

baby_names = states[0]
for state_df in states[1:]:
    baby_names = pd.concat([baby_names, state_df])
baby_names = baby_names.reset_index().iloc[:, 1:]
```


```python
len(baby_names)
```




    6215834




```python
baby_names.head()
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
      <th>State</th>
      <th>Sex</th>
      <th>Year</th>
      <th>Name</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AK</td>
      <td>F</td>
      <td>1910</td>
      <td>Mary</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AK</td>
      <td>F</td>
      <td>1910</td>
      <td>Annie</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AK</td>
      <td>F</td>
      <td>1910</td>
      <td>Anna</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AK</td>
      <td>F</td>
      <td>1910</td>
      <td>Margaret</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AK</td>
      <td>F</td>
      <td>1910</td>
      <td>Helen</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



### Selection Examples on Baby Names

As with our synthetic fruit dataset, we can use `loc` and `iloc` to select rows and columns of interest from our dataset.


```python
baby_names.loc[2:5, 'Name']# Series
```




    2        Anna
    3    Margaret
    4       Helen
    5       Elsie
    Name: Name, dtype: object



Notice the difference between the following cell and the previous one, just passing in `'Name'` returns a Series while `['Name']` returns a DataFrame.


```python
baby_names.loc[2:5, ['Name']] #df
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
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Anna</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Margaret</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Helen</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Elsie</td>
    </tr>
  </tbody>
</table>
</div>



The code below collects the rows in positions 1 through 3, and the column in position 3 ("Name").


```python
baby_names.iloc[1:4, [3]]
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
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Annie</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Anna</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Margaret</td>
    </tr>
  </tbody>
</table>
</div>



### Question 4

Use `.loc` to select `Name` and `Year` **in that order** from the `baby_names` table.

<!--
BEGIN QUESTION
name: q4
-->


```python
name_and_year = baby_names.loc[:, ['Name', 'Year']]
name_and_year[:5]
# 版本问题
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
      <th>Name</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>1910</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Annie</td>
      <td>1910</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Anna</td>
      <td>1910</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Margaret</td>
      <td>1910</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Helen</td>
      <td>1910</td>
    </tr>
  </tbody>
</table>
</div>




```python
grader.check("q4")
```




<p><strong style='color: red;'><pre style='display: inline;'>q4</pre> results:</strong></p><p><strong><pre style='display: inline;'>q4 - 1</pre> result:</strong></p><pre>    ✅ Test case passed</pre><p><strong><pre style='display: inline;'>q4 - 2</pre> result:</strong></p><pre>    ✅ Test case passed</pre><p><strong><pre style='display: inline;'>q4 - 3</pre> result:</strong></p><pre>    ❌ Test case failed
    Trying:
        name_and_year.loc[0, "Year"] 
    Expecting:
        1910
    **********************************************************************
    Line 1, in q4 2
    Failed example:
        name_and_year.loc[0, "Year"] 
    Expected:
        1910
    Got:
        np.int64(1910)
</pre>



Now repeat the same selection using the plain `[]` notation.

接受一个list of column


```python
name_and_year = baby_names[['Name','Year']]
name_and_year[:5]
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
      <th>Name</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>1910</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Annie</td>
      <td>1910</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Anna</td>
      <td>1910</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Margaret</td>
      <td>1910</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Helen</td>
      <td>1910</td>
    </tr>
  </tbody>
</table>
</div>



## Filtering Data

### Review: Filtering with boolean arrays

Filtering is the process of removing unwanted material.  In your quest for cleaner data, you will undoubtedly filter your data at some point: whether it be for clearing up cases with missing values, for culling out fishy outliers, or for analyzing subgroups of your data set.  Example usage looks like `df[df['column name'] < 5]`.

For your reference, some commonly used comparison operators are given below.

Symbol | Usage      | Meaning 
------ | ---------- | -------------------------------------
==   | a == b   | Does a equal b?
<=   | a <= b   | Is a less than or equal to b?
&gt;=   | a >= b   | Is a greater than or equal to b?
<    | a < b    | Is a less than b?
&#62;    | a &#62; b    | Is a greater than b?
~    | ~p       | Returns negation of p
&#124; | p &#124; q | p OR q
&    | p & q    | p AND q
^  | p ^ q | p XOR q (exclusive or)

In the following we construct the DataFrame containing only names registered in California


```python
ca = baby_names[baby_names['State'] == 'CA']
ca.head(5)
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
      <th>State</th>
      <th>Sex</th>
      <th>Year</th>
      <th>Name</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>390635</th>
      <td>CA</td>
      <td>F</td>
      <td>1910</td>
      <td>Mary</td>
      <td>295</td>
    </tr>
    <tr>
      <th>390636</th>
      <td>CA</td>
      <td>F</td>
      <td>1910</td>
      <td>Helen</td>
      <td>239</td>
    </tr>
    <tr>
      <th>390637</th>
      <td>CA</td>
      <td>F</td>
      <td>1910</td>
      <td>Dorothy</td>
      <td>220</td>
    </tr>
    <tr>
      <th>390638</th>
      <td>CA</td>
      <td>F</td>
      <td>1910</td>
      <td>Margaret</td>
      <td>163</td>
    </tr>
    <tr>
      <th>390639</th>
      <td>CA</td>
      <td>F</td>
      <td>1910</td>
      <td>Frances</td>
      <td>134</td>
    </tr>
  </tbody>
</table>
</div>



### Question 5
Using a boolean array, select the names in Year 2000 (from `baby_names`) that have larger than 3000 counts. Keep all columns from the original `baby_names` DataFrame.

Note: Note that compound expressions have to be grouped with parentheses. That is, any time you use `p & q` to filter the DataFrame, make sure to use `df[(df[p]) & (df[q])]` or `df.loc[(df[p]) & (df[q])]`. 

You may use either `[]` or `loc`. Both will achieve the same result. For more on `[]` vs. `loc` see the stack overflow links from the intro portion of this lab.

<!--
BEGIN QUESTION
name: q5
-->


```python
result = baby_names[(baby_names['Year'] == 2000) & (baby_names['Count'] > 3000)]
result.head()
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
      <th>State</th>
      <th>Sex</th>
      <th>Year</th>
      <th>Name</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>725638</th>
      <td>CA</td>
      <td>M</td>
      <td>2000</td>
      <td>Daniel</td>
      <td>4342</td>
    </tr>
    <tr>
      <th>725639</th>
      <td>CA</td>
      <td>M</td>
      <td>2000</td>
      <td>Anthony</td>
      <td>3839</td>
    </tr>
    <tr>
      <th>725640</th>
      <td>CA</td>
      <td>M</td>
      <td>2000</td>
      <td>Jose</td>
      <td>3804</td>
    </tr>
    <tr>
      <th>725641</th>
      <td>CA</td>
      <td>M</td>
      <td>2000</td>
      <td>Andrew</td>
      <td>3600</td>
    </tr>
    <tr>
      <th>725642</th>
      <td>CA</td>
      <td>M</td>
      <td>2000</td>
      <td>Michael</td>
      <td>3572</td>
    </tr>
  </tbody>
</table>
</div>




```python
grader.check("q5")
# 依旧是版本问题
```




<p><strong style='color: red;'><pre style='display: inline;'>q5</pre> results:</strong></p><p><strong><pre style='display: inline;'>q5 - 1</pre> result:</strong></p><pre>    ✅ Test case passed</pre><p><strong><pre style='display: inline;'>q5 - 2</pre> result:</strong></p><pre>    ❌ Test case failed
    Trying:
        result["Count"].sum()
    Expecting:
        39000
    **********************************************************************
    Line 1, in q5 1
    Failed example:
        result["Count"].sum()
    Expected:
        39000
    Got:
        np.int64(39000)
</pre><p><strong><pre style='display: inline;'>q5 - 3</pre> result:</strong></p><pre>    ❌ Test case failed
    Trying:
        result["Count"].iloc[0]
    Expecting:
        4342
    **********************************************************************
    Line 1, in q5 2
    Failed example:
        result["Count"].iloc[0]
    Expected:
        4342
    Got:
        np.int64(4342)
</pre>



#### Query Review

Recall that pandas also has a query command. For example, we can get California baby names with the code below.


```python
ca = baby_names.query('State == "CA"')
ca.head(5)
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
      <th>State</th>
      <th>Sex</th>
      <th>Year</th>
      <th>Name</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>390635</th>
      <td>CA</td>
      <td>F</td>
      <td>1910</td>
      <td>Mary</td>
      <td>295</td>
    </tr>
    <tr>
      <th>390636</th>
      <td>CA</td>
      <td>F</td>
      <td>1910</td>
      <td>Helen</td>
      <td>239</td>
    </tr>
    <tr>
      <th>390637</th>
      <td>CA</td>
      <td>F</td>
      <td>1910</td>
      <td>Dorothy</td>
      <td>220</td>
    </tr>
    <tr>
      <th>390638</th>
      <td>CA</td>
      <td>F</td>
      <td>1910</td>
      <td>Margaret</td>
      <td>163</td>
    </tr>
    <tr>
      <th>390639</th>
      <td>CA</td>
      <td>F</td>
      <td>1910</td>
      <td>Frances</td>
      <td>134</td>
    </tr>
  </tbody>
</table>
</div>



Using the `query` command, select the names in Year 2000 (from `baby_names`) that have larger than 3000 counts.


```python
result_using_query = baby_names.query("Count > 3000 and Year == 2000")
result_using_query.head(5)
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
      <th>State</th>
      <th>Sex</th>
      <th>Year</th>
      <th>Name</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>725638</th>
      <td>CA</td>
      <td>M</td>
      <td>2000</td>
      <td>Daniel</td>
      <td>4342</td>
    </tr>
    <tr>
      <th>725639</th>
      <td>CA</td>
      <td>M</td>
      <td>2000</td>
      <td>Anthony</td>
      <td>3839</td>
    </tr>
    <tr>
      <th>725640</th>
      <td>CA</td>
      <td>M</td>
      <td>2000</td>
      <td>Jose</td>
      <td>3804</td>
    </tr>
    <tr>
      <th>725641</th>
      <td>CA</td>
      <td>M</td>
      <td>2000</td>
      <td>Andrew</td>
      <td>3600</td>
    </tr>
    <tr>
      <th>725642</th>
      <td>CA</td>
      <td>M</td>
      <td>2000</td>
      <td>Michael</td>
      <td>3572</td>
    </tr>
  </tbody>
</table>
</div>



## Groupby

Let's now turn to using groupby from lecture 4.

**Note:** This [slide](https://docs.google.com/presentation/d/1FC-cs5MTGSkDzI_7R_ZENgwoHQ4aVamxFOpJuWT0fo0/edit#slide=id.g477ed0f02e_0_390) provides a visual picture of how `groupby.agg` works if you'd like a reference.

### Question 6: Elections

**Review**: Let's start by reading in the election dataset from the pandas lectures.


```python
# run this cell
elections = pd.read_csv("data/elections.csv")
elections.head(5)
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
      <th>Year</th>
      <th>Candidate</th>
      <th>Party</th>
      <th>Popular vote</th>
      <th>Result</th>
      <th>%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1824</td>
      <td>Andrew Jackson</td>
      <td>Democratic-Republican</td>
      <td>151271</td>
      <td>loss</td>
      <td>57.210122</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1824</td>
      <td>John Quincy Adams</td>
      <td>Democratic-Republican</td>
      <td>113142</td>
      <td>win</td>
      <td>42.789878</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1828</td>
      <td>Andrew Jackson</td>
      <td>Democratic</td>
      <td>642806</td>
      <td>win</td>
      <td>56.203927</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1828</td>
      <td>John Quincy Adams</td>
      <td>National Republican</td>
      <td>500897</td>
      <td>loss</td>
      <td>43.796073</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1832</td>
      <td>Andrew Jackson</td>
      <td>Democratic</td>
      <td>702735</td>
      <td>win</td>
      <td>54.574789</td>
    </tr>
  </tbody>
</table>
</div>



As we saw, we can groupby a specific column, e.g. "Party". It turns out that using some syntax we didn't cover in lecture, we can print out the subframes that result. This isn't something you'll do for any practical purpose. However, it may help you get an understanding of **what groupby is actually doing**.

An example is given below for elections since 1980.


```python
# run this cell
for n, g in elections.query("Year >= 1980").groupby("Party"):
    print(f"Name: {n}") # by the way this is an "f string", a relatively new and great feature of Python
    display(g)
```

    Name: Citizens
    


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
      <th>Year</th>
      <th>Candidate</th>
      <th>Party</th>
      <th>Popular vote</th>
      <th>Result</th>
      <th>%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>127</th>
      <td>1980</td>
      <td>Barry Commoner</td>
      <td>Citizens</td>
      <td>233052</td>
      <td>loss</td>
      <td>0.270182</td>
    </tr>
  </tbody>
</table>
</div>


    Name: Constitution
    


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
      <th>Year</th>
      <th>Candidate</th>
      <th>Party</th>
      <th>Popular vote</th>
      <th>Result</th>
      <th>%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>160</th>
      <td>2004</td>
      <td>Michael Peroutka</td>
      <td>Constitution</td>
      <td>143630</td>
      <td>loss</td>
      <td>0.117542</td>
    </tr>
    <tr>
      <th>164</th>
      <td>2008</td>
      <td>Chuck Baldwin</td>
      <td>Constitution</td>
      <td>199750</td>
      <td>loss</td>
      <td>0.152398</td>
    </tr>
    <tr>
      <th>172</th>
      <td>2016</td>
      <td>Darrell Castle</td>
      <td>Constitution</td>
      <td>203091</td>
      <td>loss</td>
      <td>0.149640</td>
    </tr>
  </tbody>
</table>
</div>


    Name: Democratic
    


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
      <th>Year</th>
      <th>Candidate</th>
      <th>Party</th>
      <th>Popular vote</th>
      <th>Result</th>
      <th>%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>129</th>
      <td>1980</td>
      <td>Jimmy Carter</td>
      <td>Democratic</td>
      <td>35480115</td>
      <td>loss</td>
      <td>41.132848</td>
    </tr>
    <tr>
      <th>134</th>
      <td>1984</td>
      <td>Walter Mondale</td>
      <td>Democratic</td>
      <td>37577352</td>
      <td>loss</td>
      <td>40.729429</td>
    </tr>
    <tr>
      <th>137</th>
      <td>1988</td>
      <td>Michael Dukakis</td>
      <td>Democratic</td>
      <td>41809074</td>
      <td>loss</td>
      <td>45.770691</td>
    </tr>
    <tr>
      <th>140</th>
      <td>1992</td>
      <td>Bill Clinton</td>
      <td>Democratic</td>
      <td>44909806</td>
      <td>win</td>
      <td>43.118485</td>
    </tr>
    <tr>
      <th>144</th>
      <td>1996</td>
      <td>Bill Clinton</td>
      <td>Democratic</td>
      <td>47400125</td>
      <td>win</td>
      <td>49.296938</td>
    </tr>
    <tr>
      <th>151</th>
      <td>2000</td>
      <td>Al Gore</td>
      <td>Democratic</td>
      <td>50999897</td>
      <td>loss</td>
      <td>48.491813</td>
    </tr>
    <tr>
      <th>158</th>
      <td>2004</td>
      <td>John Kerry</td>
      <td>Democratic</td>
      <td>59028444</td>
      <td>loss</td>
      <td>48.306775</td>
    </tr>
    <tr>
      <th>162</th>
      <td>2008</td>
      <td>Barack Obama</td>
      <td>Democratic</td>
      <td>69498516</td>
      <td>win</td>
      <td>53.023510</td>
    </tr>
    <tr>
      <th>168</th>
      <td>2012</td>
      <td>Barack Obama</td>
      <td>Democratic</td>
      <td>65915795</td>
      <td>win</td>
      <td>51.258484</td>
    </tr>
    <tr>
      <th>176</th>
      <td>2016</td>
      <td>Hillary Clinton</td>
      <td>Democratic</td>
      <td>65853514</td>
      <td>loss</td>
      <td>48.521539</td>
    </tr>
    <tr>
      <th>178</th>
      <td>2020</td>
      <td>Joseph Biden</td>
      <td>Democratic</td>
      <td>81268924</td>
      <td>win</td>
      <td>51.311515</td>
    </tr>
  </tbody>
</table>
</div>


    Name: Green
    


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
      <th>Year</th>
      <th>Candidate</th>
      <th>Party</th>
      <th>Popular vote</th>
      <th>Result</th>
      <th>%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>149</th>
      <td>1996</td>
      <td>Ralph Nader</td>
      <td>Green</td>
      <td>685297</td>
      <td>loss</td>
      <td>0.712721</td>
    </tr>
    <tr>
      <th>155</th>
      <td>2000</td>
      <td>Ralph Nader</td>
      <td>Green</td>
      <td>2882955</td>
      <td>loss</td>
      <td>2.741176</td>
    </tr>
    <tr>
      <th>156</th>
      <td>2004</td>
      <td>David Cobb</td>
      <td>Green</td>
      <td>119859</td>
      <td>loss</td>
      <td>0.098088</td>
    </tr>
    <tr>
      <th>165</th>
      <td>2008</td>
      <td>Cynthia McKinney</td>
      <td>Green</td>
      <td>161797</td>
      <td>loss</td>
      <td>0.123442</td>
    </tr>
    <tr>
      <th>170</th>
      <td>2012</td>
      <td>Jill Stein</td>
      <td>Green</td>
      <td>469627</td>
      <td>loss</td>
      <td>0.365199</td>
    </tr>
    <tr>
      <th>177</th>
      <td>2016</td>
      <td>Jill Stein</td>
      <td>Green</td>
      <td>1457226</td>
      <td>loss</td>
      <td>1.073699</td>
    </tr>
    <tr>
      <th>181</th>
      <td>2020</td>
      <td>Howard Hawkins</td>
      <td>Green</td>
      <td>405035</td>
      <td>loss</td>
      <td>0.255731</td>
    </tr>
  </tbody>
</table>
</div>


    Name: Independent
    


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
      <th>Year</th>
      <th>Candidate</th>
      <th>Party</th>
      <th>Popular vote</th>
      <th>Result</th>
      <th>%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>130</th>
      <td>1980</td>
      <td>John B. Anderson</td>
      <td>Independent</td>
      <td>5719850</td>
      <td>loss</td>
      <td>6.631143</td>
    </tr>
    <tr>
      <th>143</th>
      <td>1992</td>
      <td>Ross Perot</td>
      <td>Independent</td>
      <td>19743821</td>
      <td>loss</td>
      <td>18.956298</td>
    </tr>
    <tr>
      <th>161</th>
      <td>2004</td>
      <td>Ralph Nader</td>
      <td>Independent</td>
      <td>465151</td>
      <td>loss</td>
      <td>0.380663</td>
    </tr>
    <tr>
      <th>167</th>
      <td>2008</td>
      <td>Ralph Nader</td>
      <td>Independent</td>
      <td>739034</td>
      <td>loss</td>
      <td>0.563842</td>
    </tr>
    <tr>
      <th>174</th>
      <td>2016</td>
      <td>Evan McMullin</td>
      <td>Independent</td>
      <td>732273</td>
      <td>loss</td>
      <td>0.539546</td>
    </tr>
  </tbody>
</table>
</div>


    Name: Libertarian
    


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
      <th>Year</th>
      <th>Candidate</th>
      <th>Party</th>
      <th>Popular vote</th>
      <th>Result</th>
      <th>%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>128</th>
      <td>1980</td>
      <td>Ed Clark</td>
      <td>Libertarian</td>
      <td>921128</td>
      <td>loss</td>
      <td>1.067883</td>
    </tr>
    <tr>
      <th>132</th>
      <td>1984</td>
      <td>David Bergland</td>
      <td>Libertarian</td>
      <td>228111</td>
      <td>loss</td>
      <td>0.247245</td>
    </tr>
    <tr>
      <th>138</th>
      <td>1988</td>
      <td>Ron Paul</td>
      <td>Libertarian</td>
      <td>431750</td>
      <td>loss</td>
      <td>0.472660</td>
    </tr>
    <tr>
      <th>139</th>
      <td>1992</td>
      <td>Andre Marrou</td>
      <td>Libertarian</td>
      <td>290087</td>
      <td>loss</td>
      <td>0.278516</td>
    </tr>
    <tr>
      <th>146</th>
      <td>1996</td>
      <td>Harry Browne</td>
      <td>Libertarian</td>
      <td>485759</td>
      <td>loss</td>
      <td>0.505198</td>
    </tr>
    <tr>
      <th>153</th>
      <td>2000</td>
      <td>Harry Browne</td>
      <td>Libertarian</td>
      <td>384431</td>
      <td>loss</td>
      <td>0.365525</td>
    </tr>
    <tr>
      <th>159</th>
      <td>2004</td>
      <td>Michael Badnarik</td>
      <td>Libertarian</td>
      <td>397265</td>
      <td>loss</td>
      <td>0.325108</td>
    </tr>
    <tr>
      <th>163</th>
      <td>2008</td>
      <td>Bob Barr</td>
      <td>Libertarian</td>
      <td>523715</td>
      <td>loss</td>
      <td>0.399565</td>
    </tr>
    <tr>
      <th>169</th>
      <td>2012</td>
      <td>Gary Johnson</td>
      <td>Libertarian</td>
      <td>1275971</td>
      <td>loss</td>
      <td>0.992241</td>
    </tr>
    <tr>
      <th>175</th>
      <td>2016</td>
      <td>Gary Johnson</td>
      <td>Libertarian</td>
      <td>4489235</td>
      <td>loss</td>
      <td>3.307714</td>
    </tr>
    <tr>
      <th>180</th>
      <td>2020</td>
      <td>Jo Jorgensen</td>
      <td>Libertarian</td>
      <td>1865724</td>
      <td>loss</td>
      <td>1.177979</td>
    </tr>
  </tbody>
</table>
</div>


    Name: Natural Law
    


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
      <th>Year</th>
      <th>Candidate</th>
      <th>Party</th>
      <th>Popular vote</th>
      <th>Result</th>
      <th>%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>148</th>
      <td>1996</td>
      <td>John Hagelin</td>
      <td>Natural Law</td>
      <td>113670</td>
      <td>loss</td>
      <td>0.118219</td>
    </tr>
  </tbody>
</table>
</div>


    Name: New Alliance
    


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
      <th>Year</th>
      <th>Candidate</th>
      <th>Party</th>
      <th>Popular vote</th>
      <th>Result</th>
      <th>%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>136</th>
      <td>1988</td>
      <td>Lenora Fulani</td>
      <td>New Alliance</td>
      <td>217221</td>
      <td>loss</td>
      <td>0.237804</td>
    </tr>
  </tbody>
</table>
</div>


    Name: Populist
    


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
      <th>Year</th>
      <th>Candidate</th>
      <th>Party</th>
      <th>Popular vote</th>
      <th>Result</th>
      <th>%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>141</th>
      <td>1992</td>
      <td>Bo Gritz</td>
      <td>Populist</td>
      <td>106152</td>
      <td>loss</td>
      <td>0.101918</td>
    </tr>
  </tbody>
</table>
</div>


    Name: Reform
    


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
      <th>Year</th>
      <th>Candidate</th>
      <th>Party</th>
      <th>Popular vote</th>
      <th>Result</th>
      <th>%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>150</th>
      <td>1996</td>
      <td>Ross Perot</td>
      <td>Reform</td>
      <td>8085294</td>
      <td>loss</td>
      <td>8.408844</td>
    </tr>
    <tr>
      <th>154</th>
      <td>2000</td>
      <td>Pat Buchanan</td>
      <td>Reform</td>
      <td>448895</td>
      <td>loss</td>
      <td>0.426819</td>
    </tr>
  </tbody>
</table>
</div>


    Name: Republican
    


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
      <th>Year</th>
      <th>Candidate</th>
      <th>Party</th>
      <th>Popular vote</th>
      <th>Result</th>
      <th>%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>131</th>
      <td>1980</td>
      <td>Ronald Reagan</td>
      <td>Republican</td>
      <td>43903230</td>
      <td>win</td>
      <td>50.897944</td>
    </tr>
    <tr>
      <th>133</th>
      <td>1984</td>
      <td>Ronald Reagan</td>
      <td>Republican</td>
      <td>54455472</td>
      <td>win</td>
      <td>59.023326</td>
    </tr>
    <tr>
      <th>135</th>
      <td>1988</td>
      <td>George H. W. Bush</td>
      <td>Republican</td>
      <td>48886597</td>
      <td>win</td>
      <td>53.518845</td>
    </tr>
    <tr>
      <th>142</th>
      <td>1992</td>
      <td>George H. W. Bush</td>
      <td>Republican</td>
      <td>39104550</td>
      <td>loss</td>
      <td>37.544784</td>
    </tr>
    <tr>
      <th>145</th>
      <td>1996</td>
      <td>Bob Dole</td>
      <td>Republican</td>
      <td>39197469</td>
      <td>loss</td>
      <td>40.766036</td>
    </tr>
    <tr>
      <th>152</th>
      <td>2000</td>
      <td>George W. Bush</td>
      <td>Republican</td>
      <td>50456002</td>
      <td>win</td>
      <td>47.974666</td>
    </tr>
    <tr>
      <th>157</th>
      <td>2004</td>
      <td>George W. Bush</td>
      <td>Republican</td>
      <td>62040610</td>
      <td>win</td>
      <td>50.771824</td>
    </tr>
    <tr>
      <th>166</th>
      <td>2008</td>
      <td>John McCain</td>
      <td>Republican</td>
      <td>59948323</td>
      <td>loss</td>
      <td>45.737243</td>
    </tr>
    <tr>
      <th>171</th>
      <td>2012</td>
      <td>Mitt Romney</td>
      <td>Republican</td>
      <td>60933504</td>
      <td>loss</td>
      <td>47.384076</td>
    </tr>
    <tr>
      <th>173</th>
      <td>2016</td>
      <td>Donald Trump</td>
      <td>Republican</td>
      <td>62984828</td>
      <td>win</td>
      <td>46.407862</td>
    </tr>
    <tr>
      <th>179</th>
      <td>2020</td>
      <td>Donald Trump</td>
      <td>Republican</td>
      <td>74216154</td>
      <td>loss</td>
      <td>46.858542</td>
    </tr>
  </tbody>
</table>
</div>


    Name: Taxpayers
    


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
      <th>Year</th>
      <th>Candidate</th>
      <th>Party</th>
      <th>Popular vote</th>
      <th>Result</th>
      <th>%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>147</th>
      <td>1996</td>
      <td>Howard Phillips</td>
      <td>Taxpayers</td>
      <td>184656</td>
      <td>loss</td>
      <td>0.192045</td>
    </tr>
  </tbody>
</table>
</div>


Recall that once we've formed groups, we can aggregate each sub-dataframe (a.k.a. group) into a single row using an aggregation function. For example, if we use `.agg(np.mean)` on the groups above, we get back a single DataFrame where each group has been replaced by a single row. In each column for that aggregate row, the value that appears is the average of all values in that group.

For columns which are non-numeric, e.g. "Result", the column is dropped because we cannot compute the mean of the Result.


```python
elections.query("Year >= 1980").groupby("Party").agg(np.mean)
```

    C:\Users\86135\AppData\Local\Temp\ipykernel_61736\4206656687.py:1: FutureWarning: The provided callable <function mean at 0x00000133CD918040> is currently using DataFrameGroupBy.mean. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "mean" instead.
      elections.query("Year >= 1980").groupby("Party").agg(np.mean)
    


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:1942, in GroupBy._agg_py_fallback(self, how, values, ndim, alt)
       1941 try:
    -> 1942     res_values = self._grouper.agg_series(ser, alt, preserve_dtype=True)
       1943 except Exception as err:
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\ops.py:864, in BaseGrouper.agg_series(self, obj, func, preserve_dtype)
        862     preserve_dtype = True
    --> 864 result = self._aggregate_series_pure_python(obj, func)
        866 npvalues = lib.maybe_convert_objects(result, try_float=False)
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\ops.py:885, in BaseGrouper._aggregate_series_pure_python(self, obj, func)
        884 for i, group in enumerate(splitter):
    --> 885     res = func(group)
        886     res = extract_result(res)
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:2454, in GroupBy.mean.<locals>.<lambda>(x)
       2451 else:
       2452     result = self._cython_agg_general(
       2453         "mean",
    -> 2454         alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only),
       2455         numeric_only=numeric_only,
       2456     )
       2457     return result.__finalize__(self.obj, method="groupby")
    

    File d:\miniconda3\Lib\site-packages\pandas\core\series.py:6549, in Series.mean(self, axis, skipna, numeric_only, **kwargs)
       6541 @doc(make_doc("mean", ndim=1))
       6542 def mean(
       6543     self,
       (...)
       6547     **kwargs,
       6548 ):
    -> 6549     return NDFrame.mean(self, axis, skipna, numeric_only, **kwargs)
    

    File d:\miniconda3\Lib\site-packages\pandas\core\generic.py:12420, in NDFrame.mean(self, axis, skipna, numeric_only, **kwargs)
      12413 def mean(
      12414     self,
      12415     axis: Axis | None = 0,
       (...)
      12418     **kwargs,
      12419 ) -> Series | float:
    > 12420     return self._stat_function(
      12421         "mean", nanops.nanmean, axis, skipna, numeric_only, **kwargs
      12422     )
    

    File d:\miniconda3\Lib\site-packages\pandas\core\generic.py:12377, in NDFrame._stat_function(self, name, func, axis, skipna, numeric_only, **kwargs)
      12375 validate_bool_kwarg(skipna, "skipna", none_allowed=False)
    > 12377 return self._reduce(
      12378     func, name=name, axis=axis, skipna=skipna, numeric_only=numeric_only
      12379 )
    

    File d:\miniconda3\Lib\site-packages\pandas\core\series.py:6457, in Series._reduce(self, op, name, axis, skipna, numeric_only, filter_type, **kwds)
       6453     raise TypeError(
       6454         f"Series.{name} does not allow {kwd_name}={numeric_only} "
       6455         "with non-numeric dtypes."
       6456     )
    -> 6457 return op(delegate, skipna=skipna, **kwds)
    

    File d:\miniconda3\Lib\site-packages\pandas\core\nanops.py:147, in bottleneck_switch.__call__.<locals>.f(values, axis, skipna, **kwds)
        146 else:
    --> 147     result = alt(values, axis=axis, skipna=skipna, **kwds)
        149 return result
    

    File d:\miniconda3\Lib\site-packages\pandas\core\nanops.py:404, in _datetimelike_compat.<locals>.new_func(values, axis, skipna, mask, **kwargs)
        402     mask = isna(values)
    --> 404 result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)
        406 if datetimelike:
    

    File d:\miniconda3\Lib\site-packages\pandas\core\nanops.py:720, in nanmean(values, axis, skipna, mask)
        719 the_sum = values.sum(axis, dtype=dtype_sum)
    --> 720 the_sum = _ensure_numeric(the_sum)
        722 if axis is not None and getattr(the_sum, "ndim", False):
    

    File d:\miniconda3\Lib\site-packages\pandas\core\nanops.py:1701, in _ensure_numeric(x)
       1699 if isinstance(x, str):
       1700     # GH#44008, GH#36703 avoid casting e.g. strings to numeric
    -> 1701     raise TypeError(f"Could not convert string '{x}' to numeric")
       1702 try:
    

    TypeError: Could not convert string 'Barry Commoner' to numeric

    
    The above exception was the direct cause of the following exception:
    

    TypeError                                 Traceback (most recent call last)

    Cell In[88], line 1
    ----> 1 elections.query("Year >= 1980").groupby("Party").agg(np.mean)
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\generic.py:1432, in DataFrameGroupBy.aggregate(self, func, engine, engine_kwargs, *args, **kwargs)
       1429     kwargs["engine_kwargs"] = engine_kwargs
       1431 op = GroupByApply(self, func, args=args, kwargs=kwargs)
    -> 1432 result = op.agg()
       1433 if not is_dict_like(func) and result is not None:
       1434     # GH #52849
       1435     if not self.as_index and is_list_like(func):
    

    File d:\miniconda3\Lib\site-packages\pandas\core\apply.py:199, in Apply.agg(self)
        197     if f and not args and not kwargs:
        198         warn_alias_replacement(obj, func, f)
    --> 199         return getattr(obj, f)()
        201 # caller can react
        202 return None
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:2452, in GroupBy.mean(self, numeric_only, engine, engine_kwargs)
       2445     return self._numba_agg_general(
       2446         grouped_mean,
       2447         executor.float_dtype_mapping,
       2448         engine_kwargs,
       2449         min_periods=0,
       2450     )
       2451 else:
    -> 2452     result = self._cython_agg_general(
       2453         "mean",
       2454         alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only),
       2455         numeric_only=numeric_only,
       2456     )
       2457     return result.__finalize__(self.obj, method="groupby")
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:1998, in GroupBy._cython_agg_general(self, how, alt, numeric_only, min_count, **kwargs)
       1995     result = self._agg_py_fallback(how, values, ndim=data.ndim, alt=alt)
       1996     return result
    -> 1998 new_mgr = data.grouped_reduce(array_func)
       1999 res = self._wrap_agged_manager(new_mgr)
       2000 if how in ["idxmin", "idxmax"]:
    

    File d:\miniconda3\Lib\site-packages\pandas\core\internals\managers.py:1469, in BlockManager.grouped_reduce(self, func)
       1465 if blk.is_object:
       1466     # split on object-dtype blocks bc some columns may raise
       1467     #  while others do not.
       1468     for sb in blk._split():
    -> 1469         applied = sb.apply(func)
       1470         result_blocks = extend_blocks(applied, result_blocks)
       1471 else:
    

    File d:\miniconda3\Lib\site-packages\pandas\core\internals\blocks.py:393, in Block.apply(self, func, **kwargs)
        387 @final
        388 def apply(self, func, **kwargs) -> list[Block]:
        389     """
        390     apply the function to my values; return a block if we are not
        391     one
        392     """
    --> 393     result = func(self.values, **kwargs)
        395     result = maybe_coerce_values(result)
        396     return self._split_op_result(result)
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:1995, in GroupBy._cython_agg_general.<locals>.array_func(values)
       1992     return result
       1994 assert alt is not None
    -> 1995 result = self._agg_py_fallback(how, values, ndim=data.ndim, alt=alt)
       1996 return result
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:1946, in GroupBy._agg_py_fallback(self, how, values, ndim, alt)
       1944     msg = f"agg function failed [how->{how},dtype->{ser.dtype}]"
       1945     # preserve the kind of exception that raised
    -> 1946     raise type(err)(msg) from err
       1948 if ser.dtype == object:
       1949     res_values = res_values.astype(object, copy=False)
    

    TypeError: agg function failed [how->mean,dtype->object]


Equivalently we can use one of the shorthand aggregation functions, e.g. `.mean()`: 


```python
elections.query("Year >= 1980").groupby("Party").mean()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:1942, in GroupBy._agg_py_fallback(self, how, values, ndim, alt)
       1941 try:
    -> 1942     res_values = self._grouper.agg_series(ser, alt, preserve_dtype=True)
       1943 except Exception as err:
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\ops.py:864, in BaseGrouper.agg_series(self, obj, func, preserve_dtype)
        862     preserve_dtype = True
    --> 864 result = self._aggregate_series_pure_python(obj, func)
        866 npvalues = lib.maybe_convert_objects(result, try_float=False)
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\ops.py:885, in BaseGrouper._aggregate_series_pure_python(self, obj, func)
        884 for i, group in enumerate(splitter):
    --> 885     res = func(group)
        886     res = extract_result(res)
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:2454, in GroupBy.mean.<locals>.<lambda>(x)
       2451 else:
       2452     result = self._cython_agg_general(
       2453         "mean",
    -> 2454         alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only),
       2455         numeric_only=numeric_only,
       2456     )
       2457     return result.__finalize__(self.obj, method="groupby")
    

    File d:\miniconda3\Lib\site-packages\pandas\core\series.py:6549, in Series.mean(self, axis, skipna, numeric_only, **kwargs)
       6541 @doc(make_doc("mean", ndim=1))
       6542 def mean(
       6543     self,
       (...)
       6547     **kwargs,
       6548 ):
    -> 6549     return NDFrame.mean(self, axis, skipna, numeric_only, **kwargs)
    

    File d:\miniconda3\Lib\site-packages\pandas\core\generic.py:12420, in NDFrame.mean(self, axis, skipna, numeric_only, **kwargs)
      12413 def mean(
      12414     self,
      12415     axis: Axis | None = 0,
       (...)
      12418     **kwargs,
      12419 ) -> Series | float:
    > 12420     return self._stat_function(
      12421         "mean", nanops.nanmean, axis, skipna, numeric_only, **kwargs
      12422     )
    

    File d:\miniconda3\Lib\site-packages\pandas\core\generic.py:12377, in NDFrame._stat_function(self, name, func, axis, skipna, numeric_only, **kwargs)
      12375 validate_bool_kwarg(skipna, "skipna", none_allowed=False)
    > 12377 return self._reduce(
      12378     func, name=name, axis=axis, skipna=skipna, numeric_only=numeric_only
      12379 )
    

    File d:\miniconda3\Lib\site-packages\pandas\core\series.py:6457, in Series._reduce(self, op, name, axis, skipna, numeric_only, filter_type, **kwds)
       6453     raise TypeError(
       6454         f"Series.{name} does not allow {kwd_name}={numeric_only} "
       6455         "with non-numeric dtypes."
       6456     )
    -> 6457 return op(delegate, skipna=skipna, **kwds)
    

    File d:\miniconda3\Lib\site-packages\pandas\core\nanops.py:147, in bottleneck_switch.__call__.<locals>.f(values, axis, skipna, **kwds)
        146 else:
    --> 147     result = alt(values, axis=axis, skipna=skipna, **kwds)
        149 return result
    

    File d:\miniconda3\Lib\site-packages\pandas\core\nanops.py:404, in _datetimelike_compat.<locals>.new_func(values, axis, skipna, mask, **kwargs)
        402     mask = isna(values)
    --> 404 result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)
        406 if datetimelike:
    

    File d:\miniconda3\Lib\site-packages\pandas\core\nanops.py:720, in nanmean(values, axis, skipna, mask)
        719 the_sum = values.sum(axis, dtype=dtype_sum)
    --> 720 the_sum = _ensure_numeric(the_sum)
        722 if axis is not None and getattr(the_sum, "ndim", False):
    

    File d:\miniconda3\Lib\site-packages\pandas\core\nanops.py:1701, in _ensure_numeric(x)
       1699 if isinstance(x, str):
       1700     # GH#44008, GH#36703 avoid casting e.g. strings to numeric
    -> 1701     raise TypeError(f"Could not convert string '{x}' to numeric")
       1702 try:
    

    TypeError: Could not convert string 'Barry Commoner' to numeric

    
    The above exception was the direct cause of the following exception:
    

    TypeError                                 Traceback (most recent call last)

    Cell In[85], line 1
    ----> 1 elections.query("Year >= 1980").groupby("Party").mean()
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:2452, in GroupBy.mean(self, numeric_only, engine, engine_kwargs)
       2445     return self._numba_agg_general(
       2446         grouped_mean,
       2447         executor.float_dtype_mapping,
       2448         engine_kwargs,
       2449         min_periods=0,
       2450     )
       2451 else:
    -> 2452     result = self._cython_agg_general(
       2453         "mean",
       2454         alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only),
       2455         numeric_only=numeric_only,
       2456     )
       2457     return result.__finalize__(self.obj, method="groupby")
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:1998, in GroupBy._cython_agg_general(self, how, alt, numeric_only, min_count, **kwargs)
       1995     result = self._agg_py_fallback(how, values, ndim=data.ndim, alt=alt)
       1996     return result
    -> 1998 new_mgr = data.grouped_reduce(array_func)
       1999 res = self._wrap_agged_manager(new_mgr)
       2000 if how in ["idxmin", "idxmax"]:
    

    File d:\miniconda3\Lib\site-packages\pandas\core\internals\managers.py:1469, in BlockManager.grouped_reduce(self, func)
       1465 if blk.is_object:
       1466     # split on object-dtype blocks bc some columns may raise
       1467     #  while others do not.
       1468     for sb in blk._split():
    -> 1469         applied = sb.apply(func)
       1470         result_blocks = extend_blocks(applied, result_blocks)
       1471 else:
    

    File d:\miniconda3\Lib\site-packages\pandas\core\internals\blocks.py:393, in Block.apply(self, func, **kwargs)
        387 @final
        388 def apply(self, func, **kwargs) -> list[Block]:
        389     """
        390     apply the function to my values; return a block if we are not
        391     one
        392     """
    --> 393     result = func(self.values, **kwargs)
        395     result = maybe_coerce_values(result)
        396     return self._split_op_result(result)
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:1995, in GroupBy._cython_agg_general.<locals>.array_func(values)
       1992     return result
       1994 assert alt is not None
    -> 1995 result = self._agg_py_fallback(how, values, ndim=data.ndim, alt=alt)
       1996 return result
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:1946, in GroupBy._agg_py_fallback(self, how, values, ndim, alt)
       1944     msg = f"agg function failed [how->{how},dtype->{ser.dtype}]"
       1945     # preserve the kind of exception that raised
    -> 1946     raise type(err)(msg) from err
       1948 if ser.dtype == object:
       1949     res_values = res_values.astype(object, copy=False)
    

    TypeError: agg function failed [how->mean,dtype->object]


Note that the index of the dataframe returned by an `groupby.agg` call is no longer a set of numeric indices from 0 to N-1. Instead, we see that the index for the example above is now the `Party`. If we want to restore our DataFrame so that `Party` is a column rather than the index, we can use `reset_index`.


```python
elections.query("Year >= 1980").groupby("Party").mean().reset_index()
```

**IMPORTANT NOTE:** Notice that the code above consists of a series of chained method calls. This sort of code is very very common in Pandas programming and in data science in general. Such chained method calls can sometimes go many layers deep, in which case you might consider adding newlines between lines of code for clarity. For example, we could instead write the code above as:


```python
# pandas method chaining
(
elections.query("Year >= 1980").groupby("Party") 
                               .mean()            ## computes the mean values by party
                               .reset_index()     ## reset to a numerical index
)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:1942, in GroupBy._agg_py_fallback(self, how, values, ndim, alt)
       1941 try:
    -> 1942     res_values = self._grouper.agg_series(ser, alt, preserve_dtype=True)
       1943 except Exception as err:
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\ops.py:864, in BaseGrouper.agg_series(self, obj, func, preserve_dtype)
        862     preserve_dtype = True
    --> 864 result = self._aggregate_series_pure_python(obj, func)
        866 npvalues = lib.maybe_convert_objects(result, try_float=False)
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\ops.py:885, in BaseGrouper._aggregate_series_pure_python(self, obj, func)
        884 for i, group in enumerate(splitter):
    --> 885     res = func(group)
        886     res = extract_result(res)
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:2454, in GroupBy.mean.<locals>.<lambda>(x)
       2451 else:
       2452     result = self._cython_agg_general(
       2453         "mean",
    -> 2454         alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only),
       2455         numeric_only=numeric_only,
       2456     )
       2457     return result.__finalize__(self.obj, method="groupby")
    

    File d:\miniconda3\Lib\site-packages\pandas\core\series.py:6549, in Series.mean(self, axis, skipna, numeric_only, **kwargs)
       6541 @doc(make_doc("mean", ndim=1))
       6542 def mean(
       6543     self,
       (...)
       6547     **kwargs,
       6548 ):
    -> 6549     return NDFrame.mean(self, axis, skipna, numeric_only, **kwargs)
    

    File d:\miniconda3\Lib\site-packages\pandas\core\generic.py:12420, in NDFrame.mean(self, axis, skipna, numeric_only, **kwargs)
      12413 def mean(
      12414     self,
      12415     axis: Axis | None = 0,
       (...)
      12418     **kwargs,
      12419 ) -> Series | float:
    > 12420     return self._stat_function(
      12421         "mean", nanops.nanmean, axis, skipna, numeric_only, **kwargs
      12422     )
    

    File d:\miniconda3\Lib\site-packages\pandas\core\generic.py:12377, in NDFrame._stat_function(self, name, func, axis, skipna, numeric_only, **kwargs)
      12375 validate_bool_kwarg(skipna, "skipna", none_allowed=False)
    > 12377 return self._reduce(
      12378     func, name=name, axis=axis, skipna=skipna, numeric_only=numeric_only
      12379 )
    

    File d:\miniconda3\Lib\site-packages\pandas\core\series.py:6457, in Series._reduce(self, op, name, axis, skipna, numeric_only, filter_type, **kwds)
       6453     raise TypeError(
       6454         f"Series.{name} does not allow {kwd_name}={numeric_only} "
       6455         "with non-numeric dtypes."
       6456     )
    -> 6457 return op(delegate, skipna=skipna, **kwds)
    

    File d:\miniconda3\Lib\site-packages\pandas\core\nanops.py:147, in bottleneck_switch.__call__.<locals>.f(values, axis, skipna, **kwds)
        146 else:
    --> 147     result = alt(values, axis=axis, skipna=skipna, **kwds)
        149 return result
    

    File d:\miniconda3\Lib\site-packages\pandas\core\nanops.py:404, in _datetimelike_compat.<locals>.new_func(values, axis, skipna, mask, **kwargs)
        402     mask = isna(values)
    --> 404 result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)
        406 if datetimelike:
    

    File d:\miniconda3\Lib\site-packages\pandas\core\nanops.py:720, in nanmean(values, axis, skipna, mask)
        719 the_sum = values.sum(axis, dtype=dtype_sum)
    --> 720 the_sum = _ensure_numeric(the_sum)
        722 if axis is not None and getattr(the_sum, "ndim", False):
    

    File d:\miniconda3\Lib\site-packages\pandas\core\nanops.py:1701, in _ensure_numeric(x)
       1699 if isinstance(x, str):
       1700     # GH#44008, GH#36703 avoid casting e.g. strings to numeric
    -> 1701     raise TypeError(f"Could not convert string '{x}' to numeric")
       1702 try:
    

    TypeError: Could not convert string 'Barry Commoner' to numeric

    
    The above exception was the direct cause of the following exception:
    

    TypeError                                 Traceback (most recent call last)

    Cell In[89], line 4
          1 # pandas method chaining
          2 (
          3 elections.query("Year >= 1980").groupby("Party") 
    ----> 4                                .mean()            ## computes the mean values by party
          5                                .reset_index()     ## reset to a numerical index
          6 )
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:2452, in GroupBy.mean(self, numeric_only, engine, engine_kwargs)
       2445     return self._numba_agg_general(
       2446         grouped_mean,
       2447         executor.float_dtype_mapping,
       2448         engine_kwargs,
       2449         min_periods=0,
       2450     )
       2451 else:
    -> 2452     result = self._cython_agg_general(
       2453         "mean",
       2454         alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only),
       2455         numeric_only=numeric_only,
       2456     )
       2457     return result.__finalize__(self.obj, method="groupby")
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:1998, in GroupBy._cython_agg_general(self, how, alt, numeric_only, min_count, **kwargs)
       1995     result = self._agg_py_fallback(how, values, ndim=data.ndim, alt=alt)
       1996     return result
    -> 1998 new_mgr = data.grouped_reduce(array_func)
       1999 res = self._wrap_agged_manager(new_mgr)
       2000 if how in ["idxmin", "idxmax"]:
    

    File d:\miniconda3\Lib\site-packages\pandas\core\internals\managers.py:1469, in BlockManager.grouped_reduce(self, func)
       1465 if blk.is_object:
       1466     # split on object-dtype blocks bc some columns may raise
       1467     #  while others do not.
       1468     for sb in blk._split():
    -> 1469         applied = sb.apply(func)
       1470         result_blocks = extend_blocks(applied, result_blocks)
       1471 else:
    

    File d:\miniconda3\Lib\site-packages\pandas\core\internals\blocks.py:393, in Block.apply(self, func, **kwargs)
        387 @final
        388 def apply(self, func, **kwargs) -> list[Block]:
        389     """
        390     apply the function to my values; return a block if we are not
        391     one
        392     """
    --> 393     result = func(self.values, **kwargs)
        395     result = maybe_coerce_values(result)
        396     return self._split_op_result(result)
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:1995, in GroupBy._cython_agg_general.<locals>.array_func(values)
       1992     return result
       1994 assert alt is not None
    -> 1995 result = self._agg_py_fallback(how, values, ndim=data.ndim, alt=alt)
       1996 return result
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:1946, in GroupBy._agg_py_fallback(self, how, values, ndim, alt)
       1944     msg = f"agg function failed [how->{how},dtype->{ser.dtype}]"
       1945     # preserve the kind of exception that raised
    -> 1946     raise type(err)(msg) from err
       1948 if ser.dtype == object:
       1949     res_values = res_values.astype(object, copy=False)
    

    TypeError: agg function failed [how->mean,dtype->object]


Note that I've surrounded the entire call by a big set of parentheses so that Python doesn't complain about the indentation. An alternative is to use the \ symbol to indicate to Python that your code continues on to the next line.


```python
# pandas method chaining (alternative)
elections.query("Year >= 1980").groupby("Party") \
                               .mean() \
                               .reset_index()     
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:1942, in GroupBy._agg_py_fallback(self, how, values, ndim, alt)
       1941 try:
    -> 1942     res_values = self._grouper.agg_series(ser, alt, preserve_dtype=True)
       1943 except Exception as err:
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\ops.py:864, in BaseGrouper.agg_series(self, obj, func, preserve_dtype)
        862     preserve_dtype = True
    --> 864 result = self._aggregate_series_pure_python(obj, func)
        866 npvalues = lib.maybe_convert_objects(result, try_float=False)
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\ops.py:885, in BaseGrouper._aggregate_series_pure_python(self, obj, func)
        884 for i, group in enumerate(splitter):
    --> 885     res = func(group)
        886     res = extract_result(res)
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:2454, in GroupBy.mean.<locals>.<lambda>(x)
       2451 else:
       2452     result = self._cython_agg_general(
       2453         "mean",
    -> 2454         alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only),
       2455         numeric_only=numeric_only,
       2456     )
       2457     return result.__finalize__(self.obj, method="groupby")
    

    File d:\miniconda3\Lib\site-packages\pandas\core\series.py:6549, in Series.mean(self, axis, skipna, numeric_only, **kwargs)
       6541 @doc(make_doc("mean", ndim=1))
       6542 def mean(
       6543     self,
       (...)
       6547     **kwargs,
       6548 ):
    -> 6549     return NDFrame.mean(self, axis, skipna, numeric_only, **kwargs)
    

    File d:\miniconda3\Lib\site-packages\pandas\core\generic.py:12420, in NDFrame.mean(self, axis, skipna, numeric_only, **kwargs)
      12413 def mean(
      12414     self,
      12415     axis: Axis | None = 0,
       (...)
      12418     **kwargs,
      12419 ) -> Series | float:
    > 12420     return self._stat_function(
      12421         "mean", nanops.nanmean, axis, skipna, numeric_only, **kwargs
      12422     )
    

    File d:\miniconda3\Lib\site-packages\pandas\core\generic.py:12377, in NDFrame._stat_function(self, name, func, axis, skipna, numeric_only, **kwargs)
      12375 validate_bool_kwarg(skipna, "skipna", none_allowed=False)
    > 12377 return self._reduce(
      12378     func, name=name, axis=axis, skipna=skipna, numeric_only=numeric_only
      12379 )
    

    File d:\miniconda3\Lib\site-packages\pandas\core\series.py:6457, in Series._reduce(self, op, name, axis, skipna, numeric_only, filter_type, **kwds)
       6453     raise TypeError(
       6454         f"Series.{name} does not allow {kwd_name}={numeric_only} "
       6455         "with non-numeric dtypes."
       6456     )
    -> 6457 return op(delegate, skipna=skipna, **kwds)
    

    File d:\miniconda3\Lib\site-packages\pandas\core\nanops.py:147, in bottleneck_switch.__call__.<locals>.f(values, axis, skipna, **kwds)
        146 else:
    --> 147     result = alt(values, axis=axis, skipna=skipna, **kwds)
        149 return result
    

    File d:\miniconda3\Lib\site-packages\pandas\core\nanops.py:404, in _datetimelike_compat.<locals>.new_func(values, axis, skipna, mask, **kwargs)
        402     mask = isna(values)
    --> 404 result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)
        406 if datetimelike:
    

    File d:\miniconda3\Lib\site-packages\pandas\core\nanops.py:720, in nanmean(values, axis, skipna, mask)
        719 the_sum = values.sum(axis, dtype=dtype_sum)
    --> 720 the_sum = _ensure_numeric(the_sum)
        722 if axis is not None and getattr(the_sum, "ndim", False):
    

    File d:\miniconda3\Lib\site-packages\pandas\core\nanops.py:1701, in _ensure_numeric(x)
       1699 if isinstance(x, str):
       1700     # GH#44008, GH#36703 avoid casting e.g. strings to numeric
    -> 1701     raise TypeError(f"Could not convert string '{x}' to numeric")
       1702 try:
    

    TypeError: Could not convert string 'Barry Commoner' to numeric

    
    The above exception was the direct cause of the following exception:
    

    TypeError                                 Traceback (most recent call last)

    Cell In[90], line 3
          1 # pandas method chaining (alternative)
          2 elections.query("Year >= 1980").groupby("Party") \
    ----> 3                                .mean() \
          4                                .reset_index()     
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:2452, in GroupBy.mean(self, numeric_only, engine, engine_kwargs)
       2445     return self._numba_agg_general(
       2446         grouped_mean,
       2447         executor.float_dtype_mapping,
       2448         engine_kwargs,
       2449         min_periods=0,
       2450     )
       2451 else:
    -> 2452     result = self._cython_agg_general(
       2453         "mean",
       2454         alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only),
       2455         numeric_only=numeric_only,
       2456     )
       2457     return result.__finalize__(self.obj, method="groupby")
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:1998, in GroupBy._cython_agg_general(self, how, alt, numeric_only, min_count, **kwargs)
       1995     result = self._agg_py_fallback(how, values, ndim=data.ndim, alt=alt)
       1996     return result
    -> 1998 new_mgr = data.grouped_reduce(array_func)
       1999 res = self._wrap_agged_manager(new_mgr)
       2000 if how in ["idxmin", "idxmax"]:
    

    File d:\miniconda3\Lib\site-packages\pandas\core\internals\managers.py:1469, in BlockManager.grouped_reduce(self, func)
       1465 if blk.is_object:
       1466     # split on object-dtype blocks bc some columns may raise
       1467     #  while others do not.
       1468     for sb in blk._split():
    -> 1469         applied = sb.apply(func)
       1470         result_blocks = extend_blocks(applied, result_blocks)
       1471 else:
    

    File d:\miniconda3\Lib\site-packages\pandas\core\internals\blocks.py:393, in Block.apply(self, func, **kwargs)
        387 @final
        388 def apply(self, func, **kwargs) -> list[Block]:
        389     """
        390     apply the function to my values; return a block if we are not
        391     one
        392     """
    --> 393     result = func(self.values, **kwargs)
        395     result = maybe_coerce_values(result)
        396     return self._split_op_result(result)
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:1995, in GroupBy._cython_agg_general.<locals>.array_func(values)
       1992     return result
       1994 assert alt is not None
    -> 1995 result = self._agg_py_fallback(how, values, ndim=data.ndim, alt=alt)
       1996 return result
    

    File d:\miniconda3\Lib\site-packages\pandas\core\groupby\groupby.py:1946, in GroupBy._agg_py_fallback(self, how, values, ndim, alt)
       1944     msg = f"agg function failed [how->{how},dtype->{ser.dtype}]"
       1945     # preserve the kind of exception that raised
    -> 1946     raise type(err)(msg) from err
       1948 if ser.dtype == object:
       1949     res_values = res_values.astype(object, copy=False)
    

    TypeError: agg function failed [how->mean,dtype->object]


**IMPORTANT NOTE:** You should NEVER NEVER solve problems like the one above using loops or list comprehensions. This is slow and also misses the entire point of this part of DS100. 

Before we continue, we'll print out the election dataset again for your convenience. 


```python
elections.head(5)
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
      <th>Year</th>
      <th>Candidate</th>
      <th>Party</th>
      <th>Popular vote</th>
      <th>Result</th>
      <th>%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1824</td>
      <td>Andrew Jackson</td>
      <td>Democratic-Republican</td>
      <td>151271</td>
      <td>loss</td>
      <td>57.210122</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1824</td>
      <td>John Quincy Adams</td>
      <td>Democratic-Republican</td>
      <td>113142</td>
      <td>win</td>
      <td>42.789878</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1828</td>
      <td>Andrew Jackson</td>
      <td>Democratic</td>
      <td>642806</td>
      <td>win</td>
      <td>56.203927</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1828</td>
      <td>John Quincy Adams</td>
      <td>National Republican</td>
      <td>500897</td>
      <td>loss</td>
      <td>43.796073</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1832</td>
      <td>Andrew Jackson</td>
      <td>Democratic</td>
      <td>702735</td>
      <td>win</td>
      <td>54.574789</td>
    </tr>
  </tbody>
</table>
</div>



### Question 6a
Using `groupby.agg` or one of the shorthand methods (`groupby.min`, `groupby.first`, etc.), create a Series `best_result_percentage_only` that returns a Series showing the entire best result for every party, sorted in decreasing order. Your Series should include only parties which have earned at least 10% of the vote in some election. Your result should look like this:

```shell
Party
Democratic               61.344703
Republican               60.907806
Democratic-Republican    57.210122
National Union           54.951512
Whig                     53.051213
Liberal Republican       44.071406
National Republican      43.796073
Northern Democratic      29.522311
Progressive              27.457433
American                 21.554001
Independent              18.956298
Southern Democratic      18.138998
American Independent     13.571218
Constitutional Union     12.639283
Free Soil                10.138474
Name: %, dtype: float64
```

A list of named `groupby.agg` shorthand methods is [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#aggregation) (you'll have to scroll down about one page).

<!--
BEGIN QUESTION
name: q6a
-->


```python
best_result_percentage_only = elections[elections['%']>=10].groupby('Party')['%'].agg(max).sort_values(ascending=False)
# put your code above this line
best_result_percentage_only
```

    C:\Users\86135\AppData\Local\Temp\ipykernel_61736\687541662.py:1: FutureWarning: The provided callable <built-in function max> is currently using SeriesGroupBy.max. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "max" instead.
      best_result_percentage_only = elections[elections['%']>=10].groupby('Party')['%'].agg(max).sort_values(ascending=False)
    




    Party
    Democratic               61.344703
    Republican               60.907806
    Democratic-Republican    57.210122
    National Union           54.951512
    Whig                     53.051213
    Liberal Republican       44.071406
    National Republican      43.796073
    Northern Democratic      29.522311
    Progressive              27.457433
    American                 21.554001
    Independent              18.956298
    Southern Democratic      18.138998
    American Independent     13.571218
    Constitutional Union     12.639283
    Free Soil                10.138474
    Name: %, dtype: float64




```python
grader.check("q6a")
```




<p><strong style='color: red;'><pre style='display: inline;'>q6a</pre> results:</strong></p><p><strong><pre style='display: inline;'>q6a - 1</pre> result:</strong></p><pre>    ✅ Test case passed</pre><p><strong><pre style='display: inline;'>q6a - 2</pre> result:</strong></p><pre>    ❌ Test case failed
    Trying:
        best_result_percentage_only["Independent"].sum()
    Expecting:
        18.95629754
    **********************************************************************
    Line 1, in q6a 1
    Failed example:
        best_result_percentage_only["Independent"].sum()
    Expected:
        18.95629754
    Got:
        np.float64(18.95629754)
</pre><p><strong><pre style='display: inline;'>q6a - 3</pre> result:</strong></p><pre>    ❌ Test case failed
    Trying:
        best_result_percentage_only.iloc[0]
    Expecting:
        61.34470329
    **********************************************************************
    Line 1, in q6a 2
    Failed example:
        best_result_percentage_only.iloc[0]
    Expected:
        61.34470329
    Got:
        np.float64(61.34470329)
</pre>



### Question 6b  
Repeat Question 6a. However, this time, your result should be a DataFrame showing all available information rather than only the percentage as a series.

This question is trickier than Question 6a. Make sure to check the Lecture 4 slides if you're stuck! It's very easy to make a subtle mistake that shows Woodrow Wilson and Howard Taft both winning the 2020 election.

For example, the first 3 rows of your table should be:

|Party | Year | Candidate      | Popular Vote | Result | %         |
|------|------|----------------|--------------|--------|-----------|
|**Democratic**  | 1964 | Lyndon Johnson | 43127041      | win   | 61.344703 |
|**Republican**  | 1972 | Richard Nixon | 47168710      | win   | 60.907806 |
|**Democratic-Republican**  | 1824 | Andrew Jackson | 151271      | loss   | 57.210122 |

Note that the index is `Party`. In other words, don't use `reset_index`.

<!--
BEGIN QUESTION
name: q6b
-->


```python
best_result = elections[elections['%']>=10].sort_values(by='%',ascending=False).groupby(['Party']).agg(lambda x: x.iloc[0]).sort_values(by='%',ascending=False)
# @ 52:03 in the video of Lecture 4
# put your code above this line
best_result
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
      <th>Year</th>
      <th>Candidate</th>
      <th>Popular vote</th>
      <th>Result</th>
      <th>%</th>
    </tr>
    <tr>
      <th>Party</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Democratic</th>
      <td>1964</td>
      <td>Lyndon Johnson</td>
      <td>43127041</td>
      <td>win</td>
      <td>61.344703</td>
    </tr>
    <tr>
      <th>Republican</th>
      <td>1972</td>
      <td>Richard Nixon</td>
      <td>47168710</td>
      <td>win</td>
      <td>60.907806</td>
    </tr>
    <tr>
      <th>Democratic-Republican</th>
      <td>1824</td>
      <td>Andrew Jackson</td>
      <td>151271</td>
      <td>loss</td>
      <td>57.210122</td>
    </tr>
    <tr>
      <th>National Union</th>
      <td>1864</td>
      <td>Abraham Lincoln</td>
      <td>2211317</td>
      <td>win</td>
      <td>54.951512</td>
    </tr>
    <tr>
      <th>Whig</th>
      <td>1840</td>
      <td>William Henry Harrison</td>
      <td>1275583</td>
      <td>win</td>
      <td>53.051213</td>
    </tr>
    <tr>
      <th>Liberal Republican</th>
      <td>1872</td>
      <td>Horace Greeley</td>
      <td>2834761</td>
      <td>loss</td>
      <td>44.071406</td>
    </tr>
    <tr>
      <th>National Republican</th>
      <td>1828</td>
      <td>John Quincy Adams</td>
      <td>500897</td>
      <td>loss</td>
      <td>43.796073</td>
    </tr>
    <tr>
      <th>Northern Democratic</th>
      <td>1860</td>
      <td>Stephen A. Douglas</td>
      <td>1380202</td>
      <td>loss</td>
      <td>29.522311</td>
    </tr>
    <tr>
      <th>Progressive</th>
      <td>1912</td>
      <td>Theodore Roosevelt</td>
      <td>4122721</td>
      <td>loss</td>
      <td>27.457433</td>
    </tr>
    <tr>
      <th>American</th>
      <td>1856</td>
      <td>Millard Fillmore</td>
      <td>873053</td>
      <td>loss</td>
      <td>21.554001</td>
    </tr>
    <tr>
      <th>Independent</th>
      <td>1992</td>
      <td>Ross Perot</td>
      <td>19743821</td>
      <td>loss</td>
      <td>18.956298</td>
    </tr>
    <tr>
      <th>Southern Democratic</th>
      <td>1860</td>
      <td>John C. Breckinridge</td>
      <td>848019</td>
      <td>loss</td>
      <td>18.138998</td>
    </tr>
    <tr>
      <th>American Independent</th>
      <td>1968</td>
      <td>George Wallace</td>
      <td>9901118</td>
      <td>loss</td>
      <td>13.571218</td>
    </tr>
    <tr>
      <th>Constitutional Union</th>
      <td>1860</td>
      <td>John Bell</td>
      <td>590901</td>
      <td>loss</td>
      <td>12.639283</td>
    </tr>
    <tr>
      <th>Free Soil</th>
      <td>1848</td>
      <td>Martin Van Buren</td>
      <td>291501</td>
      <td>loss</td>
      <td>10.138474</td>
    </tr>
  </tbody>
</table>
</div>




```python
grader.check("q6b")
```




<p><strong style='color: red;'><pre style='display: inline;'>q6b</pre> results:</strong></p><p><strong><pre style='display: inline;'>q6b - 1</pre> result:</strong></p><pre>    ✅ Test case passed</pre><p><strong><pre style='display: inline;'>q6b - 2</pre> result:</strong></p><pre>    ❌ Test case failed
    Trying:
        best_result["Popular vote"].sum() 
    Expecting:
        135020916
    **********************************************************************
    Line 1, in q6b 1
    Failed example:
        best_result["Popular vote"].sum() 
    Expected:
        135020916
    Got:
        np.int64(135020916)
</pre><p><strong><pre style='display: inline;'>q6b - 3</pre> result:</strong></p><pre>    ✅ Test case passed</pre>



### Question 6c

Our DataFrame contains a number of parties which have never had a successful presidential run. For example, the 2020 elections included candiates from the Libertarian and Green parties, neither of which have elected a president.


```python
# just run this cell
elections.tail(5)
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
      <th>Year</th>
      <th>Candidate</th>
      <th>Party</th>
      <th>Popular vote</th>
      <th>Result</th>
      <th>%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>177</th>
      <td>2016</td>
      <td>Jill Stein</td>
      <td>Green</td>
      <td>1457226</td>
      <td>loss</td>
      <td>1.073699</td>
    </tr>
    <tr>
      <th>178</th>
      <td>2020</td>
      <td>Joseph Biden</td>
      <td>Democratic</td>
      <td>81268924</td>
      <td>win</td>
      <td>51.311515</td>
    </tr>
    <tr>
      <th>179</th>
      <td>2020</td>
      <td>Donald Trump</td>
      <td>Republican</td>
      <td>74216154</td>
      <td>loss</td>
      <td>46.858542</td>
    </tr>
    <tr>
      <th>180</th>
      <td>2020</td>
      <td>Jo Jorgensen</td>
      <td>Libertarian</td>
      <td>1865724</td>
      <td>loss</td>
      <td>1.177979</td>
    </tr>
    <tr>
      <th>181</th>
      <td>2020</td>
      <td>Howard Hawkins</td>
      <td>Green</td>
      <td>405035</td>
      <td>loss</td>
      <td>0.255731</td>
    </tr>
  </tbody>
</table>
</div>



Suppose we were conducting an analysis trying to focus our attention on parties that had elected a president. 

The most natural approach is to use `groupby.filter`. This is an incredibly powerful but subtle tool for filtering data.

As a reminder of how filter works, see [this slide](https://docs.google.com/presentation/d/1FC-cs5MTGSkDzI_7R_ZENgwoHQ4aVamxFOpJuWT0fo0/edit#slide=id.g5ff184b7f5_0_507). 
The code below accomplishes the task at hand. It does this by creating a function that returns True if and only if a sub-dataframe (a.k.a. group) contains at least one winner. This function in turn uses the [Pandas function "any"](https://pandas.pydata.org/docs/reference/api/pandas.Series.any.html).


```python
# just run this cell
def at_least_one_candidate_in_the_frame_has_won(frame):
    """Returns df with rows only kept for parties that have
    won at least one election
    """
    return (frame["Result"] == 'win').any()

winners_only = (
    elections
        .groupby("Party")
        .filter(at_least_one_candidate_in_the_frame_has_won)
)
winners_only.tail(5)
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
      <th>Year</th>
      <th>Candidate</th>
      <th>Party</th>
      <th>Popular vote</th>
      <th>Result</th>
      <th>%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>171</th>
      <td>2012</td>
      <td>Mitt Romney</td>
      <td>Republican</td>
      <td>60933504</td>
      <td>loss</td>
      <td>47.384076</td>
    </tr>
    <tr>
      <th>173</th>
      <td>2016</td>
      <td>Donald Trump</td>
      <td>Republican</td>
      <td>62984828</td>
      <td>win</td>
      <td>46.407862</td>
    </tr>
    <tr>
      <th>176</th>
      <td>2016</td>
      <td>Hillary Clinton</td>
      <td>Democratic</td>
      <td>65853514</td>
      <td>loss</td>
      <td>48.521539</td>
    </tr>
    <tr>
      <th>178</th>
      <td>2020</td>
      <td>Joseph Biden</td>
      <td>Democratic</td>
      <td>81268924</td>
      <td>win</td>
      <td>51.311515</td>
    </tr>
    <tr>
      <th>179</th>
      <td>2020</td>
      <td>Donald Trump</td>
      <td>Republican</td>
      <td>74216154</td>
      <td>loss</td>
      <td>46.858542</td>
    </tr>
  </tbody>
</table>
</div>



Alternately we could have used a `lambda` function instead of explicitly defining a named function using `def`. 


```python
# just run this cell (alternative)
winners_only = (
    elections
        .groupby("Party")
        .filter(lambda x : (x["Result"] == "win").any())
)
winners_only.tail(5)
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
      <th>Year</th>
      <th>Candidate</th>
      <th>Party</th>
      <th>Popular vote</th>
      <th>Result</th>
      <th>%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>171</th>
      <td>2012</td>
      <td>Mitt Romney</td>
      <td>Republican</td>
      <td>60933504</td>
      <td>loss</td>
      <td>47.384076</td>
    </tr>
    <tr>
      <th>173</th>
      <td>2016</td>
      <td>Donald Trump</td>
      <td>Republican</td>
      <td>62984828</td>
      <td>win</td>
      <td>46.407862</td>
    </tr>
    <tr>
      <th>176</th>
      <td>2016</td>
      <td>Hillary Clinton</td>
      <td>Democratic</td>
      <td>65853514</td>
      <td>loss</td>
      <td>48.521539</td>
    </tr>
    <tr>
      <th>178</th>
      <td>2020</td>
      <td>Joseph Biden</td>
      <td>Democratic</td>
      <td>81268924</td>
      <td>win</td>
      <td>51.311515</td>
    </tr>
    <tr>
      <th>179</th>
      <td>2020</td>
      <td>Donald Trump</td>
      <td>Republican</td>
      <td>74216154</td>
      <td>loss</td>
      <td>46.858542</td>
    </tr>
  </tbody>
</table>
</div>



For your exercise, you'll do a less restrictive filtering of the elections data.

**Exercise**: Using `filter`, create a DataFrame `major_party_results_since_1988` that includes all election results starting in 1988, but only show a row if the Party it belongs to has earned at least 1% of the popular vote in ANY election since 1988.

For example, in 1988, you should not include the `New Alliance` candidate, since this party has not earned 1% of the vote since 1988. However, you should include the `Libertarian` candidate from 1988 despite only having 0.47 percent of the vote in 1988, because in 2016 and 2020, the Libertarian candidates Gary Johnson and Jo Jorgensen exceeded 1% of the vote.

For example, the first three rows of the table you generate should look like:

|     |   Year | Candidate         | Party       |   Popular vote | Result   |         % |
|----:|-------:|:------------------|:------------|---------------:|:---------|----------:|
| 135 |   1988 | George H. W. Bush | Republican  |       48886597 | win      | 53.5188   |
| 137 |   1988 | Michael Dukakis   | Democratic  |       41809074 | loss     | 45.7707   |
| 138 |   1988 | Ron Paul          | Libertarian |         431750 | loss     |  0.47266  |

<!--
BEGIN QUESTION
name: q6c
-->


```python
major_party_results_since_1988 = elections[(elections['Year']>=1988)].groupby('Party').filter(lambda x: (x['%'] >= 1).any())
major_party_results_since_1988.head()
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
      <th>Year</th>
      <th>Candidate</th>
      <th>Party</th>
      <th>Popular vote</th>
      <th>Result</th>
      <th>%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>135</th>
      <td>1988</td>
      <td>George H. W. Bush</td>
      <td>Republican</td>
      <td>48886597</td>
      <td>win</td>
      <td>53.518845</td>
    </tr>
    <tr>
      <th>137</th>
      <td>1988</td>
      <td>Michael Dukakis</td>
      <td>Democratic</td>
      <td>41809074</td>
      <td>loss</td>
      <td>45.770691</td>
    </tr>
    <tr>
      <th>138</th>
      <td>1988</td>
      <td>Ron Paul</td>
      <td>Libertarian</td>
      <td>431750</td>
      <td>loss</td>
      <td>0.472660</td>
    </tr>
    <tr>
      <th>139</th>
      <td>1992</td>
      <td>Andre Marrou</td>
      <td>Libertarian</td>
      <td>290087</td>
      <td>loss</td>
      <td>0.278516</td>
    </tr>
    <tr>
      <th>140</th>
      <td>1992</td>
      <td>Bill Clinton</td>
      <td>Democratic</td>
      <td>44909806</td>
      <td>win</td>
      <td>43.118485</td>
    </tr>
  </tbody>
</table>
</div>




```python
grader.check("q6c")
```




<p><strong style='color: red;'><pre style='display: inline;'>q6c</pre> results:</strong></p><p><strong><pre style='display: inline;'>q6c - 1</pre> result:</strong></p><pre>    ✅ Test case passed</pre><p><strong><pre style='display: inline;'>q6c - 2</pre> result:</strong></p><pre>    ❌ Test case failed
    Trying:
        major_party_results_since_1988["%"].min() 
    Expecting:
        0.098088334
    **********************************************************************
    Line 1, in q6c 1
    Failed example:
        major_party_results_since_1988["%"].min() 
    Expected:
        0.098088334
    Got:
        np.float64(0.098088334)
</pre><p><strong><pre style='display: inline;'>q6c - 3</pre> result:</strong></p><pre>    ✅ Test case passed</pre>



### Question 7

Pandas provides special purpose functions for working with specific common data types such as strings and dates. For example, the code below provides the length of every Candidate's name from our elections dataset.


```python
elections["Candidate"].str.len()
```




    0      14
    1      17
    2      14
    3      17
    4      14
           ..
    177    10
    178    12
    179    12
    180    12
    181    14
    Name: Candidate, Length: 182, dtype: int64



**Exercise**: Using `.str.split`. Create a new DataFrame called `elections_with_first_name` with a new column `First Name` that is equal to the Candidate's first name.

See the Pandas `str` [documentation](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.split.html) for documentation on using `str.split`.

Hint: Use `[0]` somewhere in your code.

<!--
BEGIN QUESTION
name: q7
-->


```python
elections_with_first_name = elections.copy()
# your code here
elections_with_first_name['First Name'] = elections_with_first_name['Candidate'].str.split(' ').str[0].to_frame()
# elections_with_first_name['First Name'] = 
# end your code
elections_with_first_name
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
      <th>Year</th>
      <th>Candidate</th>
      <th>Party</th>
      <th>Popular vote</th>
      <th>Result</th>
      <th>%</th>
      <th>First Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1824</td>
      <td>Andrew Jackson</td>
      <td>Democratic-Republican</td>
      <td>151271</td>
      <td>loss</td>
      <td>57.210122</td>
      <td>Andrew</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1824</td>
      <td>John Quincy Adams</td>
      <td>Democratic-Republican</td>
      <td>113142</td>
      <td>win</td>
      <td>42.789878</td>
      <td>John</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1828</td>
      <td>Andrew Jackson</td>
      <td>Democratic</td>
      <td>642806</td>
      <td>win</td>
      <td>56.203927</td>
      <td>Andrew</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1828</td>
      <td>John Quincy Adams</td>
      <td>National Republican</td>
      <td>500897</td>
      <td>loss</td>
      <td>43.796073</td>
      <td>John</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1832</td>
      <td>Andrew Jackson</td>
      <td>Democratic</td>
      <td>702735</td>
      <td>win</td>
      <td>54.574789</td>
      <td>Andrew</td>
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
    </tr>
    <tr>
      <th>177</th>
      <td>2016</td>
      <td>Jill Stein</td>
      <td>Green</td>
      <td>1457226</td>
      <td>loss</td>
      <td>1.073699</td>
      <td>Jill</td>
    </tr>
    <tr>
      <th>178</th>
      <td>2020</td>
      <td>Joseph Biden</td>
      <td>Democratic</td>
      <td>81268924</td>
      <td>win</td>
      <td>51.311515</td>
      <td>Joseph</td>
    </tr>
    <tr>
      <th>179</th>
      <td>2020</td>
      <td>Donald Trump</td>
      <td>Republican</td>
      <td>74216154</td>
      <td>loss</td>
      <td>46.858542</td>
      <td>Donald</td>
    </tr>
    <tr>
      <th>180</th>
      <td>2020</td>
      <td>Jo Jorgensen</td>
      <td>Libertarian</td>
      <td>1865724</td>
      <td>loss</td>
      <td>1.177979</td>
      <td>Jo</td>
    </tr>
    <tr>
      <th>181</th>
      <td>2020</td>
      <td>Howard Hawkins</td>
      <td>Green</td>
      <td>405035</td>
      <td>loss</td>
      <td>0.255731</td>
      <td>Howard</td>
    </tr>
  </tbody>
</table>
<p>182 rows × 7 columns</p>
</div>




```python
grader.check("q7")
```




<p><strong><pre style='display: inline;'>q7</pre></strong> passed! 🙌</p>



### Question 8

The code below creates a table with the frequency of all names from 2020.


```python
# just run this cell
baby_names_2020 = (
    baby_names.query('Year == 2020')
              .groupby("Name")
              .sum()[["Count"]]
              .reset_index()
)
baby_names_2020
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
      <th>Name</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aaden</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Aadhira</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Aadhvik</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Aadhya</td>
      <td>186</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Aadi</td>
      <td>14</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8697</th>
      <td>Zymere</td>
      <td>6</td>
    </tr>
    <tr>
      <th>8698</th>
      <td>Zymir</td>
      <td>74</td>
    </tr>
    <tr>
      <th>8699</th>
      <td>Zyon</td>
      <td>130</td>
    </tr>
    <tr>
      <th>8700</th>
      <td>Zyra</td>
      <td>33</td>
    </tr>
    <tr>
      <th>8701</th>
      <td>Zyrah</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>8702 rows × 2 columns</p>
</div>



**Exercise**: Using the `pd.merge` function described in lecture, combine the `baby_names_2020` table with the `elections_with_first_name` table you created earlier to form `presidential_candidates_and_name_popularity`.

<!--
BEGIN QUESTION
name: q8
-->


```python
presidential_candidates_and_name_popularity = pd.merge(elections_with_first_name,baby_names_2020,  left_on='First Name', right_on='Name')
presidential_candidates_and_name_popularity
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
      <th>Year</th>
      <th>Candidate</th>
      <th>Party</th>
      <th>Popular vote</th>
      <th>Result</th>
      <th>%</th>
      <th>First Name</th>
      <th>Name</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1824</td>
      <td>Andrew Jackson</td>
      <td>Democratic-Republican</td>
      <td>151271</td>
      <td>loss</td>
      <td>57.210122</td>
      <td>Andrew</td>
      <td>Andrew</td>
      <td>5991</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1824</td>
      <td>John Quincy Adams</td>
      <td>Democratic-Republican</td>
      <td>113142</td>
      <td>win</td>
      <td>42.789878</td>
      <td>John</td>
      <td>John</td>
      <td>8180</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1828</td>
      <td>Andrew Jackson</td>
      <td>Democratic</td>
      <td>642806</td>
      <td>win</td>
      <td>56.203927</td>
      <td>Andrew</td>
      <td>Andrew</td>
      <td>5991</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1828</td>
      <td>John Quincy Adams</td>
      <td>National Republican</td>
      <td>500897</td>
      <td>loss</td>
      <td>43.796073</td>
      <td>John</td>
      <td>John</td>
      <td>8180</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1832</td>
      <td>Andrew Jackson</td>
      <td>Democratic</td>
      <td>702735</td>
      <td>win</td>
      <td>54.574789</td>
      <td>Andrew</td>
      <td>Andrew</td>
      <td>5991</td>
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
    </tr>
    <tr>
      <th>148</th>
      <td>2016</td>
      <td>Hillary Clinton</td>
      <td>Democratic</td>
      <td>65853514</td>
      <td>loss</td>
      <td>48.521539</td>
      <td>Hillary</td>
      <td>Hillary</td>
      <td>20</td>
    </tr>
    <tr>
      <th>149</th>
      <td>2020</td>
      <td>Joseph Biden</td>
      <td>Democratic</td>
      <td>81268924</td>
      <td>win</td>
      <td>51.311515</td>
      <td>Joseph</td>
      <td>Joseph</td>
      <td>8349</td>
    </tr>
    <tr>
      <th>150</th>
      <td>2020</td>
      <td>Donald Trump</td>
      <td>Republican</td>
      <td>74216154</td>
      <td>loss</td>
      <td>46.858542</td>
      <td>Donald</td>
      <td>Donald</td>
      <td>407</td>
    </tr>
    <tr>
      <th>151</th>
      <td>2020</td>
      <td>Jo Jorgensen</td>
      <td>Libertarian</td>
      <td>1865724</td>
      <td>loss</td>
      <td>1.177979</td>
      <td>Jo</td>
      <td>Jo</td>
      <td>6</td>
    </tr>
    <tr>
      <th>152</th>
      <td>2020</td>
      <td>Howard Hawkins</td>
      <td>Green</td>
      <td>405035</td>
      <td>loss</td>
      <td>0.255731</td>
      <td>Howard</td>
      <td>Howard</td>
      <td>131</td>
    </tr>
  </tbody>
</table>
<p>153 rows × 9 columns</p>
</div>




```python
grader.check("q8")
```




<p><strong style='color: red;'><pre style='display: inline;'>q8</pre> results:</strong></p><p><strong><pre style='display: inline;'>q8 - 1</pre> result:</strong></p><pre>    ✅ Test case passed</pre><p><strong><pre style='display: inline;'>q8 - 2</pre> result:</strong></p><pre>    ✅ Test case passed</pre><p><strong><pre style='display: inline;'>q8 - 3</pre> result:</strong></p><pre>    ❌ Test case failed
    Trying:
        presidential_candidates_and_name_popularity[presidential_candidates_and_name_popularity["Candidate"] == "Jo Jorgensen"].iloc[0]["Count"]
    Expecting:
        6
    **********************************************************************
    Line 1, in q8 2
    Failed example:
        presidential_candidates_and_name_popularity[presidential_candidates_and_name_popularity["Candidate"] == "Jo Jorgensen"].iloc[0]["Count"]
    Expected:
        6
    Got:
        np.int64(6)
</pre>



Just for fun: Which historical presidential candidates have names that were the least and most popular in 2020? Note: Here you'll observe a common problem in data science -- one of the least popular names is actually due to the fact that one recent president was so commonly known by his nickname that he appears named as such in the database from which you pulled election results.


```python
# your optional code here
...
```

## Bonus Exercises

The following exercises are optional and use the `ca_baby_names` dataset defined below.


```python
# just run this cell
ca_baby_names = baby_names.query('State == "CA"')
ca_baby_names
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
      <th>State</th>
      <th>Sex</th>
      <th>Year</th>
      <th>Name</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>390635</th>
      <td>CA</td>
      <td>F</td>
      <td>1910</td>
      <td>Mary</td>
      <td>295</td>
    </tr>
    <tr>
      <th>390636</th>
      <td>CA</td>
      <td>F</td>
      <td>1910</td>
      <td>Helen</td>
      <td>239</td>
    </tr>
    <tr>
      <th>390637</th>
      <td>CA</td>
      <td>F</td>
      <td>1910</td>
      <td>Dorothy</td>
      <td>220</td>
    </tr>
    <tr>
      <th>390638</th>
      <td>CA</td>
      <td>F</td>
      <td>1910</td>
      <td>Margaret</td>
      <td>163</td>
    </tr>
    <tr>
      <th>390639</th>
      <td>CA</td>
      <td>F</td>
      <td>1910</td>
      <td>Frances</td>
      <td>134</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>784809</th>
      <td>CA</td>
      <td>M</td>
      <td>2020</td>
      <td>Ziaan</td>
      <td>5</td>
    </tr>
    <tr>
      <th>784810</th>
      <td>CA</td>
      <td>M</td>
      <td>2020</td>
      <td>Ziad</td>
      <td>5</td>
    </tr>
    <tr>
      <th>784811</th>
      <td>CA</td>
      <td>M</td>
      <td>2020</td>
      <td>Ziaire</td>
      <td>5</td>
    </tr>
    <tr>
      <th>784812</th>
      <td>CA</td>
      <td>M</td>
      <td>2020</td>
      <td>Zidan</td>
      <td>5</td>
    </tr>
    <tr>
      <th>784813</th>
      <td>CA</td>
      <td>M</td>
      <td>2020</td>
      <td>Zymir</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>394179 rows × 5 columns</p>
</div>



#### Sorted Female Name Counts

Create a Series `female_name_since_2000_count` which gives the total number of occurrences of each name for female babies born in California from the year 2000 or later. The index should be the name, and the value should be the total number of births. Your Series should be ordered in decreasing order of count. For example, your first row should have index "Emily" and value 52334, because 52334 Emilys have been born since the year 2000 in California.


```python
female_name_since_2000_count = 
female_name_since_2000_count
```

#### Counts for All Names

Using `groupby`, create a Series `count_for_names_2020` listing all baby names from 2020 in California, in decreasing order of popularity. The result should not be broken down by sex! If a name is used by both male and female babies, the number you provide should be the total.

**Note:** *In this question we are now computing the number of registered babies with a given name.* 

For example, `count_for_names_2020["Noah"]` should be the number 2631 because in 2018 there were 2631 Noahs born (23 female and 2608 male).


```python
...
count_for_names_2020
```

### Extra: Explore the Data Set

The popularity of some baby names may be influenced by cultural phenomena, such as a political figure coming to power.  Below, we plot the popularity of name Hillary for female babies in Calfiornia over time. What do you notice about this plot? What real-world events in the U.S. occurred when there was a steep drop in babies named Hillary?


```python
hillary_baby_name = baby_names[(baby_names['Name'] == 'Hillary') & (baby_names['State'] == 'CA') & (baby_names['Sex'] == 'F')]
plt.plot(hillary_baby_name['Year'], hillary_baby_name['Count'])
plt.title("Hillary Popularity Over Time")
plt.xlabel('Year')
plt.ylabel('Count');
```

The code above is hard coded to generate a dataframe representing the popularity of the female name Hillary in the state of California. While this approach works, it's inelegant.

Here we'll use a more elegant approach that builds a dataframe such that:
1. It contains ALL names.
2. The counts are summed across all 50 states, not just California.

To do this, we use `groupby`, though here we're grouping on **two columns** ("Name" and "Year") instead of just one. After grouping, we use the `sum` aggregation function.


```python
# just run this cell
counts_aggregated_by_name_and_year = baby_names.groupby(["Name", "Year"]).sum()
counts_aggregated_by_name_and_year
```

Note that the resulting DataFrame is multi-indexed, i.e. it has two indices. The outer index is the Name, and the inner index is the Year. 

In order to visualize this data, we'll use `reset_index` in order to set the index back to an integer and transform the Name and Year back into columnar data.


```python
# just run this cell
counts_aggregated_by_name_and_year = counts_aggregated_by_name_and_year.reset_index()
counts_aggregated_by_name_and_year
```

Similar to before, we can plot the popularity of a given name by selecting the name we want to visualize. The code below is very similar to the plotting code above, except that we use query to get the name of interest instead of using a boolean array. 

**Note**: Here we use a special syntax `@name_of_interest` to tell the query command to use the python variable `name_of_interest`.

Try out some other names and see what trends you observe. Note that since this is the American social security database, international names are not well represented.


```python
# just run this cell
name_of_interest = 'Hillary'
chosen_baby_name = counts_aggregated_by_name_and_year.query("Name == @name_of_interest")
plt.plot(chosen_baby_name['Year'], chosen_baby_name['Count'])
plt.title(f"Popularity Of {name_of_interest} Over Time")
plt.xlabel('Year')
plt.ylabel('Count');
```

---

To double-check your work, the cell below will rerun all of the autograder tests.


```python
grader.check_all()
```

## Submission

Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output. The cell below will generate a zip file for you to submit. **Please save before exporting!**


```python
# Save your notebook first, then run this cell to export your submission.
grader.export(pdf=False)
```

 

