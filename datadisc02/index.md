# Datadisc02


# Discussion 2: Pandas Practice

We will begin our discussion of [Pandas](https://pandas.pydata.org/). You will practice:

* Selecting columns
* Filtering with boolean conditions 
* Counting with `value_counts`


```python
import pandas as pd
import numpy as np
```

## Pandas Practise

In the first Pandas question, we will be working with the `elections` dataset from lecture.


```python
elections = pd.read_csv("elections.csv") # read in the elections data into a pandas dataframe!
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



### Question 5

We want to select the "Popular vote" column as a `pd.Series`. Which of the following lines of code will error?

- `elections['Popular vote']`
- `elections.iloc['Popular vote']`
- `elections.loc['Popular vote']`
- `elections.loc[:, 'Popular vote']`
- `elections.iloc[:, 'Popular vote']`

Run each line in the cell below and see for yourself!


```python
# elections.iloc['Popular vote'] # wrong
# elections.iloc[:, 'popular votes'] # wrong
# elections['Popular vote'] # right
# elections.loc['Popular vote'] # ket error
# elections.loc[:,'Popular vote'] # right

```




    0        151271
    1        113142
    2        642806
    3        500897
    4        702735
             ...   
    173    62984828
    174      732273
    175     4489235
    176    65853514
    177     1457226
    Name: Popular vote, Length: 178, dtype: int64



### Question 6

Write one line of Pandas code that returns a `pd.DataFrame` that only contains election results from the 1900s.


```python
elections[(elections['Year'] >= 1900) & (elections['Year'] < 2000)] # 注意是 &  
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
      <th>54</th>
      <td>1900</td>
      <td>John G. Woolley</td>
      <td>Prohibition</td>
      <td>210864</td>
      <td>loss</td>
      <td>1.526821</td>
    </tr>
    <tr>
      <th>55</th>
      <td>1900</td>
      <td>William Jennings Bryan</td>
      <td>Democratic</td>
      <td>6370932</td>
      <td>loss</td>
      <td>46.130540</td>
    </tr>
    <tr>
      <th>56</th>
      <td>1900</td>
      <td>William McKinley</td>
      <td>Republican</td>
      <td>7228864</td>
      <td>win</td>
      <td>52.342640</td>
    </tr>
    <tr>
      <th>57</th>
      <td>1904</td>
      <td>Alton B. Parker</td>
      <td>Democratic</td>
      <td>5083880</td>
      <td>loss</td>
      <td>37.685116</td>
    </tr>
    <tr>
      <th>58</th>
      <td>1904</td>
      <td>Eugene V. Debs</td>
      <td>Socialist</td>
      <td>402810</td>
      <td>loss</td>
      <td>2.985897</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>147</th>
      <td>1996</td>
      <td>Howard Phillips</td>
      <td>Taxpayers</td>
      <td>184656</td>
      <td>loss</td>
      <td>0.192045</td>
    </tr>
    <tr>
      <th>148</th>
      <td>1996</td>
      <td>John Hagelin</td>
      <td>Natural Law</td>
      <td>113670</td>
      <td>loss</td>
      <td>0.118219</td>
    </tr>
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
      <th>150</th>
      <td>1996</td>
      <td>Ross Perot</td>
      <td>Reform</td>
      <td>8085294</td>
      <td>loss</td>
      <td>8.408844</td>
    </tr>
  </tbody>
</table>
<p>97 rows × 6 columns</p>
</div>



### Question 7

Write one line of Pandas code that returns a `pd.Series`, where the index is the Party, and the values are how many times that party won an election. Hint: use `value_counts`.


```python
# Your answer here
elections['Party'].value_counts()
```




    Party
    Democratic               46
    Republican               40
    Prohibition              11
    Libertarian              11
    Socialist                10
    Independent               6
    Whig                      6
    Green                     6
    Progressive               4
    Populist                  3
    Constitution              3
    American Independent      3
    American                  2
    National Republican       2
    Democratic-Republican     2
    Reform                    2
    Free Soil                 2
    Anti-Masonic              1
    National Union            1
    Constitutional Union      1
    National Democratic       1
    Union Labor               1
    Greenback                 1
    Anti-Monopoly             1
    Liberal Republican        1
    Southern Democratic       1
    Northern Democratic       1
    Farmer–Labor              1
    Dixiecrat                 1
    States' Rights            1
    Communist                 1
    Union                     1
    Taxpayers                 1
    New Alliance              1
    Citizens                  1
    Natural Law               1
    Name: count, dtype: int64



## Grading Assistance (Bonus)

Fernando is writing a grading script to compute grades for students in Data 101. Recall that many factors go into computing a student’s final grade, including homework, discussion, exams, and labs. In this question, we will help Fernando compute the homework grades for all students using a DataFrame, `hw_grades`, provided by Gradescope.

The Pandas DataFrame `hw_grades` contains homework grades for all students for all homework assignments, with one row for each combination of student and homework assignment. Any assignments that are incomplete are denoted by NaN (missing) values, and any late assignments are denoted by a True boolean value in the Late column. You may assume that the names of students are unique. Below is a sample of `hw_grades`.


```python
hw_grades = pd.read_csv("hw_grades.csv")
hw_grades.sample(5, random_state = 0)
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
      <th>Assignment</th>
      <th>Grade</th>
      <th>Late</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28</th>
      <td>Sid</td>
      <td>Homework 9</td>
      <td>82.517998</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Ash</td>
      <td>Homework 2</td>
      <td>78.264844</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Ash</td>
      <td>Homework 1</td>
      <td>98.421049</td>
      <td>False</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Emily</td>
      <td>Homework 2</td>
      <td>62.900313</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Meg</td>
      <td>Homework 3</td>
      <td>89.785619</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### Question 8a

Assuming there is a late penalty that causes a 10% grade reduction to the student’s
current score (i.e. a 65% score would become a 65% - 6.5% = 58.5%), write a line
of Pandas code to calculate all the homework grades, including the late penalty if
applicable, and store it in a column named `’LPGrade’`.


```python
# Your answer here
hw_grades['LPGrade'] = hw_grades['Grade'] * (1 - hw_grades['Late'] * 0.1) # 用个隐式转换
hw_grades.head()
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
      <th>Assignment</th>
      <th>Grade</th>
      <th>Late</th>
      <th>LPGrade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Meg</td>
      <td>Homework 1</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Meg</td>
      <td>Homework 2</td>
      <td>64.191844</td>
      <td>False</td>
      <td>64.191844</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Meg</td>
      <td>Homework 3</td>
      <td>89.785619</td>
      <td>False</td>
      <td>89.785619</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Meg</td>
      <td>Homework 4</td>
      <td>74.420033</td>
      <td>False</td>
      <td>74.420033</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Meg</td>
      <td>Homework 5</td>
      <td>74.372434</td>
      <td>True</td>
      <td>66.935190</td>
    </tr>
  </tbody>
</table>
</div>



### Question 8b

Which of the following expressions outputs the students’ names and number of late
assignments, from least to greatest number of late assignments?

- `hw_grades.groupby([’Name’]).sum().sort_values()`
- `hw_grades.groupby([’Name’, ’Late’]).sum().sort_values()`
- `hw_grades.groupby([’Name’]).sum()[’Late’].sort_values()`
- `hw_grades.groupby([’Name’]).sum().sort_values()[’Late’]`


```python
# Your answer here
# hw_grades.groupby(['Name']).sum().sort_values() # <---- Try to sort on df, but have to  give 'by=...' into sort_values()
hw_grades.groupby(['Name']).sum()['Late'].sort_values()
```




    Name
    Sid      1
    Emily    2
    Meg      2
    Ash      3
    Smith    3
    Name: Late, dtype: int64



### Question 8c

If each assignment is weighted equally, fill in the blanks below to calculate each student’s overall homework grade, including late penalties for any applicable assignments.

*Hint:* Recall that incomplete assignments have NaN values. How can we use `fillna` to replace these null values?

```
hw_grades._________(_______) \
         .groupby(___________)[____________] \
         .agg(____________)
```


```python
# Your answer here
hw_grades.fillna(0)\
    .groupby(['Name'])['LPGrade']\
    .agg('mean')
# Python中，反斜杠 \ 用作行续字符，它允许你将一行代码分割成多行，以提高代码的可读性。这在编写较长的一行代码时特别有用，可以避免代码过于拥挤，使得代码更易于阅读和维护。
```




    Name
    Ash      80.830657
    Emily    84.297725
    Meg      69.218137
    Sid      63.020729
    Smith    58.332233
    Name: LPGrade, dtype: float64



### Question 8d

Of all the homework assignments, which are the most difficult in terms of the ***median*** grade? Order by the median grade, from lowest to greatest. Do not consider incomplete assignments or late penalties in this calculation.

Fill in the blanks below to answer this question.

*Hint:* Recall that incomplete assignments have NaN values. How can we use `dropna` to remove these null values?

```
hw_grades._________() \
         .groupby(___________)[____________] \
         .agg(____________) \
         .sort_values()
```


```python
# Your answer here
hw_grades.dropna()\
    .groupby('Assignment')['Grade']\
    .agg('median')\
    .sort_values()
```




    Assignment
    Homework 2     64.160918
    Homework 10    66.366211
    Homework 5     74.372434
    Homework 8     76.362904
    Homework 4     78.207572
    Homework 3     78.348163
    Homework 9     82.517998
    Homework 6     84.369535
    Homework 1     85.473281
    Homework 7     92.200688
    Name: Grade, dtype: float64



