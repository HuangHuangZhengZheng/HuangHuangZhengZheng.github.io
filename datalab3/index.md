# DATA100-lab3: Data Cleaning and EDA


```python
# Initialize Otter
import otter
grader = otter.Notebook("lab03.ipynb")
```

# Lab 3: Data Cleaning and EDA

In this lab you will be working on visualizing a dataset from the City of Berkeley containing data on calls to the Berkeley Police Department. Information about the dataset can be found [at this link](https://data.cityofberkeley.info/Public-Safety/Berkeley-PD-Calls-for-Service/k2nh-s5h5).


### Content Warning
This lab includes an analysis of crime in Berkeley. If you feel uncomfortable with this topic, **please contact your GSI or the instructors.**

---
## Setup

In this lab, we'll perform Exploratory Data Analysis and learn some preliminary tips for working with matplotlib (a Python plotting library). Note that we configure a custom default figure size. Virtually every default aspect of matplotlib [can be customized](https://matplotlib.org/users/customizing.html).

**Collaborators:** *list names here*


```python
import pandas as pd
import numpy as np
import zipfile
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12, 9)
```


```python
fig = plt.figure()
plt.show(fig)
```


    <Figure size 1200x900 with 0 Axes>


<br/><br/>
<hr style="border: 5px solid #003262;" />
<hr style="border: 1px solid #fdb515;" />

# Part 1: Acquire the Data

**1. Obtain data**<br/>
To retrieve the dataset, we will use the `ds100_utils.fetch_and_cache` utility.


```python
# just run this cell
import ds100_utils

data_dir = 'data'
data_url = 'http://www.ds100.org/sp22/resources/assets/datasets/lab03_data_sp22.zip'
file_name = 'lab03_data_sp22.zip'

dest_path = ds100_utils.fetch_and_cache(data_url=data_url, file=file_name, data_dir=data_dir)
print(f'Located at {dest_path}')
```

    Using cached version that was downloaded (UTC): Sun Jul 28 01:04:24 2024
    Located at data\lab03_data_sp22.zip
    

**2. Unzip file**<br/>
We will now directly unzip the ZIP archive and start working with the uncompressed files.


```python
# just run this cell
my_zip = zipfile.ZipFile(dest_path, 'r')
my_zip.extractall(data_dir)
```

Note: There is no single right answer regarding whether to work with compressed files in their compressed state or to uncompress them on disk permanently. For example, if you need to work with multiple tools on the same files, or write many notebooks to analyze them—and they are not too large—it may be more convenient to uncompress them once.  But you may also have situations where you find it preferable to work with the compressed data directly.  

Python gives you tools for both approaches, and you should know how to perform both tasks in order to choose the one that best suits the problem at hand.

**3. View files**

Now, we'll use the `os` package to list all files in the `data` directory. `os.walk()` recursively traverses the directory, and `os.path.join()` creates the full pathname of each file.

If you're interested in learning more, check out the Python3 documentation pages for `os.walk` ([link](https://docs.python.org/3/library/os.html#os.walk)) and `os.path` ([link](https://docs.python.org/3/library/os.path.html#os.path.join)).

We use Python 3 [format strings](https://docs.python.org/3/tutorial/inputoutput.html) to nicely format the printed variables `dpath` and `fpath`.


```python
# just run this cell
# two for loop in the same time... ? not that funny yet?
import os

for root, directories, filenames in os.walk(data_dir):
    # first, print out all directories ---> "secret"
    for directory in directories:
        dpath = os.path.join(root, directory)
        print(f"d {dpath}")
        
    # next, print out all files
    for filename in filenames:  
        fpath = os.path.join(root,filename)
        print(f"  {fpath}")
```

    d data\secret
      data\ben_kurtovic.py
      data\Berkeley_PD_-_Calls_for_Service.csv
      data\dummy.txt
      data\hello_world.py
      data\lab03_data_sp22.zip
      data\secret\do_not_readme.md
    

In this Lab, we'll be working with the `Berkeley_PD_-_Calls_for_Service.csv` file. Feel free to check out the other files, though.

<br/><br/>
<hr style="border: 5px solid #003262;" />
<hr style="border: 1px solid #fdb515;" />

# Part 2: Clean and Explore the Data

Let's now load the CSV file we have into a `pandas.DataFrame` object and start exploring the data.


```python
# just run this cell
calls = pd.read_csv("data/Berkeley_PD_-_Calls_for_Service.csv")
calls.head()
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
      <th>CASENO</th>
      <th>OFFENSE</th>
      <th>EVENTDT</th>
      <th>EVENTTM</th>
      <th>CVLEGEND</th>
      <th>CVDOW</th>
      <th>InDbDate</th>
      <th>Block_Location</th>
      <th>BLKADDR</th>
      <th>City</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21014296</td>
      <td>THEFT MISD. (UNDER $950)</td>
      <td>04/01/2021 12:00:00 AM</td>
      <td>10:58</td>
      <td>LARCENY</td>
      <td>4</td>
      <td>06/15/2021 12:00:00 AM</td>
      <td>Berkeley, CA\n(37.869058, -122.270455)</td>
      <td>NaN</td>
      <td>Berkeley</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21014391</td>
      <td>THEFT MISD. (UNDER $950)</td>
      <td>04/01/2021 12:00:00 AM</td>
      <td>10:38</td>
      <td>LARCENY</td>
      <td>4</td>
      <td>06/15/2021 12:00:00 AM</td>
      <td>Berkeley, CA\n(37.869058, -122.270455)</td>
      <td>NaN</td>
      <td>Berkeley</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21090494</td>
      <td>THEFT MISD. (UNDER $950)</td>
      <td>04/19/2021 12:00:00 AM</td>
      <td>12:15</td>
      <td>LARCENY</td>
      <td>1</td>
      <td>06/15/2021 12:00:00 AM</td>
      <td>2100 BLOCK HASTE ST\nBerkeley, CA\n(37.864908,...</td>
      <td>2100 BLOCK HASTE ST</td>
      <td>Berkeley</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21090204</td>
      <td>THEFT FELONY (OVER $950)</td>
      <td>02/13/2021 12:00:00 AM</td>
      <td>17:00</td>
      <td>LARCENY</td>
      <td>6</td>
      <td>06/15/2021 12:00:00 AM</td>
      <td>2600 BLOCK WARRING ST\nBerkeley, CA\n(37.86393...</td>
      <td>2600 BLOCK WARRING ST</td>
      <td>Berkeley</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21090179</td>
      <td>BURGLARY AUTO</td>
      <td>02/08/2021 12:00:00 AM</td>
      <td>6:20</td>
      <td>BURGLARY - VEHICLE</td>
      <td>1</td>
      <td>06/15/2021 12:00:00 AM</td>
      <td>2700 BLOCK GARBER ST\nBerkeley, CA\n(37.86066,...</td>
      <td>2700 BLOCK GARBER ST</td>
      <td>Berkeley</td>
      <td>CA</td>
    </tr>
  </tbody>
</table>
</div>



We see that the fields include a case number, the offense type, the date and time of the offense, the "CVLEGEND" which appears to be related to the offense type, a "CVDOW" which has no apparent meaning, a date added to the database, and the location spread across four fields. We can read more about each field from the City of the Berkeley's [open dataset webpage](https://data.cityofberkeley.info/Public-Safety/Berkeley-PD-Calls-for-Service/k2nh-s5h5).

Let's also check some basic information about this DataFrame using the `DataFrame.info` ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.info.html)) and `DataFrame.describe` methods ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html)).


```python
# df.info() displays 
# name and type of each column,
# number of non-null entries, and
# size of dataframe
calls.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2632 entries, 0 to 2631
    Data columns (total 11 columns):
     #   Column          Non-Null Count  Dtype 
    ---  ------          --------------  ----- 
     0   CASENO          2632 non-null   int64 
     1   OFFENSE         2632 non-null   object
     2   EVENTDT         2632 non-null   object
     3   EVENTTM         2632 non-null   object
     4   CVLEGEND        2632 non-null   object
     5   CVDOW           2632 non-null   int64 
     6   InDbDate        2632 non-null   object
     7   Block_Location  2632 non-null   object
     8   BLKADDR         2612 non-null   object
     9   City            2632 non-null   object
     10  State           2632 non-null   object
    dtypes: int64(2), object(9)
    memory usage: 226.3+ KB
    

Note that the `BLKADDR` column only has 2612 non-null entries, while the other columns all have 2632 entries. This is because the `.info()` method only counts non-null entries.


```python
calls.describe()
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
      <th>CASENO</th>
      <th>CVDOW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.632000e+03</td>
      <td>2632.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.095978e+07</td>
      <td>3.071049</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.452665e+05</td>
      <td>1.984136</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.005721e+07</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.100568e+07</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.101431e+07</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.102256e+07</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.109066e+07</td>
      <td>6.000000</td>
    </tr>
  </tbody>
</table>
</div>



Notice that the functions above reveal type information for the columns, as well as some basic statistics about the numerical columns found in the DataFrame. However, we still need more information about what each column represents. Let's explore the data further in Question 1.

Before we go over the fields to see their meanings, the cell below will verify that all the events happened in Berkeley by grouping on the `City` and `State` columns. You should see that all of our data falls into one group.


```python
calls.groupby(["City","State"]).count()
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
      <th></th>
      <th>CASENO</th>
      <th>OFFENSE</th>
      <th>EVENTDT</th>
      <th>EVENTTM</th>
      <th>CVLEGEND</th>
      <th>CVDOW</th>
      <th>InDbDate</th>
      <th>Block_Location</th>
      <th>BLKADDR</th>
    </tr>
    <tr>
      <th>City</th>
      <th>State</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Berkeley</th>
      <th>CA</th>
      <td>2632</td>
      <td>2632</td>
      <td>2632</td>
      <td>2632</td>
      <td>2632</td>
      <td>2632</td>
      <td>2632</td>
      <td>2632</td>
      <td>2612</td>
    </tr>
  </tbody>
</table>
</div>



When we called `head()` on the Dataframe `calls`, it seemed like `OFFENSE` and `CVLEGEND` both contained information about the type of event reported. What is the difference in meaning between the two columns? One way to probe this is to look at the `value_counts` for each Series.


```python
calls['OFFENSE'].value_counts().head(10)
```




    OFFENSE
    THEFT MISD. (UNDER $950)    559
    VEHICLE STOLEN              277
    BURGLARY AUTO               218
    THEFT FELONY (OVER $950)    215
    DISTURBANCE                 204
    BURGLARY RESIDENTIAL        178
    VANDALISM                   166
    THEFT FROM AUTO             163
    ASSAULT/BATTERY MISD.       116
    ROBBERY                      90
    Name: count, dtype: int64




```python
calls['CVLEGEND'].value_counts().head(10)
```




    CVLEGEND
    LARCENY                   782
    MOTOR VEHICLE THEFT       277
    BURGLARY - VEHICLE        218
    DISORDERLY CONDUCT        204
    BURGLARY - RESIDENTIAL    178
    VANDALISM                 166
    LARCENY - FROM VEHICLE    163
    ASSAULT                   150
    FRAUD                      93
    ROBBERY                    90
    Name: count, dtype: int64



It seems like `OFFENSE` is more specific than `CVLEGEND`, e.g. "LARCENY" vs. "THEFT FELONY (OVER $950)". If you're unfamiliar with the term, "larceny" is a legal term for theft of personal property.

To get a sense of how many subcategories there are for each `OFFENSE`, we will set `calls_by_cvlegend_and_offense` equal to a multi-indexed series where the data is first indexed on the `CVLEGEND` and then on the `OFFENSE`, and the data is equal to the number of offenses in the database that match the respective `CVLEGEND` and `OFFENSE`. As you can see, `calls_by_cvlegend_and_offense["LARCENY", "THEFT FROM PERSON"]` returns 8 which means there are 8 instances of larceny with offense of type "THEFT FROM PERSON" in the database.


```python
calls_by_cvlegend_and_offense = calls.groupby(["CVLEGEND", "OFFENSE"]).size()
calls_by_cvlegend_and_offense["LARCENY", "THEFT FROM PERSON"]
```




    np.int64(8)



<br/>

<hr style="border: 1px solid #fdb515;" />

## Question 1

In the cell below, set `answer1` equal to a list of strings corresponding to the possible values (which means unique :thinking: ) for `OFFENSE` when `CVLEGEND` is "LARCENY". You can type the answer manually, or you can create an expression that automatically extracts the names.

<!--
BEGIN QUESTION
name: q1
-->


```python
answer1 = list(calls[calls['CVLEGEND'] == 'LARCENY']['OFFENSE'].unique())
answer1
```




    ['THEFT MISD. (UNDER $950)', 'THEFT FELONY (OVER $950)', 'THEFT FROM PERSON']




```python
grader.check("q1")
```




<p><strong><pre style='display: inline;'>q1</pre></strong> passed! ✨</p>



<br/><br/>
<hr style="border: 5px solid #003262;" />
<hr style="border: 1px solid #fdb515;" />

# Part 3: Visualize the Data


### Matplotlib demo

You've seen some `matplotlib` in this class already, but now we will explain how to work with the ***object-oriented*** plotting API mentioned in this [matplotlib.pyplot tutorial](https://matplotlib.org/tutorials/introductory/pyplot.html) useful. In matplotlib, plotting occurs on a set of Axes which are associated with a Figure. An analogy is that on a blank canvas (Figure), you choose a location to plot (`Axes`) and then fill it in (plot).

There are two approaches to labeling and manipulating figure contents, which we'll discuss below. Approach 1 is closest to the plotting paradigm of MATLAB, the namesake of matplotlib; Approach 2 is also common because many matplotlib-based packages (such as Seaborn) explicitly return the current set of axes after plotting data. Both are essentially equivalent, and at the end of this class you'll be comfortable with both. 

**Approach 1**: matplotlib (or Seaborn) will auto-plot onto the current set of Axes or (if none exists) create a new figure/set of default axes. You can plot data using methods from `plt`, which is shorthand for the `matplotlib.pyplot` package. Then subsequent `plt` calls all edit the same set of default-created axes.

**Approach 2**:  
After creating the initial plot, you can also use `plt.gca()` to explicitly get the current set of axes, and then edit those specific axes using axes methods. Note the method naming is slightly different!


As an example of the built-in plotting functionality of pandas, the following example uses `plot` method of the `Series` class to generate a `barh` plot type to visually display the value counts for `CVLEGEND`.

There are also many other plots that we will explore throughout the lab.

**Side note:** Pandas also offers basic functionality for plotting. For example, the `DataFrame` and `Series` classes both have a `plot` method, which uses matplotlib under the hood. For now we'll focus on matplotlib itself so you get used to the syntax, but just know that convenient Pandas plotting methods exist for your own future data science exploration.

Below, we show both approaches by generating a horizontal bar plot to visually display the value counts for `CVLEGEND`. See the `barh`[documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html?highlight=barh#matplotlib.pyplot.barh) for more details.


```python
# DEMO CELL: assign demo to 1 or 2.
demo = 1

calls_cvlegend = calls['CVLEGEND'].value_counts()

if demo == 1:
    plt.barh(calls_cvlegend.index, calls_cvlegend) # creates figure and axes (y,x) not (x,y)!
    print(f"Demo {demo}: Using plt methods to update plot")
    plt.ylabel("Crime Category")               # uses most recently plotted axes
    plt.xlabel("Number of Calls")
    plt.title("Number of Calls by Crime Type")
elif demo == 2:
    print(f"Demo {demo}: Using axes methods to update plot")
    plt.barh(calls_cvlegend.index, calls_cvlegend) # creates figure and axes
    ax = plt.gca()
    ax.set_ylabel("Crime Category")
    ax.set_xlabel("Number of Calls")
    ax.set_title("Axes methods: Number of Calls by Crime Type")
else:
    print("Error: Please assign the demo variable to 1 or 2.")

plt.show()
```

    Demo 1: Using plt methods to update plot
    


    
![png](lab03_files/lab03_34_1.png)
    


<br/>

### An Additional Note on Plotting in Jupyter Notebooks

You may have noticed that many of our plotting code cells end with a semicolon `;` or `plt.show()`. The former prevents any extra output from the last line of the cell; the latter explicitly returns (and outputs) the figure. Try adding this to your own code in the following questions!

<br/>
<hr style="border: 1px solid #fdb515;" />

## Question 2

Now it is your turn to make a plot using `matplotlib`.  Let's start by transforming the data so that it is easier to work with.

The `CVDOW` field isn't named helpfully and it is hard to see the meaning from the data alone. According to the website [linked](https://data.cityofberkeley.info/Public-Safety/Berkeley-PD-Calls-for-Service/k2nh-s5h5) at the top of this notebook, `CVDOW` is actually indicating the day that events happened. 0->Sunday, 1->Monday ... 6->Saturday. 

## Question 2a

Add a new column `Day` into the `calls` dataframe that has the string weekday (eg. 'Sunday') for the corresponding value in CVDOW. For example, if the first 3 values of `CVDOW` are `[3, 6, 0]`, then the first 3 values of the `Day` column should be `["Wednesday", "Saturday", "Sunday"]`.

**Hint:** *Try using the [Series.map](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html) function on `calls["CVDOW"]`.  Can you assign this to the new column `calls["Day"]`?*

<!--
BEGIN QUESTION
name: q2a
-->


```python
days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
day_indices = range(7)
indices_to_days_dict = dict(zip(day_indices, days)) # Should look like {0:"Sunday", 1:"Monday", ..., 6:"Saturday"}

calls["Day"] = calls["CVDOW"].map(indices_to_days_dict)
```


```python
grader.check("q2a")
```




<p><strong><pre style='display: inline;'>q2a</pre></strong> passed! 🌟</p>




```python
# just run this example cell
ax = calls['CVLEGEND'].value_counts().plot(kind='barh')
ax.set_ylabel("Crime Category")
ax.set_xlabel("Number of Calls")
ax.set_title("Number of Calls By Crime Type");
```


    
![png](lab03_files/lab03_39_0.png)
    


**Challenge (OPTIONAL):** You could also accomplish this part as a table left join with `pd.merge` ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge.html?highlight=merge#pandas.merge)), instead of using `Series.map`. You would need to merge `calls` with a new dataframe that just contains the days of the week. If you have time, try it out in the below cell!

merge操作在接下来的lab有详细介绍！:wink:


```python
# scratch space for optional challenge
dow_df = pd.DataFrame(days, columns=["Day"])

...
```




    Ellipsis



---
## Question 2b

Now let's look at the `EVENTTM` column which indicates the time for events. Since it contains hour and minute information, let's extract the hour info and create a new column named `Hour` in the `calls` dataframe. You should save the hour as an `int`.


**Hint:** *Your code should only require one line*.<br/>
**Hint 2:** The vectorized `Series.str[ind]` performs integer indexing on an array entry.

<!--
BEGIN QUESTION
name: q2b
-->


```python
calls["Hour"] = calls["EVENTTM"].str.split(" ").str[0].str.split(":").str[0].astype(int)
calls["Hour"]
```




    0       10
    1       10
    2       12
    3       17
    4        6
            ..
    2627    12
    2628    15
    2629     0
    2630    18
    2631     2
    Name: Hour, Length: 2632, dtype: int64




```python
grader.check("q2b")
```

---
## Question 2c

Using `matplotlib`, construct a line plot with the count of the number of calls (entries in the table) for each hour of the day  **ordered by the time** (eg. `12:00 AM`, `1:00 AM`, ...). Please use the provided variable `hours` in your answer. Be sure that your axes are labeled and that your plot is titled.

**Hint**: Check out the `plt.plot` method in the matplotlib [tutorial](https://matplotlib.org/stable/tutorials/introductory/pyplot.html#intro-to-pyplot), as well as our demo above.

<!--
BEGIN QUESTION
name: q2c

-->


```python
hours = list(range(24))
calls_index = calls['Hour'].value_counts().sort_index()
# calls_index
plt.plot(hours, calls_index)
plt.xlabel("Hour")
plt.ylabel("Number of Calls")
plt.title("Number of Calls per Hour")

# Leave this for grading purposes
ax_3d = plt.gca()
```


    
![png](lab03_files/lab03_46_0.png)
    



```python
grader.check("q2c")
```




<p><strong><pre style='display: inline;'>q2c</pre></strong> passed! 🌟</p>





To better understand the time of day a report occurs we could **stratify the analysis by the day of the week.**  To do this we will use **violin plots** (a variation of a **box plot**), which you will learn in more detail next week.

For now, just know that a violin plot shows an estimated distribution of quantitative data (e.g., distribution of calls by hour) over a categorical variable (day of the week). More calls occur in hours corresponding to the fatter part of each violin; the median hour of all calls in a particular day is marked by the white dot in the corresponding violin.


```python
# for now, just run this cell.
# we will learn the seaborn visualization library next week.

import seaborn as sns
ax = sns.violinplot(data=calls.sort_values("CVDOW"),
                    x="Day", y="Hour",
                    saturation=0.5, palette="Set2")
ax.set_title("Stratified Analysis of Phone Calls by Day");
```

    C:\Users\86135\AppData\Local\Temp\ipykernel_19968\4044437892.py:5: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      ax = sns.violinplot(data=calls.sort_values("CVDOW"),
    


    
![png](lab03_files/lab03_49_1.png)
    


---
## Question 2d

Based on your line plot and our violin plot above, what observations can you make about the patterns of calls? Here are some dimensions to consider:
* Are there more calls in the day or at night?
* What are the most and least popular times?
* Do call patterns vary by day of the week?

<!--
BEGIN QUESTION
name: q2d
-->

<br/>
<hr style="border: 1px solid #fdb515;" />

## Question 3
In this last part of the lab, let's extract the GPS coordinates (latitude, longitude) from the `Block_Location` of each record.


```python
# an example block location entry
calls.loc[4, 'Block_Location']
```




    '2700 BLOCK GARBER ST\nBerkeley, CA\n(37.86066, -122.253407)'



---
## Question 3a: Regular Expressions


Use regular expressions to create a dataframe `calls_lat_lon` that has two columns titled `Lat` and `Lon`, containing the respective latitude and longitude of each record in `calls`. You should use the `Block_Location` column to extract the latitude and longitude coordinates.

**Hint**: Check out the `Series.str.extract` [documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.extract.html?highlight=extract#pandas.Series.str.extract).


<!--
BEGIN QUESTION
name: q3a
-->


```python
calls_lat_lon = calls['Block_Location'].str.extract(r'(\d+\.\d+),\s([-]*\d+\.\d+)')
# 注意捕获组涵盖范围！
calls_lat_lon.columns = ['Lat', 'Lon']

calls_lat_lon.head(10)

len(calls_lat_lon)
```




    2632




```python
grader.check("q3a")
```




<p><strong><pre style='display: inline;'>q3a</pre></strong> passed! 🚀</p>



---

## Question 3b: Join Tables

Let's include the GPS data into our `calls` data. In the below cell, use `calls_lat_lon` to add two new columns called `Lat` and `Lon` to the `calls` dataframe.

**Hint**: `pd.merge` ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html?highlight=merge#pandas.DataFrame.merge)) could be useful here. Note that the order of records in `calls` and `calls_lat_lon` are the same.

<!--
BEGIN QUESTION
name: q3b
-->


```python
import pandas as pd

# 创建两个长度相同的DataFrame
df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df2 = pd.DataFrame({'C': [7, 8, 9], 'D': [10, 11, 12]})

# 基于索引合并

# 使用'Key'列合并
merged_df = pd.merge(df1, df2, left_index=True, right_index=True)  # 根据你的需要选择合并类型
print(merged_df)
```

       A  B  C   D
    0  1  4  7  10
    1  2  5  8  11
    2  3  6  9  12
    


```python
# # 用merge合并两个dataframe，注意此时没有共同列！只能索引拼接！
# print(calls.shape)
# print(calls_lat_lon.shape)
calls = pd.merge(calls, calls_lat_lon, left_index=True, right_index=True)

# print(calls.shape)
calls.sample(5)      # random rows

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
      <th>CASENO</th>
      <th>OFFENSE</th>
      <th>EVENTDT</th>
      <th>EVENTTM</th>
      <th>CVLEGEND</th>
      <th>CVDOW</th>
      <th>InDbDate</th>
      <th>Block_Location</th>
      <th>BLKADDR</th>
      <th>City</th>
      <th>State</th>
      <th>Day</th>
      <th>Hour</th>
      <th>Lat</th>
      <th>Lon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1173</th>
      <td>21024375</td>
      <td>DISTURBANCE</td>
      <td>06/02/2021 12:00:00 AM</td>
      <td>9:00</td>
      <td>DISORDERLY CONDUCT</td>
      <td>3</td>
      <td>06/15/2021 12:00:00 AM</td>
      <td>2020 KITTREDGE ST\nBerkeley, CA\n(37.868356, -...</td>
      <td>2020 KITTREDGE ST</td>
      <td>Berkeley</td>
      <td>CA</td>
      <td>Wednesday</td>
      <td>9</td>
      <td>37.868356</td>
      <td>-122.268904</td>
    </tr>
    <tr>
      <th>2573</th>
      <td>21005894</td>
      <td>DISTURBANCE</td>
      <td>02/11/2021 12:00:00 AM</td>
      <td>14:12</td>
      <td>DISORDERLY CONDUCT</td>
      <td>4</td>
      <td>06/15/2021 12:00:00 AM</td>
      <td>2400 BLOCK DWIGHT WAY\nBerkeley, CA\n(37.86482...</td>
      <td>2400 BLOCK DWIGHT WAY</td>
      <td>Berkeley</td>
      <td>CA</td>
      <td>Thursday</td>
      <td>14</td>
      <td>37.864826</td>
      <td>-122.260719</td>
    </tr>
    <tr>
      <th>990</th>
      <td>21022043</td>
      <td>ROBBERY</td>
      <td>05/18/2021 12:00:00 AM</td>
      <td>20:15</td>
      <td>ROBBERY</td>
      <td>2</td>
      <td>06/15/2021 12:00:00 AM</td>
      <td>2521 TELEGRAPH AVE\nBerkeley, CA\n(37.864705, ...</td>
      <td>2521 TELEGRAPH AVE</td>
      <td>Berkeley</td>
      <td>CA</td>
      <td>Tuesday</td>
      <td>20</td>
      <td>37.864705</td>
      <td>-122.258463</td>
    </tr>
    <tr>
      <th>908</th>
      <td>21017272</td>
      <td>THEFT MISD. (UNDER $950)</td>
      <td>04/19/2021 12:00:00 AM</td>
      <td>20:20</td>
      <td>LARCENY</td>
      <td>1</td>
      <td>06/15/2021 12:00:00 AM</td>
      <td>2800 BLOCK ADELINE ST\nBerkeley, CA\n(37.85811...</td>
      <td>2800 BLOCK ADELINE ST</td>
      <td>Berkeley</td>
      <td>CA</td>
      <td>Monday</td>
      <td>20</td>
      <td>37.858116</td>
      <td>-122.268002</td>
    </tr>
    <tr>
      <th>597</th>
      <td>21090359</td>
      <td>BURGLARY AUTO</td>
      <td>03/26/2021 12:00:00 AM</td>
      <td>0:00</td>
      <td>BURGLARY - VEHICLE</td>
      <td>5</td>
      <td>06/15/2021 12:00:00 AM</td>
      <td>2100 BLOCK 5TH ST\nBerkeley, CA\n(37.86626, -1...</td>
      <td>2100 BLOCK 5TH ST</td>
      <td>Berkeley</td>
      <td>CA</td>
      <td>Friday</td>
      <td>0</td>
      <td>37.86626</td>
      <td>-122.298335</td>
    </tr>
  </tbody>
</table>
</div>




```python
grader.check("q3b")
```




<p><strong><pre style='display: inline;'>q3b</pre></strong> passed! 🚀</p>



---
## Question 3c: Check for Missing Values

It seems like every record has valid GPS coordinates:


```python
# just run this cell
# fraction of valid lat/lon entries
(~calls[["Lat", "Lon"]].isna()).mean()
```




    Lat    1.0
    Lon    1.0
    dtype: float64



However, a closer examination of the data reveals something else. Here's the first few records of our data again:


```python
calls.head(5)
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
      <th>CASENO</th>
      <th>OFFENSE</th>
      <th>EVENTDT</th>
      <th>EVENTTM</th>
      <th>CVLEGEND</th>
      <th>CVDOW</th>
      <th>InDbDate</th>
      <th>Block_Location</th>
      <th>BLKADDR</th>
      <th>City</th>
      <th>State</th>
      <th>Day</th>
      <th>Hour</th>
      <th>Lat</th>
      <th>Lon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21014296</td>
      <td>THEFT MISD. (UNDER $950)</td>
      <td>04/01/2021 12:00:00 AM</td>
      <td>10:58</td>
      <td>LARCENY</td>
      <td>4</td>
      <td>06/15/2021 12:00:00 AM</td>
      <td>Berkeley, CA\n(37.869058, -122.270455)</td>
      <td>NaN</td>
      <td>Berkeley</td>
      <td>CA</td>
      <td>Thursday</td>
      <td>10</td>
      <td>37.869058</td>
      <td>-122.270455</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21014391</td>
      <td>THEFT MISD. (UNDER $950)</td>
      <td>04/01/2021 12:00:00 AM</td>
      <td>10:38</td>
      <td>LARCENY</td>
      <td>4</td>
      <td>06/15/2021 12:00:00 AM</td>
      <td>Berkeley, CA\n(37.869058, -122.270455)</td>
      <td>NaN</td>
      <td>Berkeley</td>
      <td>CA</td>
      <td>Thursday</td>
      <td>10</td>
      <td>37.869058</td>
      <td>-122.270455</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21090494</td>
      <td>THEFT MISD. (UNDER $950)</td>
      <td>04/19/2021 12:00:00 AM</td>
      <td>12:15</td>
      <td>LARCENY</td>
      <td>1</td>
      <td>06/15/2021 12:00:00 AM</td>
      <td>2100 BLOCK HASTE ST\nBerkeley, CA\n(37.864908,...</td>
      <td>2100 BLOCK HASTE ST</td>
      <td>Berkeley</td>
      <td>CA</td>
      <td>Monday</td>
      <td>12</td>
      <td>37.864908</td>
      <td>-122.267289</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21090204</td>
      <td>THEFT FELONY (OVER $950)</td>
      <td>02/13/2021 12:00:00 AM</td>
      <td>17:00</td>
      <td>LARCENY</td>
      <td>6</td>
      <td>06/15/2021 12:00:00 AM</td>
      <td>2600 BLOCK WARRING ST\nBerkeley, CA\n(37.86393...</td>
      <td>2600 BLOCK WARRING ST</td>
      <td>Berkeley</td>
      <td>CA</td>
      <td>Saturday</td>
      <td>17</td>
      <td>37.863934</td>
      <td>-122.250262</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21090179</td>
      <td>BURGLARY AUTO</td>
      <td>02/08/2021 12:00:00 AM</td>
      <td>6:20</td>
      <td>BURGLARY - VEHICLE</td>
      <td>1</td>
      <td>06/15/2021 12:00:00 AM</td>
      <td>2700 BLOCK GARBER ST\nBerkeley, CA\n(37.86066,...</td>
      <td>2700 BLOCK GARBER ST</td>
      <td>Berkeley</td>
      <td>CA</td>
      <td>Monday</td>
      <td>6</td>
      <td>37.86066</td>
      <td>-122.253407</td>
    </tr>
  </tbody>
</table>
</div>



There is another field that tells us whether we have a valid `Block_Location` entry per record---i.e., with GPS coordinates (latitude, longitude) that match the listed block location. What is it?

In the below cell, use the field you found to create a new dataframe, `missing_lat_lon`, that contains only the rows of `calls` that have invalid latitude and longitude data. Your new dataframe should have all the same columns of `calls`.

<!--
BEGIN QUESTION
name: q3c
-->


```python
missing_lat_lon = calls[calls['Lat'] == '37.869058'] # 理论上应该是lat和lon，但是这里只取了lat，然后注意类型隐式转换！
missing_lat_lon.head()

# print(missing_lat_lon.shape)
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
      <th>CASENO</th>
      <th>OFFENSE</th>
      <th>EVENTDT</th>
      <th>EVENTTM</th>
      <th>CVLEGEND</th>
      <th>CVDOW</th>
      <th>InDbDate</th>
      <th>Block_Location</th>
      <th>BLKADDR</th>
      <th>City</th>
      <th>State</th>
      <th>Day</th>
      <th>Hour</th>
      <th>Lat</th>
      <th>Lon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21014296</td>
      <td>THEFT MISD. (UNDER $950)</td>
      <td>04/01/2021 12:00:00 AM</td>
      <td>10:58</td>
      <td>LARCENY</td>
      <td>4</td>
      <td>06/15/2021 12:00:00 AM</td>
      <td>Berkeley, CA\n(37.869058, -122.270455)</td>
      <td>NaN</td>
      <td>Berkeley</td>
      <td>CA</td>
      <td>Thursday</td>
      <td>10</td>
      <td>37.869058</td>
      <td>-122.270455</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21014391</td>
      <td>THEFT MISD. (UNDER $950)</td>
      <td>04/01/2021 12:00:00 AM</td>
      <td>10:38</td>
      <td>LARCENY</td>
      <td>4</td>
      <td>06/15/2021 12:00:00 AM</td>
      <td>Berkeley, CA\n(37.869058, -122.270455)</td>
      <td>NaN</td>
      <td>Berkeley</td>
      <td>CA</td>
      <td>Thursday</td>
      <td>10</td>
      <td>37.869058</td>
      <td>-122.270455</td>
    </tr>
    <tr>
      <th>215</th>
      <td>21019124</td>
      <td>BURGLARY RESIDENTIAL</td>
      <td>04/30/2021 12:00:00 AM</td>
      <td>10:00</td>
      <td>BURGLARY - RESIDENTIAL</td>
      <td>5</td>
      <td>06/15/2021 12:00:00 AM</td>
      <td>Berkeley, CA\n(37.869058, -122.270455)</td>
      <td>NaN</td>
      <td>Berkeley</td>
      <td>CA</td>
      <td>Friday</td>
      <td>10</td>
      <td>37.869058</td>
      <td>-122.270455</td>
    </tr>
    <tr>
      <th>260</th>
      <td>21000289</td>
      <td>VEHICLE STOLEN</td>
      <td>01/01/2021 12:00:00 AM</td>
      <td>12:00</td>
      <td>MOTOR VEHICLE THEFT</td>
      <td>5</td>
      <td>06/15/2021 12:00:00 AM</td>
      <td>Berkeley, CA\n(37.869058, -122.270455)</td>
      <td>NaN</td>
      <td>Berkeley</td>
      <td>CA</td>
      <td>Friday</td>
      <td>12</td>
      <td>37.869058</td>
      <td>-122.270455</td>
    </tr>
    <tr>
      <th>633</th>
      <td>21013362</td>
      <td>BURGLARY AUTO</td>
      <td>03/27/2021 12:00:00 AM</td>
      <td>4:20</td>
      <td>BURGLARY - VEHICLE</td>
      <td>6</td>
      <td>06/15/2021 12:00:00 AM</td>
      <td>Berkeley, CA\n(37.869058, -122.270455)</td>
      <td>NaN</td>
      <td>Berkeley</td>
      <td>CA</td>
      <td>Saturday</td>
      <td>4</td>
      <td>37.869058</td>
      <td>-122.270455</td>
    </tr>
  </tbody>
</table>
</div>




```python
grader.check("q3c")
```

---

## Question 3d: Check Missing Values

Now let us explore if there is a pattern to which types of records have missing latitude and longitude entries.

We've implemented the plotting code for you below, but read through it and verify you understand what we're doing (we've thrown in a bonus `plt.subplots()` call, documentation [here](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html#stacking-subplots-in-one-direction)).


```python
# just run this cell
missing_by_time = (pd.to_datetime(missing_lat_lon['EVENTDT'])
                   .value_counts()
                   .sort_index()
                  )
missing_by_crime = (missing_lat_lon['CVLEGEND']
                    .value_counts() 
                    / calls['CVLEGEND'].value_counts()
                   ).dropna().sort_values(ascending=False)

fig, ax = plt.subplots(2)
ax[0].bar(missing_by_time.index, missing_by_time)
ax[0].set_ylabel("Calls with Missing Data")
ax[1].barh(missing_by_crime.index, missing_by_crime)
ax[1].set_xlabel("Fraction of Missing Data per Event Type")
fig.suptitle("Characteristics of Missing Lat/Lon Data")
plt.show()
```

    C:\Users\86135\AppData\Local\Temp\ipykernel_19968\1218153592.py:2: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
      missing_by_time = (pd.to_datetime(missing_lat_lon['EVENTDT'])
    


    
![png](lab03_files/lab03_68_1.png)
    


<!--
BEGIN QUESTION
name: q3d
-->

Based on the plots above, are there any patterns among entries that are missing latitude/longitude data? The dataset information [linked](https://data.cityofberkeley.info/Public-Safety/Berkeley-PD-Calls-for-Service/k2nh-s5h5) at the top of this notebook may also give more context.

_Type your answer here, replacing this text._

## Question 3d: Explore

The below cell plots a map of phonecalls by GPS coordinates (latitude, longitude); we drop missing location data.


```python
# just run this cell
import folium
import folium.plugins

SF_COORDINATES = (37.87, -122.28)
sf_map = folium.Map(location=SF_COORDINATES, zoom_start=13)
locs = calls.drop(missing_lat_lon.index)[['Lat', 'Lon']].astype('float').values
heatmap = folium.plugins.HeatMap(locs.tolist(), radius=10)
sf_map.add_child(heatmap)
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe srcdoc="&lt;!DOCTYPE html&gt;
&lt;html&gt;
&lt;head&gt;

    &lt;meta http-equiv=&quot;content-type&quot; content=&quot;text/html; charset=UTF-8&quot; /&gt;

        &lt;script&gt;
            L_NO_TOUCH = false;
            L_DISABLE_3D = false;
        &lt;/script&gt;

    &lt;style&gt;html, body {width: 100%;height: 100%;margin: 0;padding: 0;}&lt;/style&gt;
    &lt;style&gt;#map {position:absolute;top:0;bottom:0;right:0;left:0;}&lt;/style&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://code.jquery.com/jquery-3.7.1.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js&quot;&gt;&lt;/script&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap-glyphicons.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css&quot;/&gt;

            &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width,
                initial-scale=1.0, maximum-scale=1.0, user-scalable=no&quot; /&gt;
            &lt;style&gt;
                #map_88c9cc54858720e4970fd98017e5f89c {
                    position: relative;
                    width: 100.0%;
                    height: 100.0%;
                    left: 0.0%;
                    top: 0.0%;
                }
                .leaflet-container { font-size: 1rem; }
            &lt;/style&gt;

    &lt;script src=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium@main/folium/templates/leaflet_heat.min.js&quot;&gt;&lt;/script&gt;
&lt;/head&gt;
&lt;body&gt;


            &lt;div class=&quot;folium-map&quot; id=&quot;map_88c9cc54858720e4970fd98017e5f89c&quot; &gt;&lt;/div&gt;

&lt;/body&gt;
&lt;script&gt;


            var map_88c9cc54858720e4970fd98017e5f89c = L.map(
                &quot;map_88c9cc54858720e4970fd98017e5f89c&quot;,
                {
                    center: [37.87, -122.28],
                    crs: L.CRS.EPSG3857,
                    zoom: 13,
                    zoomControl: true,
                    preferCanvas: false,
                }
            );





            var tile_layer_5c1b5d102963d518c29b5daffdcb46e7 = L.tileLayer(
                &quot;https://tile.openstreetmap.org/{z}/{x}/{y}.png&quot;,
                {&quot;attribution&quot;: &quot;\u0026copy; \u003ca href=\&quot;https://www.openstreetmap.org/copyright\&quot;\u003eOpenStreetMap\u003c/a\u003e contributors&quot;, &quot;detectRetina&quot;: false, &quot;maxNativeZoom&quot;: 19, &quot;maxZoom&quot;: 19, &quot;minZoom&quot;: 0, &quot;noWrap&quot;: false, &quot;opacity&quot;: 1, &quot;subdomains&quot;: &quot;abc&quot;, &quot;tms&quot;: false}
            );


            tile_layer_5c1b5d102963d518c29b5daffdcb46e7.addTo(map_88c9cc54858720e4970fd98017e5f89c);


            var heat_map_94f090f0653a72aceaedb4f9d4cd9e06 = L.heatLayer(
                [[37.864908, -122.267289], [37.863934, -122.250262], [37.86066, -122.253407], [37.881957, -122.269551], [37.867426, -122.269138], [37.858116, -122.268002], [37.868355, -122.274953], [37.851491, -122.28563], [37.882033, -122.296381], [37.868714, -122.259189], [37.868785, -122.272701], [37.871828, -122.270516], [37.855076, -122.292412], [37.880376, -122.268183], [37.856769, -122.27984], [37.887344, -122.277321], [37.851516, -122.280088], [37.876897, -122.28868], [37.8719, -122.268389], [37.852174, -122.267824], [37.878642, -122.279173], [37.876595, -122.267789], [37.850001, -122.275963], [37.857876, -122.286598], [37.858116, -122.268002], [37.862059, -122.281167], [37.868667, -122.313656], [37.868913, -122.28608], [37.87325, -122.293558], [37.885691, -122.27282], [37.88383, -122.266309], [37.879689, -122.271614], [37.858116, -122.268002], [37.875084, -122.300897], [37.858145, -122.277491], [37.868714, -122.259189], [37.864061, -122.29877], [37.856488, -122.257329], [37.856111, -122.260248], [37.870107, -122.276593], [37.892804, -122.285696], [37.850434, -122.272607], [37.848606, -122.279588], [37.867972, -122.263699], [37.871167, -122.268285], [37.870948, -122.27733], [37.859802, -122.267177], [37.870054, -122.284263], [37.887843, -122.270011], [37.865202, -122.257795], [37.881003, -122.293212], [37.847442, -122.281175], [37.852174, -122.267824], [37.862927, -122.258784], [37.871246, -122.274991], [37.871461, -122.270706], [37.869363, -122.268028], [37.868815, -122.292131], [37.864535, -122.262993], [37.876428, -122.291736], [37.857745, -122.23991], [37.858673, -122.273365], [37.881003, -122.293212], [37.877863, -122.308855], [37.857336, -122.290797], [37.891214, -122.282158], [37.870096, -122.283932], [37.870054, -122.284263], [37.872725, -122.277729], [37.858116, -122.268002], [37.880228, -122.295798], [37.861843, -122.269644], [37.866426, -122.269762], [37.871167, -122.268285], [37.867501, -122.291709], [37.88014, -122.297498], [37.878373, -122.262902], [37.876897, -122.28868], [37.858116, -122.268002], [37.886766, -122.282493], [37.88014, -122.297498], [37.847908, -122.277685], [37.871167, -122.268285], [37.871167, -122.268285], [37.852683, -122.276556], [37.85948, -122.266689], [37.882033, -122.296381], [37.88014, -122.297498], [37.88014, -122.297498], [37.860372, -122.25981], [37.863823, -122.252575], [37.868263, -122.296013], [37.86825, -122.300093], [37.871246, -122.274991], [37.891214, -122.282158], [37.865793, -122.301779], [37.861285, -122.259979], [37.871369, -122.292954], [37.853861, -122.26598], [37.870054, -122.284263], [37.878629, -122.260883], [37.851503, -122.278518], [37.871167, -122.268285], [37.85968, -122.255796], [37.852079, -122.278653], [37.873288, -122.299396], [37.885744, -122.278017], [37.857099, -122.263785], [37.865529, -122.282628], [37.865748, -122.253396], [37.862024, -122.251212], [37.863611, -122.317566], [37.871246, -122.274991], [37.887344, -122.277321], [37.857387, -122.261536], [37.862169, -122.267084], [37.866296, -122.28996], [37.88014, -122.297498], [37.853203, -122.279642], [37.849376, -122.294952], [37.865772, -122.267643], [37.88055, -122.304962], [37.855435, -122.259841], [37.87325, -122.293558], [37.858116, -122.268002], [37.870054, -122.284263], [37.904235, -122.26951], [37.865202, -122.257795], [37.858116, -122.268002], [37.856614, -122.267596], [37.852136, -122.273695], [37.855935, -122.250579], [37.868785, -122.272701], [37.862927, -122.258784], [37.875505, -122.298797], [37.865511, -122.309967], [37.868913, -122.28608], [37.871246, -122.274991], [37.866761, -122.258779], [37.88308, -122.274259], [37.881366, -122.289688], [37.868641, -122.29415], [37.867176, -122.267802], [37.860189, -122.294048], [37.868532, -122.274764], [37.875053, -122.26548], [37.860105, -122.261901], [37.864385, -122.290697], [37.862927, -122.258784], [37.881149, -122.267124], [37.856121, -122.26876], [37.855684, -122.263491], [37.891867, -122.272043], [37.870311, -122.300756], [37.882033, -122.296381], [37.875308, -122.30592], [37.8808, -122.294036], [37.871167, -122.268285], [37.868714, -122.259189], [37.86825, -122.300093], [37.857254, -122.262649], [37.858116, -122.268002], [37.884902, -122.24847], [37.880312, -122.296641], [37.867513, -122.26127], [37.869167, -122.284138], [37.867176, -122.267802], [37.891095, -122.284252], [37.867307, -122.300468], [37.852435, -122.270917], [37.877696, -122.273684], [37.871246, -122.274991], [37.870054, -122.284263], [37.868263, -122.296013], [37.871167, -122.268285], [37.863679, -122.269631], [37.898747, -122.266109], [37.873157, -122.274468], [37.870639, -122.272468], [37.861387, -122.259001], [37.891594, -122.264883], [37.878373, -122.262902], [37.865795, -122.28044], [37.882033, -122.296381], [37.870538, -122.297407], [37.867176, -122.267802], [37.869105, -122.270064], [37.871167, -122.268285], [37.866563, -122.297217], [37.870054, -122.284263], [37.870205, -122.292581], [37.862927, -122.258784], [37.862024, -122.251212], [37.85177, -122.276489], [37.865748, -122.253396], [37.860993, -122.262104], [37.862927, -122.258784], [37.862118, -122.25338], [37.858525, -122.266906], [37.874254, -122.272927], [37.869363, -122.268028], [37.880075, -122.270476], [37.857776, -122.286576], [37.863369, -122.257622], [37.867176, -122.267802], [37.882033, -122.296381], [37.863415, -122.256801], [37.869993, -122.294774], [37.885559, -122.258485], [37.867176, -122.267802], [37.870054, -122.284263], [37.854318, -122.287777], [37.876897, -122.28868], [37.853552, -122.291561], [37.879921, -122.271646], [37.878868, -122.290083], [37.88014, -122.297498], [37.868641, -122.29415], [37.858116, -122.268002], [37.871246, -122.274991], [37.869363, -122.268028], [37.853221, -122.280832], [37.867176, -122.267802], [37.893448, -122.272137], [37.847908, -122.277685], [37.855435, -122.259841], [37.878935, -122.293437], [37.858116, -122.268002], [37.860993, -122.262104], [37.880636, -122.264757], [37.893213, -122.280436], [37.869105, -122.270064], [37.868641, -122.29415], [37.858448, -122.282308], [37.871607, -122.284336], [37.868913, -122.28608], [37.867513, -122.281165], [37.863679, -122.269631], [37.880522, -122.26245], [37.870145, -122.276287], [37.88014, -122.297498], [37.857856, -122.279721], [37.858116, -122.268002], [37.869363, -122.268028], [37.850809, -122.291042], [37.870205, -122.292581], [37.850114, -122.27974], [37.861843, -122.269644], [37.875053, -122.26548], [37.865772, -122.267643], [37.86825, -122.300093], [37.855671, -122.262461], [37.881957, -122.269551], [37.886936, -122.249198], [37.856088, -122.29274], [37.86542, -122.25618], [37.849244, -122.294722], [37.869764, -122.28655], [37.870298, -122.275101], [37.868641, -122.29415], [37.86769, -122.259939], [37.860212, -122.249833], [37.857336, -122.290797], [37.85062, -122.278313], [37.869293, -122.296976], [37.889295, -122.278407], [37.878407, -122.267962], [37.863157, -122.283185], [37.857784, -122.272998], [37.864827, -122.258577], [37.849099, -122.275932], [37.859006, -122.277874], [37.862763, -122.262639], [37.858047, -122.245306], [37.875738, -122.275607], [37.868108, -122.277222], [37.869067, -122.292043], [37.860459, -122.266139], [37.870054, -122.284263], [37.863443, -122.256304], [37.903991, -122.26953], [37.863679, -122.269631], [37.8598, -122.264351], [37.862763, -122.262639], [37.870417, -122.292485], [37.895819, -122.263384], [37.858392, -122.275421], [37.870054, -122.284263], [37.862927, -122.258784], [37.871501, -122.301134], [37.850899, -122.276174], [37.861118, -122.259948], [37.858535, -122.274368], [37.857714, -122.288536], [37.870054, -122.284263], [37.878239, -122.288374], [37.891867, -122.272043], [37.881004, -122.283225], [37.873327, -122.273214], [37.865511, -122.309967], [37.864079, -122.266509], [37.853864, -122.285643], [37.874787, -122.276046], [37.861677, -122.26716], [37.862927, -122.258784], [37.868058, -122.278332], [37.870924, -122.277518], [37.868356, -122.268904], [37.869293, -122.296976], [37.893448, -122.272137], [37.87965, -122.273873], [37.861387, -122.259001], [37.863292, -122.256293], [37.870205, -122.292581], [37.859589, -122.257019], [37.865816, -122.281601], [37.867176, -122.267802], [37.889482, -122.281718], [37.865945, -122.250471], [37.8719, -122.268389], [37.873687, -122.268616], [37.881445, -122.274077], [37.857869, -122.248398], [37.848812, -122.278043], [37.853576, -122.287202], [37.853221, -122.280832], [37.860768, -122.278249], [37.888064, -122.256304], [37.848606, -122.279588], [37.891095, -122.284252], [37.867513, -122.25195], [37.862927, -122.258784], [37.862927, -122.258784], [37.856111, -122.260248], [37.857495, -122.275256], [37.863353, -122.272097], [37.858628, -122.250783], [37.85717, -122.252209], [37.866936, -122.296218], [37.851569, -122.286424], [37.864701, -122.260693], [37.871167, -122.268285], [37.862927, -122.258784], [37.858116, -122.268002], [37.868641, -122.29415], [37.884257, -122.262636], [37.870054, -122.284263], [37.855678, -122.274429], [37.862927, -122.258784], [37.865845, -122.260009], [37.858116, -122.268002], [37.855824, -122.25502], [37.868913, -122.28608], [37.87304, -122.289659], [37.85489, -122.252564], [37.8598, -122.264351], [37.868815, -122.292131], [37.873976, -122.282257], [37.867708, -122.250801], [37.868667, -122.313656], [37.858116, -122.268002], [37.862927, -122.258784], [37.869764, -122.28655], [37.904331, -122.269512], [37.865868, -122.251595], [37.871167, -122.268285], [37.868641, -122.29415], [37.862938, -122.274352], [37.8531, -122.266131], [37.848453, -122.273607], [37.864707, -122.261652], [37.868164, -122.256314], [37.862927, -122.258784], [37.876045, -122.260336], [37.849609, -122.282], [37.894676, -122.285563], [37.867852, -122.258699], [37.887747, -122.264514], [37.86604, -122.2836], [37.872725, -122.277729], [37.879968, -122.296885], [37.868714, -122.259189], [37.88427, -122.276737], [37.88014, -122.297498], [37.868815, -122.292131], [37.872757, -122.291886], [37.870603, -122.270612], [37.868356, -122.268904], [37.881957, -122.269551], [37.899985, -122.265733], [37.846404, -122.275263], [37.876897, -122.28868], [37.869888, -122.300618], [37.871167, -122.268285], [37.867852, -122.258699], [37.858116, -122.268002], [37.858116, -122.268002], [37.870652, -122.27958], [37.865748, -122.253396], [37.871246, -122.274991], [37.876307, -122.268923], [37.879273, -122.255986], [37.860225, -122.269453], [37.854517, -122.281755], [37.883153, -122.292104], [37.864701, -122.260693], [37.859881, -122.285128], [37.863893, -122.251422], [37.876045, -122.260336], [37.880803, -122.274005], [37.862927, -122.258784], [37.880756, -122.303056], [37.850809, -122.291042], [37.868667, -122.313656], [37.869448, -122.281783], [37.869293, -122.296976], [37.870924, -122.277518], [37.858116, -122.268002], [37.887843, -122.270011], [37.856132, -122.271292], [37.853959, -122.284067], [37.872656, -122.292748], [37.880787, -122.277865], [37.858392, -122.275421], [37.849609, -122.282], [37.870054, -122.284263], [37.871246, -122.274991], [37.865511, -122.309967], [37.883948, -122.296991], [37.858525, -122.266906], [37.867176, -122.267802], [37.890977, -122.259745], [37.872499, -122.286632], [37.867176, -122.267802], [37.871698, -122.300095], [37.868913, -122.28608], [37.858116, -122.268002], [37.871265, -122.295043], [37.8719, -122.268389], [37.872175, -122.267835], [37.873976, -122.282257], [37.862927, -122.258784], [37.878405, -122.306072], [37.868785, -122.272701], [37.870205, -122.292581], [37.880245, -122.296973], [37.877863, -122.308855], [37.854534, -122.281798], [37.869764, -122.28655], [37.868532, -122.274764], [37.869113, -122.250903], [37.863611, -122.317566], [37.86939, -122.267883], [37.870287, -122.316238], [37.868667, -122.313656], [37.881957, -122.269551], [37.859259, -122.275787], [37.85968, -122.255796], [37.871167, -122.268285], [37.858392, -122.275421], [37.864036, -122.250272], [37.88014, -122.297498], [37.867822, -122.266003], [37.868815, -122.292131], [37.878571, -122.282954], [37.870054, -122.284263], [37.854534, -122.281798], [37.870054, -122.284263], [37.875113, -122.273413], [37.856848, -122.288122], [37.878407, -122.267962], [37.869688, -122.272805], [37.871167, -122.268285], [37.869105, -122.270064], [37.88014, -122.297498], [37.873687, -122.268616], [37.878571, -122.282954], [37.88014, -122.297498], [37.864827, -122.258577], [37.883072, -122.291168], [37.868815, -122.292131], [37.865443, -122.263189], [37.856111, -122.260248], [37.881788, -122.270678], [37.865748, -122.253396], [37.899602, -122.271102], [37.867513, -122.281165], [37.87325, -122.293558], [37.877309, -122.266712], [37.863839, -122.281391], [37.876307, -122.268923], [37.869067, -122.292043], [37.861129, -122.273879], [37.851017, -122.291088], [37.887298, -122.264457], [37.870287, -122.316238], [37.869363, -122.268028], [37.871167, -122.268285], [37.867843, -122.247802], [37.853552, -122.291561], [37.868164, -122.256314], [37.861573, -122.257452], [37.860732, -122.291369], [37.880228, -122.295798], [37.878239, -122.288374], [37.892524, -122.256323], [37.861677, -122.26716], [37.879273, -122.255986], [37.876428, -122.291736], [37.870948, -122.27733], [37.880228, -122.295798], [37.864827, -122.258577], [37.852304, -122.270045], [37.856088, -122.29274], [37.868641, -122.29415], [37.881957, -122.269551], [37.870054, -122.284263], [37.871167, -122.268285], [37.878056, -122.258553], [37.870639, -122.272468], [37.873807, -122.269211], [37.894661, -122.27503], [37.869067, -122.292043], [37.865772, -122.267643], [37.864701, -122.260693], [37.854833, -122.279839], [37.878997, -122.264607], [37.869688, -122.272805], [37.858116, -122.268002], [37.864238, -122.265263], [37.863811, -122.267412], [37.891095, -122.284247], [37.870054, -122.284263], [37.85583, -122.27339], [37.888053, -122.253497], [37.869688, -122.272805], [37.867176, -122.267802], [37.870549, -122.275186], [37.864238, -122.265263], [37.848152, -122.275807], [37.850899, -122.276174], [37.858116, -122.268002], [37.871167, -122.268285], [37.859589, -122.257019], [37.864827, -122.258577], [37.902692, -122.264479], [37.868352, -122.254459], [37.868641, -122.29415], [37.880312, -122.296641], [37.869363, -122.268028], [37.858116, -122.268002], [37.893148, -122.274809], [37.858116, -122.268002], [37.88014, -122.297498], [37.858147, -122.2848], [37.861843, -122.269644], [37.882899, -122.292918], [37.868815, -122.292131], [37.866513, -122.27474], [37.866025, -122.258623], [37.892524, -122.256323], [37.884902, -122.24847], [37.87795, -122.259386], [37.883948, -122.296991], [37.864827, -122.258577], [37.873687, -122.268616], [37.874929, -122.267601], [37.868785, -122.272701], [37.857452, -122.259548], [37.858116, -122.268002], [37.869688, -122.272805], [37.889989, -122.252226], [37.853554, -122.276925], [37.869332, -122.25019], [37.878407, -122.267962], [37.871167, -122.268285], [37.871486, -122.300027], [37.871246, -122.274991], [37.866293, -122.30551], [37.864701, -122.260693], [37.868714, -122.259189], [37.877528, -122.275956], [37.868714, -122.259189], [37.863353, -122.272097], [37.857787, -122.257013], [37.88788, -122.272336], [37.868595, -122.273835], [37.865149, -122.256487], [37.868706, -122.266279], [37.881164, -122.292378], [37.880227, -122.26936], [37.854286, -122.271015], [37.864238, -122.265263], [37.86626, -122.298335], [37.866074, -122.26331], [37.870145, -122.276287], [37.868815, -122.292131], [37.871167, -122.268285], [37.880787, -122.277865], [37.871246, -122.274991], [37.858116, -122.268002], [37.866936, -122.296218], [37.865772, -122.267643], [37.883103, -122.261459], [37.874251, -122.280203], [37.869084, -122.299245], [37.861843, -122.269644], [37.882033, -122.296381], [37.860732, -122.291369], [37.88014, -122.297498], [37.871246, -122.274991], [37.860768, -122.278249], [37.871246, -122.274991], [37.844763, -122.284231], [37.873017, -122.275481], [37.860225, -122.269453], [37.87325, -122.293558], [37.873687, -122.268616], [37.880477, -122.265919], [37.873085, -122.273187], [37.871167, -122.268285], [37.861081, -122.275984], [37.869688, -122.272805], [37.856111, -122.260248], [37.85489, -122.252564], [37.855435, -122.259841], [37.877047, -122.286183], [37.876921, -122.265567], [37.870205, -122.292581], [37.865511, -122.309967], [37.855935, -122.250579], [37.862763, -122.262639], [37.866929, -122.295042], [37.853143, -122.264085], [37.866969, -122.26553], [37.877247, -122.27708], [37.880312, -122.296641], [37.865349, -122.28377], [37.870054, -122.284263], [37.871167, -122.268285], [37.863823, -122.252575], [37.875053, -122.26548], [37.871167, -122.268285], [37.869105, -122.270064], [37.863934, -122.250262], [37.868706, -122.266279], [37.890706, -122.267186], [37.879689, -122.271614], [37.889482, -122.281718], [37.866568, -122.254084], [37.88014, -122.297498], [37.880266, -122.269032], [37.868263, -122.296013], [37.880228, -122.295798], [37.869888, -122.300618], [37.863353, -122.272097], [37.884743, -122.250296], [37.894233, -122.260552], [37.882033, -122.296381], [37.86771, -122.298466], [37.853959, -122.28016], [37.871167, -122.268285], [37.851263, -122.252477], [37.869839, -122.252365], [37.855998, -122.271278], [37.863072, -122.260352], [37.896776, -122.281277], [37.882825, -122.297836], [37.870652, -122.27958], [37.851653, -122.289194], [37.881191, -122.271769], [37.865202, -122.257795], [37.851503, -122.278518], [37.88055, -122.304962], [37.859309, -122.259291], [37.853552, -122.291561], [37.896703, -122.284274], [37.87199, -122.268396], [37.85306, -122.281362], [37.853864, -122.285643], [37.862927, -122.258784], [37.871265, -122.295043], [37.858116, -122.268002], [37.867212, -122.281739], [37.896956, -122.261613], [37.863292, -122.256293], [37.862927, -122.258784], [37.853554, -122.276925], [37.86771, -122.298466], [37.863385, -122.290219], [37.869105, -122.270064], [37.865149, -122.256487], [37.858448, -122.282308], [37.883555, -122.272036], [37.876307, -122.268923], [37.868352, -122.254459], [37.867708, -122.250801], [37.870639, -122.272468], [37.867822, -122.266003], [37.858116, -122.268002], [37.854534, -122.281798], [37.861078, -122.289632], [37.875391, -122.271141], [37.856719, -122.266672], [37.896395, -122.285494], [37.864701, -122.260693], [37.854985, -122.293982], [37.870054, -122.284263], [37.865149, -122.256487], [37.869105, -122.270064], [37.870054, -122.284263], [37.875922, -122.29441], [37.862817, -122.298359], [37.882033, -122.296381], [37.880312, -122.296641], [37.88014, -122.297498], [37.846404, -122.275263], [37.881141, -122.275177], [37.880228, -122.295798], [37.858116, -122.268002], [37.867947, -122.257926], [37.862927, -122.258784], [37.876339, -122.285012], [37.891332, -122.279975], [37.880248, -122.285626], [37.853576, -122.287202], [37.874489, -122.271072], [37.866739, -122.267299], [37.868759, -122.297933], [37.896431, -122.278418], [37.875391, -122.271141], [37.868699, -122.287718], [37.854536, -122.266403], [37.869084, -122.299245], [37.856755, -122.255248], [37.891214, -122.282158], [37.858116, -122.268002], [37.87325, -122.293558], [37.877596, -122.28657], [37.855969, -122.236484], [37.85062, -122.278313], [37.889482, -122.281718], [37.875189, -122.294176], [37.88014, -122.297498], [37.861571, -122.271722], [37.868815, -122.292131], [37.865772, -122.267643], [37.869363, -122.268028], [37.866426, -122.269762], [37.868714, -122.259189], [37.864827, -122.258577], [37.864226, -122.277937], [37.862927, -122.258784], [37.868263, -122.296013], [37.868667, -122.313656], [37.867852, -122.258699], [37.864826, -122.260719], [37.858116, -122.268002], [37.883948, -122.296991], [37.868785, -122.272701], [37.867551, -122.297541], [37.868263, -122.296013], [37.861078, -122.289632], [37.898168, -122.262253], [37.891332, -122.279975], [37.887344, -122.277321], [37.858116, -122.268002], [37.879968, -122.296885], [37.855408, -122.276149], [37.880859, -122.287774], [37.870054, -122.284263], [37.862635, -122.293663], [37.865793, -122.301779], [37.868785, -122.272701], [37.857099, -122.263785], [37.870396, -122.281585], [37.86542, -122.25618], [37.873687, -122.268616], [37.871167, -122.268285], [37.863679, -122.269631], [37.863934, -122.250262], [37.858759, -122.264112], [37.86771, -122.298466], [37.853111, -122.266049], [37.86476, -122.297852], [37.885032, -122.274324], [37.877663, -122.274831], [37.852618, -122.283677], [37.871246, -122.274991], [37.870287, -122.316238], [37.858448, -122.282308], [37.878407, -122.267962], [37.867501, -122.291709], [37.864061, -122.29877], [37.880803, -122.274005], [37.870287, -122.316238], [37.848152, -122.275807], [37.894468, -122.265464], [37.866563, -122.297217], [37.860105, -122.261901], [37.857899, -122.254371], [37.879968, -122.296885], [37.858116, -122.268002], [37.888072, -122.272344], [37.872499, -122.286632], [37.871167, -122.268285], [37.870054, -122.284263], [37.854833, -122.279839], [37.863611, -122.317566], [37.861387, -122.259001], [37.869888, -122.300618], [37.877482, -122.256109], [37.848774, -122.271171], [37.880312, -122.296641], [37.877678, -122.281631], [37.881164, -122.292378], [37.866629, -122.300423], [37.867708, -122.250801], [37.850387, -122.276278], [37.859906, -122.284931], [37.856488, -122.257329], [37.868815, -122.292131], [37.846443, -122.274971], [37.89977, -122.275237], [37.868164, -122.256314], [37.854215, -122.269167], [37.880376, -122.268183], [37.85306, -122.281362], [37.88014, -122.297498], [37.869067, -122.292043], [37.878868, -122.290083], [37.867176, -122.267802], [37.860225, -122.269453], [37.871167, -122.268285], [37.848774, -122.271171], [37.864705, -122.258463], [37.870948, -122.27733], [37.854186, -122.279157], [37.877835, -122.260354], [37.866174, -122.26454], [37.888199, -122.250826], [37.888842, -122.254155], [37.863679, -122.269631], [37.893448, -122.272137], [37.871369, -122.292954], [37.868913, -122.28608], [37.850385, -122.270897], [37.86466, -122.257744], [37.880266, -122.269032], [37.858116, -122.268002], [37.862927, -122.258784], [37.880228, -122.295798], [37.854833, -122.279839], [37.86825, -122.300093], [37.860225, -122.269453], [37.867852, -122.258699], [37.887344, -122.277321], [37.858116, -122.268002], [37.867176, -122.267802], [37.869688, -122.272805], [37.856088, -122.29274], [37.899349, -122.260994], [37.88014, -122.297498], [37.880312, -122.296641], [37.868641, -122.29415], [37.87662, -122.29269], [37.871167, -122.268285], [37.867176, -122.267802], [37.867551, -122.297541], [37.867947, -122.257926], [37.861409, -122.253481], [37.889399, -122.262594], [37.858116, -122.268002], [37.890928, -122.287251], [37.858116, -122.268002], [37.868641, -122.29415], [37.857776, -122.286576], [37.861118, -122.259948], [37.855042, -122.284746], [37.870948, -122.27733], [37.876595, -122.267789], [37.858116, -122.268002], [37.862927, -122.258784], [37.867513, -122.25195], [37.870054, -122.284263], [37.86939, -122.267883], [37.869688, -122.272805], [37.867176, -122.267802], [37.899602, -122.271102], [37.869764, -122.28655], [37.861283, -122.273911], [37.870924, -122.277518], [37.881957, -122.269551], [37.891332, -122.279975], [37.862024, -122.251212], [37.863611, -122.317566], [37.854536, -122.266403], [37.868263, -122.296013], [37.858116, -122.268002], [37.876921, -122.265567], [37.866924, -122.283899], [37.858116, -122.268002], [37.879058, -122.282584], [37.850001, -122.275963], [37.869332, -122.25019], [37.861843, -122.269644], [37.858116, -122.268002], [37.862927, -122.258784], [37.872175, -122.267835], [37.874781, -122.268758], [37.875308, -122.30592], [37.88014, -122.297498], [37.898162, -122.286279], [37.870054, -122.284263], [37.862927, -122.258784], [37.858392, -122.275421], [37.870948, -122.27733], [37.856968, -122.279544], [37.855435, -122.259841], [37.881004, -122.283225], [37.855969, -122.236484], [37.870924, -122.277518], [37.852211, -122.286336], [37.881957, -122.269551], [37.869113, -122.250903], [37.869113, -122.250903], [37.871167, -122.268285], [37.88014, -122.297498], [37.865511, -122.309967], [37.862024, -122.251212], [37.856111, -122.260248], [37.866761, -122.258779], [37.881788, -122.270678], [37.870205, -122.292581], [37.859216, -122.26855], [37.871246, -122.274991], [37.869764, -122.28655], [37.870054, -122.284263], [37.862059, -122.274073], [37.865511, -122.309967], [37.867176, -122.267802], [37.867513, -122.25195], [37.869363, -122.268028], [37.867643, -122.280643], [37.852304, -122.270045], [37.860766, -122.255895], [37.864827, -122.258577], [37.904235, -122.26951], [37.850175, -122.293099], [37.851653, -122.289194], [37.871167, -122.268285], [37.854215, -122.269167], [37.892137, -122.269182], [37.855998, -122.271278], [37.871828, -122.270516], [37.85525, -122.277126], [37.891594, -122.264883], [37.869105, -122.270064], [37.859184, -122.289022], [37.859665, -122.27169], [37.881957, -122.269551], [37.858116, -122.268002], [37.865059, -122.272291], [37.877247, -122.27708], [37.871265, -122.295043], [37.877951, -122.271395], [37.858116, -122.268002], [37.865202, -122.257795], [37.859665, -122.27169], [37.867501, -122.291709], [37.873739, -122.275557], [37.858116, -122.268002], [37.866761, -122.258779], [37.862763, -122.262639], [37.88055, -122.304962], [37.866568, -122.254084], [37.863679, -122.269631], [37.859195, -122.261915], [37.85108, -122.293322], [37.855293, -122.266502], [37.864705, -122.258463], [37.882706, -122.263356], [37.882033, -122.296381], [37.857495, -122.275256], [37.861571, -122.271722], [37.870086, -122.265901], [37.866969, -122.26553], [37.867176, -122.267802], [37.85525, -122.277126], [37.858116, -122.268002], [37.893448, -122.272137], [37.849609, -122.282], [37.857452, -122.259548], [37.86939, -122.267883], [37.868641, -122.29415], [37.878837, -122.285555], [37.865034, -122.297939], [37.868785, -122.272701], [37.868641, -122.29415], [37.869105, -122.270064], [37.882033, -122.296381], [37.867176, -122.267802], [37.876307, -122.268923], [37.882033, -122.296381], [37.847908, -122.277685], [37.869764, -122.28655], [37.855408, -122.276149], [37.867176, -122.267802], [37.851203, -122.289129], [37.875391, -122.271141], [37.856111, -122.260248], [37.879677, -122.299779], [37.865793, -122.301779], [37.869688, -122.272805], [37.873607, -122.270903], [37.869385, -122.28953], [37.867176, -122.267802], [37.874581, -122.277744], [37.871369, -122.292954], [37.883948, -122.296991], [37.882825, -122.297836], [37.878407, -122.267962], [37.858518, -122.252653], [37.869888, -122.300618], [37.862927, -122.258784], [37.897333, -122.276042], [37.855684, -122.263491], [37.858214, -122.269252], [37.869688, -122.272805], [37.868108, -122.277222], [37.869289, -122.28081], [37.847888, -122.275752], [37.863611, -122.317566], [37.867176, -122.267802], [37.854186, -122.279157], [37.848812, -122.278043], [37.863072, -122.260352], [37.883948, -122.296991], [37.868532, -122.274764], [37.873017, -122.275481], [37.882033, -122.296381], [37.865511, -122.309967], [37.869764, -122.28655], [37.877625, -122.294534], [37.873327, -122.273214], [37.868714, -122.259189], [37.863679, -122.269631], [37.876921, -122.265567], [37.881957, -122.269551], [37.867176, -122.267802], [37.871246, -122.274991], [37.880734, -122.265359], [37.858392, -122.275421], [37.88014, -122.297498], [37.882482, -122.261872], [37.864079, -122.266509], [37.876428, -122.291736], [37.851477, -122.278651], [37.88014, -122.297498], [37.869688, -122.272805], [37.868931, -122.281731], [37.867176, -122.267802], [37.862927, -122.258784], [37.86331, -122.316113], [37.8559, -122.283101], [37.867176, -122.267802], [37.849786, -122.269782], [37.865149, -122.256487], [37.86771, -122.298466], [37.891214, -122.282158], [37.870256, -122.298419], [37.885683, -122.308114], [37.874251, -122.280203], [37.857792, -122.258537], [37.853386, -122.263973], [37.8559, -122.283101], [37.862927, -122.258784], [37.856111, -122.260248], [37.86277, -122.297218], [37.892804, -122.285696], [37.868913, -122.28608], [37.855435, -122.259841], [37.880652, -122.283181], [37.858116, -122.268002], [37.885032, -122.274324], [37.865529, -122.282628], [37.857694, -122.281628], [37.897393, -122.28115], [37.866037, -122.265618], [37.87935, -122.276123], [37.862927, -122.258784], [37.867176, -122.267802], [37.850675, -122.286265], [37.849609, -122.282], [37.849376, -122.294952], [37.868785, -122.272701], [37.858392, -122.275421], [37.863839, -122.281391], [37.861129, -122.273879], [37.86466, -122.267381], [37.866037, -122.265618], [37.870924, -122.277518], [37.865511, -122.309967], [37.862927, -122.258784], [37.881164, -122.292378], [37.868667, -122.313656], [37.853929, -122.289679], [37.86825, -122.300093], [37.876045, -122.260336], [37.896776, -122.281277], [37.866037, -122.265618], [37.865134, -122.258331], [37.899278, -122.283222], [37.862927, -122.258784], [37.858116, -122.268002], [37.855815, -122.280413], [37.88014, -122.297498], [37.865134, -122.258331], [37.867501, -122.291709], [37.865511, -122.309967], [37.850681, -122.270619], [37.859309, -122.259291], [37.85717, -122.252209], [37.868815, -122.292131], [37.864723, -122.263034], [37.867852, -122.258699], [37.853683, -122.261744], [37.864925, -122.273436], [37.902692, -122.264479], [37.870205, -122.292581], [37.865984, -122.293289], [37.871828, -122.270516], [37.875922, -122.29441], [37.877417, -122.300713], [37.860225, -122.269453], [37.869167, -122.284138], [37.866969, -122.26553], [37.867176, -122.267802], [37.860225, -122.269453], [37.866074, -122.26331], [37.867935, -122.258021], [37.856968, -122.279544], [37.853576, -122.287202], [37.864908, -122.267289], [37.871167, -122.268285], [37.88014, -122.297498], [37.869423, -122.281951], [37.875166, -122.280125], [37.866739, -122.267299], [37.868714, -122.259189], [37.855632, -122.256606], [37.855684, -122.263491], [37.882006, -122.283736], [37.865443, -122.263189], [37.866074, -122.26331], [37.859006, -122.277874], [37.862516, -122.27853], [37.865868, -122.251595], [37.881957, -122.269551], [37.880636, -122.264757], [37.870219, -122.273994], [37.880228, -122.295798], [37.862334, -122.264828], [37.868356, -122.268904], [37.868164, -122.256314], [37.861387, -122.259001], [37.858116, -122.268002], [37.859804, -122.251257], [37.865772, -122.267643], [37.866739, -122.267299], [37.886101, -122.273116], [37.878113, -122.269114], [37.865984, -122.293289], [37.871167, -122.268285], [37.867708, -122.250801], [37.858116, -122.268002], [37.89977, -122.275237], [37.889399, -122.262594], [37.854215, -122.269167], [37.853275, -122.279078], [37.88014, -122.297498], [37.850385, -122.270897], [37.859674, -122.286861], [37.860993, -122.262104], [37.852211, -122.286336], [37.857452, -122.259548], [37.868667, -122.313656], [37.881957, -122.269551], [37.871167, -122.268285], [37.851107, -122.28324], [37.866568, -122.254084], [37.856291, -122.234692], [37.859066, -122.284541], [37.860225, -122.269453], [37.871369, -122.292954], [37.862652, -122.293668], [37.871132, -122.276743], [37.879677, -122.299779], [37.861689, -122.255732], [37.849376, -122.294952], [37.858116, -122.268002], [37.853221, -122.280832], [37.88014, -122.297498], [37.867176, -122.267802], [37.865034, -122.297939], [37.858628, -122.250783], [37.855935, -122.250579], [37.862817, -122.298359], [37.871369, -122.292954], [37.867176, -122.267802], [37.871167, -122.268285], [37.870603, -122.270612], [37.867501, -122.291709], [37.865748, -122.253396], [37.857856, -122.279721], [37.863679, -122.269631], [37.881957, -122.269551], [37.88014, -122.297498], [37.902033, -122.265247], [37.868356, -122.268904], [37.858116, -122.268002], [37.863369, -122.257622], [37.856111, -122.260248], [37.880227, -122.26936], [37.857452, -122.259548], [37.8559, -122.283101], [37.866568, -122.254084], [37.855435, -122.259841], [37.855389, -122.264552], [37.859259, -122.275787], [37.870639, -122.272468], [37.852956, -122.266195], [37.864908, -122.267289], [37.850385, -122.270897], [37.880228, -122.295798], [37.858116, -122.268002], [37.859826, -122.270996], [37.85525, -122.277126], [37.869105, -122.270064], [37.857792, -122.258537], [37.852435, -122.270917], [37.899668, -122.27375], [37.870054, -122.284263], [37.864701, -122.260693], [37.850541, -122.286037], [37.869363, -122.268028], [37.878722, -122.295312], [37.850899, -122.276174], [37.869764, -122.28655], [37.868815, -122.292131], [37.854186, -122.279157], [37.850681, -122.270619], [37.861571, -122.271722], [37.867366, -122.296346], [37.861078, -122.289632], [37.885541, -122.274771], [37.863811, -122.267412], [37.862927, -122.258784], [37.863611, -122.317566], [37.879773, -122.30623], [37.867501, -122.291709], [37.869688, -122.272805], [37.864583, -122.275705], [37.854272, -122.268418], [37.854612, -122.281178], [37.864238, -122.265263], [37.862562, -122.262598], [37.880551, -122.304085], [37.864385, -122.290697], [37.866145, -122.300016], [37.877047, -122.286183], [37.882033, -122.296381], [37.860993, -122.262104], [37.858116, -122.268002], [37.854517, -122.281755], [37.871167, -122.268285], [37.870205, -122.292581], [37.851477, -122.278651], [37.851263, -122.252477], [37.879968, -122.296885], [37.857713, -122.250709], [37.851503, -122.278518], [37.85968, -122.255796], [37.891332, -122.279975], [37.854102, -122.251059], [37.851017, -122.291088], [37.862491, -122.2646], [37.869688, -122.272805], [37.862493, -122.24708], [37.858116, -122.268002], [37.866508, -122.261057], [37.867501, -122.291709], [37.891332, -122.279975], [37.885541, -122.274771], [37.864827, -122.258577], [37.867513, -122.25195], [37.860189, -122.294048], [37.880227, -122.26936], [37.88788, -122.272336], [37.871167, -122.268285], [37.868352, -122.254459], [37.871246, -122.274991], [37.88014, -122.297498], [37.851477, -122.278651], [37.86097, -122.262273], [37.871246, -122.274991], [37.855671, -122.262461], [37.870243, -122.275149], [37.854104, -122.271109], [37.852174, -122.267824], [37.885691, -122.27282], [37.858628, -122.250783], [37.858116, -122.268002], [37.853683, -122.261744], [37.852304, -122.270045], [37.877247, -122.27708], [37.852491, -122.243412], [37.854091, -122.256132], [37.866037, -122.265618], [37.889482, -122.281718], [37.871167, -122.268285], [37.861387, -122.259001], [37.897393, -122.28115], [37.871461, -122.270706], [37.862516, -122.27853], [37.856769, -122.27984], [37.871167, -122.268285], [37.872907, -122.266975], [37.868667, -122.313656], [37.852871, -122.26799], [37.8559, -122.283101], [37.863934, -122.250262], [37.86007, -122.28931], [37.849786, -122.269782], [37.855998, -122.271278], [37.85525, -122.277126], [37.865845, -122.260009], [37.875391, -122.271141], [37.885413, -122.268753], [37.876921, -122.265567], [37.867852, -122.258699], [37.868667, -122.313656], [37.854247, -122.24375], [37.862024, -122.251212], [37.878973, -122.279366], [37.863611, -122.317566], [37.881957, -122.269551], [37.880228, -122.295798], [37.880228, -122.295798], [37.848292, -122.271773], [37.870054, -122.284263], [37.88113, -122.276389], [37.855026, -122.266475], [37.863611, -122.317566], [37.853576, -122.287202], [37.864226, -122.277937], [37.865443, -122.263189], [37.862901, -122.261616], [37.885559, -122.258485], [37.867176, -122.267802], [37.889989, -122.252226], [37.865748, -122.253396], [37.866881, -122.299689], [37.866265, -122.278162], [37.871167, -122.268285], [37.860334, -122.280567], [37.849027, -122.26926], [37.868355, -122.274953], [37.854154, -122.272171], [37.855408, -122.276149], [37.853221, -122.280832], [37.866563, -122.297217], [37.865149, -122.256487], [37.861409, -122.253481], [37.867176, -122.267802], [37.876897, -122.28868], [37.868574, -122.270415], [37.891095, -122.284252], [37.866936, -122.296218], [37.864826, -122.260719], [37.875738, -122.275607], [37.85177, -122.276489], [37.858116, -122.268002], [37.846404, -122.275263], [37.870603, -122.270612], [37.881366, -122.289688], [37.880228, -122.295798], [37.8719, -122.268389], [37.854891, -122.280148], [37.851176, -122.253096], [37.879451, -122.300901], [37.894468, -122.265464], [37.851919, -122.286667], [37.858116, -122.268002], [37.859195, -122.261915], [37.864827, -122.258577], [37.858116, -122.268002], [37.869105, -122.270064], [37.88014, -122.297498], [37.869764, -122.28655], [37.86249, -122.29124], [37.874251, -122.280203], [37.867513, -122.281165], [37.852066, -122.272994], [37.857387, -122.261536], [37.872753, -122.291926], [37.856968, -122.279544], [37.869888, -122.300618], [37.858116, -122.268002], [37.861078, -122.289632], [37.869067, -122.292043], [37.87325, -122.293558], [37.858116, -122.268002], [37.853576, -122.287202], [37.876897, -122.28868], [37.882033, -122.296381], [37.865202, -122.257795], [37.869385, -122.28953], [37.863292, -122.256293], [37.871461, -122.270706], [37.867176, -122.267802], [37.870054, -122.284263], [37.850675, -122.286265], [37.880312, -122.296641], [37.862927, -122.258784], [37.85062, -122.278313], [37.862927, -122.258784], [37.880312, -122.296641], [37.872499, -122.286632], [37.857452, -122.280961], [37.856848, -122.288122], [37.869363, -122.268028], [37.854442, -122.277107], [37.868714, -122.259189], [37.871246, -122.274991], [37.858116, -122.268002], [37.862927, -122.258784], [37.862927, -122.258784], [37.869688, -122.272805], [37.855824, -122.25502], [37.868108, -122.277222], [37.871167, -122.268285], [37.883512, -122.27103], [37.880037, -122.303902], [37.866293, -122.30551], [37.874251, -122.280203], [37.858527, -122.293006], [37.863811, -122.267412], [37.878839, -122.259307], [37.885559, -122.258485], [37.853221, -122.280832], [37.864701, -122.260693], [37.877863, -122.308855], [37.876297, -122.27124], [37.853552, -122.268286], [37.847281, -122.27545], [37.877951, -122.271395], [37.86825, -122.300093], [37.868263, -122.296013], [37.880636, -122.264757], [37.880027, -122.279976], [37.878935, -122.293437], [37.867176, -122.267802], [37.855969, -122.236484], [37.88014, -122.297498], [37.860105, -122.261901], [37.851107, -122.28324], [37.872158, -122.282066], [37.883072, -122.291168], [37.870287, -122.316238], [37.852211, -122.286336], [37.871167, -122.268285], [37.871167, -122.268285], [37.871167, -122.268285], [37.868263, -122.296013], [37.871167, -122.268285], [37.867176, -122.267802], [37.870603, -122.270612], [37.861118, -122.259948], [37.866761, -122.258779], [37.858116, -122.268002], [37.870639, -122.272468], [37.865793, -122.301779], [37.891332, -122.279975], [37.855435, -122.259841], [37.868108, -122.277222], [37.861107, -122.289485], [37.85489, -122.252564], [37.867852, -122.258699], [37.849512, -122.272671], [37.857336, -122.290797], [37.852529, -122.247255], [37.904235, -122.26951], [37.866074, -122.26331], [37.869888, -122.300618], [37.865149, -122.256487], [37.869067, -122.292043], [37.877482, -122.256109], [37.862927, -122.258784], [37.857714, -122.288536], [37.878722, -122.295312], [37.881957, -122.269551], [37.864827, -122.258577], [37.869839, -122.252365], [37.88674, -122.262299], [37.882353, -122.267136], [37.894676, -122.285563], [37.86549, -122.295702], [37.852529, -122.247255], [37.87325, -122.293558], [37.858116, -122.268002], [37.862264, -122.250972], [37.88548, -122.2713], [37.880228, -122.295798], [37.862927, -122.258784], [37.864238, -122.265263], [37.88014, -122.297498], [37.858392, -122.275421], [37.8559, -122.287818], [37.863811, -122.267412], [37.857787, -122.257013], [37.853959, -122.284067], [37.869246, -122.244474], [37.898747, -122.266109], [37.853275, -122.279078], [37.8719, -122.268389], [37.867176, -122.267802], [37.882033, -122.296381], [37.869888, -122.300618], [37.882469, -122.294567], [37.873687, -122.268616], [37.858116, -122.268002], [37.871167, -122.268285], [37.864827, -122.258577], [37.878239, -122.288374], [37.885683, -122.308114], [37.856195, -122.288053], [37.864827, -122.258577], [37.858116, -122.268002], [37.88055, -122.304962], [37.883948, -122.296991], [37.869067, -122.292043], [37.87325, -122.293558], [37.873687, -122.268616], [37.86531, -122.25699], [37.868164, -122.256314], [37.868714, -122.259189], [37.856035, -122.240397], [37.858448, -122.282308], [37.848812, -122.278043], [37.858777, -122.29006], [37.88222, -122.247955], [37.867185, -122.267803], [37.871167, -122.268285], [37.871167, -122.268285], [37.862927, -122.258784], [37.871544, -122.272714], [37.865772, -122.267643], [37.881957, -122.269551], [37.871544, -122.272714], [37.875084, -122.300897], [37.857295, -122.243843], [37.87965, -122.273873], [37.867176, -122.267802], [37.859682, -122.267008], [37.872563, -122.284648], [37.855748, -122.290281], [37.858116, -122.268002], [37.862491, -122.2646], [37.865141, -122.265441], [37.862927, -122.258784], [37.855065, -122.249937], [37.872175, -122.267835], [37.880228, -122.295798], [37.861387, -122.259001], [37.88265, -122.279975], [37.871167, -122.268285], [37.858116, -122.268002], [37.862516, -122.27853], [37.863611, -122.317566], [37.865772, -122.267643], [37.866206, -122.29129], [37.868714, -122.259189], [37.880383, -122.285574], [37.894636, -122.284305], [37.861571, -122.271722], [37.880228, -122.295798], [37.868164, -122.256314], [37.871265, -122.295043], [37.88014, -122.297498], [37.853552, -122.291561], [37.850798, -122.294689], [37.856111, -122.260248], [37.8598, -122.264351], [37.868334, -122.303753], [37.879921, -122.271646], [37.865511, -122.309967], [37.896703, -122.284274], [37.855076, -122.292412], [37.8871, -122.251321], [37.85717, -122.252209], [37.88014, -122.297498], [37.863099, -122.27207], [37.863593, -122.276751], [37.869385, -122.28953], [37.880228, -122.295798], [37.875189, -122.294176], [37.867176, -122.267802], [37.871716, -122.252796], [37.871167, -122.268285], [37.862927, -122.258784], [37.855576, -122.282153], [37.868714, -122.259189], [37.862927, -122.258784], [37.856698, -122.290343], [37.870219, -122.273994], [37.854833, -122.279839], [37.88189, -122.29825], [37.871167, -122.268285], [37.869306, -122.268487], [37.880756, -122.303056], [37.871544, -122.272714], [37.851921, -122.269813], [37.861387, -122.259001], [37.868815, -122.292131], [37.861627, -122.285466], [37.896395, -122.285494], [37.896431, -122.278418], [37.867852, -122.258699], [37.880667, -122.29489], [37.862552, -122.289951], [37.880636, -122.264757], [37.865772, -122.267643], [37.86156, -122.285969], [37.868161, -122.291751], [37.864827, -122.258577], [37.850444, -122.288866], [37.871167, -122.268285], [37.866074, -122.26331], [37.847281, -122.27545], [37.868192, -122.248515], [37.852683, -122.276556], [37.859906, -122.284931], [37.865748, -122.253396], [37.887344, -122.277321], [37.891332, -122.279975], [37.874581, -122.277744], [37.860507, -122.280446], [37.854186, -122.279157], [37.875391, -122.271141], [37.88222, -122.247955], [37.868161, -122.291751], [37.881957, -122.269551], [37.858448, -122.282308], [37.863611, -122.317566], [37.868334, -122.303753], [37.867176, -122.267802], [37.863611, -122.317566], [37.855065, -122.249937], [37.865795, -122.28044], [37.870205, -122.292581], [37.869385, -122.28953], [37.871167, -122.268285], [37.88014, -122.297498], [37.853864, -122.285643], [37.869067, -122.292043], [37.856853, -122.26482], [37.880227, -122.26936], [37.856132, -122.271292], [37.858116, -122.268002], [37.878407, -122.267962], [37.894256, -122.281603], [37.866969, -122.26553], [37.865868, -122.251595], [37.869764, -122.28655], [37.851919, -122.286667], [37.892875, -122.268748], [37.870054, -122.284263], [37.855832, -122.272369], [37.858116, -122.268002], [37.869105, -122.270064], [37.865868, -122.251595], [37.858427, -122.282485], [37.871167, -122.268285], [37.870298, -122.275101], [37.888679, -122.280432], [37.867212, -122.281739], [37.84905, -122.269098], [37.855798, -122.252707], [37.859665, -122.27169], [37.858116, -122.268002], [37.860372, -122.25981], [37.868667, -122.313656], [37.881366, -122.289688], [37.871369, -122.292954], [37.85175, -122.29379], [37.88674, -122.262299], [37.870948, -122.27733], [37.860372, -122.25981], [37.851017, -122.291088], [37.85948, -122.266689], [37.855798, -122.252707], [37.872656, -122.292748], [37.878864, -122.265718], [37.84905, -122.269098], [37.869688, -122.272805], [37.878405, -122.306072], [37.870054, -122.284263], [37.871167, -122.268285], [37.891755, -122.269881], [37.856769, -122.27984], [37.903781, -122.273576], [37.882033, -122.296381], [37.871167, -122.268285], [37.867212, -122.281739], [37.852348, -122.271972], [37.879968, -122.296885], [37.871167, -122.268285], [37.871246, -122.274991], [37.859881, -122.285128], [37.857784, -122.272998], [37.85488, -122.269079], [37.869688, -122.272805], [37.85717, -122.252209], [37.858888, -122.264137], [37.858116, -122.268002], [37.848774, -122.271171], [37.854612, -122.281178], [37.872499, -122.286632], [37.882016, -122.281271], [37.858147, -122.2848], [37.868641, -122.29415], [37.871167, -122.268285], [37.880376, -122.268183], [37.867176, -122.267802], [37.871917, -122.298961], [37.866936, -122.296218], [37.866025, -122.258623], [37.863611, -122.317566], [37.865772, -122.267643], [37.857336, -122.290797], [37.889658, -122.270137], [37.883798, -122.268567], [37.887246, -122.278286], [37.864925, -122.273436], [37.878986, -122.277231], [37.852315, -122.272231], [37.891214, -122.282158], [37.858392, -122.275421], [37.880592, -122.268207], [37.88014, -122.297498], [37.869363, -122.268028], [37.867176, -122.267802], [37.889482, -122.281718], [37.855969, -122.236484], [37.871167, -122.268285], [37.869764, -122.28655], [37.857856, -122.279721], [37.882482, -122.261872], [37.867708, -122.250801], [37.850629, -122.28065], [37.880334, -122.27627], [37.866074, -122.26331], [37.857452, -122.259548], [37.859413, -122.288733], [37.868206, -122.277232], [37.8611, -122.297826], [37.870867, -122.270635], [37.863823, -122.252575], [37.897333, -122.276042], [37.849431, -122.278174], [37.869186, -122.283943], [37.866293, -122.30551], [37.887843, -122.270011], [37.861573, -122.257452], [37.867501, -122.291709], [37.851503, -122.278518], [37.875053, -122.26548], [37.858116, -122.268002], [37.854857, -122.262314], [37.856111, -122.260248], [37.876615, -122.291393], [37.888768, -122.279287], [37.871167, -122.268285], [37.888702, -122.26371], [37.858448, -122.282308], [37.865141, -122.265441], [37.88014, -122.297498], [37.868913, -122.28608], [37.871161, -122.298716], [37.850798, -122.294689], [37.878051, -122.285222], [37.865793, -122.301779], [37.862393, -122.248851], [37.866563, -122.297217], [37.858392, -122.275421], [37.870396, -122.281585], [37.850629, -122.28065], [37.864827, -122.258577], [37.852934, -122.294575], [37.869741, -122.279479], [37.869104, -122.256017], [37.867513, -122.25195], [37.869067, -122.292043], [37.871828, -122.270516], [37.866761, -122.258779], [37.870205, -122.292581], [37.862024, -122.251212], [37.881957, -122.269551], [37.865772, -122.267643], [37.88014, -122.297498], [37.868714, -122.259189], [37.865134, -122.258331], [37.858116, -122.268002], [37.869764, -122.28655], [37.899959, -122.282042], [37.864827, -122.258577], [37.858116, -122.268002], [37.897731, -122.271759], [37.865149, -122.256487], [37.862927, -122.258784], [37.896688, -122.27456], [37.891332, -122.279975], [37.858116, -122.268002], [37.892598, -122.280094], [37.865511, -122.309967], [37.863072, -122.260352], [37.867176, -122.267802], [37.871461, -122.270706], [37.868714, -122.259189], [37.870205, -122.292581], [37.851919, -122.286667], [37.869764, -122.28655], [37.897333, -122.276042], [37.894233, -122.260552], [37.871167, -122.268285], [37.864385, -122.290697], [37.881227, -122.29097], [37.869113, -122.250903], [37.869363, -122.268028], [37.885653, -122.269953], [37.875113, -122.273413], [37.871167, -122.268285], [37.870054, -122.284263], [37.869764, -122.28655], [37.866761, -122.258779], [37.875113, -122.273413], [37.853203, -122.279642], [37.856111, -122.260248], [37.878904, -122.258814], [37.852871, -122.26799], [37.857452, -122.259548], [37.868714, -122.259189], [37.870185, -122.299609], [37.888072, -122.272344], [37.862927, -122.258784], [37.863839, -122.281391], [37.856111, -122.260248], [37.891594, -122.264883], [37.853552, -122.291561], [37.860459, -122.266139], [37.865324, -122.256883], [37.855998, -122.271278], [37.85177, -122.276489], [37.870054, -122.284263], [37.881957, -122.269551], [37.872158, -122.282066], [37.848812, -122.278043], [37.868714, -122.259189], [37.865748, -122.253396], [37.86549, -122.295702], [37.878113, -122.269114], [37.867176, -122.267802], [37.865748, -122.253396], [37.87783, -122.298557], [37.873976, -122.282257], [37.871167, -122.268285], [37.861677, -122.26716], [37.876339, -122.285012], [37.862024, -122.251212], [37.867373, -122.249515], [37.88014, -122.297498], [37.879188, -122.276107], [37.855389, -122.264552], [37.852422, -122.278571], [37.891755, -122.269881], [37.869993, -122.294774], [37.858116, -122.268002], [37.88113, -122.276389], [37.880756, -122.303056], [37.879058, -122.282584], [37.891383, -122.257303], [37.866293, -122.30551], [37.85062, -122.278313], [37.860225, -122.269453], [37.878571, -122.282954], [37.858116, -122.268002], [37.870054, -122.284263], [37.88014, -122.297498], [37.855293, -122.266502], [37.85976, -122.257034], [37.858116, -122.268002], [37.880228, -122.295798], [37.857452, -122.259548], [37.875826, -122.273495], [37.868815, -122.292131], [37.860225, -122.269453], [37.857714, -122.288536], [37.869962, -122.277411], [37.853576, -122.287202], [37.883608, -122.285672], [37.862927, -122.258784], [37.868995, -122.279598], [37.877247, -122.27708], [37.875963, -122.296521], [37.850385, -122.270897], [37.871167, -122.268285], [37.868815, -122.292131], [37.854833, -122.279839], [37.889482, -122.281718], [37.891214, -122.282158], [37.898189, -122.286977], [37.871167, -122.268285], [37.878662, -122.291965], [37.861387, -122.259001], [37.897333, -122.276042], [37.858047, -122.245306], [37.875571, -122.268722], [37.857869, -122.248398], [37.848798, -122.296242], [37.873687, -122.268616], [37.866969, -122.26553], [37.868714, -122.259189], [37.864826, -122.260719], [37.862927, -122.258784], [37.875865, -122.261749], [37.865134, -122.258331], [37.897333, -122.276042], [37.868785, -122.272701], [37.867176, -122.267802], [37.868663, -122.301325], [37.871167, -122.268285], [37.856756, -122.273832], [37.882033, -122.296381], [37.87783, -122.298557], [37.849609, -122.282], [37.871167, -122.268285], [37.878405, -122.306072], [37.86939, -122.267883], [37.871167, -122.268285], [37.873017, -122.275481], [37.870287, -122.316238], [37.86771, -122.298466], [37.880027, -122.279976], [37.850175, -122.293099], [37.864238, -122.265263], [37.862808, -122.27633], [37.868058, -122.278332], [37.856853, -122.26482], [37.868194, -122.276063], [37.871167, -122.268285], [37.87325, -122.293558], [37.870054, -122.284263], [37.879188, -122.276107], [37.885381, -122.272269], [37.865816, -122.281601], [37.854833, -122.279839], [37.863938, -122.253735], [37.878407, -122.267962], [37.888512, -122.266746], [37.865202, -122.257795], [37.854891, -122.280148], [37.863072, -122.260352], [37.867176, -122.267802], [37.890977, -122.259745], [37.866206, -122.29129], [37.870185, -122.299609], [37.863839, -122.281391], [37.856769, -122.27984], [37.853221, -122.280832], [37.858047, -122.245306], [37.863369, -122.257622], [37.853576, -122.287202], [37.858116, -122.268002], [37.856111, -122.260248], [37.865349, -122.28377], [37.870054, -122.284263], [37.864976, -122.266752], [37.883512, -122.27103], [37.872175, -122.267835], [37.890928, -122.287251], [37.88014, -122.297498], [37.86549, -122.295702], [37.859556, -122.273545], [37.85941, -122.25919], [37.858759, -122.264112], [37.88014, -122.297498], [37.873025, -122.275438], [37.865945, -122.250471], [37.883791, -122.249709], [37.862927, -122.258784], [37.869363, -122.268028], [37.864226, -122.277937], [37.869363, -122.268028], [37.873607, -122.270903], [37.871167, -122.268285], [37.877696, -122.273684], [37.852211, -122.286336], [37.870603, -122.270612], [37.868667, -122.313656], [37.856111, -122.260248], [37.875391, -122.271141], [37.858116, -122.268002], [37.863072, -122.260352], [37.848292, -122.271773], [37.882015, -122.292643], [37.894172, -122.253946], [37.872158, -122.282066], [37.87241, -122.277692], [37.866206, -122.29129], [37.869701, -122.287039], [37.866528, -122.290227], [37.861627, -122.285466], [37.871246, -122.274991], [37.867935, -122.258021], [37.867176, -122.267802], [37.878405, -122.306072], [37.858628, -122.250783], [37.864827, -122.258577], [37.901596, -122.270187], [37.868714, -122.259189], [37.863292, -122.256293], [37.868263, -122.296013], [37.866508, -122.261057], [37.858116, -122.268002], [37.883512, -122.27103], [37.857745, -122.23991], [37.862927, -122.258784], [37.873639, -122.27064], [37.869764, -122.28655], [37.848152, -122.275807], [37.871167, -122.268285], [37.869385, -122.28953], [37.864827, -122.258577], [37.869385, -122.28953], [37.869993, -122.294774], [37.890928, -122.287251], [37.891095, -122.284247], [37.862927, -122.258784], [37.88014, -122.297498], [37.899602, -122.271102], [37.870396, -122.281585], [37.862927, -122.258784], [37.858433, -122.280183], [37.858427, -122.282485], [37.883798, -122.268567], [37.8683, -122.262426], [37.878571, -122.282954], [37.867259, -122.26324], [37.867668, -122.260109], [37.869363, -122.268028], [37.862817, -122.298359], [37.869402, -122.297654], [37.877482, -122.256109], [37.8549, -122.27945], [37.865443, -122.263189], [37.862927, -122.258784], [37.871167, -122.268285], [37.880228, -122.295798], [37.88014, -122.297498], [37.866405, -122.272443], [37.881957, -122.269551], [37.870538, -122.297407], [37.879869, -122.269192], [37.870205, -122.292581], [37.872408, -122.26843], [37.881943, -122.284971], [37.884743, -122.250296], [37.883621, -122.269734], [37.857592, -122.269186], [37.885381, -122.272269], [37.902647, -122.277413], [37.870603, -122.270612], [37.8598, -122.264351], [37.866293, -122.30551], [37.882875, -122.288595], [37.863977, -122.267307], [37.870054, -122.284263], [37.880636, -122.264757], [37.870243, -122.275149], [37.866145, -122.300016], [37.875053, -122.26548], [37.891117, -122.280265], [37.858525, -122.266906], [37.872134, -122.284582], [37.872656, -122.292748], [37.858116, -122.268002], [37.858116, -122.268002], [37.876595, -122.267789], [37.880227, -122.26936], [37.896395, -122.285494], [37.871167, -122.268285], [37.878405, -122.306072], [37.868714, -122.259189], [37.857246, -122.277347], [37.868108, -122.277222], [37.855523, -122.266529], [37.862927, -122.258784], [37.867595, -122.253254], [37.862927, -122.258784], [37.87965, -122.273873], [37.859556, -122.273545], [37.868714, -122.259189], [37.875076, -122.299959], [37.856968, -122.279544], [37.869839, -122.252365], [37.88014, -122.297498], [37.858116, -122.268002], [37.862927, -122.258784], [37.863611, -122.317566], [37.860677, -122.279], [37.880893, -122.263673], [37.862927, -122.258784], [37.862927, -122.258784], [37.849786, -122.269782], [37.868641, -122.29415], [37.857452, -122.259548], [37.865059, -122.272291], [37.853861, -122.26598], [37.865772, -122.267643], [37.870054, -122.284263], [37.870337, -122.282043], [37.867176, -122.267802], [37.888842, -122.254155], [37.868898, -122.243093], [37.898747, -122.266109], [37.858116, -122.268002], [37.87325, -122.293558], [37.870948, -122.27733], [37.871167, -122.268285], [37.856132, -122.271292], [37.862562, -122.262598], [37.883621, -122.269734], [37.848774, -122.271171], [37.853951, -122.273198], [37.869764, -122.28655], [37.871461, -122.270706], [37.8559, -122.283101], [37.860172, -122.282847], [37.85444, -122.296619], [37.892638, -122.284804], [37.848413, -122.277936], [37.862169, -122.267084], [37.862927, -122.258784], [37.871175, -122.304755], [37.864238, -122.265263], [37.857714, -122.288536], [37.871265, -122.295043], [37.872499, -122.286632], [37.867307, -122.300468], [37.880227, -122.26936], [37.874182, -122.265937], [37.865797, -122.252741], [37.855798, -122.252707], [37.884902, -122.24847], [37.858392, -122.275421], [37.868263, -122.296013], [37.868352, -122.254459], [37.870205, -122.292581], [37.862059, -122.281167], [37.855435, -122.259841], [37.868263, -122.296013], [37.860687, -122.26454], [37.871246, -122.274991], [37.871317, -122.286575], [37.867176, -122.267802], [37.871167, -122.268285], [37.850704, -122.230786], [37.858116, -122.268002], [37.878239, -122.288374], [37.877442, -122.266726], [37.865748, -122.253396], [37.86825, -122.300093], [37.871167, -122.268285], [37.872725, -122.277729], [37.873687, -122.268616], [37.870054, -122.284263], [37.864826, -122.260719], [37.850798, -122.294689], [37.85976, -122.257034], [37.858628, -122.250783], [37.862516, -122.27853], [37.873393, -122.286956], [37.869067, -122.292043], [37.866568, -122.254084], [37.86771, -122.298466], [37.8719, -122.268389], [37.865748, -122.253396], [37.896395, -122.285494], [37.856111, -122.260248], [37.879677, -122.299779], [37.853216, -122.281082], [37.859006, -122.277874], [37.863611, -122.317566], [37.857714, -122.288536], [37.869993, -122.294774], [37.870054, -122.284263], [37.855798, -122.252707], [37.869363, -122.268028], [37.862024, -122.251212], [37.879968, -122.296885], [37.869067, -122.292043], [37.868263, -122.296013], [37.854833, -122.279839], [37.853959, -122.284067], [37.896453, -122.260331], [37.871246, -122.274991], [37.868714, -122.259189], [37.88014, -122.297498], [37.858116, -122.268002], [37.867176, -122.267802], [37.867972, -122.263699], [37.852491, -122.243412], [37.881957, -122.269551], [37.869332, -122.25019], [37.859215, -122.268555], [37.870205, -122.292581], [37.876897, -122.28868], [37.861081, -122.275984], [37.856132, -122.271292], [37.879451, -122.300901], [37.867176, -122.267802], [37.858756, -122.253219], [37.861843, -122.269644], [37.867513, -122.25195], [37.86771, -122.298466], [37.858116, -122.268002], [37.895076, -122.264743], [37.85976, -122.257034], [37.848774, -122.271171], [37.865748, -122.253396], [37.857452, -122.259548], [37.85732, -122.284575], [37.862927, -122.258784], [37.869363, -122.268028], [37.866761, -122.258779], [37.878372, -122.296373], [37.890928, -122.287251], [37.855167, -122.258753], [37.881003, -122.293212], [37.879027, -122.288477], [37.867501, -122.291709], [37.867176, -122.267802], [37.855293, -122.266502], [37.892524, -122.256323], [37.866084, -122.292775], [37.867513, -122.26127], [37.904224, -122.272694], [37.868785, -122.272701], [37.859216, -122.26855], [37.870639, -122.272468], [37.878864, -122.265718], [37.853576, -122.287202], [37.877793, -122.260709], [37.882033, -122.296381], [37.853723, -122.259975], [37.867176, -122.267802], [37.86771, -122.298466], [37.873327, -122.273214], [37.867501, -122.291709], [37.871167, -122.268285], [37.867717, -122.249901], [37.88014, -122.297498], [37.862512, -122.290094], [37.861118, -122.259948], [37.865748, -122.253396], [37.864701, -122.260693], [37.864079, -122.266509], [37.880228, -122.295798], [37.880734, -122.265359], [37.880228, -122.295798], [37.867176, -122.267802], [37.849431, -122.278174], [37.858116, -122.268002], [37.851653, -122.289194], [37.868667, -122.313656], [37.864701, -122.260693], [37.879933, -122.283098], [37.858116, -122.268002], [37.848152, -122.275807], [37.867513, -122.26127], [37.868714, -122.259189], [37.898898, -122.274322], [37.864535, -122.262993], [37.867513, -122.26127], [37.866724, -122.288804], [37.883621, -122.269734], [37.877951, -122.271395], [37.858116, -122.268002], [37.855293, -122.266502], [37.868815, -122.292131], [37.850675, -122.286265], [37.865034, -122.297939], [37.863611, -122.317566], [37.857787, -122.257013], [37.868355, -122.274953], [37.861129, -122.273879], [37.865511, -122.309967], [37.867176, -122.267802], [37.88014, -122.297498], [37.871167, -122.268285], [37.867513, -122.25195], [37.866563, -122.297217], [37.846404, -122.275263], [37.871461, -122.270706], [37.879888, -122.285306], [37.898747, -122.266109], [37.867822, -122.266003], [37.868714, -122.259189], [37.891827, -122.275094], [37.872175, -122.267835], [37.858525, -122.266906], [37.88014, -122.297498], [37.868667, -122.313656], [37.859367, -122.291421], [37.869744, -122.28181], [37.853576, -122.287202], [37.871167, -122.268285], [37.87199, -122.273062], [37.866513, -122.27474], [37.859802, -122.267177], [37.860687, -122.26454], [37.88014, -122.297498], [37.873393, -122.286956], [37.862927, -122.258784], [37.85062, -122.278313], [37.882033, -122.296381], [37.858116, -122.268002], [37.864827, -122.258577], [37.880027, -122.279976], [37.852435, -122.270917], [37.869105, -122.270064], [37.85274, -122.283711], [37.854186, -122.279157], [37.88014, -122.297498], [37.85177, -122.276489], [37.868913, -122.28608], [37.866145, -122.300016], [37.868815, -122.292131], [37.878644, -122.303432], [37.869688, -122.272805], [37.852764, -122.26165], [37.859364, -122.288914], [37.869113, -122.250903], [37.851107, -122.28324], [37.881788, -122.270678], [37.867708, -122.250801], [37.850285, -122.273794], [37.881003, -122.293212], [37.887961, -122.269822], [37.876644, -122.299354], [37.855042, -122.284746], [37.859665, -122.27169], [37.862562, -122.262598], [37.865149, -122.256487], [37.869067, -122.292043], [37.852304, -122.270045], [37.870287, -122.316238], [37.867176, -122.267802], [37.854442, -122.277107], [37.873687, -122.268616], [37.864108, -122.259461], [37.871167, -122.268285], [37.878629, -122.260883], [37.870396, -122.281585], [37.859259, -122.275787], [37.862092, -122.289804], [37.868714, -122.259189], [37.858116, -122.268002], [37.849786, -122.269782], [37.880075, -122.270476], [37.853552, -122.291561], [37.859309, -122.259291], [37.864908, -122.267289], [37.888199, -122.250826], [37.867176, -122.267802], [37.85062, -122.278313], [37.881788, -122.270678], [37.882033, -122.296381], [37.858116, -122.268002], [37.871167, -122.268285], [37.86825, -122.300093], [37.877636, -122.264323], [37.863292, -122.256293], [37.868532, -122.274764], [37.862817, -122.298359], [37.888702, -122.26371], [37.869688, -122.272805], [37.868667, -122.313656], [37.870205, -122.292581], [37.865202, -122.257795], [37.853275, -122.279078], [37.871369, -122.292954], [37.876307, -122.268923], [37.875281, -122.293681], [37.860766, -122.255895], [37.857099, -122.263785], [37.865511, -122.309967], [37.86939, -122.267883], [37.86939, -122.267883], [37.885744, -122.278017], [37.887344, -122.277321], [37.86825, -122.300093], [37.869993, -122.294774], [37.856326, -122.259709], [37.879713, -122.299159], [37.850798, -122.294689], [37.872656, -122.292748], [37.876307, -122.268923], [37.858455, -122.293374], [37.867176, -122.267802], [37.870054, -122.284263], [37.877863, -122.308855], [37.856088, -122.29274], [37.871167, -122.268285], [37.866924, -122.283899], [37.866426, -122.269762], [37.870911, -122.289684], [37.871167, -122.268285], [37.864827, -122.258577], [37.878722, -122.295312], [37.86771, -122.298466], [37.879968, -122.296885], [37.865511, -122.309967], [37.857714, -122.288536], [37.894256, -122.281603], [37.858116, -122.268002], [37.867176, -122.267802], [37.862169, -122.267084], [37.857254, -122.262649], [37.870226, -122.277441], [37.856088, -122.29274], [37.882016, -122.281271], [37.869937, -122.295043], [37.855389, -122.264552], [37.851921, -122.269813], [37.881957, -122.269551], [37.891095, -122.284247], [37.863811, -122.267412], [37.871246, -122.274991], [37.878407, -122.267962], [37.876045, -122.260336], [37.867513, -122.281165], [37.867852, -122.258699], [37.864827, -122.258577], [37.856488, -122.257329], [37.866568, -122.254084], [37.869363, -122.268028], [37.863811, -122.267412], [37.858116, -122.268002], [37.891408, -122.278186], [37.876307, -122.268923], [37.850681, -122.270619], [37.869385, -122.28953], [37.865772, -122.267643], [37.863977, -122.267307], [37.867176, -122.267802], [37.864258, -122.272202], [37.853552, -122.291561], [37.869186, -122.283943], [37.861843, -122.269644], [37.863934, -122.250262], [37.869962, -122.277411], [37.855523, -122.266529], [37.856939, -122.272438], [37.881957, -122.269551], [37.858116, -122.268002], [37.862927, -122.258784], [37.864258, -122.272202], [37.887747, -122.264514], [37.858116, -122.268002], [37.879679, -122.273638], [37.856198, -122.27972], [37.877835, -122.260354], [37.880027, -122.279976], [37.882033, -122.296381], [37.879226, -122.260975], [37.893104, -122.267605], [37.870603, -122.270612], [37.856111, -122.260248], [37.899249, -122.279642], [37.864827, -122.258577], [37.853864, -122.285643], [37.867513, -122.25195], [37.871167, -122.268285], [37.857809, -122.271478], [37.878405, -122.306072], [37.854012, -122.251243], [37.853552, -122.291561], [37.859195, -122.261915], [37.846404, -122.275263], [37.873327, -122.273214], [37.858903, -122.243972], [37.883948, -122.296991], [37.880163, -122.30676], [37.850541, -122.286037], [37.855678, -122.274429], [37.855832, -122.272369], [37.859802, -122.267177], [37.868714, -122.259189], [37.882033, -122.296381], [37.870256, -122.298419], [37.868263, -122.296013], [37.85931, -122.273495], [37.866206, -122.29129], [37.878326, -122.306549], [37.856968, -122.279544], [37.865772, -122.267643], [37.889017, -122.272389], [37.866265, -122.278162], [37.87241, -122.277692], [37.878051, -122.285222], [37.848152, -122.275807], [37.852348, -122.271972], [37.880228, -122.295798], [37.884732, -122.280918], [37.882944, -122.26847], [37.871167, -122.268285], [37.89316, -122.283937], [37.869363, -122.268028], [37.869617, -122.295925], [37.858759, -122.264112], [37.858116, -122.268002], [37.880383, -122.285574], [37.870205, -122.292581], [37.871544, -122.272714], [37.876307, -122.268923], [37.88014, -122.297498], [37.883948, -122.296991], [37.85177, -122.276489], [37.858116, -122.268002], [37.853576, -122.287202], [37.8559, -122.283101], [37.860766, -122.255895], [37.866568, -122.254084], [37.868785, -122.272701], [37.878722, -122.295312], [37.863938, -122.253735], [37.88014, -122.297498], [37.856066, -122.27019], [37.871167, -122.268285], [37.865793, -122.301779], [37.866122, -122.260981], [37.868334, -122.303753], [37.849431, -122.278174], [37.864826, -122.260719], [37.883555, -122.272036], [37.868913, -122.28608], [37.869186, -122.283943], [37.869363, -122.268028], [37.859006, -122.277874], [37.857099, -122.263785], [37.870603, -122.270612], [37.884732, -122.280918], [37.86542, -122.25618], [37.878405, -122.306072], [37.868058, -122.278332], [37.878901, -122.274916], [37.87304, -122.289659], [37.883621, -122.269734], [37.866969, -122.26553], [37.846388, -122.273111], [37.871828, -122.270516], [37.849747, -122.277907], [37.854944, -122.257583], [37.881164, -122.292378], [37.857776, -122.286576], [37.894468, -122.265464], [37.86825, -122.300093], [37.868785, -122.272701], [37.876307, -122.268923], [37.866074, -122.26331], [37.872499, -122.286632], [37.851203, -122.289129], [37.889989, -122.252226], [37.871246, -122.274991], [37.871544, -122.272714], [37.870205, -122.292581], [37.869307, -122.248958], [37.861283, -122.273911], [37.871544, -122.272714], [37.867176, -122.267802], [37.867176, -122.267802], [37.872656, -122.292748], [37.860225, -122.269453], [37.865149, -122.256487], [37.882199, -122.268386], [37.856968, -122.279544], [37.858116, -122.268002], [37.878407, -122.267962], [37.870054, -122.284263], [37.890928, -122.287251], [37.869888, -122.300618], [37.857784, -122.272998], [37.859309, -122.259291], [37.84905, -122.269098], [37.881957, -122.269551], [37.866559, -122.299584], [37.869764, -122.28655], [37.863611, -122.317566], [37.85968, -122.255796], [37.864827, -122.258577]],
                {&quot;blur&quot;: 15, &quot;maxZoom&quot;: 18, &quot;minOpacity&quot;: 0.5, &quot;radius&quot;: 10}
            );


            heat_map_94f090f0653a72aceaedb4f9d4cd9e06.addTo(map_88c9cc54858720e4970fd98017e5f89c);

&lt;/script&gt;
&lt;/html&gt;" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



Based on the above map, what could be some **drawbacks** of using the location fields in this dataset to draw conclusions about crime in Berkeley? Here are some sub-questions to consider:
* Is campus really the safest place to be?
* Why are all the calls located on the street and often at intersections?

<!--
BEGIN QUESTION
name: q3d
-->

<br/>

<hr style="border: 5px solid #003262;" />

**Important**: To make sure the test cases run correctly, click `Kernel>Restart & Run All` and make sure all of the test cases are still passing. Doing so will submit your code for you. 

If your test cases are no longer passing after restarting, it's likely because you're missing a variable, or the modifications that you'd previously made to your DataFrame are no longer taking place (perhaps because you deleted a cell). 

You may submit this assignment as many times as you'd like before the deadline.

**You must restart and run all cells before submitting. Otherwise, you may pass test cases locally, but not on our servers. We will not entertain regrade requests of the form, “my code passed all of my local test cases, but failed the autograder”.**

## Congratulations!

Congrats! You are finished with this lab.

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

