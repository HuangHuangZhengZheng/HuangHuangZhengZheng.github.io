# DATA100-lab10: SQL


```python
# Initialize Otter
import otter
grader = otter.Notebook("lab10.ipynb")
```

# Lab 10: SQL

In this lab, we are going to practice viewing, sorting, grouping, and merging tables with SQL. We will explore two datasets:
1. A "minified" version of the [Internet Movie Database](https://www.imdb.com/interfaces/) (IMDb). This SQLite database (~10MB) is a tiny sample of the much larger database (more than a few GBs). As a result, disclaimer that we may get wildly different results than if we use the whole database!

1. The money donated during the 2016 election using the [Federal Election Commission (FEC)'s public records](https://www.fec.gov/data/). You will be connecting to a SQLite database containing the data. The data we will be working with in this lab is quite small (~16MB); however, it is a sample taken from a much larger database (more than a few GBs).



```python
# Run this cell to set up your notebook
import numpy as np
import pandas as pd
import plotly.express as px
import sqlalchemy
from ds100_utils import fetch_and_cache
from pathlib import Path
%load_ext sql

# Unzip the data.
!unzip -o data.zip
```

    Archive:  data.zip
      inflating: imdbmini.db             
      inflating: fec_nyc.db              
    

## SQL Query Syntax

Throughout this lab, you will become familiar with the following syntax for the `SELECT` query:

```
SELECT [DISTINCT] 
    {* | expr [[AS] c_alias] 
    {,expr [[AS] c_alias] ...}}
FROM tableref {, tableref}
[[INNER | LEFT ] JOIN table_name
    ON qualification_list]
[WHERE search_condition]
[GROUP BY colname {,colname...}]
[HAVING search condition]
[ORDER BY column_list]
[LIMIT number]
[OFFSET number of rows];
```

<br/><br/>
<hr style="border: 5px solid #003262;" />
<hr style="border: 1px solid #fdb515;" />

# Part 0 [Tutorial]: Writing SQL in Jupyter Notebooks

## 1. `%%sql` cell magic

In lecture, we used the `sql` extension to call **`%%sql` cell magic**, which enables us to connect to SQL databses and issue SQL commands within Jupyter Notebooks.

Run the below cell to connect to a mini IMDb database.


```python
%sql sqlite:///imdbmini.db
```

<br/>

Above, prefixing our single-line command with `%sql` means that the entire line will be treated as a SQL command (this is called "line magic"). In this class we will most often write multi-line SQL, meaning we need "cell magic", where the first line has `%%sql` (note the double `%` operator).

The database `imdbmini.db` includes several tables, one of which is `Title`. Running the below cell will return first 5 lines of that table. Note that `%%sql` is on its own line.

We've also included syntax for single-line comments, which are surrounded by `--`.


```sql
%%sql
/*
 * This is a
 * multi-line comment.
 */
-- This is a single-line/inline comment. --
SELECT *
FROM Name
LIMIT 5;
```

     * sqlite:///imdbmini.db
    Done.
    




<table>
    <thead>
        <tr>
            <th>nconst</th>
            <th>primaryName</th>
            <th>birthYear</th>
            <th>deathYear</th>
            <th>primaryProfession</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>1</td>
            <td>Fred Astaire</td>
            <td>1899</td>
            <td>1987</td>
            <td>soundtrack,actor,miscellaneous</td>
        </tr>
        <tr>
            <td>2</td>
            <td>Lauren Bacall</td>
            <td>1924</td>
            <td>2014</td>
            <td>actress,soundtrack</td>
        </tr>
        <tr>
            <td>3</td>
            <td>Brigitte Bardot</td>
            <td>1934</td>
            <td>None</td>
            <td>actress,soundtrack,music_department</td>
        </tr>
        <tr>
            <td>4</td>
            <td>John Belushi</td>
            <td>1949</td>
            <td>1982</td>
            <td>actor,soundtrack,writer</td>
        </tr>
        <tr>
            <td>5</td>
            <td>Ingmar Bergman</td>
            <td>1918</td>
            <td>2007</td>
            <td>writer,director,actor</td>
        </tr>
    </tbody>
</table>



<br/><br/>

### 2. The Pandas command `pd.read_sql`

As of 2022, the `%sql` magic for Jupyter Notebooks is still in development (check out its [GitHub](https://github.com/catherinedevlin/ipython-sql). It is still missing many features that would justify real-world use with Python. In particular, its returned tables are *not* Pandas dataframes (for example, the query result from the above cell is missing an index).

The rest of this section describes how data scientists use SQL and Python in practice, using the Pandas command `pd.read_sql` ([documentation](https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html)). **You will see both `%sql` magic and `pd.read_sql` in this course**. 


The below cell connects to the same database using the SQLAlchemy Python library, which can connect to several different database management systems, including sqlite3, MySQL, PostgreSQL, and Oracle. The library also supports an advanced feature for generating queries called an [object relational mapper](https://docs.sqlalchemy.org/en/latest/orm/tutorial.html) or ORM, which we won't discuss in this course but is quite useful for application development.


```python
# important!!! run this cell
import sqlalchemy

# create a SQL Alchemy connection to the database
engine = sqlalchemy.create_engine("sqlite:///imdbmini.db")
connection = engine.connect()
```

With the SQLAlchemy object `connection`, we can then call `pd.read_sql` which takes in a `query` **string**. Note the `"""` to define our multi-line string, which allows us to have a query span multiple lines. The resulting `df` DataFrame stores the results of the same SQL query from the previous section.


```python
# just run this cell
query = """
SELECT *
FROM Title
LIMIT 5;
"""

df = pd.read_sql(query, engine)
df
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
      <th>tconst</th>
      <th>titleType</th>
      <th>primaryTitle</th>
      <th>originalTitle</th>
      <th>isAdult</th>
      <th>startYear</th>
      <th>endYear</th>
      <th>runtimeMinutes</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>417</td>
      <td>short</td>
      <td>A Trip to the Moon</td>
      <td>Le voyage dans la lune</td>
      <td>0</td>
      <td>1902</td>
      <td>None</td>
      <td>13</td>
      <td>Action,Adventure,Comedy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4972</td>
      <td>movie</td>
      <td>The Birth of a Nation</td>
      <td>The Birth of a Nation</td>
      <td>0</td>
      <td>1915</td>
      <td>None</td>
      <td>195</td>
      <td>Drama,History,War</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10323</td>
      <td>movie</td>
      <td>The Cabinet of Dr. Caligari</td>
      <td>Das Cabinet des Dr. Caligari</td>
      <td>0</td>
      <td>1920</td>
      <td>None</td>
      <td>76</td>
      <td>Fantasy,Horror,Mystery</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12349</td>
      <td>movie</td>
      <td>The Kid</td>
      <td>The Kid</td>
      <td>0</td>
      <td>1921</td>
      <td>None</td>
      <td>68</td>
      <td>Comedy,Drama,Family</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13442</td>
      <td>movie</td>
      <td>Nosferatu</td>
      <td>Nosferatu, eine Symphonie des Grauens</td>
      <td>0</td>
      <td>1922</td>
      <td>None</td>
      <td>94</td>
      <td>Fantasy,Horror</td>
    </tr>
  </tbody>
</table>
</div>



<br/>

**Long error messages**: Given that the SQL query is now in the string, the errors become more unintelligible. Consider the below (incorrect) query, which has a semicolon in the wrong place.


```python
# uncomment the below code and check out the error

# query = """
# SELECT *
# FROM Title;
# LIMIT 5
# """
# pd.read_sql(query, engine)
```


    ---------------------------------------------------------------------------

    ProgrammingError                          Traceback (most recent call last)

    File d:\miniconda3\Lib\site-packages\sqlalchemy\engine\base.py:1967, in Connection._exec_single_context(self, dialect, context, statement, parameters)
       1966     if not evt_handled:
    -> 1967         self.dialect.do_execute(
       1968             cursor, str_statement, effective_parameters, context
       1969         )
       1971 if self._has_events or self.engine._has_events:
    

    File d:\miniconda3\Lib\site-packages\sqlalchemy\engine\default.py:924, in DefaultDialect.do_execute(self, cursor, statement, parameters, context)
        923 def do_execute(self, cursor, statement, parameters, context=None):
    --> 924     cursor.execute(statement, parameters)
    

    ProgrammingError: You can only execute one statement at a time.

    
    The above exception was the direct cause of the following exception:
    

    ProgrammingError                          Traceback (most recent call last)

    Cell In[7], line 8
          1 # uncomment the below code and check out the error
          3 query = """
          4 SELECT *
          5 FROM Title;
          6 LIMIT 5
          7 """
    ----> 8 pd.read_sql(query, engine)
    

    File d:\miniconda3\Lib\site-packages\pandas\io\sql.py:734, in read_sql(sql, con, index_col, coerce_float, params, parse_dates, columns, chunksize, dtype_backend, dtype)
        724     return pandas_sql.read_table(
        725         sql,
        726         index_col=index_col,
       (...)
        731         dtype_backend=dtype_backend,
        732     )
        733 else:
    --> 734     return pandas_sql.read_query(
        735         sql,
        736         index_col=index_col,
        737         params=params,
        738         coerce_float=coerce_float,
        739         parse_dates=parse_dates,
        740         chunksize=chunksize,
        741         dtype_backend=dtype_backend,
        742         dtype=dtype,
        743     )
    

    File d:\miniconda3\Lib\site-packages\pandas\io\sql.py:1836, in SQLDatabase.read_query(self, sql, index_col, coerce_float, parse_dates, params, chunksize, dtype, dtype_backend)
       1779 def read_query(
       1780     self,
       1781     sql: str,
       (...)
       1788     dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
       1789 ) -> DataFrame | Iterator[DataFrame]:
       1790     """
       1791     Read SQL query into a DataFrame.
       1792 
       (...)
       1834 
       1835     """
    -> 1836     result = self.execute(sql, params)
       1837     columns = result.keys()
       1839     if chunksize is not None:
    

    File d:\miniconda3\Lib\site-packages\pandas\io\sql.py:1659, in SQLDatabase.execute(self, sql, params)
       1657 args = [] if params is None else [params]
       1658 if isinstance(sql, str):
    -> 1659     return self.con.exec_driver_sql(sql, *args)
       1660 return self.con.execute(sql, *args)
    

    File d:\miniconda3\Lib\site-packages\sqlalchemy\engine\base.py:1779, in Connection.exec_driver_sql(self, statement, parameters, execution_options)
       1774 execution_options = self._execution_options.merge_with(
       1775     execution_options
       1776 )
       1778 dialect = self.dialect
    -> 1779 ret = self._execute_context(
       1780     dialect,
       1781     dialect.execution_ctx_cls._init_statement,
       1782     statement,
       1783     None,
       1784     execution_options,
       1785     statement,
       1786     distilled_parameters,
       1787 )
       1789 return ret
    

    File d:\miniconda3\Lib\site-packages\sqlalchemy\engine\base.py:1846, in Connection._execute_context(self, dialect, constructor, statement, parameters, execution_options, *args, **kw)
       1844     return self._exec_insertmany_context(dialect, context)
       1845 else:
    -> 1846     return self._exec_single_context(
       1847         dialect, context, statement, parameters
       1848     )
    

    File d:\miniconda3\Lib\site-packages\sqlalchemy\engine\base.py:1986, in Connection._exec_single_context(self, dialect, context, statement, parameters)
       1983     result = context._setup_result_proxy()
       1985 except BaseException as e:
    -> 1986     self._handle_dbapi_exception(
       1987         e, str_statement, effective_parameters, cursor, context
       1988     )
       1990 return result
    

    File d:\miniconda3\Lib\site-packages\sqlalchemy\engine\base.py:2353, in Connection._handle_dbapi_exception(self, e, statement, parameters, cursor, context, is_sub_exec)
       2351 elif should_wrap:
       2352     assert sqlalchemy_exception is not None
    -> 2353     raise sqlalchemy_exception.with_traceback(exc_info[2]) from e
       2354 else:
       2355     assert exc_info[1] is not None
    

    File d:\miniconda3\Lib\site-packages\sqlalchemy\engine\base.py:1967, in Connection._exec_single_context(self, dialect, context, statement, parameters)
       1965                 break
       1966     if not evt_handled:
    -> 1967         self.dialect.do_execute(
       1968             cursor, str_statement, effective_parameters, context
       1969         )
       1971 if self._has_events or self.engine._has_events:
       1972     self.dispatch.after_cursor_execute(
       1973         self,
       1974         cursor,
       (...)
       1978         context.executemany,
       1979     )
    

    File d:\miniconda3\Lib\site-packages\sqlalchemy\engine\default.py:924, in DefaultDialect.do_execute(self, cursor, statement, parameters, context)
        923 def do_execute(self, cursor, statement, parameters, context=None):
    --> 924     cursor.execute(statement, parameters)
    

    ProgrammingError: (sqlite3.ProgrammingError) You can only execute one statement at a time.
    [SQL: 
    SELECT *
    FROM Title;
    LIMIT 5
    ]
    (Background on this error at: https://sqlalche.me/e/20/f405)


<br/>

Now that's an unruly error message!

<br/><br/>

### 3. A suggested workflow for writing SQL in Jupyter Notebooks

Which approach is better, `%sql` magic or `pd.read_sql`?

The SQL database generally contains much more data than what you would analyze in detail. As a Python-fluent data scientist, you will often query SQL databases to perform initial exploratory data analysis, a subset of which you load into Python for further processing.

In practice, you would likely use a combination of the two approaches. First, you'd try out some SQL queries with `%sql` magic to get an interesting subset of data. Then, you'd copy over the query into a `pd.read_sql` command for visualization, modeling, and export with Pandas, sklearn, and other Python libraries.

For SQL assignments in this course, to minimize unruly error messages while maximizing Python compatibility, we suggest the following "sandboxed" workflow:
1. Create a `%%sql` magic cell **below** the answer cell. You can copy in the below code:

    ```
    %% sql
    -- This is a comment. Put your code here... --
    ```
<br/>

1.  Work on the SQL query in the `%%sql` cell; e.g., `SELECT ... ;`
1. Then, once you're satisfied with your SQL query, copy it into the multi-string query in the answer cell (the one that contains the `pd.read_sql` call).

You don't have to follow the above workflow to get full credit on assignments, but we suggest it to reduce debugging headaches. We've created the scratchwork `%%sql` cells for you in this assignment, but **do not** add cells between this `%%sql` cell and the Python cell right below it. It will cause errors when we run the autograder, and it will sometimes cause a failure to generate the PDF file.


<br/><br/>
<hr style="border: 5px solid #003262;" />
<hr style="border: 1px solid #fdb515;" />

# Part 1: The IMDb (mini) Dataset

Let's explore a miniature version of the [IMDb Dataset](https://www.imdb.com/interfaces/). This is the same dataset that we will use for the upcoming homework.


Let's load in the database in two ways (using both Python and cell magic) so that we can flexibly explore the database.


```python
engine = sqlalchemy.create_engine("sqlite:///imdbmini.db")
connection = engine.connect()
```


```python
%sql sqlite:///imdbmini.db
```

<br/>


```sql
%%sql
SELECT * FROM sqlite_master WHERE type='table';
```

     * sqlite:///imdbmini.db
    Done.
    




<table>
    <thead>
        <tr>
            <th>type</th>
            <th>name</th>
            <th>tbl_name</th>
            <th>rootpage</th>
            <th>sql</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>table</td>
            <td>Title</td>
            <td>Title</td>
            <td>2</td>
            <td>CREATE TABLE &quot;Title&quot; (<br>&quot;tconst&quot; INTEGER,<br>  &quot;titleType&quot; TEXT,<br>  &quot;primaryTitle&quot; TEXT,<br>  &quot;originalTitle&quot; TEXT,<br>  &quot;isAdult&quot; TEXT,<br>  &quot;startYear&quot; TEXT,<br>  &quot;endYear&quot; TEXT,<br>  &quot;runtimeMinutes&quot; TEXT,<br>  &quot;genres&quot; TEXT<br>)</td>
        </tr>
        <tr>
            <td>table</td>
            <td>Name</td>
            <td>Name</td>
            <td>12</td>
            <td>CREATE TABLE &quot;Name&quot; (<br>&quot;nconst&quot; INTEGER,<br>  &quot;primaryName&quot; TEXT,<br>  &quot;birthYear&quot; TEXT,<br>  &quot;deathYear&quot; TEXT,<br>  &quot;primaryProfession&quot; TEXT<br>)</td>
        </tr>
        <tr>
            <td>table</td>
            <td>Role</td>
            <td>Role</td>
            <td>70</td>
            <td>CREATE TABLE &quot;Role&quot; (<br>tconst INTEGER,<br>ordering TEXT,<br>nconst INTEGER,<br>category TEXT,<br>job TEXT,<br>characters TEXT<br>)</td>
        </tr>
        <tr>
            <td>table</td>
            <td>Rating</td>
            <td>Rating</td>
            <td>41</td>
            <td>CREATE TABLE &quot;Rating&quot; (<br>tconst INTEGER,<br>averageRating TEXT,<br>numVotes TEXT<br>)</td>
        </tr>
    </tbody>
</table>



From running the above cell, we see the database has 4 tables: `Name`, `Role`, `Rating`, and `Title`.

<details>
    <summary>[<b>Click to Expand</b>] See descriptions of each table's schema.</summary>
    
**`Name`** – Contains the following information for names of people.
    
- nconst (text) - alphanumeric unique identifier of the name/person
- primaryName (text)– name by which the person is most often credited
- birthYear (integer) – in YYYY format
- deathYear (integer) – in YYYY format
    
    
**`Role`** – Contains the principal cast/crew for titles.
    
- tconst (text) - alphanumeric unique identifier of the title
- ordering (integer) – a number to uniquely identify rows for a given tconst
- nconst (text) - alphanumeric unique identifier of the name/person
- category (text) - the category of job that person was in
- characters (text) - the name of the character played if applicable, else '\\N'
    
**`Rating`** – Contains the IMDb rating and votes information for titles.
    
- tconst (text) - alphanumeric unique identifier of the title
- averageRating (real) – weighted average of all the individual user ratings
- numVotes (integer) - number of votes (i.e., ratings) the title has received
    
**`Title`** - Contains the following information for titles.
    
- tconst (text) - alphanumeric unique identifier of the title
- titleType (text) -  the type/format of the title
- primaryTitle (text) -  the more popular title / the title used by the filmmakers on promotional materials at the point of release
- isAdult (text) - 0: non-adult title; 1: adult title
- year (YYYY) – represents the release year of a title.
- runtimeMinutes (integer)  – primary runtime of the title, in minutes
    
</details>

<br/><br/>
From the above descriptions, we can conclude the following:
* `Name.nconst` and `Title.tconst` are primary keys of the `Name` and `Title` tables, respectively.
* that `Role.nconst` and `Role.tconst` are **foreign keys** that point to `Name.nconst` and `Title.tconst`, respectively.

<br/><br/>
<hr style="border: 1px solid #fdb515;" />

## Question 1

What are the different kinds of `titleType`s included in the `Title` table? Write a query to find out all the unique `titleType`s of films using the `DISTINCT` keyword.  (**You may not use `GROUP BY`.**)


```sql
%%sql
/*
 * Code in this scratchwork cell is __not graded.__
 * Copy over any SQL queries you write here into the below Python cell.
 * Do __not__ insert any new cells in between the SQL/Python cells!
 * Doing so may break the autograder.
 */
-- Write below this comment. --
SELECT DISTINCT titleType FROM Title;
```

     * sqlite:///imdbmini.db
    Done.
    




<table>
    <thead>
        <tr>
            <th>titleType</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>short</td>
        </tr>
        <tr>
            <td>movie</td>
        </tr>
        <tr>
            <td>tvSeries</td>
        </tr>
        <tr>
            <td>tvMovie</td>
        </tr>
        <tr>
            <td>tvMiniSeries</td>
        </tr>
        <tr>
            <td>video</td>
        </tr>
        <tr>
            <td>videoGame</td>
        </tr>
        <tr>
            <td>tvEpisode</td>
        </tr>
        <tr>
            <td>tvSpecial</td>
        </tr>
    </tbody>
</table>




```python
query_q1 = """
SELECT DISTINCT titleType FROM Title;
"""

res_q1 = pd.read_sql(query_q1, engine)
res_q1
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
      <th>titleType</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>short</td>
    </tr>
    <tr>
      <th>1</th>
      <td>movie</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tvSeries</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tvMovie</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tvMiniSeries</td>
    </tr>
    <tr>
      <th>5</th>
      <td>video</td>
    </tr>
    <tr>
      <th>6</th>
      <td>videoGame</td>
    </tr>
    <tr>
      <th>7</th>
      <td>tvEpisode</td>
    </tr>
    <tr>
      <th>8</th>
      <td>tvSpecial</td>
    </tr>
  </tbody>
</table>
</div>




```python
grader.check("q1")
```




<p><strong><pre style='display: inline;'>q1</pre></strong> passed! 🍀</p>



---

## Question 2

Before we proceed we want to get a better picture of the kinds of jobs that exist.  To do this examine the `Role` table by computing the number of records with each job `category`.  Present the results in descending order by the total counts.

The top of your table should look like this (however, you should have more rows):

| |category|total|
|-----|-----|-----|
|**0**|actor|21665|
|**1**|writer|13830|
|**2**|...|...|


```sql
%%sql
/*
 * Code in this scratchwork cell is __not graded.__
 * Copy over any SQL queries you write here into the below Python cell.
 * Do __not__ insert any new cells in between the SQL/Python cells!
 * Doing so may break the autograder.
 */
-- Write below this comment. --
SELECT category, COUNT(*) AS total
FROM Role
GROUP BY category
ORDER BY total DESC;
```

     * sqlite:///imdbmini.db
    Done.
    




<table>
    <thead>
        <tr>
            <th>category</th>
            <th>total</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>actor</td>
            <td>21665</td>
        </tr>
        <tr>
            <td>writer</td>
            <td>13830</td>
        </tr>
        <tr>
            <td>actress</td>
            <td>12175</td>
        </tr>
        <tr>
            <td>producer</td>
            <td>11028</td>
        </tr>
        <tr>
            <td>director</td>
            <td>6995</td>
        </tr>
        <tr>
            <td>composer</td>
            <td>4123</td>
        </tr>
        <tr>
            <td>cinematographer</td>
            <td>2747</td>
        </tr>
        <tr>
            <td>editor</td>
            <td>1558</td>
        </tr>
        <tr>
            <td>self</td>
            <td>623</td>
        </tr>
        <tr>
            <td>production_designer</td>
            <td>410</td>
        </tr>
        <tr>
            <td>archive_footage</td>
            <td>66</td>
        </tr>
        <tr>
            <td>archive_sound</td>
            <td>6</td>
        </tr>
    </tbody>
</table>




```python
query_q2 = """
SELECT category, COUNT(*) AS total
FROM Role
GROUP BY category
ORDER BY total DESC;
"""

res_q2 = pd.read_sql(query_q2, engine)
res_q2
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
      <th>category</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>actor</td>
      <td>21665</td>
    </tr>
    <tr>
      <th>1</th>
      <td>writer</td>
      <td>13830</td>
    </tr>
    <tr>
      <th>2</th>
      <td>actress</td>
      <td>12175</td>
    </tr>
    <tr>
      <th>3</th>
      <td>producer</td>
      <td>11028</td>
    </tr>
    <tr>
      <th>4</th>
      <td>director</td>
      <td>6995</td>
    </tr>
    <tr>
      <th>5</th>
      <td>composer</td>
      <td>4123</td>
    </tr>
    <tr>
      <th>6</th>
      <td>cinematographer</td>
      <td>2747</td>
    </tr>
    <tr>
      <th>7</th>
      <td>editor</td>
      <td>1558</td>
    </tr>
    <tr>
      <th>8</th>
      <td>self</td>
      <td>623</td>
    </tr>
    <tr>
      <th>9</th>
      <td>production_designer</td>
      <td>410</td>
    </tr>
    <tr>
      <th>10</th>
      <td>archive_footage</td>
      <td>66</td>
    </tr>
    <tr>
      <th>11</th>
      <td>archive_sound</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
grader.check("q2")
```




<p><strong><pre style='display: inline;'>q2</pre></strong> passed! ✨</p>



<br/>
If we computed the results correctly we should see a nice horizontal bar chart of the counts per category below:


```python
# just run this cell
px.bar(res_q2, x="total", y="category", orientation='h')
```



<br/><br/>
<hr style="border: 1px solid #fdb515;" />

## Question 3

Now that we have a better sense of the basics of our data, we can ask some more interesting questions.

The `Rating` table has the `numVotes` and the `averageRating` for each title. Which 10 films have the most ratings?

Write a SQL query that outputs three fields: the `title`, `numVotes`, and `averageRating` for the 10 films that have the highest number of ratings.  Sort the result in descending order by the number of votes.

*Hint*: The `numVotes` in the `Rating` table is not an integer! Use `CAST(Rating.numVotes AS int) AS numVotes` to convert the attribute to an integer.


```sql
%%sql
/*
 * Code in this scratchwork cell is __not graded.__
 * Copy over any SQL queries you write here into the below Python cell.
 * Do __not__ insert any new cells in between the SQL/Python cells!
 * Doing so may break the autograder.
 */
-- Write below this comment. --
SELECT primaryTitle, 
CAST(numVotes AS int) AS numVotes,
averageRating
FROM Rating, Title
WHERE Rating.tconst = Title.tconst
ORDER BY numVotes DESC;

```


```python
query_q3 = """
SELECT primaryTitle AS title, 
CAST(numVotes AS int) AS numVotes,
averageRating
FROM Rating, Title
WHERE Rating.tconst = Title.tconst
ORDER BY numVotes DESC
LIMIT 10;
"""


res_q3 = pd.read_sql(query_q3, engine)
res_q3
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
      <th>title</th>
      <th>numVotes</th>
      <th>averageRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Shawshank Redemption</td>
      <td>2462686</td>
      <td>9.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Dark Knight</td>
      <td>2417875</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Inception</td>
      <td>2169255</td>
      <td>8.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fight Club</td>
      <td>1939312</td>
      <td>8.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pulp Fiction</td>
      <td>1907561</td>
      <td>8.9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Forrest Gump</td>
      <td>1903969</td>
      <td>8.8</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Game of Thrones</td>
      <td>1874040</td>
      <td>9.2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>The Matrix</td>
      <td>1756469</td>
      <td>8.7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>The Lord of the Rings: The Fellowship of the Ring</td>
      <td>1730296</td>
      <td>8.8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>The Lord of the Rings: The Return of the King</td>
      <td>1709023</td>
      <td>8.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
grader.check("q3")
```

<br/><br/>
<hr style="border: 5px solid #003262;" />
<hr style="border: 1px solid #fdb515;" />

# Part 2: Election Donations in New York City

Finally, let's analyze the Federal Election Commission (FEC)'s public records. We connect to the database in two ways (using both Python and cell magic) so that we can flexibly explore the database.


```python
# important!!! run this cell and the next one
import sqlalchemy
# create a SQL Alchemy connection to the database
engine = sqlalchemy.create_engine("sqlite:///fec_nyc.db")
connection = engine.connect()
```


```python
%sql sqlite:///fec_nyc.db
```

## Table Descriptions

Run the below cell to explore the **schemas** of all tables saved in the database.

If you'd like, you can consult the below linked FEC pages for the descriptions of the tables themselves.

* `cand` ([link](https://www.fec.gov/campaign-finance-data/candidate-summary-file-description/)): Candidates table. Contains names and party affiliation.
* `comm` ([link](https://www.fec.gov/campaign-finance-data/committee-summary-file-description/)): Committees table. Contains committee names and types.
* `indiv_sample_nyc` ([link](https://www.fec.gov/campaign-finance-data/contributions-individuals-file-description)): All individual contributions from New York City .


```sql
%%sql
/* just run this cell */
SELECT sql FROM sqlite_master WHERE type='table';
```

     * sqlite:///fec_nyc.db
       sqlite:///imdbmini.db
    Done.
    




<table>
    <thead>
        <tr>
            <th>sql</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>CREATE TABLE &quot;cand&quot; (<br>    cand_id character varying(9),<br>    cand_name text,<br>    cand_pty_affiliation character varying(3),<br>    cand_election_yr integer,<br>    cand_office_st character varying(2),<br>    cand_office character(1),<br>    cand_office_district integer,<br>    cand_ici character(1),<br>    cand_status character(1),<br>    cand_pcc character varying(9),<br>    cand_st1 text,<br>    cand_st2 text,<br>    cand_city text,<br>    cand_st character varying(2),<br>    cand_zip character varying(10)<br>)</td>
        </tr>
        <tr>
            <td>CREATE TABLE &quot;comm&quot;(<br>  &quot;cmte_id&quot; TEXT,<br>  &quot;cmte_nm&quot; TEXT,<br>  &quot;tres_nm&quot; TEXT,<br>  &quot;cmte_st1&quot; TEXT,<br>  &quot;cmte_st2&quot; TEXT,<br>  &quot;cmte_city&quot; TEXT,<br>  &quot;cmte_st&quot; TEXT,<br>  &quot;cmte_zip&quot; TEXT,<br>  &quot;cmte_dsgn&quot; TEXT,<br>  &quot;cmte_tp&quot; TEXT,<br>  &quot;cmte_pty_affiliation&quot; TEXT,<br>  &quot;cmte_filing_freq&quot; TEXT,<br>  &quot;org_tp&quot; TEXT,<br>  &quot;connected_org_nm&quot; TEXT,<br>  &quot;cand_id&quot; TEXT<br>)</td>
        </tr>
        <tr>
            <td>CREATE TABLE indiv_sample_nyc (<br>    cmte_id character varying(9),<br>    amndt_ind character(1),<br>    rpt_tp character varying(3),<br>    transaction_pgi character(5),<br>    image_num bigint,<br>    transaction_tp character varying(3),<br>    entity_tp character varying(3),<br>    name text,<br>    city text,<br>    state character(2),<br>    zip_code character varying(12),<br>    employer text,<br>    occupation text,<br>    transaction_dt character varying(9),<br>    transaction_amt integer,<br>    other_id text,<br>    tran_id text,<br>    file_num bigint,<br>    memo_cd text,<br>    memo_text text,<br>    sub_id bigint<br>)</td>
        </tr>
    </tbody>
</table>



<br/><br/>

Let's look at the `indiv_sample_nyc` table. The below cell displays individual donations made by residents of the state of New York. We use `LIMIT 5` to avoid loading and displaying a huge table.


```sql
%%sql
/* just run this cell */
SELECT comm.cmte_id, cmte_nm, sum(transaction_amt) as total
FROM indiv_sample_nyc, comm
WHERE indiv_sample_nyc.cmte_id = comm.cmte_id AND name LIKE '%TRUMP%' AND name LIKE '%DONALD%'
GROUP BY cmte_nm
ORDER BY transaction_amt 
LIMIT 5;
```

     * sqlite:///fec_nyc.db
       sqlite:///imdbmini.db
    Done.
    




<table>
    <thead>
        <tr>
            <th>cmte_id</th>
            <th>cmte_nm</th>
            <th>total</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>C00608489</td>
            <td>GREAT AMERICA PAC</td>
            <td>75</td>
        </tr>
        <tr>
            <td>C00369033</td>
            <td>TEXANS FOR SENATOR JOHN CORNYN INC</td>
            <td>1000</td>
        </tr>
        <tr>
            <td>C00494229</td>
            <td>HELLER FOR SENATE</td>
            <td>2000</td>
        </tr>
        <tr>
            <td>C00554949</td>
            <td>FRIENDS OF DAVE BRAT INC.</td>
            <td>2600</td>
        </tr>
        <tr>
            <td>C00230482</td>
            <td>GRASSLEY COMMITTEE INC</td>
            <td>5200</td>
        </tr>
    </tbody>
</table>



You can write a SQL query to return the id and name of the first five candidates from the Democratic party, as below:


```sql
%%sql
/* just run this cell */
SELECT cand_id, cand_name
FROM cand
WHERE cand_pty_affiliation = 'DEM'
LIMIT 5;
```

     * sqlite:///fec_nyc.db
       sqlite:///imdbmini.db
    Done.
    




<table>
    <thead>
        <tr>
            <th>cand_id</th>
            <th>cand_name</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>H0AL05049</td>
            <td>CRAMER, ROBERT E &quot;BUD&quot; JR</td>
        </tr>
        <tr>
            <td>H0AL07086</td>
            <td>SEWELL, TERRYCINA ANDREA</td>
        </tr>
        <tr>
            <td>H0AL07094</td>
            <td>HILLIARD, EARL FREDERICK JR</td>
        </tr>
        <tr>
            <td>H0AR01091</td>
            <td>GREGORY, JAMES CHRISTOPHER</td>
        </tr>
        <tr>
            <td>H0AR01109</td>
            <td>CAUSEY, CHAD</td>
        </tr>
    </tbody>
</table>



<br/><br/>
<hr style="border: 1px solid #fdb515;" />

## [Tutorial] Matching Text with `LIKE`

First, let's look at 2016 election contributions made by Donald Trump, who was a New York (NY) resident during that year. The following SQL query returns the `cmte_id`, `transaction_amt`, and `name` for every contribution made by any donor with "DONALD" and "TRUMP" in their name in the `indiv_sample_nyc` table.

Notes:
* We use the `WHERE ... LIKE '...'` to match fields with text patterns. The `%` wildcard represents at least zero characters. Compare this to what you know from regex!
* We use `pd.read_sql` syntax here because we will do some EDA on the result `res`.


```python
# just run this cell
example_query = """
SELECT 
    cmte_id,
    transaction_amt,
    name
FROM indiv_sample_nyc
WHERE name LIKE '%TRUMP%' AND name LIKE '%DONALD%';
"""

example_res = pd.read_sql(example_query, engine)
example_res
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
      <th>cmte_id</th>
      <th>transaction_amt</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C00230482</td>
      <td>2600</td>
      <td>DONALD, TRUMP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C00230482</td>
      <td>2600</td>
      <td>DONALD, TRUMP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C00014498</td>
      <td>9000</td>
      <td>TRUMP, DONALD</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C00494229</td>
      <td>2000</td>
      <td>TRUMP, DONALD MR</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C00571869</td>
      <td>2700</td>
      <td>TRUMP, DONALD J.</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>152</th>
      <td>C00608489</td>
      <td>5</td>
      <td>DONALD J TRUMP FOR PRESIDENT INC</td>
    </tr>
    <tr>
      <th>153</th>
      <td>C00608489</td>
      <td>5</td>
      <td>DONALD J TRUMP FOR PRESIDENT INC</td>
    </tr>
    <tr>
      <th>154</th>
      <td>C00608489</td>
      <td>5</td>
      <td>DONALD J TRUMP FOR PRESIDENT INC</td>
    </tr>
    <tr>
      <th>155</th>
      <td>C00608489</td>
      <td>5</td>
      <td>DONALD J TRUMP FOR PRESIDENT INC</td>
    </tr>
    <tr>
      <th>156</th>
      <td>C00608489</td>
      <td>5</td>
      <td>DONALD J TRUMP FOR PRESIDENT INC</td>
    </tr>
  </tbody>
</table>
<p>157 rows × 3 columns</p>
</div>



If we look at the list above, it appears that some donations were not by Donald Trump himself, but instead by an entity called "DONALD J TRUMP FOR PRESIDENT INC". Fortunately, we see that our query only seems to have picked up one such anomalous name.


```python
# just run this cell
example_res['name'].value_counts()
```




    name
    TRUMP, DONALD J.                    133
    DONALD J TRUMP FOR PRESIDENT INC     15
    TRUMP, DONALD                         4
    DONALD, TRUMP                         2
    TRUMP, DONALD MR                      1
    TRUMP, DONALD J MR.                   1
    TRUMP, DONALD J MR                    1
    Name: count, dtype: int64



<br/><br/>

<hr style="border: 1px solid #fdb515;" />

## Question 4

Revise the above query so that the 15 anomalous donations made by "DONALD J TRUMP FOR PRESIDENT INC" do not appear. Your resulting table should have 142 rows. 

Hints:
* Consider using the above query as a starting point, or checking out the SQL query skeleton at the top of this lab. 
* The `NOT` keyword may also be useful here.



```sql
%%sql
/*
 * Code in this scratchwork cell is __not graded.__
 * Copy over any SQL queries you write here into the below Python cell.
 * Do __not__ insert any new cells in between the SQL/Python cells!
 * Doing so may break the autograder.
 */
-- Write below this comment. --
SELECT 
    cmte_id,
    transaction_amt,
    name
FROM indiv_sample_nyc
WHERE name LIKE '%TRUMP%' AND name LIKE '%DONALD%' AND name NOT LIKE '%DONALD J TRUMP FOR PRESIDENT INC%';
```


```python
query_q4 = """
SELECT 
    cmte_id,
    transaction_amt,
    name
FROM indiv_sample_nyc
WHERE name LIKE '%TRUMP%' AND name LIKE '%DONALD%' AND name NOT LIKE '%DONALD J TRUMP FOR PRESIDENT INC%';
"""

res_q4 = pd.read_sql(query_q4, engine)
res_q4
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
      <th>cmte_id</th>
      <th>transaction_amt</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C00230482</td>
      <td>2600</td>
      <td>DONALD, TRUMP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C00230482</td>
      <td>2600</td>
      <td>DONALD, TRUMP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C00014498</td>
      <td>9000</td>
      <td>TRUMP, DONALD</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C00494229</td>
      <td>2000</td>
      <td>TRUMP, DONALD MR</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C00571869</td>
      <td>2700</td>
      <td>TRUMP, DONALD J.</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>137</th>
      <td>C00580100</td>
      <td>9752</td>
      <td>TRUMP, DONALD J.</td>
    </tr>
    <tr>
      <th>138</th>
      <td>C00580100</td>
      <td>2574</td>
      <td>TRUMP, DONALD J.</td>
    </tr>
    <tr>
      <th>139</th>
      <td>C00580100</td>
      <td>23775</td>
      <td>TRUMP, DONALD J.</td>
    </tr>
    <tr>
      <th>140</th>
      <td>C00580100</td>
      <td>2000000</td>
      <td>TRUMP, DONALD J.</td>
    </tr>
    <tr>
      <th>141</th>
      <td>C00580100</td>
      <td>2574</td>
      <td>TRUMP, DONALD J.</td>
    </tr>
  </tbody>
</table>
<p>142 rows × 3 columns</p>
</div>




```python
grader.check("q4")
```




<p><strong><pre style='display: inline;'>q4</pre></strong> passed! ✨</p>



<br/><br/>
<hr style="border: 1px solid #fdb515;" />

## Question 5: `JOIN`ing Tables

Let's explore the other two tables in our database: `cand` and `comm`.

The `cand` table contains summary financial information about each candidate registered with the FEC or appearing on an official state ballot for House, Senate or President.


```sql
%%sql
/* just run this cell */
SELECT *
FROM indiv_sample_nyc
LIMIT 5;
```

     * sqlite:///fec_nyc.db
       sqlite:///imdbmini.db
    Done.
    




<table>
    <thead>
        <tr>
            <th>cmte_id</th>
            <th>amndt_ind</th>
            <th>rpt_tp</th>
            <th>transaction_pgi</th>
            <th>image_num</th>
            <th>transaction_tp</th>
            <th>entity_tp</th>
            <th>name</th>
            <th>city</th>
            <th>state</th>
            <th>zip_code</th>
            <th>employer</th>
            <th>occupation</th>
            <th>transaction_dt</th>
            <th>transaction_amt</th>
            <th>other_id</th>
            <th>tran_id</th>
            <th>file_num</th>
            <th>memo_cd</th>
            <th>memo_text</th>
            <th>sub_id</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>C00445015</td>
            <td>N</td>
            <td>Q1</td>
            <td>P    </td>
            <td>15951128130</td>
            <td>15</td>
            <td>IND</td>
            <td>SINGER, TRIPP MR.</td>
            <td>NEW YORK</td>
            <td>NY</td>
            <td>100214505</td>
            <td>ATLANTIC MAILBOXES, INC.</td>
            <td>OWNER</td>
            <td>01302015</td>
            <td>1000</td>
            <td></td>
            <td>A-CF13736</td>
            <td>1002485</td>
            <td></td>
            <td></td>
            <td>4041420151241812398</td>
        </tr>
        <tr>
            <td>C00510461</td>
            <td>N</td>
            <td>Q1</td>
            <td>P    </td>
            <td>15951129284</td>
            <td>15E</td>
            <td>IND</td>
            <td>SIMON, DANIEL A</td>
            <td>NEW YORK</td>
            <td>NY</td>
            <td>100237940</td>
            <td>N/A</td>
            <td>RETIRED</td>
            <td>03292015</td>
            <td>400</td>
            <td>C00401224</td>
            <td>VN8JBDDJBA8</td>
            <td>1002590</td>
            <td></td>
            <td>* EARMARKED CONTRIBUTION: SEE BELOW</td>
            <td>4041420151241813640</td>
        </tr>
        <tr>
            <td>C00422410</td>
            <td>N</td>
            <td>Q1</td>
            <td>P    </td>
            <td>15970352211</td>
            <td>15</td>
            <td>IND</td>
            <td>ABDUL RAUF, FEISAL</td>
            <td>NEW YORK</td>
            <td>NY</td>
            <td>101150010</td>
            <td>THE CORDOBA INITIATIVE</td>
            <td>CHAIRMAN</td>
            <td>03042015</td>
            <td>250</td>
            <td></td>
            <td>VN8A3DBSYG6</td>
            <td>1003643</td>
            <td></td>
            <td></td>
            <td>4041620151241914560</td>
        </tr>
        <tr>
            <td>C00510461</td>
            <td>N</td>
            <td>Q1</td>
            <td>P    </td>
            <td>15951129280</td>
            <td>15</td>
            <td>IND</td>
            <td>SCHWARZER, FRANK</td>
            <td>NEW YORK</td>
            <td>NY</td>
            <td>100145135</td>
            <td>METRO HYDRAULIC JACK CO</td>
            <td>SALES</td>
            <td>01162015</td>
            <td>100</td>
            <td></td>
            <td>VN8JBDAP4C4</td>
            <td>1002590</td>
            <td></td>
            <td>* EARMARKED CONTRIBUTION: SEE BELOW</td>
            <td>4041420151241813630</td>
        </tr>
        <tr>
            <td>C00510461</td>
            <td>N</td>
            <td>Q1</td>
            <td>P    </td>
            <td>15951129281</td>
            <td>15</td>
            <td>IND</td>
            <td>SCHWARZER, FRANK</td>
            <td>NEW YORK</td>
            <td>NY</td>
            <td>100145135</td>
            <td>METRO HYDRAULIC JACK CO</td>
            <td>SALES</td>
            <td>02162015</td>
            <td>100</td>
            <td></td>
            <td>VN8JBDBRDG3</td>
            <td>1002590</td>
            <td></td>
            <td>* EARMARKED CONTRIBUTION: SEE BELOW</td>
            <td>4041420151241813632</td>
        </tr>
    </tbody>
</table>



The `comm` table contains summary financial information about each committee registered with the FEC. Committees are organizations that spend money for political action or parties, or spend money for or against political candidates.


```sql
%%sql
/* just run this cell */
SELECT *
FROM comm
LIMIT 5;
```

     * sqlite:///fec_nyc.db
       sqlite:///imdbmini.db
    Done.
    




<table>
    <thead>
        <tr>
            <th>cmte_id</th>
            <th>cmte_nm</th>
            <th>tres_nm</th>
            <th>cmte_st1</th>
            <th>cmte_st2</th>
            <th>cmte_city</th>
            <th>cmte_st</th>
            <th>cmte_zip</th>
            <th>cmte_dsgn</th>
            <th>cmte_tp</th>
            <th>cmte_pty_affiliation</th>
            <th>cmte_filing_freq</th>
            <th>org_tp</th>
            <th>connected_org_nm</th>
            <th>cand_id</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>C00000059</td>
            <td>HALLMARK CARDS PAC</td>
            <td>ERIN BROWER</td>
            <td>2501 MCGEE</td>
            <td>MD#288</td>
            <td>KANSAS CITY</td>
            <td>MO</td>
            <td>64108</td>
            <td>U</td>
            <td>Q</td>
            <td>UNK</td>
            <td>M</td>
            <td>C</td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>C00000422</td>
            <td>AMERICAN MEDICAL ASSOCIATION POLITICAL ACTION COMMITTEE</td>
            <td>WALKER, KEVIN</td>
            <td>25 MASSACHUSETTS AVE, NW</td>
            <td>SUITE 600</td>
            <td>WASHINGTON</td>
            <td>DC</td>
            <td>20001</td>
            <td>B</td>
            <td>Q</td>
            <td></td>
            <td>M</td>
            <td>M</td>
            <td>AMERICAN MEDICAL ASSOCIATION</td>
            <td></td>
        </tr>
        <tr>
            <td>C00000489</td>
            <td>D R I V E POLITICAL FUND CHAPTER 886</td>
            <td>TOM RITTER</td>
            <td>3528 W RENO</td>
            <td></td>
            <td>OKLAHOMA CITY</td>
            <td>OK</td>
            <td>73107</td>
            <td>U</td>
            <td>N</td>
            <td></td>
            <td>Q</td>
            <td>L</td>
            <td>TEAMSTERS LOCAL UNION 886</td>
            <td></td>
        </tr>
        <tr>
            <td>C00000547</td>
            <td>KANSAS MEDICAL SOCIETY POLITICAL ACTION COMMITTEE</td>
            <td>C. RICHARD BONEBRAKE, M.D.</td>
            <td>623 SW 10TH AVE</td>
            <td></td>
            <td>TOPEKA</td>
            <td>KS</td>
            <td>66612</td>
            <td>U</td>
            <td>Q</td>
            <td>UNK</td>
            <td>Q</td>
            <td>T</td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>C00000638</td>
            <td>INDIANA STATE MEDICAL ASSOCIATION POLITICAL ACTION COMMITTEE</td>
            <td>VIDYA KORA, M.D.</td>
            <td>322 CANAL WALK, CANAL LEVEL</td>
            <td></td>
            <td>INDIANAPOLIS</td>
            <td>IN</td>
            <td>46202</td>
            <td>U</td>
            <td>Q</td>
            <td></td>
            <td>Q</td>
            <td>M</td>
            <td></td>
            <td></td>
        </tr>
    </tbody>
</table>



---

### Question 5a

Notice that both the `cand` and `comm` tables have a `cand_id` column. Let's try joining these two tables on this column to print out committee information for candidates.

List the first 5 candidate names (`cand_name`) in reverse lexicographic order by `cand_name`, along with their corresponding committee names. **Only select rows that have a matching `cand_id` in both tables.**

Your output should look similar to the following:

|    |cand_name|cmte_nm|
|----|----|----|
|**0**|ZUTLER, DANIEL PAUL MR|CITIZENS TO ELECT DANIEL P ZUTLER FOR PRESIDENT|
|**1**|ZUMWALT, JAMES|ZUMWALT FOR CONGRESS|
|**...**|...|...|

Consider starting from the following query skeleton, which uses the `AS` keyword to rename the `cand` and `comm` tables to `c1` and `c2`, respectively.
Which join is most appropriate?

    SELECT ...
    FROM cand AS c1
        [INNER | {LEFT |RIGHT | FULL } {OUTER}] JOIN comm AS c2
        ON ...
    ...
    ...;



```sql
%%sql
/*
 * Code in this scratchwork cell is __not graded.__
 * Copy over any SQL queries you write here into the below Python cell.
 * Do __not__ insert any new cells in between the SQL/Python cells!
 * Doing so may break the autograder.
 */
-- Write below this comment. --
    SELECT cand_name, cmte_nm
    FROM cand AS c1
        INNER JOIN comm AS c2
        ON c1.cand_id = c2.cand_id
    ORDER BY cand_name DESC;
```


```python
query_q5a = """
    SELECT cand_name, cmte_nm
    FROM cand AS c1
        INNER JOIN comm AS c2
        ON c1.cand_id = c2.cand_id
    ORDER BY cand_name DESC
    LIMIT 5;
"""

res_q5a = pd.read_sql(query_q5a, engine)
res_q5a
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
      <th>cand_name</th>
      <th>cmte_nm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ZUTLER, DANIEL PAUL MR</td>
      <td>CITIZENS TO ELECT DANIEL P ZUTLER FOR PRESIDENT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ZUMWALT, JAMES</td>
      <td>ZUMWALT FOR CONGRESS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ZUKOWSKI, ANDREW GEORGE</td>
      <td>ZUKOWSKI FOR CONGRESS</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ZUCCOLO, JOE</td>
      <td>JOE ZUCCOLO FOR CONGRESS</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ZORN, ROBERT ERWIN</td>
      <td>CONSTITUTIONAL COMMITTEE</td>
    </tr>
  </tbody>
</table>
</div>




```python
grader.check("q5a")
```




<p><strong><pre style='display: inline;'>q5a</pre></strong> passed! 🎉</p>



<br/><br/>

---

### Question 5b

Suppose we modify the query from the previous part to include *all* candidates, **including those that don't have a committee.**


List the first 5 candidate names (`cand_name`) in reverse lexicographic order by `cand_name`, along with their corresponding committee names. If the candidate has no committee in the `comm` table, then `cmte_nm` should be NULL (or None in the Python representation).

Your output should look similar to the following:

|    |cand_name|cmte_nm|
|----|----|----|
|**0**|ZUTLER, DANIEL PAUL MR|CITIZENS TO ELECT DANIEL P ZUTLER FOR PRESIDENT|
|**...**|...|...|
|**4**|ZORNOW, TODD MR|None|

Hint: Start from the same query skeleton as the previous part. 
Which join is most appropriate?


```sql
%%sql
/*
 * Code in this scratchwork cell is __not graded.__
 * Copy over any SQL queries you write here into the below Python cell.
 * Do __not__ insert any new cells in between the SQL/Python cells!
 * Doing so may break the autograder.
 */
-- Write below this comment. --
    SELECT cand_name, cmte_nm
    FROM cand AS c1
        LEFT JOIN comm AS c2
        ON c1.cand_id = c2.cand_id
    ORDER BY cand_name DESC
    LIMIT 5;
```

     * sqlite:///fec_nyc.db
       sqlite:///imdbmini.db
    Done.
    




<table>
    <thead>
        <tr>
            <th>cand_name</th>
            <th>cmte_nm</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>ZUTLER, DANIEL PAUL MR</td>
            <td>CITIZENS TO ELECT DANIEL P ZUTLER FOR PRESIDENT</td>
        </tr>
        <tr>
            <td>ZUMWALT, JAMES</td>
            <td>ZUMWALT FOR CONGRESS</td>
        </tr>
        <tr>
            <td>ZUKOWSKI, ANDREW GEORGE</td>
            <td>ZUKOWSKI FOR CONGRESS</td>
        </tr>
        <tr>
            <td>ZUCCOLO, JOE</td>
            <td>JOE ZUCCOLO FOR CONGRESS</td>
        </tr>
        <tr>
            <td>ZORNOW, TODD MR</td>
            <td>None</td>
        </tr>
    </tbody>
</table>




```python
query_q5b = """
    SELECT cand_name, cmte_nm
    FROM cand AS c1
        LEFT JOIN comm AS c2
        ON c1.cand_id = c2.cand_id
    ORDER BY cand_name DESC
    LIMIT 5;
"""

res_q5b = pd.read_sql(query_q5b, engine)
res_q5b
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
      <th>cand_name</th>
      <th>cmte_nm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ZUTLER, DANIEL PAUL MR</td>
      <td>CITIZENS TO ELECT DANIEL P ZUTLER FOR PRESIDENT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ZUMWALT, JAMES</td>
      <td>ZUMWALT FOR CONGRESS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ZUKOWSKI, ANDREW GEORGE</td>
      <td>ZUKOWSKI FOR CONGRESS</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ZUCCOLO, JOE</td>
      <td>JOE ZUCCOLO FOR CONGRESS</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ZORNOW, TODD MR</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>




```python
grader.check("q5b")
```




<p><strong><pre style='display: inline;'>q5b</pre></strong> passed! 🚀</p>



<br/><br/>
<hr style="border: 1px solid #fdb515;" />

## Question 6: Subqueries and Grouping

If we return to our results from Question 4, we see that many of the contributions were to the same committee:


```python
# Your SQL query result from Question 4
# reprinted for your convenience
res_q4
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
      <th>cmte_id</th>
      <th>transaction_amt</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C00230482</td>
      <td>2600</td>
      <td>DONALD, TRUMP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C00230482</td>
      <td>2600</td>
      <td>DONALD, TRUMP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C00014498</td>
      <td>9000</td>
      <td>TRUMP, DONALD</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C00494229</td>
      <td>2000</td>
      <td>TRUMP, DONALD MR</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C00571869</td>
      <td>2700</td>
      <td>TRUMP, DONALD J.</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>137</th>
      <td>C00580100</td>
      <td>9752</td>
      <td>TRUMP, DONALD J.</td>
    </tr>
    <tr>
      <th>138</th>
      <td>C00580100</td>
      <td>2574</td>
      <td>TRUMP, DONALD J.</td>
    </tr>
    <tr>
      <th>139</th>
      <td>C00580100</td>
      <td>23775</td>
      <td>TRUMP, DONALD J.</td>
    </tr>
    <tr>
      <th>140</th>
      <td>C00580100</td>
      <td>2000000</td>
      <td>TRUMP, DONALD J.</td>
    </tr>
    <tr>
      <th>141</th>
      <td>C00580100</td>
      <td>2574</td>
      <td>TRUMP, DONALD J.</td>
    </tr>
  </tbody>
</table>
<p>142 rows × 3 columns</p>
</div>



<br/>

---

Create a new SQL query that returns the total amount that Donald Trump contributed to each committee.

Your table should have four columns: `cmte_id`, `total_amount` (total amount contributed to that committee), `num_donations` (total number of donations), and `cmte_nm` (name of the committee). Your table should be sorted in **decreasing order** of `total_amount`.

**This is a hard question!** Don't be afraid to reference the lecture slides, or the overall SQL query skeleton at the top of this lab.

Here are some other hints:

* Note that committee names are not available in `indiv_sample_nyc`, so you will have to obtain information somehow from the `comm` table (perhaps a `JOIN` would be useful).
* Remember that you can compute summary statistics after grouping by using aggregates like `COUNT(*)`, `SUM()` as output fields.
* A **subquery** may be useful to break your question down into subparts. Consider the following query skeleton, which uses the `WITH` operator to store a subquery's results in a temporary table named `donations`.

        WITH donations AS (
            SELECT ...
            ...
        )
        SELECT ...
        FROM donations
        GROUP BY ...
        ORDER BY ...;


```sql
%%sql
/* just run this cell */
SELECT comm.cmte_id, sum(transaction_amt) as total_amount, COUNT(*) as num_donations, cmte_nm
FROM indiv_sample_nyc, comm
WHERE indiv_sample_nyc.cmte_id = comm.cmte_id AND name LIKE '%TRUMP%' AND name LIKE '%DONALD%'
GROUP BY cmte_nm
ORDER BY total_amount DESC
LIMIT 10;
```

     * sqlite:///fec_nyc.db
       sqlite:///imdbmini.db
    Done.
    




<table>
    <thead>
        <tr>
            <th>cmte_id</th>
            <th>total_amount</th>
            <th>num_donations</th>
            <th>cmte_nm</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>C00580100</td>
            <td>18633157</td>
            <td>131</td>
            <td>DONALD J. TRUMP FOR PRESIDENT, INC.</td>
        </tr>
        <tr>
            <td>C00055582</td>
            <td>10000</td>
            <td>1</td>
            <td>NY REPUBLICAN FEDERAL CAMPAIGN COMMITTEE</td>
        </tr>
        <tr>
            <td>C00014498</td>
            <td>9000</td>
            <td>1</td>
            <td>REPUBLICAN PARTY OF IOWA</td>
        </tr>
        <tr>
            <td>C00571869</td>
            <td>5400</td>
            <td>2</td>
            <td>DONOVAN FOR CONGRESS</td>
        </tr>
        <tr>
            <td>C00230482</td>
            <td>5200</td>
            <td>2</td>
            <td>GRASSLEY COMMITTEE INC</td>
        </tr>
        <tr>
            <td>C00034033</td>
            <td>5000</td>
            <td>1</td>
            <td>SOUTH CAROLINA REPUBLICAN PARTY</td>
        </tr>
        <tr>
            <td>C00136457</td>
            <td>5000</td>
            <td>1</td>
            <td>NEW HAMPSHIRE REPUBLICAN STATE COMMITTEE</td>
        </tr>
        <tr>
            <td>C00554949</td>
            <td>2600</td>
            <td>1</td>
            <td>FRIENDS OF DAVE BRAT INC.</td>
        </tr>
        <tr>
            <td>C00494229</td>
            <td>2000</td>
            <td>1</td>
            <td>HELLER FOR SENATE</td>
        </tr>
        <tr>
            <td>C00369033</td>
            <td>1000</td>
            <td>1</td>
            <td>TEXANS FOR SENATOR JOHN CORNYN INC</td>
        </tr>
    </tbody>
</table>




```sql
%%sql
/*
 * Code in this scratchwork cell is __not graded.__
 * Copy over any SQL queries you write here into the below Python cell.
 * Do __not__ insert any new cells in between the SQL/Python cells!
 * Doing so may break the autograder.
 */
-- Write below this comment. --
        WITH donations AS (
            SELECT c1.cmte_id,transaction_amt, cmte_nm
            FROM indiv_sample_nyc AS c1, comm AS c2
            WHERE c1.cmte_id = c2.cmte_id AND name LIKE '%TRUMP%' AND name LIKE '%DONALD%' AND transaction_amt > 0
        )
        SELECT cmte_id, SUM(transaction_amt) AS total_amount, count(*) AS num_donations, cmte_nm
        FROM donations
        GROUP BY cmte_nm
        ORDER BY total_amount
        LIMIT 10;
```

     * sqlite:///fec_nyc.db
       sqlite:///imdbmini.db
    Done.
    




<table>
    <thead>
        <tr>
            <th>cmte_id</th>
            <th>total_amount</th>
            <th>num_donations</th>
            <th>cmte_nm</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>C00608489</td>
            <td>75</td>
            <td>15</td>
            <td>GREAT AMERICA PAC</td>
        </tr>
        <tr>
            <td>C00369033</td>
            <td>1000</td>
            <td>1</td>
            <td>TEXANS FOR SENATOR JOHN CORNYN INC</td>
        </tr>
        <tr>
            <td>C00494229</td>
            <td>2000</td>
            <td>1</td>
            <td>HELLER FOR SENATE</td>
        </tr>
        <tr>
            <td>C00554949</td>
            <td>2600</td>
            <td>1</td>
            <td>FRIENDS OF DAVE BRAT INC.</td>
        </tr>
        <tr>
            <td>C00136457</td>
            <td>5000</td>
            <td>1</td>
            <td>NEW HAMPSHIRE REPUBLICAN STATE COMMITTEE</td>
        </tr>
        <tr>
            <td>C00034033</td>
            <td>5000</td>
            <td>1</td>
            <td>SOUTH CAROLINA REPUBLICAN PARTY</td>
        </tr>
        <tr>
            <td>C00230482</td>
            <td>5200</td>
            <td>2</td>
            <td>GRASSLEY COMMITTEE INC</td>
        </tr>
        <tr>
            <td>C00571869</td>
            <td>5400</td>
            <td>2</td>
            <td>DONOVAN FOR CONGRESS</td>
        </tr>
        <tr>
            <td>C00014498</td>
            <td>9000</td>
            <td>1</td>
            <td>REPUBLICAN PARTY OF IOWA</td>
        </tr>
        <tr>
            <td>C00055582</td>
            <td>10000</td>
            <td>1</td>
            <td>NY REPUBLICAN FEDERAL CAMPAIGN COMMITTEE</td>
        </tr>
    </tbody>
</table>




```python
query_q6 = """
SELECT comm.cmte_id, sum(transaction_amt) as total_amount, COUNT(*) as num_donations, cmte_nm
FROM indiv_sample_nyc, comm
WHERE indiv_sample_nyc.cmte_id = comm.cmte_id AND name LIKE '%TRUMP%' AND name LIKE '%DONALD%'
GROUP BY cmte_nm
ORDER BY total_amount DESC
LIMIT 10;
"""


res_q6 = pd.read_sql(query_q6, engine)
res_q6
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
      <th>cmte_id</th>
      <th>total_amount</th>
      <th>num_donations</th>
      <th>cmte_nm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C00580100</td>
      <td>18633157</td>
      <td>131</td>
      <td>DONALD J. TRUMP FOR PRESIDENT, INC.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C00055582</td>
      <td>10000</td>
      <td>1</td>
      <td>NY REPUBLICAN FEDERAL CAMPAIGN COMMITTEE</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C00014498</td>
      <td>9000</td>
      <td>1</td>
      <td>REPUBLICAN PARTY OF IOWA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C00571869</td>
      <td>5400</td>
      <td>2</td>
      <td>DONOVAN FOR CONGRESS</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C00230482</td>
      <td>5200</td>
      <td>2</td>
      <td>GRASSLEY COMMITTEE INC</td>
    </tr>
    <tr>
      <th>5</th>
      <td>C00034033</td>
      <td>5000</td>
      <td>1</td>
      <td>SOUTH CAROLINA REPUBLICAN PARTY</td>
    </tr>
    <tr>
      <th>6</th>
      <td>C00136457</td>
      <td>5000</td>
      <td>1</td>
      <td>NEW HAMPSHIRE REPUBLICAN STATE COMMITTEE</td>
    </tr>
    <tr>
      <th>7</th>
      <td>C00554949</td>
      <td>2600</td>
      <td>1</td>
      <td>FRIENDS OF DAVE BRAT INC.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>C00494229</td>
      <td>2000</td>
      <td>1</td>
      <td>HELLER FOR SENATE</td>
    </tr>
    <tr>
      <th>9</th>
      <td>C00369033</td>
      <td>1000</td>
      <td>1</td>
      <td>TEXANS FOR SENATOR JOHN CORNYN INC</td>
    </tr>
  </tbody>
</table>
</div>




```python
grader.check("q6")
```




<p><strong><pre style='display: inline;'>q6</pre></strong> passed! 🎉</p>



# Congratulations! You finished the lab!

---



