# CS186-L21: MapReduce and Spark



## Motivation
only scaling up relational databases is challenging :s

## MapReduce Data and Programming Model
Target
![alt text](image.png)

### Map phase
![alt text](image-1.png)
map function will not keep the state of the intermediate results, so it can be parallelized easily

### Reduce phase
![alt text](image-2.png)
for example, wanna count the number of occurrences of each word in the input data, we can use the reduce function to sum up the values of the same key
![alt text](image-3.png)

## Implementation of MapReduce
### fault tolerance
by writing intermediate results to disk...
- mappers can write their output to local disk
- reducers can read the output of mappers from local disk and combine them, if the reduce task is restarted, the reduce task is restarted on another server

### implementation
![alt text](image-4.png)
how to handle the stragglers?
![alt text](image-5.png)

## Implementing Relational Operators
![alt text](image-6.png)

![alt text](image-7.png)

![alt text](image-8.png)

![alt text](image-9.png)

## Introduction to Spark
why MR sucks?
- hard to write more complex queries
- slow for writing all intermediate results to disk


![alt text](image-10.png)

![alt text](image-11.png)

## Programming in Spark
collections in spark
![alt text](image-12.png)

```java
JavaSparkContext s = SparkSession.builder().appName("MyApp").getOrCreate();
JavaRDD<String> lines = s.read().textFile("input.txt");
JavaRDD<String> errors = lines.filter(line -> line.contains("error")); // lazy
errors.collect() // eager
```

similar steps in spark and MR
![alt text](image-13.png)

### Persistence
![alt text](image-14.png)
API in Java
![alt text](image-15.png)

## Spark 2.0
has DataFrame API :astonished:

and have Datasets API :astonished:

![alt text](image-16.png)

like DATA100 python!
