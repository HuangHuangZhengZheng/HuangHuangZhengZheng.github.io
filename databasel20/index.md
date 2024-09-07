# CS186-L20: NoSQL


## Scaling Relational Databases isn't always the best option
including *partitioning* and *replication*

BUT, consistency is hard to enforce!

![alt text](image.png)

## Taxonomy of NoSQL Data Models
### Key-Value Stores
```java
Map<Key, Value>
get/put
```
Distribution / Partitioning, just using hash function
- if no replication, key k is stored on $h(k)$ node
- if multi-way replication, key k is stored on $h_i(k), i=1,2,...,n$ nodes

![alt text](image-1.png)

### Extensible Record Stores
![alt text](image-2.png)
the idea is that do not use a simple key to lookup :thinking:

### Document Stores
#### JSON Documents
using JSON as example
![alt text](image-3.png)

![alt text](image-4.png)

![alt text](image-5.png)
do not store replicated key!

JSON is a Tree :wood:, Self-describing :speech_balloon:, and Flexible :fire:

can store Json in RDBMS
```sql
SELECT # FROM people
WHERE person @> '{"name": "John Doe", "age": 30}';
```

#### mapping between JSON and Relational Data
Relational Data Model ===> JSON Document
easy, note that replicated key can be handled by using a array [  ]

JSON Document ===> Relational Data Model
- using NULL to represent missing values
- nested or replicated data? hard to handle! **multi-tables** may help :thinking:

![alt text](image-6.png)

## Introduction to MongoDB
![alt text](image-7.png)
基本语法
### select and find
![alt text](image-8.png)
![alt text](image-9.png)
```mongo
db.collection.find(<predicate>, optional<projection>) 
db.inventory.find({}) // return all documents
```
![alt text](image-10.png)
![alt text](image-11.png)

### Limit and sort
![alt text](image-12.png)


## MQL Aggregations and Updates
![alt text](image-13.png)
![alt text](image-14.png)

### unwind
![alt text](image-15.png)

### update
![alt text](image-16.png)
![alt text](image-17.png)

## MongoDB internals
![alt text](image-18.png)
