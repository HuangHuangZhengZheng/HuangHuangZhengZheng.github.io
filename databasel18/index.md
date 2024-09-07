# CS186-L18: Parallel Query Processing


## Intro to Parallelism
![alt text](image.png)
## Architectures and Software Structures
![alt text](image-1.png)
we will focus on the shared-nothing here :yum:

## Kinds of Query Parallelism
![alt text](image-3.png)
side note:
- intra: single
- inter: multiple at the same level

## Parallel Data Acceess
### Data Partitioning across Machines
![alt text](image-4.png)
Round robin means that each machine haves the same shuffled data
### parallel scans
scan and merge

$\sigma_p$ : an operator that skip entire sites that have no matching tuples in *range or hash partitioning*

### lookup by key
if data partitioned on function of key, then Route lookup only to the relevant nodes

otherwise, broadcast lookup to all nodes

### insert
if on function of key, insert only to the relevant nodes

else insert to any nodes

insert an unique key seems to be same

### parallel hash join
#### naive hash join
![alt text](image-5.png)

#### grace hash join
Pass one is like hashing above, but do it 2x-- once for each relation being joined

Pass two is local grace hash join per node

![alt text](image-6.png)

## sort-merge join
![alt text](image-7.png)
回到均分问题了

然后和上面一样读取分配两次for join

## parallel aggregation/grouping
![alt text](image-8.png)
naive group by:
![alt text](image-9.png)

## Symmetric Hash Joins
sort and hash can break the pipeline......
![alt text](image-10.png)

## one-sided and Broadcast Joins
### one-sided joins
one is sorted/hashed
![alt text](image-11.png)
### broadcast joins
one is small
![alt text](image-12.png)
