# CS186-L19:  Distributed Transactions


distributed == parallel shared nothing architecture

## Intro
![alt text](image.png)

## Distributed Locking
each nodes has lock table locally, can manage the pages/tuples easily, but when it comes to Table, there should be a global lock table （or distributed lock tables）and a *coordinator* to manage the access to the table.

## Distributed Deadlocks Detection
![alt text](image-1.png)
合并全局waits
![alt text](image-2.png)

## Distributed Commit
全局投票
![alt text](image-3.png)
### 2PC
![alt text](image-4.png)
![alt text](image-5.png)

## The Recovery Processes
crash situations
![alt text](image-7.png)
![alt text](image-6.png)

##  2PC, Locking and Availability

2PC + Strict 2PL locking
![alt text](image-8.png)

what if a node is down? some locks can still be held by other nodes......
![alt text](image-9.png)

## Summary
![alt text](image-10.png)
