# CS186-L17: Recovery


## Need for Atomicity and Durability, SQL support for Transactions
![alt text](image.png)
![alt text](image-1.png)

## Strawman Solution
![alt text](image-2.png)
**No Steal/Force policy**

seem like no a good choice for recovery
- not scalable in buffer
- if crash in 2a, inconsistencies will occur

## STEAL / NO FORCE, UNDO and REDO

### STEAL/NO FORCE
- no force: 
  - problem: sys crash before dirty page of a committed transaction is written to disk
  - solution: flush as little as possible, in a convenient space, prior to commit. allows REDOing modifications

- STEAL:
  - must remember the old value of flushed pages to support ***undo***

### pattern
![alt text](image-3.png)

## Intro to Write-Ahead Logging (WAL)
- Log: a ordered list of log records to allow redo/undo
  - log records: **<XID, pageID, offset, length, old data, new data>**
  - and other info

- Write-Ahead Logging (WAL):
- 1. force the log record before the data page is written to disk
- 2. force all log records before a transaction is committed
- #1 with UNDO guarantee Atomicity and #2 with REDO guarantee Durability ===> steal/no force implementation

对于每个log record，有一个对应的Log Sequence Number (LSN)来标识它在日志中的位置，我们对最近（lately）的LSN们感兴趣，flushedLSN, pageLSN等等
![alt text](image-4.png)
## Undo logging
Rule:
![alt text](image-5.png)
和WAL有点不一样，注意U2，COMMIT放在最后！
![alt text](image-6.png)
presudo code:
![alt text](image-7.png)

## Redo logging
No steal/no force
![alt text](image-8.png)
from beginning to end, redo all log records that are committed

incomplete? do nothing!

两者对比
![alt text](image-9.png)
## ARIES logging
Log records format belike:
![alt text](image-10.png)


## ARIES and Checkpointing
## ARIES logging during normal execution

