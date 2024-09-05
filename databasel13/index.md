# CS186-L13: Transactions & Concurrency I


![alt text](image.png) :tada:

## Intro 
transaction's principle ACID
![alt text](image-1.png)

### Isolation (Concurrency)
![alt text](image-2.png)
however, do not consider serial execution :sweat_smile:
### Atomicity and Durability
![alt text](image-3.png)
### Consistency
![alt text](image-4.png)

## Concurrency Control
基本符号表达
![alt text](image-5.png)


序列等价性：
- $Def1:$ **Serial Schedule**
  - each transaction executes in a serial order, one after the other, without any intervening
- $Def2:$ schedules **Equivalent**
  - involve same transaction
  - each transaction's actions are the same order
  - both transactions have the same effect on the database's final state
- $Def3:$ **Serializable**
  - if a schedule is serializable, then it is equivalent to some serial schedule

### Conflict Serializability
#### conflict operations?
![alt text](image-6.png)
![alt text](image-7.png)
**Intuitive Understanding of Conflict Serializable**
![alt text](image-8.png)

**Conflict Dependency Graph**
![alt text](image-9.png)


### View Serializability
![alt text](image-10.png)

## Conclusion
Neither definition allows all schedules that are actually serializable.

because they can not check the meaning of the operation :smiling_imp:
