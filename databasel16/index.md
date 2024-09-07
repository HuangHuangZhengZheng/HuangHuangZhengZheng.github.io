# CS186-L16: DB Design: FDs and Normalization


## Functional Dependencies
big picture
![alt text](image.png)
### Def

- **X -> Y** means X determines Y, X and Y can be a single column or multiple columns
- **F+** means that to be the set of all FDs that are implied by F
### terminology
![alt text](image-1.png)

## Anomalies
可以用FD分解relation从而避免冗余
![alt text](image-2.png)

## Armstrongs Axioms
![alt text](image-4.png)

## Attribute Closure
wanna check if X->Y is in F+
![alt text](image-5.png)

## BCNF and other Normal Forms
### Basic Normal Form
NF is a def of data model!
![alt text](image-6.png)

### Boyce-Codd Normal Form
![alt text](image-7.png)


## Lossless Join Decompositions
Def: decomposition won't create new attributes, and will cover the original attributes (不是完全无重叠分割)

### Problems with Decompositions
- can ***loss*** info and unable to reconstruct the original data
    - do not loss data actually, in fact, we gain some dirty data
    - ![alt text](image-8.png)
- Dependency check may require *joins*
- some queries may be more expensive, since *join* is required

### Lossless Join Decompositions
定义
![alt text](image-9.png)
定理
![alt text](image-10.png)

## Dependency Preservation and BCNF Decomposition
Def: Projection of set of FDs F:
![alt text](image-11.png)
Def: Dependency Preserving Decomposition
![alt text](image-12.png)

### BCNF Decomposition
![alt text](image-13.png)
###### 没有听懂
但是dependency没有保留
![alt text](image-14.png)
所以BCNF可以lossless，但是不一定保留所有的dependency

