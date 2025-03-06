# L7-GNN t1

# L7-GNN theory 1

所有gnn的原型在这里

1. 一个layer
![alt text](image.png)

2. layers之间的交互
![alt text](image-1.png)

3. input graph -> computational graph的构建
![alt text](image-2.png)

GO!!!!
## a simple layer of GNN

![alt text](image-3.png)

1. msg computations
原神处理dddd

2. aggregation
核心：order invariance 焯这就是pointnet的核心之一啊！！！

issue： 容易忽略自己节点
![alt text](image-4.png)


### some examples of GNNs
#### GCN
![alt text](image-5.png)

#### GraphSAGE
![alt text](image-6.png)

![alt text](image-7.png)
这里的细节在于如何聚合邻居的信息

同时每层来个norm

#### GAT
![alt text](image-8.png)
首先解释一下什么是attention mechanism
![alt text](image-9.png)

![alt text](image-10.png)
单个attention的计算
![alt text](image-11.png)
多头attention的计算（更容易收敛）
![alt text](image-12.png)


![alt text](image-13.png)

### GNN layers in practice
glidar呼之欲出了 :scream:
![alt text](image-14.png)




有意思的东西:yum:
![alt text](image-15.png)

## stacking GNN layers






