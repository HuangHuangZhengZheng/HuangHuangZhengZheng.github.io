# L17-huge GNN

# Scaling up GNNs

![alt text](image.png)

直接load全部nodes又不太可能【naive approach】4090 / A100带不动

## neighbor sampling

![alt text](image-1.png)
对hub node的思考
![alt text](image-2.png)

![alt text](image-3.png)
see the paper 

## cluster-GCN

![alt text](image-4.png)

![alt text](image-5.png)

![alt text](image-6.png)

advanced
![alt text](image-7.png)


## Simplified GCN

舍弃了GCN的non-linearity，直接用linear layer

同质性？但是我想知道和glidar的区别？
![alt text](image-8.png)

![alt text](image-9.png)


