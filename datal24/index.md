# DATA100-L24: Clustering


# introduction to clustering
![alt text](image.png)
![alt text](image-1.png)
no label at all :cry:

# K-means clustering
[算法动画演示](https://docs.google.com/presentation/d/1qYThwhMXKjCH390AQ29Ob27bUDjV5DWBTmGEzby-Bto/edit#slide=id.p)

K-Means vs KNN
![alt text](image-2.png)

# minimizing inertia
convex?? 损失函数不一定凸，梯度下降难顶  how to see which one is better :question:
![alt text](image-3.png)
![alt text](image-4.png)
但是找到全局最优解非常困难
![alt text](image-5.png)
# agglomerative clustering
演示见上面链接以及lec code！

和CS61B的minimum spanning tree类似，每次合并两个最近的点，直到终止条件

![alt text](image-6.png)
![alt text](image-7.png)
![alt text](image-8.png)
outlier 有时忽略处理或者自成一类
# picking K
![alt text](image-9.png)

![alt text](image-10.png)
Smax？
![alt text](image-11.png)

can s be negative?
![alt text](image-12.png)
![alt text](image-14.png)
![alt text](image-13.png)


## summary
![alt text](image-15.png)
