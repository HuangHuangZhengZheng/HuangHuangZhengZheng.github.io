# L3-Node Embeddings

# Node Embeddings
https://web.stanford.edu/class/cs224w/slides/02-nodeemb.pdf

![alt text](image.png)

## encoder and decoder
![alt text](image-1.png)

### encoder: simple example

![alt text](image-2.png)
？？注意这里矩阵是one column per node， 这里似乎解释通了为什么glidar里面node在encode的过程中数量不变，换句话说就是 **not scalable**



呼之欲出啊啊啊啊啊 :scream:

***以下内容非常具有启发性***

![alt text](image-3.png)


![alt text](image-4.png)

![alt text](image-5.png)


## Random walks

![alt text](image-6.png)

![alt text](image-7.png)
怎么理解高效率？


对特征学习的考量
![alt text](image-8.png)
提出损失函数
![alt text](image-9.png)

![alt text](image-10.png)

用了一个近似来化简 （不约而同走到了noise-denoise）
![alt text](image-11.png)
k在5~20之间！又是glidar的论文！


### summary
![alt text](image-12.png)


### node2vec
![alt text](image-13.png)
![alt text](image-14.png)

![alt text](image-15.png)


## embedding the entire graph

SKIP



