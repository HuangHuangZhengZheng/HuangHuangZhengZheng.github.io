# DATA100-L22: Logistic Regression II


# logistic regression model continued
## sklearn demo
go to see lec code! 
## MLE: high-level, detailed (recorded)

# linear separability and regularization
线性可分性：如果存在一个 **超平面（hyperplane）** 可以将数据集分割成两部分，那么这就是线性可分的。

超平面的维度和数据集的维度相同
![alt text](image-2.png)
$$
C
$$
注意对“push”的理解！

#### 另一种理解正则化的角度
![alt text](image-4.png)
![alt text](image-3.png)
这里是避免loss出现无限大的情况（梯度爆炸？），避免出现使前面情况发生的参数（infinite theta）出现，所以在loss里面预先加入正则化项。

# performance metrics
## accuracy
```python
# using sklearn
model.score(X_test, y_test)
```
## imbalanced data, precision, recall
`Acc` is not a good metric for imbalanced data, use precision and recall instead!!!
![alt text](image-5.png)
$$
acc= \frac{TP+TN}{n}\\
precision(精确率)=\frac{TP}{TP+FP}\\
recall(召回率)=\frac{TP}{TP+FN}
$$
![alt text](image-6.png)
![alt text](image-7.png)

# adjusting the classification threshold(阈值界限)
## a case study
![alt text](image-9.png)
变界限可能是因为imbalanced data导致的
## ROC curves and AUC
怎么选择阈值？
![alt text](image-10.png)
![alt text](image-11.png)
![alt text](image-12.png)
![alt text](image-13.png)
![alt text](image-14.png)
![alt text](image-15.png)

# [extra] detailed MLE, gradient descent, PR curves

## Why cross-entropy?
- KL散度: https://www.textbook.ds100.org/ch/24/classification_cost_justification.html?highlight=divergence
- MLE

以下讨论MLE，二分类的话以 **伯努利** 举例
![alt text](image.png)
![alt text](image-1.png)


## PR curves
![alt text](image-16.png)
false positive在T变大的时候增加得更快，所以可能slightly decrease
![alt text](image-17.png)
![alt text](image-18.png)
考虑PR
![alt text](image-19.png)
![alt text](image-20.png)


## 插曲
似乎自然科学所有学科都可以被解构为 “观测到的知识点（context）” + 信息数理化（math & computer science） ？

换言之只需要一方面不断扩充数据/知识点，另一方面提出高明的信息数理化分析方法，就可以推动科学的进步？:thinking: :thinking: :question:
![alt text](image-8.png)
https://docs.google.com/presentation/d/1YsxPERhul760_0TrLhawljbWWqDbtIp5tUm05irfkmw/edit#slide=id.g12444cd4007_0_537
