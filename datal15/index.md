# DATA100-L15: Cross Validation, Regularization


# Cross Validation
## the holdout method
![alt text](image.png)
```python
from sklearn.utils import shuffle
training_set, dev_set = np.split(shuffle(data), [int(.8*len(data))])
```
比较validation error和training error，选择最优的模型。


## K-fold cross validation
![alt text](image-1.png)
K=1 is equivalent to holdout method.


## Test sets
provide an unbiased estimate of the model's performance on new, unseen data.
![alt text](image-2.png)
# Regularization
## L2 regularization (Ridge)
![alt text](image-3.png)
the small the ball, the simpler the model
![alt text](image-4.png)
拉格朗日思想，$\alpha$ 越大，约束越强，模型越简单。
![alt text](image-5.png)
岭回归
## scaling data for regularization
标准化数据，be on the same scale
## L1 regularization (Lasso)
![alt text](image-6.png)

### summary
![alt text](image-7.png)
