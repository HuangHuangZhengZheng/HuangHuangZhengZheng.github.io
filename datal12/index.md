# DATA100-L12: Gradient Descent, sklearn

开始调包！:smirk:
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(df[["total_bill"]], df["tip"])
df["predicted_tip"] = model.predict(df[["total_bill"]])
```

所有的机器学习似乎都在最小化loss function，而梯度下降就是一种优化算法，它通过迭代的方式不断更新模型参数，使得loss function的值不断减小。

详情见NNDL栏目
