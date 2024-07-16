# DATA100-L7: Visualization Ⅰ


## distribution 定义
![define](image.png)

## bar plots for distribution
### data8 example
![table](image-1.png)
### compound way 
![compound](image-2.png)
### *seaborn* example
![seaborn](image-3.png)
```python
import seaborn as sns
sns.countplot(x='variable', data=df)
# rug plot
sns.rugplot(x='variable', data=df, color='black')
```

### plotly example
![plotly](image-4.png)

## 处理异常值（outliers）和峰值（mode）
### density curve
密度曲线看峰
![alt text](image-5.png)
- 箱型图
![alt text](image-6.png)
![alt text](image-7.png)
```python
import seaborn as sns
sns.boxplot(x='variable', data=df)
```

- violin plot
和箱型图对比来看，violin plot宽度有意义
```python
import seaborn as sns
sns.violinplot(x='variable', data=df)
```

![alt text](image-8.png)
处理overplotting
random jitter
![alt text](image-9.png)
