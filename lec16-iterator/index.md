# Lec16-Iterator

# Iterator 

```python
iter(iterable)
next(iterator)
```

## List

![alt text](image.png)
`list(iterator)` 创建一个新的列表，包含迭代器中的所有元素

## Dictionary
values, keys, items can be iterated using `iter()` and `next()` functions


![alt text](image-1.png)

迭代的时候不要改变字典的结构（长度），否则会导致迭代出错


## for r in range...


![alt text](image-2.png)

if use iterator in for statement, it will not be able to use again, because it will be exhausted after first iteration

## built-in functions in Python

LAZY MODE:

map / filter / zip / reversed

![alt text](image-3.png)
### map / filter
![alt text](image-4.png)

filter see data100 :yum:

### zip

![alt text](image-5.png)

unpack the zip object into multiple variables(can be useful!)


## why Iterator?

![alt text](image-6.png)

