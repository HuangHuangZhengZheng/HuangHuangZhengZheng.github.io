# Lec28-Calculator

# Calculator

## Exception

![alt text](image.png)

```python
raise Exception("Invalid input")
```

```python
try:
    # code that may raise an exception
except Exception as e:
    print(e)
```
见java try-catch :smirk:

```python
float('inf') # positive infinity
float('-inf') # negative infinity
```

## Programming Languages
- Programs are trees... and the way interpreters work is through a tree recursion.

![alt text](image-1.png)

## Parsing

把文本转化为抽象语法树（Abstract Syntax Tree，AST）

base case: only symbols and numbers

recursive case: expressions and statements

## Scheme-Syntax Calculator

![alt text](image-2.png)

using Python `Pair` to describe pairs of expressions and statements
### the eval function

![alt text](image-3.png)

```python
def calc_apply(op, args):
    """
    args: Iterable
    """
    if op == '+':
        ...
    elif op == '-':
        ...
    elif op == '*':
        ...
    elif op == '/':
        ...
    else:
        raise Exception("Invalid operator")
```

### interactive cli
Read-Eval-Print-Loop (REPL) :open_mouth:

![alt text](image-4.png)

### raise exception

![alt text](image-5.png)

