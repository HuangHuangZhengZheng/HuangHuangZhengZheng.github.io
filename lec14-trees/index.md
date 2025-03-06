# Lec14-Trees

# Trees
A tree has a root label and a list of branches and each branch is a tree itself

```python
def tree(label, branches=[]):
    for branch in branches:
        assert is_tree(branch), 'branches must be trees'
    return [label] + list(branches) # make sure branches is a list

def label(tree):
    return tree[0]

def branches(tree):
    return tree[1:]

def is_tree(tree):
    if type(tree)!= list or len(tree) < 1:
        return False
    for branch in branches(tree):
        if not is_tree(branch):
            return False
    return True
```

methods:

```python
def is_leaf(tree):
    return not branches(tree)

def fib_tree(n):
    if n <= 1:
        return tree(n)
    else:
        left, right = fib_tree(n-2), fib_tree(n-1)
        return tree(label(left) + label(right), [left, right])

def count_leaves(tree):
    if is_leaf(tree):
        return 1
    else:
        return sum(count_leaves(branch) for branch in branches(tree))

def leaves(tree):
    """
    >>> leaves(fib_tree(5))
    [1, 0, 1, 0, 1, 1, 0, 1]
    """
    if is_leaf(tree):
        return [label(tree)]
    else:
        return sum([leaves(branch) for branch in branches(tree)], [])

def increment_leaves(tree):
    if is_leaf(tree):
        return tree(label(tree) + 1)
    else:
        return tree(label(tree), [increment_leaves(branch) for branch in branches(tree)])

def increment(tree):
    return tree(label(tree) + 1, [increment(branch) for branch in branches(tree)])

def print_tree(tree, indent=0):
    print(' ' * indent + str(label(tree)))
    for branch in branches(tree):
        print_tree(branch, indent + 1)


def print_sum(tree, so_far=0):
    so_far += label(tree)
    if is_leaf(tree):
        print(so_far)
    else:
        for branch in branches(tree):
            print_sum(branch, so_far)

def count_paths(tree, total):
    if label(tree) == total:
        found = 1
    else:
        found = 0
    return found + sum(count_paths(branch, total - label(tree)) for branch in branches(tree))

``` 






