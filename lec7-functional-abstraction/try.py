a = 1
def f(g):
    a = 2
    return lambda y: a * g(y) # a is 2

print(f(lambda y: a + y)(a)) # a is 1, so should be 4