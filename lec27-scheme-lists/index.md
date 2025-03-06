# Lec27-Scheme Lists

# Scheme Lists

## cons / car / cdr / nil

![alt text](image.png)

```scheme
(null? nil) ; #t
(null? (cons 1 nil)) ; #f
(car (cons 1 2)) ; 1
(list 1 2 3) ; (1 2 3)
```

## Symbolic Programming

Lisp is a symbolic programming language, which uses in AI for a long time...?

  
![alt text](image-1.png)
注意单引号
```scheme
(car (cdr (car (cdr '(1 (2 3) 4))))) ; 3
```

## List Processing
![alt text](image-2.png)

![alt text](image-3.png)

纯看scheme属实有点抽象了

![alt text](image-4.png)

helper function 化简之

![alt text](image-5.png)

进一步化简

![alt text](image-6.png)

语法稍微了解一些，似乎interpreter才是重点 :thinking:


