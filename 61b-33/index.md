# 61B-33: Quick Sort


# Backstory, Partitioning
![alt text](image.png)
![alt text](image-1.png)

# Quick Sort
Partition Sort, a.k.a. Quicksort
![alt text](image-2.png)

# Quicksort Runtime
![alt text](image-3.png)
![alt text](image-4.png)

Theoretical analysis:
- Best case: Θ(N log N)
- Worst case: Θ(N2)

Compare this to Mergesort.
- Best case: Θ(N log N)
- Worst case: Θ(N log N)

Recall that Θ(N log N) vs. Θ(N2) is a really big deal. So how can Quicksort be the fastest sort empirically? *Because on average it is Θ(N log N).*
Rigorous proof requires probability theory + calculus, but intuition + empirical analysis will hopefully convince you.
![alt text](image-5.png)
***Argument #2: Quicksort is BST Sort*** :thinking: 
![alt text](image-6.png)

![alt text](image-7.png)

### so far summary
![alt text](image-8.png)

# Avoiding the Quicksort Worst Case
![alt text](image-9.png)



## summary so far
![alt text](image-10.png)
