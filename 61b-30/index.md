# 61B-30: Minimum Spanning Trees


# warm up
![alt text](image.png)
![alt text](image-1.png)

# MST, Cut Property, Generic MST Algorithm
![alt text](image-2.png)
## MST vs SPT
![alt text](image-3.png)
A shortest paths tree depends on the start vertex:
- Because it tells you how to get from a source to EVERYTHING.

There is no source for a MST.

Nonetheless, the MST sometimes happens to be an SPT for a specific vertex.

两者关系不大？
![alt text](image-4.png)
## Cut Property
![alt text](image-5.png)
简单证明 cross bridge 一定在 MST 中。
![alt text](image-6.png)
## Generic MST Algorithm
Start with no edges in the MST.
- Find a cut that has no crossing edges in the MST. 
- Add smallest crossing edge to the MST.
- Repeat until V-1 edges.

This should work, but we need some way of finding a cut with no crossing edges!
- Random isn’t a very good idea.

# Prim’s Algorithm
![alt text](image-7.png)
https://docs.google.com/presentation/d/1NFLbVeCuhhaZAM1z3s9zIYGGnhT4M4PWwAc-TLmCJjc/edit#slide=id.g9a60b2f52_0_0

![alt text](image-8.png)
https://docs.google.com/presentation/d/1GPizbySYMsUhnXSXKvbqV4UhPCvrt750MiqPPgU-eCY/edit#slide=id.g9a60b2f52_0_0
## Prim’s vs. Dijkstra’s
Prim’s and Dijkstra’s algorithms are exactly the same, except Dijkstra’s considers “distance from the source”, and Prim’s considers “distance from the tree.”

Visit order:
- Dijkstra’s algorithm visits vertices in order of distance from the source.
- Prim’s algorithm visits vertices in order of distance from the MST under construction.

Relaxation:
- Relaxation in Dijkstra’s considers an edge better based on distance to source.
- Relaxation in Prim’s considers an edge better based on distance to tree.

## pseudocode

![alt text](image-9.png)

![alt text](image-10.png)


## runtime
![alt text](image-11.png)


# Kruskal’s Algorithm
![alt text](image-12.png)

[conceptual](https://docs.google.com/presentation/d/1RhRSYs9Jbc335P24p7vR-6PLXZUl-1EmeDtqieL9ad8/edit?usp=sharing)
[real](https://docs.google.com/presentation/d/1KpNiR7aLIEG9sm7HgX29nvf3yLD8_vdQEPa0ktQfuYc/edit?usp=sharing)

## Kruskal’s Algorithm Pseudocode
![alt text](image-13.png)
## runtime
![alt text](image-14.png)



## Summary
![alt text](image-15.png)
https://docs.google.com/presentation/d/1I8GSEL0CgT09_JjSUF7MfoRMJkyzPjo8lKRd8XdOaRA/edit#slide=id.g772f8a8e2_0_117

SOTA of compare-based MST algorithms :up:
