# CS186-L10: Iterators & Joins


## Intro
![alt text](image-1.png)
![alt text](image-2.png)
relational operator: tuples(in other way, relations) in, tuples out
```java
abstract class Iterator {
    // set up the children and the dataflow graph    
    void setup(List<Iterator> inputs); 
    void init(args); // state
    tuple next(); // returns the next tuple
    void close();
}
```
![alt text](image.png)

### presudo code
####  select
on the fly :thinking:
```java
    init() {
        child.init();
        pred = predicate;
        current = null;
    }

    next() {
        while (current != EOF && !pred(current)) {
            current = child.next();
        }
    }

    close() {
        child.close();
    }
```

#### heap scan
want to find out the empty record id 
```java
    init(relation) {
        heap = open heap file for the relation;
        cur_page = heap.first_page();
        cur_slot = cur_page.first_slot();
    }

    next() {
        if (cur_page == null) return EOF;
        current = [cur_page, cur_slot]; // return the id
        // advance to the next slot
        cur_slot = cur_page.next_slot(cur_slot);
        if (cur_slot == null) {
            // advance to the next page, first slot
            cur_page = heap.next_page(cur_page);
            if (cur_page != null) cur_slot = cur_page.first_slot();
        }
        return current;
    }

    close() {
        heap.close();
    }
```

#### sort (two pass)
![alt text](image-3.png)

#### Group By
assume that already sorted, and notice that only contain ONE tuple at a time ===> memory efficient
![alt text](image-4.png)
![alt text](image-5.png)

### A single thread
![alt text](image-6.png)
side note:
- how does the block operator work with the streaming operator
- Sort use disk internally
- we do not store the operator output in disk ===> stream through the call stack

## Join operators
### Simple Nested Loops Join
[see the course note, not that hard to understand](https://cs186berkeley.net/sp21/resources/static/notes/n08-Joins.pdf)
![alt text](image-7.png)
$[R] + [R]|S|$
![alt text](image-8.png)
$[S] + [S]|R|$ 顺序很重要！

### Pages Nested Loops Join
![alt text](image-9.png)
$[R]+[R][S]$

### Chunk Nested Loops Join
![alt text](image-10.png)
$[R] + \lceil(\frac{[R]}{B-2})\rceil[S]$

### Index Nested Loops Join
![alt text](image-11.png)
$[R] + |R|*(cost\ of\ index\ lookup)$
![alt text](image-12.png)

#### cost of index lookup
- unclustered: (# of matching s tuples for each r **tuple**) $\times$ (access cost of per s tuple)
- clustered: (# of matching s tuples for each r **pages**) $\times$ (access cost of per s page)

### Sort-Merge Join
依次滚两个纸带，对齐，归并。
$Sort(R) + Sort(S) + ([R]+[S])$

worst $|R|[S]$ , too many dups 

#### a refinement of the sort-merge join
![alt text](image-13.png)
note that if join and sort, will cost around 9500 > 7500

so sort and join can allow us to get the ORDER BY free :thinking: here comes the refinement
![alt text](image-14.png)
重点在于对sorting最后一次merge的优化，因为可以 track R和S的最小值，于是开始join的步骤即可

### Naive Hash Join
- Requires equality predicate: equi-join and natural join
- assume that $R$ is small enough to fit in memory
- algorithm:
    - hash $R$ into hash table
    - scan $S$ (can be huge file) and probe $R$
- requires $R$ < (B-2)*hash_fill_factor

### Grace Hash Join
- Requires equality predicate: equi-join and natural join
- algorithm:
    - **partition** tuples from $R$ and $S$
    - **Build & Probe** a separate hash table for each partition
      - assume that each partition is small enough to fit in memory
        - recurse if necessary

![alt text](<屏幕截图 2024-09-03 144232.png>)
![alt text](<屏幕截图 2024-09-03 144244.png>)
cost:
$3([R]+[S])$

![alt text](image-15.png)
so it is a good choice for large $S$ and small $R$
![alt text](image-16.png)
Hybrid Hash Join is not included :thinking:

## Summary
![alt text](image-17.png)
