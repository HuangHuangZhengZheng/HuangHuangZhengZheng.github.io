# CS186-L9: Sorting and Hashing


## Why Sort?
- Rendezvous
  - eliminating duplicates (DISTINCT)
  - Grouping for summarization (GROUP BY)
  - Upcoming sort-merge join algorithms
- Ordering
  - sometimes output must be in a specific order
  - First step in bulk loading Tree indexes
- Problem: sort 100GB of data with 1GB of RAM
  - why not virtual memory? -- random IO access, too slow

## Out-of-Core Algorithms
### Single Streaming data passing through the memory
![alt text](image.png)

### Better: Double Buffering
![alt text](image-1.png)


#### 1. **主要线程处理 I/O 缓冲区中的数据**
   - **主线程**负责在一个I/O缓冲区对（即输入缓冲区和输出缓冲区）上运行f(x)函数。
   - 主线程完成计算后准备处理新的缓冲区数据时，会进行缓冲区的交换（Swap）。

#### 2. **第二个 I/O 线程并行处理未使用的 I/O 缓冲区**
   - **第二个I/O线程**并行操作，用于清空已满的输出缓冲区并填充新的输入缓冲区。
   - 这种并行性能够提高系统性能，因为I/O操作通常较为耗时，而通过并行处理可以减少主线程的等待时间，从而更高效地利用CPU资源。

#### 3. **为什么并行处理是可行的？**
   - **原因**：通常情况下，I/O操作比较慢，因此需要占用单独的线程来处理，以避免阻塞主线程。
   - **主题**：I/O处理通常需要独立的线程来管理，以提高整体处理效率。

#### 4. **图解说明**
   - 图中显示了双缓冲机制下的处理流程：输入缓冲区和输出缓冲区成对出现，其中一对缓冲区在主线程中处理，而另一对缓冲区在I/O线程中处理。当主线程处理完当前缓冲区对时，两个线程会进行缓冲区交换。

#### 总结
相比单缓冲的单次流式处理，双缓冲通过并行处理I/O操作，可以显著提高处理效率，尤其是在I/O操作较慢的情况下。主线程可以专注于计算，而不必等待I/O操作完成，进一步提升了系统的并行性和性能。





- double buffering applies to all streams!
  - assume that you have RAM buffers to spare!

## Sorting and Hashing
### Formal Specs
- a file $F$:
  - a multiset of records $R$
  - consuming $N$ blocks of storage
- two "scratch" disks
  - each with >> $N$ blocks of free storage
- a fixed amount of space in RAM
  - memory capacity equivalent to $B$ blocks of disk

As for sorting:
- produce an output file $F_S$
  - with content $R$ stored in order by a given sorting criterion

As for hashing:
- produce an output file $F_H$
  - with content $R$, *arranged on disk so that no 2 records that have the same hash value are separated by a record with a different hash value*
  - i.e., *consecutively* stored on disk


### Sorting
#### Strawman Algorithm
注意左侧是没有sort的，右侧是sort之后的。
![alt text](image-2.png)

#### General External Merge Sort
![alt text](image-3.png)
side note: 
- length = $B$，最后一个是变长的block
- $B$ pages/blocks ---> $B-1$ merge（简单归纳）
