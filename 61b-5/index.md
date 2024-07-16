# 61B-5: DLLists, Arrays


# Doubly Linked Lists
![alt text](image.png)

![alt text](image-1.png)

![alt text](image-2.png)
注意只有sentinel时要讨论一些特殊情况，特别是环状链表。

# Generic Lists 泛型列表

```java
public class SLList<BleepBlorp> {
   private IntNode sentinel;
   private int size;


   public class IntNode {
      public BleepBlorp item;
      public IntNode next;
      ...
   }
   ...
}

SLList<Integer> s1 = new SLList<>(5);
s1.insertFront(10);
 
SLList<String> s2 = new SLList<>("hi");
s2.insertFront("apple");
```
![alt text](image-3.png)

# Arrays, AList
![alt text](image-4.png)
介绍了`System.arraycopy()`用来resize
![alt text](image-5.png)

# 2D Arrays
![alt text](image-6.png)

# Arrays vs. Classes
array的runtime动态索引（和cpp不一样）
![alt text](image-7.png)
class runtime
![](image-8.png)
![alt text](image-9.png)
