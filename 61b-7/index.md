# 61B-7: Testing


# Ad Hoc Testing vs. JUnit
![](image.png)
```java
public class TestSort {
  /** Tests the sort method of the Sort class. */  
  public static testSort() {
    String[] input = {"cows", "dwell", "above", "clouds"};
    String[] expected = {"above", "cows", "clouds", "dwell"};
    Sort.sort(input);
 
    org.junit.Assert.assertArrayEquals(expected, input);
  }
 
  public static void main(String[] args) {
    testSort();
  }
}
```

# Selection Sort
简单介绍一下了，关注点在junit
![alt text](image-1.png)

# Simpler JUnit Tests 
![](image-2.png)
![alt text](image-3.png)
![alt text](image-4.png)
# ADD, TDD, Integration Testing
![alt text](image-5.png)

# More On JUnit (Extra)
![alt text](image-6.png)
![alt text](image-7.png)
