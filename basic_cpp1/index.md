# C++ ptr

# learning pointer(advanced version)
~~为了防止搞混而写~~
### 下标为0?首地址?
```c++
void test0(){
    int arr[] = {1, 2, 3};
    cout << &arr[0] << endl;
    cout << &arr << endl;
    cout << arr << endl;
}
```
- `arr` `&arr` `&arr[0]` 存储的都是相同的地址
- `arr` 常量指针不能被改变 
### 指向数组元素的指针(不一定是首元素)以用[]来访问数组元素
```c++
void test2() {
    int a[3] = {1,2,3};
    int *p = a;
    p++;
    cout << p[0] << endl; // 2
}
```

### 数组类型的指针


```c++
void test2(){
    int arr[] = {1, 2, 3};
    int (*p)[] = &arr;       // 定义一个指向数组的指针
    cout << (*p)[0] << endl; // 输出数组首地址
    cout << p[0] << endl;    // 输出数组首地址
    cout << p[0][0] << endl; // 输出数组首元素
}
```


- `int *p[] = &arr` vs `int (*p)[] = &arr`???? `[ ]`优先级高于`*`


```shell
int (*p)[] = &arr;
*p --> 一个指针
（*p）[] --> 指向数组的指针
int (*p)[] --> 指向的数组的元素是int类型
```


- `p` = `&arr` 定义了一个指向数组的指针，`(*p)` = `arr` 解引用指针得到数组首地址，`(*p)[0]` = `arr[0]` 访问数组首元素
- `p[0]` = `arr` 访问数组首地址，`p[0][0]` = `arr[0]` 访问数组首元素

### 那么`int *(*p)[] = { };`水到渠成了

### 内存映像图
内存映像图|
:--:|
|1|
|2|
|...|


**内存地址从上往下递增**

~~和CSAPP里面的栈画法有点不一样~~

### delete

1. 申请一个连续的内存块，然后将其视为二维数组：
   ```cpp
   int** arr = new int*[rows];
   for (int i = 0; i < rows; ++i) {
       arr[i] = new int[cols];
   }
   ```
   释放时，你需要先释放每一行的内存，然后释放行指针数组：
   ```cpp
   for (int i = 0; i < rows; ++i) {
       delete[] arr[i];
   }
   delete[] arr;
   ```

2. 申请一个足够大的连续内存块，然后将其视为二维数组：
   ```cpp
   int* arr = new int[rows * cols];
   ```
   在这种情况下，你只需要释放一次：
   ```cpp
   delete[] arr;
   ```
   注意，这种方式下，`arr`实际上是一个一维数组，但是你可以像访问二维数组一样使用它（例如，`arr[i][j]`实际上是`arr[i * cols + j]`）。

确保在释放内存后将指针设置为`nullptr`，以避免悬垂指针问题：
```cpp
delete[] arr;
arr = nullptr; // 或者使用智能指针自动管理内存
```



### char?
```c++
int main() {
    char **p, *city[] = {"aaa","bbb"};
    for (p = city; p < city + 2; ++p) {
        cout << *p << endl;
    }
    return 0;
}
```
结果为：
```shell
aaa
bbb
```

