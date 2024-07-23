# 61B-35: Sorting and Algorithmic Bounds


## math problems

$$
N! ∈ \Omega (N^{N}) ?
$$
√

$$
log(N!) ∈ \Omega (NlogN) ?
$$
√

$$
NlogN∈ \Omega (log(N!))  ?
$$
√

所以可以推出：

$$
NlogN ∈ \Theta (logN!)
$$

$$
log(N!)  ∈ \Theta (NlogN)
$$

## TUCS用时 上下界？
the ultimate comparison sort run time

$$
\Omega(NlogN) 
$$

$$
O(NlogN)
$$

下面开始证明：
考虑下界，对n个物体进行排序，有N！种可能，用两两比大小，考虑决策树的高度$$
H = \log_2 N!
$$
因此下界为
$$
\Omega (log(N!))
$$
或者
$$
\Omega (NlogN)
$$
上界通过TUCS的性质可以通过具体示例反证得到，比如用merge sort
