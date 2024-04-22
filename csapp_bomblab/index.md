# csapp_bomblab


# csapp_bomblab
~~都是汇编语言，没有什么好说的~~
*注意GDB调试*

## 核心概念之一：寻址
- 如何寻址？ $Imm(r_1,r_2,factor)$ 注意值还是地址？
- (%rdx)取memory时，$M[R_i]$ 中M一直在最外层

## 核心概念之二：GDB调试
### 常用命令
- `run` 运行程序（注意结合数据流pipeline）
- `b` +$Addr$ 设置断点
- `delete` 删除断点
- `next` 单步执行
- `step` `stepi``finish`进入函数
- `p $eax` 打印变量
- `x /$nxb $Addr$` 打印内存
- `layout asm` 切换到汇编模式有好看的窗口
- `info registers` 打印寄存器
- `info frame` 打印栈帧
- `info args` 打印函数参数
- `info locals` 打印局部变量
- `info breakpoints` 打印断点信息
- `continue` 继续运行
- `quit` `stop`退出调试

## 一些些技巧
- `mov`一些奇奇怪怪的地址时，很可能是线索，可以用`x /$nxb $Addr$`查看内存
- `jne`之类的能不能直接取等擦边通过
- 常见的基础语句（条件/循环）有一些固定的范式，可以用`x /6i $PC`等查看指令
