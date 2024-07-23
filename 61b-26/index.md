# 61B-26: Midterm 2 Review


*For BSTs, the most inefficient way to add is in to put it in order.*

在二叉搜索树（BST）中，最低效的方式是按升序或降序插入节点，因为这种方式会导致树变成一条链表，从而使其性能退化到O(n)的时间复杂度。

这里是一个用Java实现的例子，展示了按升序插入节点的二叉搜索树（BST）会变成链表的情况：

```java
// 定义树的节点
class TreeNode {
    int value;
    TreeNode left;
    TreeNode right;

    TreeNode(int value) {
        this.value = value;
        this.left = null;
        this.right = null;
    }
}

// 定义二叉搜索树
class BinarySearchTree {
    private TreeNode root;

    // 插入节点
    public void insert(int value) {
        root = insertRec(root, value);
    }

    private TreeNode insertRec(TreeNode root, int value) {
        if (root == null) {
            root = new TreeNode(value);
            return root;
        }
        if (value < root.value) {
            root.left = insertRec(root.left, value);
        } else {
            root.right = insertRec(root.right, value);
        }
        return root;
    }

    // 打印树的中序遍历
    public void inorderTraversal() {
        inorderRec(root);
        System.out.println();
    }

    private void inorderRec(TreeNode root) {
        if (root != null) {
            inorderRec(root.left);
            System.out.print(root.value + " ");
            inorderRec(root.right);
        }
    }

    // 打印树的结构（用于调试）
    public void printTree() {
        printTreeRec(root, "", true);
    }

    private void printTreeRec(TreeNode root, String indent, boolean last) {
        if (root != null) {
            System.out.print(indent);
            if (last) {
                System.out.print("R----");
                indent += "   ";
            } else {
                System.out.print("L----");
                indent += "|  ";
            }
            System.out.println(root.value);
            printTreeRec(root.left, indent, false);
            printTreeRec(root.right, indent, true);
        }
    }
}

// 测试类
public class Main {
    public static void main(String[] args) {
        BinarySearchTree bst = new BinarySearchTree();

        // 按升序插入节点（1, 2, 3, 4, 5）
        bst.insert(1);
        bst.insert(2);
        bst.insert(3);
        bst.insert(4);
        bst.insert(5);

        System.out.println("Inorder Traversal of BST:");
        bst.inorderTraversal();

        System.out.println("Tree Structure:");
        bst.printTree();
    }
}
```

### 代码解析
1. **TreeNode类**：定义了树的节点，包括节点的值以及左子节点和右子节点。
2. **BinarySearchTree类**：定义了二叉搜索树，包括插入节点的逻辑和中序遍历打印树的结构。
3. **insert方法**：将节点按升序插入。由于每个新插入的节点都是比当前节点大的，因此所有新节点都会被插入到右子树上，导致树结构像链表。
4. **printTree方法**：用于以结构化方式打印树，用于调试树的结构。

### 运行结果
插入升序节点后，树的结构将变成链表样式，即每个节点只有右子节点，没有左子节点。

```
Inorder Traversal of BST:
1 2 3 4 5 

Tree Structure:
R----1
    R----2
        R----3
            R----4
                R----5
```

可以看到，树的结构变成了一个向右倾斜的链表，说明每个节点只有一个右子节点，没有左子节点。这种结构使得树的性能退化为O(n)的时间复杂度。

more to see
https://docs.google.com/presentation/d/1TA-xr-z7df4vnJz6oo4s7OkpGLoVy1EYYTwR_8AVhxA/edit#slide=id.g35a9240b53_0_150
