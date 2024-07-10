# Git



## 文件状态
未跟踪-未修改-已修改-暂存
`git add <name>` - *->暂存
`git commit -m "message"` - 暂存->未修改
`git rm <name>` - 未修改->未跟踪
### 查看状态
``` bash
git status
```
更加细致几行几列
``` bash
git diff
```
查看历史日志
``` bash
git log --pretty=oneline
git log --graph --oneline --decorate
```
## 基本操作
### 基础配置
``` bash
git config --global user.name "your name"
git config --global user.email "your email"
```

### 创建版本库
``` bash
mkdir myproject
cd myproject
git init
```

### 克隆版本库
``` bash
git clone https://github.com/username/repository.git
```


### 跟踪文件or文件夹
``` bash
git add <filename>
```
``` bash
git rm <filename>
git rm --cache <filename>
```

### 设置缓存状态
``` bash
git add
```
``` bash
git reset HEAD <filename>
```

### 提交修改
``` bash
git commit -m "commit message str"
```
撤销非首次修改
``` bash
git reset head~ --soft
```

### 和github联系
``` bash
git remote add origin https://github.com/username/repository.git
git remote
```
``` bash
git remote rename origin old_name 
```
推到远程仓库
``` bash
git push origin master
```
***ssh连接？***

### 分支管理
#### 创建分支
``` bash
git branch --list
git branch hhzz
git checkout hhzz
```
``` bash
git checkout -b hhzz
```
#### 合并分支
``` bash
git merge hhzz
```
#### 删除分支
``` bash
git branch -d hhzz
```
### 贮藏功能 stash
~~待施工~~
### 重置、换基功能
~~待施工~~

{{< figure src="/avatar.jpg" title="Lighthouse (figure)" >}}
