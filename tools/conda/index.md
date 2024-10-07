# Conda配置备忘录

# Conda配置备忘录
## 导出与创建环境
使用YAML文件创建Conda环境的步骤如下：

1. **创建YAML文件**：首先，您需要创建一个YAML配置文件，通常命名为`environment.yml`。在该文件中，您可以定义环境的名称、所需的通道以及依赖的包。例如：

   ```yaml
   name: my_conda_env
   channels:
     - conda-forge
   dependencies:
     - numpy=1.19.5
     - pandas=1.2.3
     - scikit-learn=0.23.2
   ```

   在这个示例中，定义了一个名为`my_conda_env`的环境，并指定了要安装的包及其版本。

2. **使用Conda创建环境**：在命令行中，导航到包含`environment.yml`文件的目录，然后运行以下命令：

   ```bash
   conda env create -f environment.yml
   ```

   这条命令会根据YAML文件中的定义创建新的Conda环境。

3. **激活新环境**：创建完成后，可以使用以下命令激活新环境：

   ```bash
   conda activate my_conda_env
   ```

   现在，您可以在新环境中安装、运行和测试软件包。

4. **导出现有环境**：如果您想将当前环境导出为YAML文件，可以使用以下命令：

   ```bash
   conda env export > environment.yml
   ```

   这将创建一个包含当前环境所有包及其版本信息的YAML文件，方便在其他计算机上重建相同的环境

## 单纯创建、切换、删除环境
```bash
conda create -n YOUR_ENV_NAME python=3.8
```

```bash
conda activate YOUR_ENV_NAME
```


```bash
conda remove -n YOUR_ENV_NAME --all
```

查看环境
```bash
conda info -e
```


## pip install 第三方库
以topologylayer==0.0.0为例，直接pip将会报错，因为该库没有发布到conda中

```bash
git clone https://github.com/bruel-gabrielsson/TopologyLayer.git
cd TopologyLayer
# 注意找到setup.py文件
pip install -e . # pip install .
```





