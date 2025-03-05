# Useful Linux CLI Commands



## uncond train, and eval
### train
```bash
# vae
OMP_NUM_THREADS=3 python main.py -b configs/autoencoder/kitti/autoencoder_c2_p4.yaml -t --gpus 0,1, -r logs/kitti/2025-02-27T11-19-07_autoencoder_c2_p4

OMP_NUM_THREADS=3 python main.py -b configs/autoencoder/kitti/autoencoder_c2_p4.yaml -t --gpus 3,
# lidm both for cond and uncond
OMP_NUM_THREADS=3 python main.py -b configs/lidar_diffusion/kitti/uncond_vig_dec0.yaml -t --gpus 0,

OMP_NUM_THREADS=3 python main.py -b configs/lidar_diffusion/kitti/cond_t2l_vig_dec0.yaml -t --gpus 1,3 -r logs/kitti/2025-02-25T12-14-50_cond_t2l_vig_dec0
```



### eval
```bash
# sample and eval
CUDA_VISIBLE_DEVICES=2 python scripts/sample.py -d kitti -r models/lidm/kitti/vig_dec0/epoch=000027.ckpt -n 2000 -s 250 --eval # for current best model, and specify the seed

CUDA_VISIBLE_DEVICES=1 python scripts/sample.py -d kitti -r models/lidm/kitti/vig_dec0/epoch=000027.ckpt -n 2000 -c 25 --eval # for inference steps

CUDA_VISIBLE_DEVICES=1 python scripts/sample.py -d kitti -r models/lidm/kitti/vig_dec0/last.ckpt -n 2000 --eval

# eval only
CUDA_VISIBLE_DEVICES=1 python scripts/sample.py -d kitti -f models/lidm/kitti/gcn_tr1d/samples/00452624/2025-02-03-15-44-42/numpy/samples.pcd --eval
```


#### conditional sample
```bash
# text2lidar
OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=1 python scripts/text2lidar.py -r models/lidm/kitti/t2l/epoch=000020.ckpt -d kitti -p "there is a crosswalk in front of the ego vehicle" # 有个十字路口的场景
```


## docker setup



## unix helpful commands
### find and delete folders containing sth
在Unix系统中，你可以使用find命令结合rm命令来删除所有包含break的文件夹。这里有一个命令示例，它会查找当前路径下所有包含break的文件夹，并删除它们：
```bash
find . -type d -name '*2025-03-02-10-04-43*' -exec rm -rf {} +
```
```bash
find . -type f -name '*00002*' -exec rm {} +
```


下面是命令的详细解释：
find .：从当前目录开始查找。
-type d：查找的类型是目录。
-name '\*break\*'：查找名称中包含break的目录。
-exec：对找到的每个项目执行后面的命令。
rm -rf {} +：删除找到的每个目录及其内容。{}是一个占位符，代表find命令找到的每个项目，+表示将所有找到的项目传递给rm命令，而不是一次传递一个。


### zip
```bash
zip -r topo_1000.zip *.txt
```


### peek the gpu usage
```bash
watch -n 1 nvidia-smi
```

### peek the cpu usage
```bash
top
```
```bash
OMP_NUM_THREADS=3
```

### kill a process
```bash
kill -9 <pid>
```
在你提供的截图中，通过 ps aux | grep python 命令列出了包含 "python" 关键词的进程。
如果你想要终止所有正在运行的 Python 进程，你可以使用 pkill 命令：
```bash
pkill -f python
```

### peek the disk usage
```bash
df -h
```

### ssh to a remote server
```bash
ssh -P <port> <username>@<remote_server_ip>
```

### nohup
```bash
nohup python GLiDR_kitti.py --data /data/data_odometry_static_dynamic/scan/ --exp_name find_shapes_kitti_mlpdiff --beam 16 --dim 8 --batch_size 32 --mode kitti > training_mlp_diff_100.log 2>&1 &
```

### cd
```bash
cd - # 回到上一级目录
cd "/path/to/directory with spaces" # 进入带有空格的目录
```


## tmux后台使用逻辑
```bash
# install tmux
sudo apt-get update
sudo apt-get install tmux
```
### Session
- 新建会话
```bash
tmux new -s lidm
```
- 离开会话
`Ctrl+b d`

- 重连会话
```bash
tmux attach -t my_session
```

- 列出会话
```bash
tmux ls
```

- 杀死会话
`Ctrl+b :kill-session`


- 上下滚动
`Ctrl+b [`  `q`

### Window
- 创建新窗口
`Ctrl+b c`
- 切换窗口
`Ctrl+b p` （上一个窗口） `Ctrl+b n` （下一个窗口）`Ctrl+b <number>` （指定窗口）`Ctrl+b w` （窗口列表）

- 为窗口命名  
`Ctrl+b ,`

### Panes
- 切分窗格
`Ctrl+b %` 垂直分割 `Ctrl+b "` 水平分割

- 转换窗口
`Ctrl+b <arrow key>`

- 关闭当前 pane
`Ctrl+b x`



### build essentials
```bash
sudo apt-get update
sudo apt-get install build-essential
```


## ssh docker image transfer
### docker run
```bash
nvidia-docker run -it --name 'hz_lidargen' --shm-size 16g -v /data0/dataset/:/data -v /home/liujiuming/hz:/home/hz fzhiheng/deep_envs:py38-torch110-cu113-pointnet2 /bin/bash

docker start hz_lidargen # lidardiff / lidargen / ultralidar

docker exec -it hz_lidargen /bin/bash
```

### docker save and load
```bash
docker save -o my-image.tar my-image:tag
```

```bash
scp 'epoch=000037.ckpt' liujiuming@10.129.22.67:~/GLIDR/GLiDR-main/LiDAR-Diffusion-main/models/lidm/kitti/uncond_vig_dec1

# folder
scp -r my-folder user@remote-server:/path/to/destination

docker load -i /path/to/destination/my-image.tar
```


### ifconfig
```bash
ifconfig
```
```
6012 10.129.22.57
6017 10.129.22.67 
6016 10.129.22.66
```




## Python Lib
### knn_cuda
```bash
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl -i http://pypi.doubanio.com/simple/ --trusted-host pypi.doubanio.com
```

### pointnet2_ops_lib
```bash
pip install git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#subdirectory=pointnet2_ops_lib
```


### topologylayer
```bash
pip install git+https://github.com/Topologic/topologylayer.git
# topologylayer np.int --> np.int64
```

### ninja
https://kimi.moonshot.cn/chat/ctjnjk8pe77ot9u80lvg
```bash
sudo apt-get install ninja-build
```

## pip configs / 换源 / conda 换源
```bash
pip install open3d -i https://pypi.tuna.tsinghua.edu.cn/simple # 临时
```

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple # 永久
```

```shell
conda config --show channels
conda config --set show_channel_urls yes
# 加进去
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# conda-forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
# 注意windows prefers http instead of https
```

## lidargen / ultralidar commands

```bash
export PYTHONPATH=/home/hz/mmdetection3d:$PYTHONPATH # in docker
CUDA_VISIBLE_DEVICES=3 ./tools/dist_test.sh configs/ultralidar_kitti360_static_blank_code.py ./epoch_200.pth 1 --eval "mIoU"
CUDA_VISIBLE_DEVICES=3 ./tools/dist_test.sh configs/ultralidar_kitti360_gene.py ./epoch_200.pth 1 --eval "mIoU"
```

##### for viz draft
```bash
export KITTI360_DATASET=/data/KITTI-360/
OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=3 python lidargen.py --sample --exp kitti_pretrained --config kitti.yml

OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=3 python try.py --loop_count 200 --mid 4

OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=3 python try.py --loop_count 5000 --mid 4

```

