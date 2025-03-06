import os
from datetime import datetime

# 获取当前脚本所在目录的名称，用于categories字段
current_folder_name = os.path.basename(os.getcwd())

# 遍历当前目录下的文件夹
for folder in os.listdir('.'):
    if folder.startswith('Lec') and os.path.isdir(folder):  # 确保是Lec开头的文件夹
        index_path = os.path.join(folder, 'index.md')  # 构造index.md的路径

        # 获取文件夹的创建时间
        folder_path = os.path.join('.', folder)
        creation_time = os.path.getctime(folder_path)
        creation_date = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%dT%H:%M:%S+08:00')

        # 构造要插入的内容
        insert_text = f"""+++
title = '{folder}'
date = {creation_date}
draft = false
categories = ['UCB-CS61A']
+++\n"""
        

        # categories = ['{current_folder_name}']

        if os.path.exists(index_path):  # 确保index.md文件存在
            # 读取原有内容并插入新内容
            with open(index_path, 'r+', encoding='utf-8') as file:
                content = file.read()  # 读取原有内容
                file.seek(0)  # 移动文件指针到开头
                file.write(insert_text + content)  # 插入新内容并写入
                file.truncate()  # 保证文件大小正确
            print(f"已更新 {index_path}")
        else:
            print(f"未找到 {index_path}")


