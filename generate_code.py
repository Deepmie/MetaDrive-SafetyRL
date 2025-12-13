import os
from typing import Dict, List
import shutil
import time

WORKSPACE_DIR = '/workspace/metadrive-github'

def transfer_files(root: str, transfer_root: str):
    if os.path.isfile(root) and root.endswith('.py'):
        paths: List[str] = root.split('/')
        pth_str = transfer_root
        # 建立所需要的文件
        for idx, pth in enumerate(paths):
            pth_str = f'{pth_str}/{pth}'
            if idx == len(paths) - 1: break
            if not os.path.exists(pth_str): os.mkdir(pth_str)
        shutil.copy(src=root, dst=pth_str)
    elif os.path.isdir(root) and (not root.endswith('__') or not root.endswith('__')):
        for sub_path in os.listdir(root):
            transfer_files(os.path.join(root, sub_path), transfer_root)


def build_project_files(project_file_name: str = './metadrive_project'):
    if os.path.exists(project_file_name): # 如果存在就删除
        shutil.rmtree(project_file_name)
    os.mkdir(project_file_name) # 新建该文件夹

    transfer_root = os.path.join(WORKSPACE_DIR, project_file_name)
    print('prepare tansfer files...')
    time.sleep(0.5)
    os.chdir('./dp_single_version2')
    transfer_files('main.py', transfer_root=transfer_root)
    os.chdir('../')
    
    os.chdir('./metadrive/')
    transfer_files('custom2_version2', transfer_root=transfer_root)
    os.chdir('../')
    print('transfer finished!, now skip back...')
    if not os.path.samefile(os.getcwd(), WORKSPACE_DIR): os.chdir(WORKSPACE_DIR)


def get_all_file_contents(root: str) -> Dict[str, str]:
    file_contents: Dict = dict()
    # 遍历孩子节点
    for sub_path in os.listdir(root):
        sub_abs_path = os.path.join(root, sub_path)
        if os.path.isfile(sub_abs_path) and sub_path.endswith('.py'):
            with open(sub_abs_path, mode='r', encoding='utf-8') as sub_file:
                file_contents[sub_abs_path] = str(sub_file.read()).strip()
        elif os.path.isdir(sub_abs_path):
            file_contents.update(get_all_file_contents(sub_abs_path))
    return file_contents


def generate_code(root_path: str = './metadrive_project', generate_file_name: str = './total_code.md'):
    print('start to generate code!')
    generate_file = open(generate_file_name, mode='w', encoding='utf-8')
    file_contents: Dict[str, str] = get_all_file_contents(root_path)

    for name, content in file_contents.items():
        generate_file.write(
            f'{name}:\n```\n{content}\n```\n\n'
        )
    generate_file.close()
    print(f'generate code finished! in {generate_file_name}.')


if __name__ == '__main__':
    root_path = './metadrive_project'
    build_project_files(root_path)
    generate_code(root_path, generate_file_name='./total_code.md')