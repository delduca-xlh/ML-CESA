#!/usr/bin/env python3
"""
print_structure.py - 打印项目文件夹结构

Usage:
    python print_structure.py
    python print_structure.py /path/to/folder
"""

import os
import sys

IGNORE_DIRS = {
    '.git', '__pycache__', 'node_modules', '.venv', 'venv', 
    '.idea', '.vscode', '.eggs', '*.egg-info', 'dist', 'build',
    '.pytest_cache', '.mypy_cache', 'htmlcov', '.tox'
}

IGNORE_FILES = {
    '.DS_Store', '.gitignore', '*.pyc', '*.pyo', '*.so',
    'Thumbs.db', '.env'
}

def should_ignore(name, is_dir=False):
    """检查是否应该忽略"""
    if name.startswith('.'):
        return True
    if is_dir:
        return name in IGNORE_DIRS
    return name in IGNORE_FILES


def print_tree(path, prefix="", max_depth=None, current_depth=0):
    """递归打印文件夹结构"""
    if max_depth is not None and current_depth > max_depth:
        return
    
    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        return
    
    # 分离文件夹和文件，过滤忽略项
    dirs = [e for e in entries if os.path.isdir(os.path.join(path, e)) and not should_ignore(e, True)]
    files = [e for e in entries if os.path.isfile(os.path.join(path, e)) and not should_ignore(e, False)]
    
    # 合并并排序
    all_entries = [(f, False) for f in files] + [(d, True) for d in dirs]
    
    for i, (name, is_dir) in enumerate(all_entries):
        is_last = (i == len(all_entries) - 1)
        connector = "└── " if is_last else "├── "
        
        if is_dir:
            print(f"{prefix}{connector}{name}/")
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(os.path.join(path, name), new_prefix, max_depth, current_depth + 1)
        else:
            print(f"{prefix}{connector}{name}")


def main():
    # 获取路径（默认当前目录）
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = os.getcwd()
    
    # 获取最大深度（可选）
    max_depth = None
    if len(sys.argv) > 2:
        try:
            max_depth = int(sys.argv[2])
        except ValueError:
            pass
    
    # 打印根目录名
    print(f"{os.path.basename(os.path.abspath(path))}/")
    
    # 打印结构
    print_tree(path, "", max_depth)


if __name__ == "__main__":
    main()
