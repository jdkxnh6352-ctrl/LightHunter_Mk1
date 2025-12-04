import os

# === 严厉的过滤规则 ===
# 忽略的文件夹
IGNORE_DIRS = {
    'venv', 'env', '.git', '__pycache__', '.idea', '.vscode', 
    'node_modules', 'build', 'dist', 'logs', 'data', 'temp', 
    '__MACOSX'
}
# 忽略的文件后缀 (只看代码)
IGNORE_EXTS = {
    '.pyc', '.pyd', '.exe', '.dll', '.so', '.log', '.zip', '.tar', '.gz', 
    '.png', '.jpg', '.jpeg', '.svg', '.ico', '.db', '.sqlite', '.pkl'
}
# 只读取这些后缀的代码文件
ALLOW_EXTS = {
    '.py', '.js', '.html', '.css', '.json', '.sql', '.md', '.txt', 
    '.yaml', '.yml', '.ini', '.toml', '.sh', '.bat'
}

def make_snapshot(output_file='Project_Code_Snapshot.md'):
    root_dir = os.getcwd()
    file_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Project Snapshot: {os.path.basename(root_dir)}\n\n")
        
        # 1. 先画出目录树 (让我也能看懂架构)
        f.write("## 1. Directory Structure\n```text\n")
        for root, dirs, files in os.walk(root_dir):
            # 过滤文件夹
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            level = root.replace(root_dir, '').count(os.sep)
            indent = '    ' * level
            f.write(f"{indent}|-- {os.path.basename(root)}/\n")
            subindent = '    ' * (level + 1)
            for file in files:
                if any(file.endswith(ext) for ext in ALLOW_EXTS):
                    f.write(f"{subindent}|-- {file}\n")
        f.write("```\n\n")
        
        # 2. 写入文件内容
        f.write("## 2. File Contents\n\n")
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext not in ALLOW_EXTS or file == 'make_snapshot.py' or file == output_file:
                    continue
                
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, root_dir)
                
                # 写入文件名标记
                f.write(f"### 📄 File: `{rel_path}`\n")
                
                # 写入代码块
                lang = ext.replace('.', '')
                if lang == 'py': lang = 'python'
                
                f.write(f"```{lang}\n")
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as code_file:
                        content = code_file.read()
                        # 防止文件太大，限制单文件最大行数 (可选)
                        # if len(content.splitlines()) > 2000: 
                        #     content = "...(File too large, skipped)..."
                        f.write(content)
                except Exception as e:
                    f.write(f"# Error reading file: {e}")
                f.write("\n```\n\n")
                file_count += 1
                
    print(f"✅ 搞定！已处理 {file_count} 个代码文件。")
    print(f"📁 请把生成的 [{output_file}] 直接拖给 Gemini！")

if __name__ == '__main__':
    make_snapshot()