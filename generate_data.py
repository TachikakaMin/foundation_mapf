import os
import subprocess

def run_ecbs(input_dir, output_dir, weight=1.2, timeout=5):
    # 获取 input_dir 中的所有 .yaml 文件
    yaml_files = [f for f in os.listdir(input_dir) if f.endswith('.yaml')]
    
    # 如果 output_dir 不存在，创建该目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历每个 .yaml 文件并运行 ecbs 命令
    for yaml_file in yaml_files:
        input_file_path = os.path.join(input_dir, yaml_file)
        output_file_path = os.path.join(output_dir, yaml_file)
        
        # 构建命令
        command = f'./ECBS -i {input_file_path} -o {output_file_path} -w {weight}'
        
        print(f"Running command: {command}")
        
        try:
            # 执行命令，设置超时时间
            subprocess.run(command, shell=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            # 处理超时情况
            print(f"Command timed out after {timeout} seconds: {yaml_file}")
        except Exception as e:
            # 处理其他异常
            print(f"Error running command on {yaml_file}: {e}")

# 示例用法
input_file_dir = "/home/yimin/research/ITA-CBS2/map_file_ecbs/paper_maze_32_32_2"
output_file_dir = "data/"
run_ecbs(input_file_dir, output_file_dir)
