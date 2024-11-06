import os
import subprocess
import signal
from concurrent.futures import ThreadPoolExecutor

def run_single_ecbs(yaml_file, input_dir, output_dir, weight, timeout):
    input_file_path = os.path.join(input_dir, yaml_file)
    output_file_path = os.path.join(output_dir, yaml_file)
    command = f'./ECBS -i {input_file_path} -o {output_file_path} -w {weight}'
    
    print(f"Running command: {command}")
    
    # 使用新的进程组，以便在超时时终止整个进程组
    process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
    try:
        # 等待进程完成或超时
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds: {yaml_file}")
        # 超时后终止整个进程组
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except Exception as e:
        print(f"Error running command on {yaml_file}: {e}")

def run_ecbs(input_dir, output_dir, weight=1.2, timeout=5, max_workers=32):
    # 获取所有 .yaml 文件
    yaml_files = [f for f in os.listdir(input_dir) if f.endswith('.yaml')]
    
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 使用 ThreadPoolExecutor 控制并发数
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for yaml_file in yaml_files:
            # 提交任务，每个任务运行一个 ECBS 实例
            executor.submit(run_single_ecbs, yaml_file, input_dir, output_dir, weight, timeout)

# 示例用法
input_file_dir = "/home/yimin/research/ITA-CBS2/map_file_ecbs/paper_empty_32_32_for_foundation/"
output_file_dir = "single_map_data_empty/"
run_ecbs(input_file_dir, output_file_dir, weight=1.2, timeout=5, max_workers=4)
