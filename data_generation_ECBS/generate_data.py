import os
import subprocess
import signal
from concurrent.futures import ProcessPoolExecutor

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
        
        # 检查输出文件是否存在且不完整，若是，则删除
        if os.path.exists(output_file_path):
            print(f"Deleting incomplete output file: {output_file_path}")
            os.remove(output_file_path)
    except Exception as e:
        print(f"Error running command on {yaml_file}: {e}")
        if os.path.exists(output_file_path):
            os.remove(output_file_path)  # 删除可能生成的部分文件

def run_ecbs(input_dir, output_dir, weight=1.2, timeout=5, max_workers=32):
    # 获取所有 .yaml 文件
    yaml_files = [f for f in os.listdir(input_dir) if f.endswith('.yaml')]
    
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # 使用 ProcessPoolExecutor 控制并发进程数
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交每个 YAML 文件处理任务
            futures = [executor.submit(run_single_ecbs, yaml_file, input_dir, output_dir, weight, timeout)
                       for yaml_file in yaml_files]
            for future in futures:
                try:
                    # 等待每个任务完成
                    future.result()
                except Exception as e:
                    print(f"Task generated an exception: {e}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Cancelling all tasks...")
        # 使用 signal.SIGTERM 终止所有正在运行的子进程
        for proc in subprocess._active:  # _active 是所有活跃的进程列表
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception as e:
                print(f"Error while terminating process: {e}")
    finally:
        print("All tasks cancelled.")

# 示例用法
input_file_dir = "/home/yimin/research/ITA-CBS2/map_file_ecbs/paper_empty_32_32_for_foundation"
output_file_dir = "single_map_data_empty/"
run_ecbs(input_file_dir, output_file_dir, weight=1.2, timeout=5, max_workers=16)
