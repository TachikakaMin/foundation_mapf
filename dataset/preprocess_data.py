import numpy as np
import yaml


file_name = "data/sample.yaml"
with open(file_name, 'r') as f:
    data = yaml.safe_load(f)
    
    
statistics_array = {}
schedule_arrays = {}
for key, value in data['statistics'].items():
    statistics_array[key] = np.array(value)

agent_paths = {}
for agent_name, path in data['schedule'].items():
    agent_index = int(''.join(filter(str.isdigit, agent_name)))
    positions = np.array([[p['x'], p['y'], p['t']] for p in path])
    agent_paths[agent_index] = positions

max_agent_index = max(agent_paths.keys())

all_agent_paths = [None] * (max_agent_index + 1)

for index in range(max_agent_index + 1):
    all_agent_paths[index] = agent_paths.get(index, np.array([]))
all_agent_paths_array = np.array(all_agent_paths, dtype=object)


map_file = statistics_array["map"]
map_file = f"map_file/{map_file}" 
with open(map_file, 'r') as f:
    map_data = f.readlines()[4:]
    map_data = [d.replace("\n","") for d in map_data]
    map_array = np.zeros((len(map_data), len(map_data[0])), dtype=int)
    for i, line in enumerate(map_data):
        for j, char in enumerate(line):
            if char == '@':
                map_array[i, j] = 1  # 障碍物
            elif char == '.':
                map_array[i, j] = 0  # 可通行区域
            else:
                # 如果有其他字符，您可以根据需要进行处理
                pass

file_name = file_name.replace(".yaml", ".npz")
np.savez(file_name, map_array=map_array, agent_paths=all_agent_paths_array)
