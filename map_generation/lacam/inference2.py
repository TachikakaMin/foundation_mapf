import ctypes
import numpy as np

import os
import subprocess
if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'liblacam.so')):
    calling_script_dir = os.path.dirname(os.path.abspath(__file__))
    cmake_cmd = ['cmake', '.']
    subprocess.run(cmake_cmd, check=True, cwd=calling_script_dir)
    make_cmd = ['make', '-j8']
    subprocess.run(make_cmd, check=True, cwd=calling_script_dir)
    

def convert_paths(agent_paths):
    formatted_data = {}
    for i, agent_path in enumerate(agent_paths):
        formatted_data[f'agent{i}'] = [
            {'x': coord[0], 'y': coord[1], 't': t} 
            for t, coord in enumerate(agent_path)
        ]
    return formatted_data

def format_task_string(idx, start_xy, target_xy, map_shape):
    task_file_content = f"{idx}	tmp.map	{map_shape[0]}	{map_shape[1]}	"
    task_file_content += f"{start_xy[1]}	{start_xy[0]}	{target_xy[1]}	{target_xy[0]}	1\n"
    return task_file_content

def _parse_data(data):
    if data is None:
        return None
    lines = data.strip().split('\n')
    columns = None

    for line in lines:
        tuples = [tuple(map(int, item.split(','))) for item in line.strip().split('|') if item]
        if len(tuples) == 0:
            return None
        if columns is None:
            columns = [[] for _ in range(len(tuples))]
        for i, t in enumerate(tuples):
            columns[i].append(t[::-1])

    return columns

class LacamLib:
    def __init__(self, lib_path):
        self.lib_path = lib_path
        self.load_library()
            
    def load_library(self):
        self._lacam_lib = ctypes.CDLL(self.lib_path)

        self._lacam_lib.run_lacam.argtypes = [
            ctypes.c_char_p,  # map_name
            ctypes.c_char_p,  # scene_name
            ctypes.c_int,     # N
            ctypes.c_float    # time_limit_sec
        ]
        self._lacam_lib.run_lacam.restype = ctypes.c_char_p
    
    def run_lacam(self, map_file_content, scene_file_content, num_agents, lacam_timeouts):
        map_file_bytes = map_file_content.encode('utf-8')
        scenario_file_bytes = scene_file_content.encode('utf-8')

        num_agents_int = ctypes.c_int(num_agents)
        for time_limit_sec in lacam_timeouts:
            result = self._lacam_lib.run_lacam(
                map_file_bytes, 
                scenario_file_bytes, 
                num_agents_int,
                time_limit_sec
            )

            try:
                result_str = result.decode('utf-8')
            except Exception as e:
                print(f'Exception occured while running Lacam: {e}')
                raise e
            
            if "ERROR" in result_str:
                print(f'Lacam failed to find path with time_limit_sec={time_limit_sec} | {result_str}')
            else:
                return True, result_str
        
        return False, None


class LacamInference:
    def __init__(self):
        self.timeouts = [1.0, 5.0, 10.0, 60.0]
        self.lacam_agents = None
        self.lacam_lib = LacamLib("lacam/liblacam.so")
        

    def solve(self, observations):
        print(len(observations))
        print(observations[0].keys())
        print(observations[0])
        map_array = np.array(observations[0]['global_obstacles'])
        agent_starts_xy = [obs['global_xy'] for obs in observations]
        agent_targets_xy = [obs['global_target_xy'] for obs in observations]
        agent_number = len(agent_starts_xy)

        agent_tasks_dict = {}
        for idx, (start_xy, target_xy) in enumerate(zip(agent_starts_xy, agent_targets_xy)):
            agent_task = format_task_string(idx, start_xy, target_xy, map_shape=map_array.shape)
            agent_tasks_dict[idx] = agent_task
            
            
        task_file_content = "version 1\n"
        for idx in range(agent_number):
            task_file_content += agent_tasks_dict[idx]

        map_row = lambda row: ''.join('@' if x else '.' for x in row)
        map_content = '\n'.join(map_row(row) for row in map_array)
        map_file_content = f"type octile\nheight {map_array.shape[0]}\nwidth {map_array.shape[1]}\nmap\n{map_content}"
        solved, lacam_results = self.lacam_lib.run_lacam(map_file_content, task_file_content, len(self.lacam_agents), self.timeouts)
        if solved:
            agent_paths = _parse_data(lacam_results)
        else:
            print(f"Lacam failed to find path for {len(self.lacam_agents)} agents")
            agent_paths = [[agent_starts_xy[i] for _ in range(256)] for i in range(agent_number)] # if failed - agents just wait in start locations
        return agent_paths