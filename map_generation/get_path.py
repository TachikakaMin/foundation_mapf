from pogema_toolbox.evaluator import evaluation
from pogema_toolbox.registry import ToolboxRegistry
from pogema_toolbox.create_env import Environment

from lacam.inference import LacamInference, LacamInferenceConfig
from create_env import create_env_base

import yaml

def main():
    env_cfg_name = 'Environment'
    ToolboxRegistry.register_env(env_cfg_name, create_env_base, Environment)
    
    ToolboxRegistry.register_algorithm('LaCAM', LacamInference, LacamInferenceConfig)

    folder_names = [
        'random_maps',
        'maze_maps',
        'warehouse_maps',
    ]

    for folder in folder_names:
        maps_path = folder + "/test_map.yaml"
        print(f"Loading maps from {maps_path}")
        with open(maps_path, 'r') as f:
            maps = yaml.safe_load(f)
        ToolboxRegistry.register_maps(maps)
        
        config_path = folder + "/config.yaml"
        print(f"Loading config from {config_path}")
        with open(config_path) as f:
            evaluation_config = yaml.safe_load(f)
        
        eval_dir = folder
        print(f"Running evaluation for {eval_dir}")
        evaluation(evaluation_config, eval_dir=eval_dir)


if __name__ == '__main__':
    main()
