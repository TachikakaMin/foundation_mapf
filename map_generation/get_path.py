from pogema_toolbox.eval_utils import save_evaluation_results

from pathlib import Path

import yaml
from pogema_toolbox.evaluator import evaluation

from pogema_toolbox.registry import ToolboxRegistry

from create_env import create_env_base
from pogema_toolbox.create_env import Environment
from lacam.inference import LacamInference, LacamInferenceConfig


BASE_PATH = Path('map_generation')

def main():
    env_cfg_name = 'Environment'
    ToolboxRegistry.register_env(env_cfg_name, create_env_base, Environment)
    
    ToolboxRegistry.register_algorithm('LaCAM', LacamInference, LacamInferenceConfig)

    folder_names = [
        'random_maps',
        'maze_maps',
    ]

    for folder in folder_names:
        maps_path = BASE_PATH / folder / "maps.yaml"
        print(f"Loading maps from {maps_path}")
        with open(maps_path, 'r') as f:
            maps = yaml.safe_load(f)
        ToolboxRegistry.register_maps(maps)
        
        config_path = BASE_PATH / folder / "config.yaml"
        print(f"Loading config from {config_path}")
        with open(config_path) as f:
            evaluation_config = yaml.safe_load(f)
        
        eval_dir = BASE_PATH / folder
        print(f"Running evaluation for {eval_dir}")
        evaluation(evaluation_config, eval_dir=eval_dir)
        save_evaluation_results(eval_dir)


if __name__ == '__main__':
    main()
