import torch
from tqdm import tqdm
from .utils import construct_input_feature, parse_file_name, read_distance_map
import time
import numpy as np


def statistic_result(current_locations, goal_locations):
    total_distance = 0
    success_num = 0
    for i in range(len(current_locations)):
        total_distance += abs(current_locations[i][0] - goal_locations[i][0]) + abs(
            current_locations[i][1] - goal_locations[i][1]
        )
        if torch.equal(current_locations[i], goal_locations[i]):
            success_num += 1
    return total_distance, success_num / len(current_locations)


def move_agent(action, map_data, current_locations, temperature):
    agent_num = current_locations.shape[0]
    device = current_locations.device
    height, width = map_data.shape
    tmp_current_locs = current_locations.clone()
    # 获取每个agent位置的动作
    act_dirs = action[current_locations[:, 0], current_locations[:, 1]]
    # 创建移动方向的掩码
    up_mask = (act_dirs == 1).to(torch.int32)
    down_mask = (act_dirs == 2).to(torch.int32)
    left_mask = (act_dirs == 3).to(torch.int32)
    right_mask = (act_dirs == 4).to(torch.int32)
    # 一次性更新所有agent的位置
    tmp_current_locs[:, 1] += up_mask
    tmp_current_locs[:, 1] -= down_mask
    tmp_current_locs[:, 0] -= left_mask
    tmp_current_locs[:, 0] += right_mask
    # 使用与张量相同设备上的边界值进行裁剪
    tmp_current_locs[:, 0].clamp_(0, torch.tensor(height - 1, device=device))
    tmp_current_locs[:, 1].clamp_(0, torch.tensor(width - 1, device=device))
    collision_flag_per_agent = torch.zeros(agent_num, dtype=torch.bool)
    while True:
        collision_flag = False
        
        # Create a tensor to count occurrences of each position
        unique_positions, counts = torch.unique(tmp_current_locs, dim=0, return_counts=True)
        collision_positions = unique_positions[counts > 1]
        
        if len(collision_positions) > 0:
            # Find all agents that moved to collision positions
            for collision_pos in collision_positions:
                # Find agents at this collision position
                collision_mask = (tmp_current_locs == collision_pos).all(dim=1)
                colliding_agents = torch.where(collision_mask)[0]
                
                # Keep the agent that didn't move (if any), revert others
                for agent_idx in colliding_agents:
                    if not torch.equal(current_locations[agent_idx], tmp_current_locs[agent_idx]):
                        tmp_current_locs[agent_idx] = current_locations[agent_idx]
                        collision_flag = True
                        collision_flag_per_agent[agent_idx] = True

        
        # Create position mapping: position -> agent_id
        pos_to_new_agent = {tuple(pos.tolist()): i for i, pos in enumerate(tmp_current_locs)}
        
        # Check swaps in single pass
        for i in range(agent_num):
            if collision_flag_per_agent[i]:
                continue
            old_pos = tuple(current_locations[i].tolist())
            new_pos = tuple(tmp_current_locs[i].tolist())
            
            if old_pos != new_pos:  # Agent has moved
                if new_agent_id := pos_to_new_agent.get(old_pos):
                    if tuple(current_locations[new_agent_id].tolist()) == new_pos:
                        tmp_current_locs[i] = current_locations[i]
                        tmp_current_locs[new_agent_id] = current_locations[new_agent_id]
                        collision_flag = True
                        collision_flag_per_agent[i] = True
                        collision_flag_per_agent[new_agent_id] = True  
        if not collision_flag:
            break
    return tmp_current_locs, temperature


def sample_action(
    logits, current_locations, temperature, feature, action_choice="sample"
):
    action_dim, height, width = logits.shape[1], logits.shape[2], logits.shape[3]
    agent_num = len(current_locations)

    if action_choice == "sample":
        logits_reshaped = logits[0].permute(1, 2, 0).contiguous()
        for i in range(agent_num):
            current_x = current_locations[i][0]
            current_y = current_locations[i][1]
            # Scale logits for this agent's position by its temperature
            logits_reshaped[current_x, current_y] = (
                logits_reshaped[current_x, current_y] / temperature[i]
            )
        probs = torch.softmax(logits_reshaped.view(-1, action_dim), dim=-1)
        sampled_actions = torch.multinomial(probs, num_samples=1)
        probs = sampled_actions.view(height, width)
    elif action_choice == "max":
        probs = logits.squeeze(0).permute((1, 2, 0)).argmax(-1)

    mask = (feature[1] > 0).float()
    action = probs * mask
    return action


def calculate_metrics(current_locations, goal_locations, steps_to_target, agent_num, all_densities, total_running_time, max_step, number_of_agent_reached_target):
    """Calculate all metrics for path formation evaluation."""
    current_goal_distance, success_rate = statistic_result(current_locations, goal_locations)
    
    metrics = {
        'total_cost': steps_to_target.sum() + agent_num,  # Sum of cost
        'ep_length': sum(steps_to_target) / agent_num + 1,  # Average episode length
        'makespan': max(steps_to_target),  # Makespan
        'isr': success_rate,  # Individual Success Rate
        'csr': 1 if success_rate == 1 else 0,  # Complete Success Rate
        'final_distance': current_goal_distance,
        'avg_density': sum(all_densities) / len(all_densities) if all_densities else 0,
        'total_time': total_running_time,
        'throughput': number_of_agent_reached_target / max_step
    }
    
    return metrics


def generate_from_possible_targets(possible_positions, position):
    idx = np.random.choice(len(possible_positions))
    while tuple(possible_positions[idx]) == position:
        idx = np.random.choice(len(possible_positions))
    return possible_positions[idx]


def path_formation(model, val_loader, idx, device, feature_type, action_choice="sample", steps=300, log_file=None, lifelong=False):
    def log_print(msg):
        print(msg)
        if log_file:
            with open(log_file, "a") as f:
                f.write(msg + "\n")
                f.flush()

    # Initialize data
    all_paths = []
    all_goal_locations = []
    sample_data = val_loader.dataset[idx]
    feature = sample_data["feature"]
    file_name = sample_data["file_name"]
    map_name, path_name = parse_file_name(file_name)
    agent_num = sample_data["mask"].sum()
    log_print(f"Path Formation: {path_name}")

    map_data = feature[0]
    distance_map = read_distance_map(map_name)

    # Get initial locations
    current_locations, goal_locations = [], []
    for i in range(agent_num):
        current_pos = torch.where(feature[1] == i + 1)
        goal_pos = torch.where(feature[2] == i + 1)
        current_locations.append(torch.tensor([current_pos[0][0], current_pos[1][0]]))
        goal_locations.append(torch.tensor([goal_pos[0][0], goal_pos[1][0]]))
    
    current_locations = torch.stack(current_locations)
    goal_locations = torch.stack(goal_locations)
    
    # Initialize tracking variables
    temperature = torch.ones(agent_num)
    steps_to_target = torch.full((agent_num,), 1048, dtype=torch.int32)  # 初始化为 1048 表示未到达目标
    agent_reached_target = torch.zeros(agent_num, dtype=torch.bool)
    all_densities = []
    total_running_time = 0
    all_paths.append(current_locations.cpu().numpy())
    all_goal_locations.append(goal_locations.cpu().numpy().copy())
    number_of_agent_reached_target = 0

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    model.eval()
    # Main loop
    for i in tqdm(range(steps), desc=f"Path Formation {path_name}"):
        # Model inference and action
        start_time = time.time()
        logits, _ = model(feature.to(device).unsqueeze(0))
        action = sample_action(logits.cpu(), current_locations, temperature, feature, action_choice)
        total_running_time += time.time() - start_time

        # Move agents
        current_locations, temperature = move_agent(action, map_data, current_locations, temperature)
        feature = construct_input_feature(map_data, current_locations, goal_locations, 
                                       distance_map, feature.shape[0], feature_type)
        all_paths.append(current_locations.cpu().numpy())

        possible_positions = torch.nonzero((map_data+feature[2]) == 0, as_tuple=False).tolist()
        # Update statistics and assign new goals if lifelong is enabled
        for j in range(agent_num):
            if not agent_reached_target[j] and torch.equal(current_locations[j], goal_locations[j]):
                agent_reached_target[j] = True
                steps_to_target[j] = i
                number_of_agent_reached_target += 1
                if lifelong:
                    new_goal = generate_from_possible_targets(possible_positions, current_locations[j].tolist())
                    goal_locations[j] = torch.tensor(new_goal, device=device)
                    agent_reached_target[j] = False  # Reset target reached status
        # Create a deep copy of goal_locations when appending
        all_goal_locations.append(goal_locations.cpu().numpy().copy())
        # Calculate density
        agent_densities = calculate_step_density(current_locations, map_data)
        if agent_densities:
            all_densities.append(sum(agent_densities) / len(agent_densities))
        if not lifelong and torch.equal(current_locations, goal_locations):
            break

    # Calculate final metrics
    metrics = calculate_metrics(current_locations, goal_locations, steps_to_target, 
                              agent_num, all_densities, total_running_time, steps, number_of_agent_reached_target)
    
    # Log results
    # log_print(f"Total Running Time: {metrics['total_time']:.4f} seconds")
    # log_print(f"Average Density: {metrics['avg_density']:.4f}")
    # log_print(f"End Goal Distance: {metrics['final_distance']:.4f}, Success Rate: {metrics['isr']:.4f}")
    log_print(f"metrics: {metrics}")
    return all_paths, all_goal_locations, metrics['final_distance'], file_name


def calculate_step_density(current_locations, map_data):
    """Calculate density for each agent's local observation in current step."""
    agent_densities = []
    agent_positions = torch.zeros_like(map_data)
    for pos in current_locations:
        agent_positions[pos[0], pos[1]] = 1

    for agent_pos in current_locations:
        x, y = agent_pos[0], agent_pos[1]
        window_size = 5
        half_size = window_size // 2
        
        x_min = max(0, x - half_size)
        x_max = min(map_data.shape[0], x + half_size + 1)
        y_min = max(0, y - half_size)
        y_max = min(map_data.shape[1], y + half_size + 1)
        
        local_map = map_data[x_min:x_max, y_min:y_max]
        local_agents = agent_positions[x_min:x_max, y_min:y_max]
        
        traversable_cells = torch.sum(local_map == 0).item()
        if traversable_cells > 0:
            occupied_cells = torch.sum(local_agents).item()
            local_density = occupied_cells / traversable_cells
            agent_densities.append(local_density)
    
    return agent_densities