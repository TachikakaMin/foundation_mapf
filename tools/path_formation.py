import time

import numpy as np
import torch
from tqdm import tqdm

from .utils import (
    construct_input_feature,
    get_distance,
    parse_file_name,
    read_distance_map,
)


# Channel-to-move mapping the released checkpoint actually predicts.
# Verified with a single-agent diagnostic on an empty 32x32 grid: the
# model places highest probability on channel 1 when the goal is at
# col-1 (LEFT) and on channel 2 when the goal is at col+1 (RIGHT). The
# previous table had 1/2 labeled the opposite way, which was self-
# consistent inside this repo's own simulator (move_agent applied the
# same wrong assumption that built the training labels) but produced
# the wrong move whenever this table was consumed by anything else.
ACTION_DELTAS = torch.tensor(
    [
        [0, 0],   # stay
        [0, -1],  # 1: left  (col-1)
        [0, 1],   # 2: right (col+1)
        [-1, 0],  # 3: up    (row-1)
        [1, 0],   # 4: down  (row+1)
    ],
    dtype=torch.int64,
)
INVALID_ACTION_EPS = 1e-8


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
    action_deltas = ACTION_DELTAS.to(device)
    act_dirs = action[current_locations[:, 0], current_locations[:, 1]].long()
    tmp_current_locs = current_locations.clone() + action_deltas[act_dirs]
    # 使用与张量相同设备上的边界值进行裁剪
    tmp_current_locs[:, 0].clamp_(0, torch.tensor(height - 1, device=device))
    tmp_current_locs[:, 1].clamp_(0, torch.tensor(width - 1, device=device))
    # 障碍物碰撞检测：走进障碍物的 agent 回退到原位
    for i in range(agent_num):
        r, c = tmp_current_locs[i, 0].item(), tmp_current_locs[i, 1].item()
        if map_data[r, c] != 0:
            tmp_current_locs[i] = current_locations[i]
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


def _normalize_action_probs(action_probs):
    return action_probs / action_probs.sum(dim=1, keepdim=True).clamp_min(
        INVALID_ACTION_EPS
    )


def _extract_agent_action_probs(logits, current_locations, map_data):
    agent_rows = current_locations[:, 0].long().to(logits.device)
    agent_cols = current_locations[:, 1].long().to(logits.device)
    agent_logits = logits[0, :, agent_rows, agent_cols].transpose(0, 1).detach().cpu()
    action_probs = torch.softmax(agent_logits, dim=1)

    map_tensor = map_data.detach().cpu() if isinstance(map_data, torch.Tensor) else torch.tensor(map_data)
    deltas = ACTION_DELTAS.cpu()
    next_locations = current_locations.long().cpu().unsqueeze(1) + deltas.unsqueeze(0)
    next_rows = next_locations[:, :, 0]
    next_cols = next_locations[:, :, 1]

    valid_actions = (
        (next_rows >= 0)
        & (next_rows < map_tensor.shape[0])
        & (next_cols >= 0)
        & (next_cols < map_tensor.shape[1])
    )
    if valid_actions.any():
        in_bounds = valid_actions.clone()
        valid_rows = next_rows[valid_actions]
        valid_cols = next_cols[valid_actions]
        valid_actions[in_bounds] = map_tensor[valid_rows, valid_cols] == 0

    action_probs = torch.where(
        valid_actions,
        action_probs,
        torch.full_like(action_probs, INVALID_ACTION_EPS),
    )
    return _normalize_action_probs(action_probs)


def _convert_probs_to_preferences(action_probs, preference_type):
    if preference_type == "sorted":
        return torch.argsort(action_probs, dim=1, descending=True)
    if preference_type != "sampled":
        raise ValueError(f"unsupported CS-PIBT preference type: {preference_type}")

    sampling_weights = action_probs.clone()
    preferences = torch.empty_like(sampling_weights, dtype=torch.long)
    for rank in range(sampling_weights.shape[1]):
        chosen = torch.multinomial(sampling_weights, num_samples=1, replacement=False)
        preferences[:, rank] = chosen[:, 0]
        sampling_weights.scatter_(1, chosen, 0.0)
    return preferences


def _initialize_priorities(current_locations, goal_locations, distance_map):
    priorities = torch.tensor(
        [
            float(get_distance(distance_map, current_locations[i], goal_locations[i]))
            for i in range(len(current_locations))
        ],
        dtype=torch.float32,
    )
    max_priority = priorities.max().item() if len(priorities) > 0 else 0.0
    if max_priority > 0:
        priorities /= max_priority
    else:
        priorities.zero_()
    return priorities


def _update_priorities(priorities, current_locations, goal_locations):
    at_goal = torch.all(current_locations == goal_locations, dim=1)
    next_priorities = priorities.clone()

    remained_at_goal = (priorities <= 0) & at_goal
    reached_goal_now = (priorities > 0) & at_goal
    not_at_goal = ~at_goal

    next_priorities[remained_at_goal] -= 1
    next_priorities[reached_goal_now] = 0
    next_priorities[not_at_goal] = (
        torch.clamp(priorities[not_at_goal], min=0.0) + 1.0
    )
    return next_priorities


def _coord_key(position):
    return int(position[0]), int(position[1])


def _plan_cspibt_step(map_data, current_locations, action_preferences, agent_priorities):
    agent_num = current_locations.shape[0]
    planned_agents = torch.zeros(agent_num, dtype=torch.bool)
    next_locations = current_locations.clone()
    occupied_nodes = torch.zeros(map_data.shape, dtype=torch.bool)
    occupied_edges = {}
    current_locs_to_agent = torch.full(map_data.shape, -1, dtype=torch.long)
    current_locs_to_agent[current_locations[:, 0], current_locations[:, 1]] = torch.arange(
        agent_num
    )
    agent_order = sorted(
        range(agent_num), key=lambda agent_id: float(agent_priorities[agent_id]), reverse=True
    )

    def pibt_recursive(agent_id):
        current_pos = current_locations[agent_id]
        current_row, current_col = _coord_key(current_pos)

        for action_index in action_preferences[agent_id].tolist():
            next_pos = current_pos + ACTION_DELTAS[action_index]
            next_row, next_col = _coord_key(next_pos)

            if next_row < 0 or next_row >= map_data.shape[0]:
                continue
            if next_col < 0 or next_col >= map_data.shape[1]:
                continue
            if map_data[next_row, next_col] != 0:
                continue
            if occupied_nodes[next_row, next_col]:
                continue

            reverse_edge = (next_row, next_col, current_row, current_col)
            if occupied_edges.get(reverse_edge, False):
                continue

            next_locations[agent_id] = next_pos
            planned_agents[agent_id] = True
            occupied_nodes[next_row, next_col] = True
            edge_key = (current_row, current_col, next_row, next_col)
            occupied_edges[edge_key] = True

            conflicting_agent = int(current_locs_to_agent[next_row, next_col].item())
            if (
                conflicting_agent != -1
                and conflicting_agent != agent_id
                and not planned_agents[conflicting_agent]
            ):
                if pibt_recursive(conflicting_agent):
                    return True

                planned_agents[agent_id] = False
                occupied_edges[edge_key] = False
                continue

            return True

        next_locations[agent_id] = current_pos
        planned_agents[agent_id] = True
        occupied_nodes[current_row, current_col] = True
        return False

    for agent_id in agent_order:
        if planned_agents[agent_id]:
            continue
        if not pibt_recursive(agent_id):
            return next_locations, False

    return next_locations, True


def move_agent_cspibt(
    logits,
    map_data,
    current_locations,
    agent_priorities,
    preference_type,
):
    action_probs = _extract_agent_action_probs(logits, current_locations, map_data)
    action_preferences = _convert_probs_to_preferences(action_probs, preference_type)
    next_locations, worked = _plan_cspibt_step(
        map_data.detach().cpu() if isinstance(map_data, torch.Tensor) else torch.tensor(map_data),
        current_locations.long().cpu(),
        action_preferences.long().cpu(),
        agent_priorities.float().cpu(),
    )
    if not worked:
        raise RuntimeError("CS-PIBT failed to find a valid next configuration")
    return next_locations, torch.ones(current_locations.shape[0], dtype=torch.float32)


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
    total_cost = int(steps_to_target.sum().item() + agent_num)
    makespan = int(steps_to_target.max().item())
    
    metrics = {
        'total_cost': total_cost,  # Sum of cost
        'ep_length': float(total_cost / agent_num),  # Average episode length
        'makespan': makespan,  # Makespan
        'isr': float(success_rate),  # Individual Success Rate
        'csr': int(success_rate == 1),  # Complete Success Rate
        'final_distance': int(current_goal_distance),
        'avg_density': float(sum(all_densities) / len(all_densities)) if all_densities else 0.0,
        'total_time': float(total_running_time),
        'throughput': float(number_of_agent_reached_target / max_step),
    }
    
    return metrics


def generate_from_possible_targets(possible_positions, position):
    idx = np.random.choice(len(possible_positions))
    while tuple(possible_positions[idx]) == position:
        idx = np.random.choice(len(possible_positions))
    return possible_positions[idx]


def path_formation(
    model,
    val_loader,
    idx,
    device,
    feature_type,
    action_choice="sample",
    collision_shield="naive",
    cspibt_preference_type="sampled",
    steps=300,
    log_file=None,
    lifelong=False,
    return_metrics=False,
):
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
    agent_num = int(sample_data["mask"].sum().item())
    log_print(f"Path Formation: {path_name}")

    if agent_num <= 0:
        raise ValueError(f"inference sample has no agents: {file_name}")

    map_data = feature[0]
    distance_map = read_distance_map(map_name)

    # Get initial locations
    current_locations, goal_locations = [], []
    for i in range(agent_num):
        current_pos = torch.where(feature[1] == i + 1)
        goal_pos = torch.where(feature[2] == i + 1)
        if current_pos[0].numel() == 0 or goal_pos[0].numel() == 0:
            raise ValueError(
                f"inference sample is missing agent/goal positions for agent {i + 1}: {file_name}"
            )
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
    if collision_shield not in {"naive", "cs_pibt"}:
        raise ValueError(f"unsupported collision shield: {collision_shield}")
    agent_priorities = None
    if collision_shield == "cs_pibt":
        agent_priorities = _initialize_priorities(
            current_locations, goal_locations, distance_map
        )
    # Main loop
    for i in tqdm(range(steps), desc=f"Path Formation {path_name}"):
        # Model inference and action
        start_time = time.time()
        with torch.no_grad():
            logits, _ = model(feature.to(device).unsqueeze(0))
        total_running_time += time.time() - start_time

        # Move agents
        if collision_shield == "cs_pibt":
            agent_priorities = _update_priorities(
                agent_priorities, current_locations, goal_locations
            )
            current_locations, temperature = move_agent_cspibt(
                logits,
                map_data,
                current_locations,
                agent_priorities,
                cspibt_preference_type,
            )
        else:
            action = sample_action(
                logits.cpu(), current_locations, temperature, feature, action_choice
            )
            current_locations, temperature = move_agent(
                action, map_data, current_locations, temperature
            )
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
    if return_metrics:
        return all_paths, all_goal_locations, metrics['final_distance'], file_name, metrics
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
