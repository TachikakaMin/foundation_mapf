import random
import torch
from tqdm import tqdm
from .utils import construct_input_feature, parse_file_name

def calculate_current_goal_distance(current_locations, goal_locations):
    total_distance = 0
    for i in range(len(current_locations)):
        total_distance += abs(current_locations[i][0] - goal_locations[i][0]) + abs(
            current_locations[i][1] - goal_locations[i][1]
        )
    return total_distance


def move_agent(action, map_data, current_locations, temperature):
    agent_num = current_locations.shape[0]
    height, width = map_data.shape
    tmp_current_locs = current_locations.clone()
    for i in range(agent_num):
        x, y = current_locations[i]
        act_dir = action[x, y]
        if act_dir == 1:  # up
            y = min(height - 1, y + 1)
        if act_dir == 2:  # down
            y = max(0, y - 1)
        if act_dir == 3:  # left
            x = max(0, x - 1)
        if act_dir == 4:  # right
            x = min(width - 1, x + 1)
        tmp_current_locs[i] = torch.tensor([x, y])
    collision_flag_per_agent = torch.zeros(agent_num, dtype=torch.bool)
    while True:
        collision_flag = False
        map_mark = map_data.clone()
        # First check normal collisions
        for i in range(agent_num):
            x, y = tmp_current_locs[i]
            map_mark[x, y] += 1
        for i in range(agent_num):
            cur_x, cur_y = current_locations[i]
            x, y = tmp_current_locs[i]
            if map_mark[x, y] > 1:
                if cur_x != x or cur_y != y:
                    map_mark[x, y] -= 1
                    tmp_current_locs[i] = current_locations[i]
                    collision_flag = True
                    collision_flag_per_agent[i] = True
                    # temperature[i] += 1

        # Then check position swaps
        for i in range(agent_num):
            for j in range(i + 1, agent_num):
                if tuple(current_locations[i]) == tuple(tmp_current_locs[j]) and tuple(
                    current_locations[j]
                ) == tuple(tmp_current_locs[i]):
                    tmp_current_locs[i] = current_locations[i]
                    tmp_current_locs[j] = current_locations[j]
                    collision_flag = True
                    collision_flag_per_agent[i] = True
                    collision_flag_per_agent[j] = True
                    # temperature[i] += 1
                    # temperature[j] += 1

        if not collision_flag:
            break
    # for i in range(agent_num):
    #     if collision_flag_per_agent[i] is False:
    #         temperature[i] -= 1
    #         temperature[i] = max(temperature[i], 1)
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


def path_formation(model, val_loader, idx, device, feature_type, action_choice="sample", steps=300):
    all_paths = []
    sample_data = val_loader.dataset[idx]
    feature = sample_data["feature"].to(device)
    file_name = sample_data["file_name"]
    map_name, path_name = parse_file_name(file_name)
    agent_num = sample_data["mask"].sum()
    print(f"Path Formation: {path_name}")

    map_data = feature[0]
    distance_map = val_loader.dataset.get_distance_map(map_name)

    # Get locations based on agent IDs from feature maps
    goal_locations = []
    current_locations = []
    for i in range(agent_num):
        current_pos = torch.where(feature[1] == i + 1)  # +1 because agent IDs start from 1
        goal_pos = torch.where(feature[2] == i + 1)
        current_locations.append(torch.tensor([current_pos[0][0], current_pos[1][0]]))
        goal_locations.append(torch.tensor([goal_pos[0][0], goal_pos[1][0]]))
    
    current_locations = torch.stack(current_locations)
    goal_locations = torch.stack(goal_locations)
    
    current_goal_distances = calculate_current_goal_distance(
        current_locations, goal_locations
    )
    print("Start Goal Distance: ", current_goal_distances)
    temperature = torch.ones(agent_num)
    all_paths.append(current_locations.cpu().numpy())
    for i in tqdm(range(steps), desc=f"Path Formation {path_name}"):
        feature = feature.to(device)

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            logits, _ = model.module(feature.unsqueeze(0))
        else:
            logits, _ = model(feature.unsqueeze(0))
        action = sample_action(
            logits, current_locations, temperature, feature, action_choice
        )
        current_locations, temperature = move_agent(
            action, map_data, current_locations, temperature
        )
        feature = construct_input_feature(
            map_data,
            current_locations,
            goal_locations,
            distance_map,
            feature.shape[0],
            feature_type
        )
        all_paths.append(current_locations.cpu().numpy())
    current_goal_distance = calculate_current_goal_distance(
        current_locations, goal_locations
    )
    print(f"End Goal Distance: {current_goal_distance:.4f}")
     
    return all_paths, goal_locations.cpu().numpy(), current_goal_distance, file_name
