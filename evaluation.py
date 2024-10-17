# map 需要障碍物
import numpy as np
import torch

def generate_test(seed=0, h=16, w=16, agent_num=10, agent_dim=3):
    """
    生成测试数据，包括智能体当前位置、目标位置和地图特征信息。

    Args:
    - seed (int): 随机种子，确保生成的数据是可重复的。
    - h (int): 网格高度。
    - w (int): 网格宽度。
    - agent_num (int): 智能体数量。
    - agent_dim (int): 用于表示智能体的二进制维度。

    Returns:
    - agent_num (int): 智能体数量。
    - agent_cur (ndarray): 智能体的当前坐标。
    - agent_goals (ndarray): 智能体的目标坐标。
    - goal_loc_info (FloatTensor): 每个目标位置的二进制编码信息。
    - mp_feature (FloatTensor): 地图特征。
    """
    
    np.random.seed(seed)
    
    # 初始化地图特征
    mp_feature = np.zeros((h, w, 1), dtype=int)
    
    # 生成所有位置并随机打乱
    all_positions = [[x, y] for x in range(h) for y in range(w)]
    np.random.shuffle(all_positions)
    
    # 选择智能体的当前位置和目标位置
    agent_init = np.array(all_positions[:agent_num])
    agent_goals = np.array(all_positions[agent_num:2 * agent_num])
    
    # 初始化目标位置的二进制编码信息
    goal_loc_info = np.zeros((h, w, agent_dim), dtype=int)   # 用来存储目标位置的二进制编码信息。
    indices = np.arange(agent_num)
    
    # 为每个智能体生成二进制编码（从1开始）
    binary_strings = np.array([list(format(i + 1, f'0{agent_dim}b')) for i in indices], dtype=int)
    
    # 将目标位置填入二进制编码
    goal_loc_info[agent_goals[:, 0], agent_goals[:, 1]] = binary_strings
    
    # 转换为PyTorch张量
    goal_loc_info = torch.FloatTensor(goal_loc_info)
    mp_feature = torch.FloatTensor(mp_feature)
    
    return agent_num, agent_init, agent_goals, goal_loc_info, mp_feature


def move_agent(curr_loc, act, mp_feature):
    """
    移动智能体，根据给定的动作更新智能体的位置，同时避免智能体之间的碰撞。

    Args:
    - curr_loc (ndarray): 智能体的当前位置，形状为 (agent_num, 2)，每行是一个智能体的 [x, y] 坐标。
    - act (Tensor): 动作张量，表示每个位置对应的智能体的动作方向，形状为 (h, w)。
    - mp_feature (Tensor): 地图特征张量，表示地图的基本信息（例如障碍物位置）。

    Returns:
    - new_loc (ndarray): 更新后的智能体新位置，考虑了边界和碰撞处理。
    """
    new_loc = np.copy(curr_loc)  # 深拷贝当前智能体位置
    act = act.detach().cpu().numpy()  # 将动作从Tensor转换为NumPy数组
    agent_num = new_loc.shape[0]
    h, w = mp_feature.shape[:2]

    # 遍历每个智能体，根据动作更新其位置
    for i in range(agent_num):
        loc = new_loc[i]
        action = act[loc[0], loc[1]]
        
        if action == 0:  # left
            loc[1] = max(loc[1] - 1, 0)  # 向左，确保不越界
        if action == 1:  # right
            loc[1] = min(loc[1] + 1, w - 1)  # 向右，确保不越界
        if action == 2:  # up
            loc[0] = max(loc[0] - 1, 0)  # 向上，确保不越界
        if action == 3:  # down
            loc[0] = min(loc[0] + 1, h - 1)  # 向下，确保不越界

        new_loc[i] = loc

    # 处理智能体之间的碰撞
    while True:
        b = 1
        # 占位地图地图 mp_feature 的一个扩展，每个位置初始值是地图特征的两倍（可能为 0，或表示障碍物）。
        occupy_map = 2 * mp_feature  # 创建占位地图  # 为什么要*2
        for i in range(agent_num):
            loc = new_loc[i]
            occupy_map[loc[0], loc[1]] += 1  # 标记智能体位置
        
        for i in range(agent_num):
            loc = new_loc[i]
            if occupy_map[loc[0], loc[1]] > 1:  # 发生碰撞  # 所有该位置的智能体都回到原位置
                new_loc[i] = curr_loc[i]  # 回到原位置
                b = 0  # 表示有冲突，需要继续检测
        
        if b:  # 如果没有冲突，跳出循环
            break
        
    
    # 感觉不用所有有冲突的agent 都回到原位置，只需要一个agent回到原位置就行
    # while True:
    #     b = 1
    #     # 占位地图地图 mp_feature 的一个扩展，每个位置初始值是地图特征的两倍（可能为 0，或表示障碍物）。
    #     occupy_map = 1 * mp_feature  # 创建占位地图
    #     for i in range(agent_num):
    #         loc = new_loc[i]
    #         occupy_map[loc[0], loc[1]] += 1  # 标记智能体位置
        
    #     for i in range(agent_num):
    #         loc2 = new_loc[i]
    #         loc1 = curr_loc[i]
    #         if occupy_map[loc2[0], loc2[1]] > 1:  # 发生碰撞  # 所有该位置的智能体都回到原位置
    #             occupy_map[loc2[0], loc2[1]] -= 1
    #             occupy_map[loc1[0], loc1[1]] += 1
    #             new_loc[i] = curr_loc[i]  # 回到原位置
    #             b = 0  # 表示有冲突，需要继续检测
    #     if b:  # 如果没有冲突，跳出循环
    #         break
    return new_loc


def cal_dis(agent_cur, agent_goals):
    """_summary_
    计算所有智能体当前位置到目标位置的曼哈顿距离总和;即在二维平面上，水平方向和竖直方向的距离之和
    感觉如果地图有障碍物，这种任务完成程度或优化目标的衡量标准不是很好 (A star等算法更好?)

    Args:
        agent_cur (_type_): _description_
        agent_goals (_type_): _description_

    Returns:
        _type_: 所有智能体从当前位置到目标位置的总距离，这可以作为任务完成程度或优化目标的衡量标准。

    """
    agent_num = len(agent_cur)
    ans = 0
    for i in range(agent_num):
        ans += abs(agent_cur[i][0]-agent_goals[i][0])
        ans += abs(agent_cur[i][1]-agent_goals[i][1])
    return ans


def evaluate(args, model, device):
    model.eval()
    agent_num, agent_curr, agent_goals, goal_loc_info, mp_feature = generate_test()
    h, w = mp_feature.shape[:2]
    for i in range(100):
        current_loc_info = np.zeros((h, w, args.agent_dim), dtype=int)
        indices = np.arange(agent_num)
        binary_strings = np.array([list(format(i+1, f'0{args.agent_dim}b')) for i in indices], dtype=int)
        current_loc_info[agent_curr[:, 0], agent_curr[:, 1]] = binary_strings
        current_loc_info = torch.FloatTensor(current_loc_info)
        
        # 将地图特征 mp_feature、目标位置 goal_loc_info 和当前智能体位置 current_loc_info 拼接在一起，形成输入特征。
        in_feature = [mp_feature, goal_loc_info, current_loc_info]
        in_feature = torch.cat(in_feature, dim=-1).permute((2, 0, 1))
        in_feature = in_feature.unsqueeze(0).to(device) # 增加 batch 维度，变成 (1, channels, h, w)
        
        with torch.no_grad():
            output = model(in_feature)
        output = output.squeeze(0).permute((1, 2, 0)).argmax(-1) # 将张量维度调整为 (h, w, n_classes); 选择动作预测的最大值，即每个智能体在当前位置的最佳动作（根据模型输出的概率最高的动作）
        mask = current_loc_info.any(-1).to(device)
        action = output * mask
        agent_curr = move_agent(agent_curr, action, mp_feature)
        distance = cal_dis(agent_curr, agent_goals)
        if not distance:  # 不用返回所有时间步吗？
            return 0
            #return i+1, 0  # 只需要最后能到终点就好，不需要最优时间步数，不需要和test的路径做对比，所以不用额外记录test的完整路径
    cost = cal_dis(agent_curr, agent_goals)
    return cost
    # return 100, cost 