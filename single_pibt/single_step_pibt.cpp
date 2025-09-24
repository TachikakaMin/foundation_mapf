#include "single_step_pibt.hpp"
#include <algorithm>
#include <climits>

SingleStepPIBT::SingleStepPIBT(int map_width, int map_height, int seed)
    : map_width_(map_width), map_height_(map_height), rng_(seed) {}

std::vector<int> SingleStepPIBT::getNextActions(
    const std::vector<AgentState>& agents,
    const std::vector<std::vector<int>>& obstacle_map,
    const std::vector<std::vector<double>>& action_preferences) {
    
    int num_agents = agents.size();
    std::vector<int> actions(num_agents, 0); // 默认停留
    
    // 创建agent的副本用于规划
    std::vector<AgentState> agent_states = agents;
    
    // 创建占用地图：-1表示空，否则为agent_id
    std::vector<std::vector<int>> occupation_map(map_height_, 
        std::vector<int>(map_width_, -1));
    
    // 初始化当前位置占用
    for (int i = 0; i < num_agents; ++i) {
        const auto& pos = agent_states[i].current_pos;
        if (pos.x >= 0 && pos.x < map_height_ && pos.y >= 0 && pos.y < map_width_) {
            occupation_map[pos.x][pos.y] = i;
        }
    }
    
    // 按优先级排序agent
    std::vector<int> agent_order(num_agents);
    std::iota(agent_order.begin(), agent_order.end(), 0);
    
    std::sort(agent_order.begin(), agent_order.end(), 
        [&agent_states](int a, int b) {
            // 优先级高的先规划，然后按elapsed time，最后按距离
            if (agent_states[a].priority != agent_states[b].priority) {
                return agent_states[a].priority > agent_states[b].priority;
            }
            if (agent_states[a].elapsed_time != agent_states[b].elapsed_time) {
                return agent_states[a].elapsed_time > agent_states[b].elapsed_time;
            }
            return a < b; // tie-breaker
        });
    
    // 为每个agent规划下一步
    for (int agent_id : agent_order) {
        planAgent(agent_id, agent_states, occupation_map, obstacle_map, action_preferences);
        
        // 将规划结果转换为动作索引
        // 注意：这里的坐标系与path_formation.py一致
        // Position.x = 行索引, Position.y = 列索引
        const auto& current = agents[agent_id].current_pos;
        const auto& next = agent_states[agent_id].current_pos;
        
        if (next.x == current.x && next.y == current.y) {
            actions[agent_id] = 0; // 停留
        } else if (next.y == current.y + 1) {
            actions[agent_id] = 1; // 动作1: 列+1 (path_formation中的"up")
        } else if (next.y == current.y - 1) {
            actions[agent_id] = 2; // 动作2: 列-1 (path_formation中的"down")
        } else if (next.x == current.x - 1) {
            actions[agent_id] = 3; // 动作3: 行-1 (path_formation中的"left") 
        } else if (next.x == current.x + 1) {
            actions[agent_id] = 4; // 动作4: 行+1 (path_formation中的"right")
        } else {
            actions[agent_id] = 0; // 默认停留
        }
    }
    
    return actions;
}

std::vector<Position> SingleStepPIBT::getNeighbors(const Position& pos) {
    // 按照path_formation.py中的动作编码生成邻居
    // Position.x = 行索引, Position.y = 列索引
    std::vector<Position> neighbors = {
        {pos.x, pos.y},     // 动作0: 停留
        {pos.x, pos.y + 1}, // 动作1: 列+1 (path_formation中的"up")
        {pos.x, pos.y - 1}, // 动作2: 列-1 (path_formation中的"down") 
        {pos.x - 1, pos.y}, // 动作3: 行-1 (path_formation中的"left")
        {pos.x + 1, pos.y}  // 动作4: 行+1 (path_formation中的"right")
    };
    return neighbors;
}

bool SingleStepPIBT::isValidPosition(const Position& pos, 
                                   const std::vector<std::vector<int>>& obstacle_map) {
    return pos.x >= 0 && pos.x < map_height_ && 
           pos.y >= 0 && pos.y < map_width_ && 
           obstacle_map[pos.x][pos.y] == 0; // 0表示可通行
}

int SingleStepPIBT::manhattanDistance(const Position& a, const Position& b) {
    return std::abs(a.x - b.x) + std::abs(a.y - b.y);
}

bool SingleStepPIBT::planAgent(int agent_id, std::vector<AgentState>& agents,
                              std::vector<std::vector<int>>& occupation_map,
                              const std::vector<std::vector<int>>& obstacle_map,
                              const std::vector<std::vector<double>>& action_preferences,
                              int caller_id) {

    auto& agent = agents[agent_id];
    std::vector<Position> candidates;

    // 如果有动作偏好，直接按偏好序列构建candidates
    if (!action_preferences.empty() &&
        agent_id < action_preferences.size() &&
        action_preferences[agent_id].size() == 5) {

        // 将动作索引转换为位置
        auto action_to_pos = [&](int action) -> Position {
            const auto& curr = agent.current_pos;
            switch (action) {
                case 0: return curr;                           // 停留
                case 1: return {curr.x, curr.y + 1};         // 动作1: 列+1
                case 2: return {curr.x, curr.y - 1};         // 动作2: 列-1
                case 3: return {curr.x - 1, curr.y};         // 动作3: 行-1
                case 4: return {curr.x + 1, curr.y};         // 动作4: 行+1
                default: return curr;
            }
        };

        for (int i = 0; i < 5; i++) {
            int preferred_action = static_cast<int>(action_preferences[agent_id][i]);
            // 边界检查：确保动作在有效范围内
            if (preferred_action < 0 || preferred_action > 4) {
                preferred_action = 0;  // 默认为停留
            }
            Position candidate_pos = action_to_pos(preferred_action);
            candidates.push_back(candidate_pos);
        }
    } else {
        // 没有偏好时使用默认逻辑
        candidates = getNeighbors(agent.current_pos);
    }

    // 将位置转换为动作索引的helper函数
    auto pos_to_action = [&](const Position& pos) -> int {
        const auto& curr = agent.current_pos;
        if (pos.x == curr.x && pos.y == curr.y) return 0; // 停留
        if (pos.x == curr.x - 1) return 1; // 上
        if (pos.x == curr.x + 1) return 2; // 下
        if (pos.y == curr.y - 1) return 3; // 左
        if (pos.y == curr.y + 1) return 4; // 右
        return 0; // 默认
    };


    // 尝试每个候选位置（类似原始PIBT的循环逻辑）
    for (const auto& next_pos : candidates) {
        // 检查是否有效
        if (!isValidPosition(next_pos, obstacle_map)) continue;

        // 检查是否与调用者冲突（避免循环调用）
        if (caller_id != -1 && agents[caller_id].current_pos == next_pos) continue;

        // 检查是否已被其他agent占用
        int occupier = occupation_map[next_pos.x][next_pos.y];
        if (occupier != -1 && occupier != agent_id) {
            // 类似原始PIBT的递归调用：尝试让占用者让路
            if (!planAgent(occupier, agents, occupation_map, obstacle_map,
                          action_preferences, agent_id)) {
                continue; // 让路失败，尝试下一个候选位置
            }
        }

        // 成功分配位置
        // 清除旧位置
        if (occupation_map[agent.current_pos.x][agent.current_pos.y] == agent_id) {
            occupation_map[agent.current_pos.x][agent.current_pos.y] = -1;
        }
        // 占用新位置
        occupation_map[next_pos.x][next_pos.y] = agent_id;
        // 更新agent位置
        agent.current_pos = next_pos;

        return true; // 成功规划
    }

    // 无法找到有效位置，保持原地（类似原始PIBT失败情况）
    return false;
}
