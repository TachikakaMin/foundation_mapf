/*
 * Single Step PIBT - 只返回下一步动作的PIBT实现
 * 基于原始PIBT算法，但只执行一个时间步
 */

#pragma once
#include <vector>
#include <random>

struct Position {
    int x, y;
    Position(int x = 0, int y = 0) : x(x), y(y) {}
    bool operator==(const Position& other) const {
        return x == other.x && y == other.y;
    }
};

struct AgentState {
    int id;
    Position current_pos;
    Position goal_pos;
    double priority;
    int elapsed_time;
    
    AgentState(int id, Position current, Position goal, double priority = 0.0, int elapsed = 0)
        : id(id), current_pos(current), goal_pos(goal), priority(priority), elapsed_time(elapsed) {}
};

class SingleStepPIBT {
public:
    SingleStepPIBT(int map_width, int map_height, int seed = 0);
    ~SingleStepPIBT() = default;
    
    // 主要函数：返回所有agent的下一步动作
    // actions: 每个agent的动作索引 (0=停留, 1=上, 2=下, 3=左, 4=右)
    std::vector<int> getNextActions(
        const std::vector<AgentState>& agents,
        const std::vector<std::vector<int>>& obstacle_map,
        const std::vector<std::vector<double>>& action_preferences = {}
    );

private:
    int map_width_;
    int map_height_;
    std::mt19937 rng_;
    
    // 获取邻居位置
    std::vector<Position> getNeighbors(const Position& pos);
    
    // 检查位置是否有效
    bool isValidPosition(const Position& pos, const std::vector<std::vector<int>>& obstacle_map);
    
    // 计算Manhattan距离
    int manhattanDistance(const Position& a, const Position& b);
    
    // 核心PIBT函数
    bool planAgent(int agent_id, std::vector<AgentState>& agents, 
                   std::vector<std::vector<int>>& occupation_map,
                   const std::vector<std::vector<int>>& obstacle_map,
                   const std::vector<std::vector<double>>& action_preferences,
                   int caller_id = -1);
};
