#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <thread>
#include <mutex>
#include <filesystem>
#include <algorithm>
#include <sstream>
#include <regex>
#include <chrono>
#include <atomic>

struct Position {
    uint8_t x, y;
};

struct ScenarioData {
    uint16_t steps;
    uint16_t agent_num;
    std::vector<uint8_t> data;
    std::string file_name;
};

struct MBinFileInfo {
    std::string map_name;
    std::string agent_num;
    std::vector<std::string> path_files;
};

class PathConverter {
private:
    std::mutex output_mutex;
    std::atomic<int> processed_files{0};
    std::atomic<int> total_files{0};
    
    uint8_t getAction(const Position& cur_pos, const Position& next_pos) {
        int dx = static_cast<int>(next_pos.x) - static_cast<int>(cur_pos.x);
        int dy = static_cast<int>(next_pos.y) - static_cast<int>(cur_pos.y);
        
        if (dx == 0 && dy == 1) return 1;   // 右
        if (dx == 0 && dy == -1) return 2;  // 左
        if (dx == -1 && dy == 0) return 3;  // 上
        if (dx == 1 && dy == 0) return 4;   // 下
        return 0; // 不动
    }
    
    std::vector<Position> parseCoordinates(const std::string& coord_str) {
        std::vector<Position> positions;
        std::regex coord_regex(R"(\((\d+),(\d+)\))");
        std::sregex_iterator iter(coord_str.begin(), coord_str.end(), coord_regex);
        std::sregex_iterator end;
        
        for (; iter != end; ++iter) {
            Position pos;
            pos.x = static_cast<uint8_t>(std::stoi((*iter)[1].str()));
            pos.y = static_cast<uint8_t>(std::stoi((*iter)[2].str()));
            positions.push_back(pos);
        }
        
        return positions;
    }
    
    std::optional<ScenarioData> convertPathToScenarioData(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            return std::nullopt;
        }
        
        std::string line;
        std::vector<std::string> lines;
        while (std::getline(file, line)) {
            lines.push_back(line);
        }
        file.close();
        
        // 查找solution行
        int solution_line = -1;
        for (size_t i = 0; i < lines.size(); i++) {
            if (lines[i].find("solution=") == 0) {
                solution_line = i;
                break;
            }
        }
        
        if (solution_line == -1 || solution_line >= static_cast<int>(lines.size()) - 1) {
            return std::nullopt;
        }
        
        // 解析路径数据
        std::vector<std::vector<Position>> paths;
        for (size_t i = solution_line + 1; i < lines.size(); i++) {
            size_t colon_pos = lines[i].find(':');
            if (colon_pos != std::string::npos) {
                std::string coord_str = lines[i].substr(colon_pos + 1);
                std::vector<Position> coords = parseCoordinates(coord_str);
                if (!coords.empty()) {
                    paths.push_back(coords);
                }
            }
        }
        
        if (paths.empty() || paths[0].empty()) {
            return std::nullopt;
        }
        
        uint16_t steps = paths.size();
        uint16_t agent_num = paths[0].size();
        
        // 生成scenario数据
        std::vector<uint8_t> scenario_data;
        
        // 写入steps和agent_num
        scenario_data.push_back(steps & 0xFF);
        scenario_data.push_back((steps >> 8) & 0xFF);
        scenario_data.push_back(agent_num & 0xFF);
        scenario_data.push_back((agent_num >> 8) & 0xFF);
        
        // 写入每个时间步的数据
        for (size_t t = 0; t < steps; t++) {
            // 写入当前位置
            for (size_t agent_id = 0; agent_id < agent_num; agent_id++) {
                Position cur_pos = paths[t][agent_id];
                scenario_data.push_back(cur_pos.x);
                scenario_data.push_back(cur_pos.y);
            }
            
            // 写入动作
            for (size_t agent_id = 0; agent_id < agent_num; agent_id++) {
                Position cur_pos = paths[t][agent_id];
                Position next_pos = (t + 1 < steps) ? paths[t + 1][agent_id] : cur_pos;
                uint8_t action = getAction(cur_pos, next_pos);
                scenario_data.push_back(action);
            }
        }
        
        ScenarioData result;
        result.steps = steps;
        result.agent_num = agent_num;
        result.data = std::move(scenario_data);
        result.file_name = std::filesystem::path(file_path).filename().string();
        
        return result;
    }
    
    void createMBinFile(const MBinFileInfo& file_info) {
        std::string output_dir = "data/input_data/" + file_info.map_name + "/" + 
                                file_info.map_name + "-" + file_info.agent_num;
        std::string output_file = output_dir + "/" + file_info.map_name + "-" + 
                                 file_info.agent_num + ".mbin";
        
        // 创建输出目录
        std::filesystem::create_directories(output_dir);
        
        // 检查文件是否已存在
        if (std::filesystem::exists(output_file)) {
            {
                std::lock_guard<std::mutex> lock(output_mutex);
                processed_files += file_info.path_files.size();
                std::cout << "跳过已存在文件: " << output_file << std::endl;
            }
            return;
        }
        
        {
            std::lock_guard<std::mutex> lock(output_mutex);
            std::cout << "处理 " << file_info.map_name << "-" << file_info.agent_num 
                      << ": " << file_info.path_files.size() << " 个文件" << std::endl;
        }
        
        // 转换所有路径文件
        std::vector<ScenarioData> scenarios;
        int local_processed = 0;
        
        for (const std::string& file_path : file_info.path_files) {
            auto scenario_opt = convertPathToScenarioData(file_path);
            if (scenario_opt.has_value()) {
                scenarios.push_back(std::move(scenario_opt.value()));
            }
            
            local_processed++;
            if (local_processed % 10 == 0) {
                std::lock_guard<std::mutex> lock(output_mutex);
                processed_files += 10;
                std::cout << "\r进度: " << processed_files.load() << "/" << total_files.load() 
                          << " (" << (100 * processed_files.load() / total_files.load()) << "%)";
                std::cout.flush();
            }
        }
        
        // 处理剩余的文件计数
        {
            std::lock_guard<std::mutex> lock(output_mutex);
            processed_files += (local_processed % 10);
        }
        
        if (scenarios.empty()) {
            std::lock_guard<std::mutex> lock(output_mutex);
            std::cout << "警告: " << file_info.map_name << "-" << file_info.agent_num 
                      << " 没有有效的scenario数据" << std::endl;
            return;
        }
        
        // 写入.mbin文件
        std::ofstream out_file(output_file, std::ios::binary);
        if (!out_file.is_open()) {
            std::lock_guard<std::mutex> lock(output_mutex);
            std::cout << "错误: 无法创建文件 " << output_file << std::endl;
            return;
        }
        
        // 文件头部 (16字节)
        uint32_t num_scenarios = scenarios.size();
        out_file.write(reinterpret_cast<const char*>(&num_scenarios), 4);
        
        // 12字节预留空间
        char padding[12] = {0};
        out_file.write(padding, 12);
        
        // 计算偏移量
        size_t header_size = 16;
        size_t index_table_size = num_scenarios * 272;
        size_t data_start_offset = header_size + index_table_size;
        
        // 写入索引表
        size_t current_offset = data_start_offset;
        for (const auto& scenario : scenarios) {
            uint64_t offset = current_offset;
            uint32_t data_size = scenario.data.size();
            uint16_t steps = scenario.steps;
            uint16_t agent_num = scenario.agent_num;
            
            out_file.write(reinterpret_cast<const char*>(&offset), 8);
            out_file.write(reinterpret_cast<const char*>(&data_size), 4);
            out_file.write(reinterpret_cast<const char*>(&steps), 2);
            out_file.write(reinterpret_cast<const char*>(&agent_num), 2);
            
            // 文件名 (256字节)
            char file_name_padded[256] = {0};
            size_t name_len = std::min(scenario.file_name.length(), size_t(255));
            scenario.file_name.copy(file_name_padded, name_len);
            out_file.write(file_name_padded, 256);
            
            current_offset += data_size;
        }
        
        // 写入scenario数据
        for (const auto& scenario : scenarios) {
            out_file.write(reinterpret_cast<const char*>(scenario.data.data()), 
                          scenario.data.size());
        }
        
        out_file.close();
        
        {
            std::lock_guard<std::mutex> lock(output_mutex);
            std::cout << "✅ 生成 " << output_file << ", 包含 " << num_scenarios 
                      << " 个scenarios" << std::endl;
        }
    }
    
public:
    void processDirectory(const std::string& input_dir) {
        // 收集所有.path文件
        std::vector<std::string> path_files;
        
        for (const auto& entry : std::filesystem::recursive_directory_iterator(input_dir)) {
            if (entry.path().extension() == ".path") {
                path_files.push_back(entry.path().string());
            }
        }
        
        if (path_files.empty()) {
            std::cout << "在目录中未找到.path文件: " << input_dir << std::endl;
            return;
        }
        
        std::cout << "找到 " << path_files.size() << " 个.path文件" << std::endl;
        total_files = path_files.size();
        
        // 按地图和代理数量分组
        std::map<std::pair<std::string, std::string>, std::vector<std::string>> grouped_files;
        
        for (const std::string& file_path : path_files) {
            std::string basename = std::filesystem::path(file_path).stem().string();
            
            // 解析文件名格式: maze-32-32-60-1-75-0-16-1.path
            std::vector<std::string> parts;
            std::stringstream ss(basename);
            std::string part;
            
            while (std::getline(ss, part, '-')) {
                parts.push_back(part);
            }
            
            if (parts.size() >= 8) {
                // 重构地图名称 (maze-32-32-60-1-75-0)
                std::string map_name = parts[0];
                for (size_t i = 1; i < 7; i++) {
                    map_name += "-" + parts[i];
                }
                std::string agent_num = parts[7];
                
                auto key = std::make_pair(map_name, agent_num);
                grouped_files[key].push_back(file_path);
            }
        }
        
        std::cout << "分组后共有 " << grouped_files.size() << " 个不同的地图-代理组合" << std::endl;
        
        // 准备工作信息
        std::vector<MBinFileInfo> work_items;
        for (const auto& [key, files] : grouped_files) {
            MBinFileInfo info;
            info.map_name = key.first;
            info.agent_num = key.second;
            info.path_files = files;
            work_items.push_back(std::move(info));
        }
        
        // 多线程处理
        unsigned int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
        
        std::cout << "使用 " << num_threads << " 个线程进行处理" << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<std::thread> threads;
        std::atomic<size_t> work_index{0};
        
        for (unsigned int i = 0; i < num_threads; i++) {
            threads.emplace_back([&, this]() {
                size_t index;
                while ((index = work_index.fetch_add(1)) < work_items.size()) {
                    createMBinFile(work_items[index]);
                }
            });
        }
        
        // 等待所有线程完成
        for (auto& thread : threads) {
            thread.join();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << std::endl << "✅ 所有.mbin文件生成完成" << std::endl;
        std::cout << "总耗时: " << duration.count() << " 毫秒" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "用法: " << argv[0] << " <path_to_lacam_result_file_dir>" << std::endl;
        return 1;
    }
    
    std::string input_dir = argv[1];
    
    if (!std::filesystem::exists(input_dir)) {
        std::cout << "错误: 目录不存在 " << input_dir << std::endl;
        return 1;
    }
    
    PathConverter converter;
    converter.processDirectory(input_dir);
    
    return 0;
} 