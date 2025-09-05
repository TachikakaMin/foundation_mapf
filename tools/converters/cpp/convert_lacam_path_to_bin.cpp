#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <regex>
#include <map>
#include <set>

namespace fs = std::filesystem;

const size_t TARGET_FILE_SIZE = 5ULL * 1024 * 1024 * 1024; // 5GB

// 进度条类
class ProgressBar {
private:
    std::atomic<int> current{0};
    int total;
    std::mutex print_mutex;
    std::chrono::steady_clock::time_point start_time;
    
public:
    ProgressBar(int total_items) : total(total_items) {
        start_time = std::chrono::steady_clock::now();
    }
    
    void update() {
        int prev = current.fetch_add(1);
        if (prev % std::max(1, total / 100) == 0 || prev == total - 1) {
            print_progress();
        }
    }
    
    void print_progress() {
        std::lock_guard<std::mutex> lock(print_mutex);
        int percent = (current.load() * 100) / total;
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
        
        std::cout << "\r[";
        int bar_width = 50;
        int pos = bar_width * percent / 100;
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << std::setw(3) << percent << "% ";
        std::cout << "(" << current.load() << "/" << total << ") ";
        
        if (current.load() > 0) {
            auto remaining = elapsed * (total - current.load()) / current.load();
            std::cout << "ETA: " << std::setw(2) << remaining.count() / 60 << "m " 
                      << std::setw(2) << (remaining.count() % 60) << "s";
        }
        
        std::cout << std::flush;
    }
    
    void finish() {
        std::cout << std::endl;
    }
};

// 坐标结构
struct Coordinate {
    int x, y;
    Coordinate(int x = 0, int y = 0) : x(x), y(y) {}
};

// 单个场景的数据
struct ScenarioData {
    uint16_t steps;
    uint16_t agent_num;
    std::vector<uint8_t> data; // 原始二进制数据
    std::string original_filename;
    
    size_t get_data_size() const {
        return 4 + data.size(); // header + data
    }
};

// 合并文件头部信息
struct MergedFileHeader {
    uint32_t num_scenarios;
    uint32_t total_data_size;
    uint32_t reserved1;
    uint32_t reserved2;
};

// 场景索引信息
struct ScenarioIndex {
    uint64_t offset;        // 在文件中的偏移量
    uint32_t data_size;     // 数据大小（包含4字节头部）
    uint16_t steps;         // 步数
    uint16_t agent_num;     // 智能体数量
    char filename[256];     // 原始文件名
};

// 解析坐标字符串
std::vector<Coordinate> parse_coordinates(const std::string& coord_str) {
    std::vector<Coordinate> coords;
    std::regex coord_pattern(R"((\d+),(\d+))");
    std::sregex_iterator iter(coord_str.begin(), coord_str.end(), coord_pattern);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        int x = std::stoi((*iter)[1]);
        int y = std::stoi((*iter)[2]);
        coords.emplace_back(x, y);
    }
    
    return coords;
}

// 获取动作
int get_action(const Coordinate& cur_pos, const Coordinate& next_pos) {
    int dx = next_pos.x - cur_pos.x;
    int dy = next_pos.y - cur_pos.y;
    
    if (dx == 0 && dy == 1) return 1;      // 右
    if (dx == 0 && dy == -1) return 2;     // 左
    if (dx == -1 && dy == 0) return 3;     // 上
    if (dx == 1 && dy == 0) return 4;      // 下
    
    return 0; // 静止
}

// 从path文件创建场景数据
ScenarioData create_scenario_from_path(const std::string& file_name) {
    ScenarioData scenario;
    scenario.original_filename = fs::path(file_name).filename().string();
    
    // 读取路径文件
    std::ifstream input_file(file_name);
    if (!input_file.is_open()) {
        throw std::runtime_error("无法打开文件: " + file_name);
    }
    
    std::string line;
    std::vector<std::vector<Coordinate>> paths;
    bool found_solution = false;
    
    // 查找solution行
    while (std::getline(input_file, line)) {
        if (line.find("solution=") == 0) {
            found_solution = true;
            break;
        }
    }
    
    if (!found_solution) {
        input_file.close();
        throw std::runtime_error("未找到solution行");
    }
    
    // 解析路径数据
    while (std::getline(input_file, line)) {
        size_t colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
            std::string coord_part = line.substr(colon_pos + 1);
            auto coords = parse_coordinates(coord_part);
            if (!coords.empty()) {
                paths.push_back(std::move(coords));
            }
        }
    }
    
    input_file.close();
    
    if (paths.empty()) {
        throw std::runtime_error("路径数据为空");
    }
    
    scenario.steps = static_cast<uint16_t>(paths.size());
    scenario.agent_num = static_cast<uint16_t>(paths[0].size());
    
    // 生成二进制数据
    std::ostringstream data_stream;
    
    // 写入每个时间步的数据
    for (size_t t = 0; t < scenario.steps; ++t) {
        // 写入当前位置
        for (size_t agent = 0; agent < scenario.agent_num; ++agent) {
            uint8_t x = static_cast<uint8_t>(paths[t][agent].x);
            uint8_t y = static_cast<uint8_t>(paths[t][agent].y);
            data_stream.write(reinterpret_cast<const char*>(&x), sizeof(x));
            data_stream.write(reinterpret_cast<const char*>(&y), sizeof(y));
        }
        
        // 写入动作
        for (size_t agent = 0; agent < scenario.agent_num; ++agent) {
            Coordinate cur_pos = paths[t][agent];
            Coordinate next_pos = (t + 1 < scenario.steps) ? paths[t + 1][agent] : cur_pos;
            uint8_t action = static_cast<uint8_t>(get_action(cur_pos, next_pos));
            data_stream.write(reinterpret_cast<const char*>(&action), sizeof(action));
        }
    }
    
    std::string data_str = data_stream.str();
    scenario.data.assign(data_str.begin(), data_str.end());
    
    return scenario;
}

// 写入合并的场景文件
void write_merged_scenario_file(const std::string& output_file, 
                               const std::vector<ScenarioData>& scenarios) {
    std::ofstream file(output_file, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("无法创建输出文件: " + output_file);
    }
    
    // 写入文件头部
    MergedFileHeader header;
    header.num_scenarios = static_cast<uint32_t>(scenarios.size());
    header.total_data_size = 0;
    for (const auto& scenario : scenarios) {
        header.total_data_size += static_cast<uint32_t>(scenario.get_data_size());
    }
    header.reserved1 = 0;
    header.reserved2 = 0;
    
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    
    // 计算索引表大小
    size_t index_table_size = scenarios.size() * sizeof(ScenarioIndex);
    size_t data_start_offset = sizeof(header) + index_table_size;
    
    // 创建索引表
    std::vector<ScenarioIndex> indices(scenarios.size());
    uint64_t current_offset = data_start_offset;
    
    for (size_t i = 0; i < scenarios.size(); ++i) {
        indices[i].offset = current_offset;
        indices[i].data_size = static_cast<uint32_t>(scenarios[i].get_data_size());
        indices[i].steps = scenarios[i].steps;
        indices[i].agent_num = scenarios[i].agent_num;
        strncpy(indices[i].filename, scenarios[i].original_filename.c_str(), 
                sizeof(indices[i].filename) - 1);
        indices[i].filename[sizeof(indices[i].filename) - 1] = '\0';
        
        current_offset += scenarios[i].get_data_size();
    }
    
    // 写入索引表
    for (const auto& index : indices) {
        file.write(reinterpret_cast<const char*>(&index), sizeof(index));
    }
    
    // 写入场景数据
    for (const auto& scenario : scenarios) {
        // 写入场景头部
        file.write(reinterpret_cast<const char*>(&scenario.steps), sizeof(scenario.steps));
        file.write(reinterpret_cast<const char*>(&scenario.agent_num), sizeof(scenario.agent_num));
        
        // 写入场景数据
        file.write(reinterpret_cast<const char*>(scenario.data.data()), scenario.data.size());
        }
    
    file.close();
}

// 注意：原来的单文件转换逻辑已被场景目录合并逻辑替代

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "用法: " << argv[0] << " <lacam结果文件目录路径>" << std::endl;
        return 1;
    }
    
    std::string input_dir = argv[1];
    
    // 检查输入目录是否存在
    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        std::cerr << "错误: 目录不存在或不是有效目录: " << input_dir << std::endl;
        return 1;
    }
    
    // 递归查找所有.path文件并按场景目录分组
    std::map<std::string, std::vector<std::string>> scenario_groups;
    
    for (const auto& entry : fs::recursive_directory_iterator(input_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".path") {
            std::string path_file = entry.path().string();
            std::string scenario_dir = entry.path().parent_path().string();
            scenario_groups[scenario_dir].push_back(path_file);
        }
    }
    
    if (scenario_groups.empty()) {
        std::cout << "在目录中未找到.path文件: " << input_dir << std::endl;
        return 1;
    }
    
    size_t total_files = 0;
    for (const auto& [dir, files] : scenario_groups) {
        total_files += files.size();
    }
    
    std::cout << "找到 " << total_files << " 个.path文件，分布在 " 
              << scenario_groups.size() << " 个场景目录中" << std::endl;
    
    // 将场景组转换为向量以便并行处理
    std::vector<std::pair<std::string, std::vector<std::string>>> scenario_list;
    for (const auto& [dir, files] : scenario_groups) {
        scenario_list.emplace_back(dir, files);
    }
    
    // 创建进度条（按场景目录计数）
    ProgressBar progress_bar(scenario_list.size());
    
    // 确定线程数
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // 默认值
    
    std::cout << "使用 " << num_threads << " 个线程进行并行处理" << std::endl;
    
    // 计算每个线程处理的场景目录数
    size_t dirs_per_thread = scenario_list.size() / num_threads;
    size_t remaining_dirs = scenario_list.size() % num_threads;
    
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    std::atomic<size_t> processed_files{0};
    
    // 启动工作线程
    size_t start_idx = 0;
    for (unsigned int i = 0; i < num_threads; ++i) {
        size_t end_idx = start_idx + dirs_per_thread + (i < remaining_dirs ? 1 : 0);
        threads.emplace_back([&scenario_list, start_idx, end_idx, &progress_bar, &success_count, &processed_files]() {
            for (size_t idx = start_idx; idx < end_idx; ++idx) {
                const auto& [scenario_dir, files] = scenario_list[idx];
        try {
            // 生成输出文件路径
            std::string output_file = scenario_dir;
            size_t pos = output_file.find("path_files");
            if (pos != std::string::npos) {
                output_file.replace(pos, 10, "input_data");
            }
            
            // 使用场景目录名作为输出文件名
            std::string scenario_name = fs::path(scenario_dir).filename().string();
            output_file = fs::path(output_file).parent_path() / (scenario_name + ".mbin");
            
            // 检查输出文件是否已存在
            if (fs::exists(output_file)) {
                processed_files += files.size();
                progress_bar.update();
                success_count++;
                continue;
            }
            
            // 创建输出目录
            fs::create_directories(fs::path(output_file).parent_path());
            
            // 处理该场景目录中的所有文件
            std::vector<ScenarioData> scenarios;
            size_t current_size = 0;
            
            for (const auto& file : files) {
                try {
                    ScenarioData scenario = create_scenario_from_path(file);
                    scenarios.push_back(std::move(scenario));
                    current_size += scenarios.back().get_data_size();
                    
                    // 如果达到5GB限制，写入当前批次并开始新批次
                    if (current_size >= TARGET_FILE_SIZE && scenarios.size() > 1) {
                        // 移除最后一个场景，为下一批保留
                        ScenarioData last_scenario = std::move(scenarios.back());
                        scenarios.pop_back();
                        
                        // 写入当前批次
                        std::string batch_output = output_file;
                        batch_output.replace(batch_output.find(".mbin"), 5, 
                                           "_part" + std::to_string(scenarios.size()) + ".mbin");
                        write_merged_scenario_file(batch_output, scenarios);
                        
                        // 开始新批次
                        scenarios.clear();
                        scenarios.push_back(std::move(last_scenario));
                        current_size = scenarios[0].get_data_size();
                    }
                    
                } catch (const std::exception& e) {
                    std::cerr << "处理文件失败 " << file << ": " << e.what() << std::endl;
                }
            }
            
            // 写入剩余的场景
            if (!scenarios.empty()) {
                write_merged_scenario_file(output_file, scenarios);
            }
            
            processed_files += files.size();
            success_count++;
            
        } catch (const std::exception& e) {
            std::cerr << "处理场景目录失败 " << scenario_dir << ": " << e.what() << std::endl;
        }
        
                progress_bar.update();
            }
        });
        start_idx = end_idx;
    }
    
    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }
    
    progress_bar.finish();
    
    std::cout << "转换完成! 成功处理 " << success_count.load() << "/" 
              << scenario_groups.size() << " 个场景目录，共 " 
              << processed_files.load() << "/" << total_files << " 个文件" << std::endl;
    
    return 0;
} 