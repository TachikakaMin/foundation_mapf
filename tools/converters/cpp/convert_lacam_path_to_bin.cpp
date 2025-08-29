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

namespace fs = std::filesystem;

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

// 转换单个路径文件
bool convert_path_to_bin(const std::string& file_name, ProgressBar& progress_bar) {
    std::string output_file = file_name;
    size_t pos = output_file.find("path_files");
    if (pos != std::string::npos) {
        output_file.replace(pos, 10, "input_data");
    }
    pos = output_file.find(".path");
    if (pos != std::string::npos) {
        output_file.replace(pos, 5, ".bin");
    }
    
    // 检查输出文件是否已存在
    if (fs::exists(output_file)) {
        progress_bar.update();
        return true;
    }
    
    // 创建输出目录
    fs::path output_path(output_file);
    fs::create_directories(output_path.parent_path());
    
    // 读取路径文件
    std::ifstream input_file(file_name);
    if (!input_file.is_open()) {
        std::cerr << "无法打开文件: " << file_name << std::endl;
        progress_bar.update();
        return false;
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
        fs::remove(file_name);
        progress_bar.update();
        return false;
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
        fs::remove(file_name);
        progress_bar.update();
        return false;
    }
    
    // 写入二进制文件
    std::ofstream output_bin(output_file, std::ios::binary);
    if (!output_bin.is_open()) {
        std::cerr << "无法创建输出文件: " << output_file << std::endl;
        progress_bar.update();
        return false;
    }
    
    uint16_t steps = static_cast<uint16_t>(paths.size());
    uint16_t agent_num = static_cast<uint16_t>(paths[0].size());
    
    // 写入步骤数和智能体数
    output_bin.write(reinterpret_cast<const char*>(&steps), sizeof(steps));
    output_bin.write(reinterpret_cast<const char*>(&agent_num), sizeof(agent_num));
    
    // 写入每个时间步的数据
    for (size_t t = 0; t < steps; ++t) {
        // 写入当前位置
        for (size_t agent_id = 0; agent_id < agent_num; ++agent_id) {
            uint8_t x = static_cast<uint8_t>(paths[t][agent_id].x);
            uint8_t y = static_cast<uint8_t>(paths[t][agent_id].y);
            output_bin.write(reinterpret_cast<const char*>(&x), sizeof(x));
            output_bin.write(reinterpret_cast<const char*>(&y), sizeof(y));
        }
        
        // 写入动作
        for (size_t agent_id = 0; agent_id < agent_num; ++agent_id) {
            Coordinate cur_pos = paths[t][agent_id];
            Coordinate next_pos = (t + 1 < steps) ? paths[t + 1][agent_id] : cur_pos;
            uint8_t action = static_cast<uint8_t>(get_action(cur_pos, next_pos));
            output_bin.write(reinterpret_cast<const char*>(&action), sizeof(action));
        }
    }
    
    output_bin.close();
    progress_bar.update();
    return true;
}

// 工作线程函数
void worker_thread(const std::vector<std::string>& files, size_t start_idx, size_t end_idx, 
                   ProgressBar& progress_bar, std::atomic<int>& success_count) {
    for (size_t i = start_idx; i < end_idx; ++i) {
        if (convert_path_to_bin(files[i], progress_bar)) {
            success_count.fetch_add(1);
        }
    }
}

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
    
    // 递归查找所有.path文件
    std::vector<std::string> path_files;
    for (const auto& entry : fs::recursive_directory_iterator(input_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".path") {
            path_files.push_back(entry.path().string());
        }
    }
    
    if (path_files.empty()) {
        std::cout << "在目录中未找到.path文件: " << input_dir << std::endl;
        return 1;
    }
    
    std::cout << "找到 " << path_files.size() << " 个.path文件需要处理" << std::endl;
    
    // 创建进度条
    ProgressBar progress_bar(path_files.size());
    
    // 确定线程数
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // 默认值
    
    std::cout << "使用 " << num_threads << " 个线程进行并行处理" << std::endl;
    
    // 计算每个线程处理的文件数
    size_t files_per_thread = path_files.size() / num_threads;
    size_t remaining_files = path_files.size() % num_threads;
    
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    
    // 启动工作线程
    size_t start_idx = 0;
    for (unsigned int i = 0; i < num_threads; ++i) {
        size_t end_idx = start_idx + files_per_thread + (i < remaining_files ? 1 : 0);
        threads.emplace_back(worker_thread, std::ref(path_files), start_idx, end_idx, 
                           std::ref(progress_bar), std::ref(success_count));
        start_idx = end_idx;
    }
    
    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }
    
    progress_bar.finish();
    
    std::cout << "转换完成! 成功处理 " << success_count.load() << "/" 
              << path_files.size() << " 个文件" << std::endl;
    
    return 0;
} 