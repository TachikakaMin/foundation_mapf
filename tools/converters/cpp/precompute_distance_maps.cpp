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
#include <queue>
#include <unordered_map>
#include <unordered_set>

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

// 地图数据结构
struct MapData {
    int height;
    int width;
    std::vector<std::vector<int>> grid;
    
    MapData(int h, int w) : height(h), width(w), grid(h, std::vector<int>(w, 0)) {}
    
    bool is_valid_pos(int x, int y) const {
        return x >= 0 && x < height && y >= 0 && y < width && grid[x][y] == 0;
    }
};

// 位置结构
struct Position {
    int x, y;
    Position(int x = 0, int y = 0) : x(x), y(y) {}
    
    bool operator==(const Position& other) const {
        return x == other.x && y == other.y;
    }
};

// Position的哈希函数
struct PositionHash {
    size_t operator()(const Position& pos) const {
        return std::hash<int>()(pos.x) ^ (std::hash<int>()(pos.y) << 1);
    }
};

// 距离地图类型
using DistanceMap = std::unordered_map<Position, std::vector<std::vector<int>>, PositionHash>;

// 读取地图文件
MapData read_map_file(const std::string& map_file) {
    std::ifstream file(map_file);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开地图文件: " + map_file);
    }
    
    std::string line;
    std::getline(file, line); // type
    
    std::getline(file, line); // height
    int height = std::stoi(line.substr(line.find(' ') + 1));
    
    std::getline(file, line); // width
    int width = std::stoi(line.substr(line.find(' ') + 1));
    
    std::getline(file, line); // map
    
    MapData map_data(height, width);
    
    // 读取地图数据
    for (int i = 0; i < height; ++i) {
        std::getline(file, line);
        for (int j = 0; j < width && j < line.length(); ++j) {
            char c = line[j];
            if (c == '@' || c == 'T') {
                map_data.grid[i][j] = 1; // 障碍物
            } else {
                map_data.grid[i][j] = 0; // 可通行
            }
        }
    }
    
    file.close();
    return map_data;
}

// BFS计算从起点到所有点的距离
std::vector<std::vector<int>> compute_distances_from_position(const MapData& map_data, const Position& start) {
    const int NOT_FOUND_PATH = 2048;
    std::vector<std::vector<int>> distances(map_data.height, std::vector<int>(map_data.width, NOT_FOUND_PATH));
    
    if (!map_data.is_valid_pos(start.x, start.y)) {
        return distances;
    }
    
    std::queue<Position> queue;
    std::unordered_set<Position, PositionHash> visited;
    
    queue.push(start);
    visited.insert(start);
    distances[start.x][start.y] = 0;
    
    // 四个方向：上、下、左、右
    const int dx[] = {-1, 1, 0, 0};
    const int dy[] = {0, 0, -1, 1};
    
    while (!queue.empty()) {
        Position current = queue.front();
        queue.pop();
        
        for (int i = 0; i < 4; ++i) {
            int new_x = current.x + dx[i];
            int new_y = current.y + dy[i];
            Position new_pos(new_x, new_y);
            
            if (map_data.is_valid_pos(new_x, new_y) && visited.find(new_pos) == visited.end()) {
                visited.insert(new_pos);
                distances[new_x][new_y] = distances[current.x][current.y] + 1;
                queue.push(new_pos);
            }
        }
    }
    
    return distances;
}

// 创建完整的距离地图
DistanceMap create_distance_map(const MapData& map_data) {
    DistanceMap distance_map;
    
    // 为每个可通行的位置计算到所有其他位置的距离
    for (int x = 0; x < map_data.height; ++x) {
        for (int y = 0; y < map_data.width; ++y) {
            if (map_data.is_valid_pos(x, y)) {
                Position pos(x, y);
                distance_map[pos] = compute_distances_from_position(map_data, pos);
            }
        }
    }
    
    return distance_map;
}

// 将距离地图保存为二进制文件
void save_distance_map_binary(const std::string& output_file, const DistanceMap& distance_map, 
                              int height, int width) {
    std::ofstream file(output_file, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("无法创建输出文件: " + output_file);
    }
    
    // 写入文件头部
    uint32_t num_positions = static_cast<uint32_t>(distance_map.size());
    uint32_t map_height = static_cast<uint32_t>(height);
    uint32_t map_width = static_cast<uint32_t>(width);
    
    file.write(reinterpret_cast<const char*>(&num_positions), sizeof(num_positions));
    file.write(reinterpret_cast<const char*>(&map_height), sizeof(map_height));
    file.write(reinterpret_cast<const char*>(&map_width), sizeof(map_width));
    
    // 写入每个位置的距离数据
    for (const auto& [pos, distances] : distance_map) {
        // 写入起始位置
        uint16_t start_x = static_cast<uint16_t>(pos.x);
        uint16_t start_y = static_cast<uint16_t>(pos.y);
        file.write(reinterpret_cast<const char*>(&start_x), sizeof(start_x));
        file.write(reinterpret_cast<const char*>(&start_y), sizeof(start_y));
        
        // 写入距离矩阵
        for (int x = 0; x < height; ++x) {
            for (int y = 0; y < width; ++y) {
                uint16_t distance = static_cast<uint16_t>(std::min(distances[x][y], 65535));
                file.write(reinterpret_cast<const char*>(&distance), sizeof(distance));
            }
        }
    }
    
    file.close();
}

// 处理单个地图文件
bool process_single_map(const std::string& map_file, ProgressBar& progress_bar) {
    try {
        // 生成输出文件路径
        std::string output_file = map_file;
        size_t pos = output_file.find("map_files");
        if (pos != std::string::npos) {
            output_file.replace(pos, 9, "distance_maps");
        }
        pos = output_file.find(".map");
        if (pos != std::string::npos) {
            output_file.replace(pos, 4, ".dmap");
        }
        
        // 检查输出文件是否已存在
        if (fs::exists(output_file)) {
            progress_bar.update();
            return true;
        }
        
        // 创建输出目录
        fs::path output_path(output_file);
        fs::create_directories(output_path.parent_path());
        
        // 读取地图数据
        MapData map_data = read_map_file(map_file);
        
        // 创建距离地图
        DistanceMap distance_map = create_distance_map(map_data);
        
        // 保存距离地图
        save_distance_map_binary(output_file, distance_map, map_data.height, map_data.width);
        
        progress_bar.update();
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "处理文件失败 " << map_file << ": " << e.what() << std::endl;
        progress_bar.update();
        return false;
    }
}

// 工作线程函数
void worker_thread(const std::vector<std::string>& files, size_t start_idx, size_t end_idx, 
                   ProgressBar& progress_bar, std::atomic<int>& success_count) {
    for (size_t i = start_idx; i < end_idx; ++i) {
        if (process_single_map(files[i], progress_bar)) {
            success_count.fetch_add(1);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "用法: " << argv[0] << " <地图文件目录>" << std::endl;
        return 1;
    }
    
    std::string map_dir = argv[1];
    
    // 检查输入目录是否存在
    if (!fs::exists(map_dir) || !fs::is_directory(map_dir)) {
        std::cerr << "错误: 目录不存在或不是有效目录: " << map_dir << std::endl;
        return 1;
    }
    
    // 递归查找所有.map文件
    std::vector<std::string> map_files;
    for (const auto& entry : fs::recursive_directory_iterator(map_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".map") {
            map_files.push_back(entry.path().string());
        }
    }
    
    if (map_files.empty()) {
        std::cout << "在目录中未找到.map文件: " << map_dir << std::endl;
        return 1;
    }
    
    std::cout << "找到 " << map_files.size() << " 个.map文件需要处理" << std::endl;
    
    // 创建进度条
    ProgressBar progress_bar(map_files.size());
    
    // 确定线程数
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // 默认值
    
    std::cout << "使用 " << num_threads << " 个线程进行并行处理" << std::endl;
    
    // 计算每个线程处理的文件数
    size_t files_per_thread = map_files.size() / num_threads;
    size_t remaining_files = map_files.size() % num_threads;
    
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    
    // 启动工作线程
    size_t start_idx = 0;
    for (unsigned int i = 0; i < num_threads; ++i) {
        size_t end_idx = start_idx + files_per_thread + (i < remaining_files ? 1 : 0);
        threads.emplace_back(worker_thread, std::ref(map_files), start_idx, end_idx, 
                           std::ref(progress_bar), std::ref(success_count));
        start_idx = end_idx;
    }
    
    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }
    
    progress_bar.finish();
    
    std::cout << "处理完成! 成功处理 " << success_count.load() << "/" 
              << map_files.size() << " 个文件" << std::endl;
    
    return 0;
} 