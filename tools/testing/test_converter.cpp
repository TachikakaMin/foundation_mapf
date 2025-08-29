#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <regex>
#include <cassert>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

struct Coordinate {
    int x, y;
    Coordinate(int x = 0, int y = 0) : x(x), y(y) {}
    bool operator==(const Coordinate& other) const {
        return x == other.x && y == other.y;
    }
};

struct TestResult {
    bool passed;
    std::string message;
    
    TestResult(bool p, const std::string& m) : passed(p), message(m) {}
};

class PathConverterTester {
private:
    int tests_run = 0;
    int tests_passed = 0;
    
public:
    // 解析坐标字符串（与C++转换器相同的逻辑）
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
    
    // 获取动作（与C++转换器相同的逻辑）
    int get_action(const Coordinate& cur_pos, const Coordinate& next_pos) {
        int dx = next_pos.x - cur_pos.x;
        int dy = next_pos.y - cur_pos.y;
        
        if (dx == 0 && dy == 1) return 1;      // 右
        if (dx == 0 && dy == -1) return 2;     // 左
        if (dx == -1 && dy == 0) return 3;     // 上
        if (dx == 1 && dy == 0) return 4;      // 下
        
        return 0; // 静止
    }
    
    // 从.path文件读取原始路径数据
    std::vector<std::vector<Coordinate>> read_path_file(const std::string& path_file) {
        std::ifstream file(path_file);
        if (!file.is_open()) {
            throw std::runtime_error("无法打开路径文件: " + path_file);
        }
        
        std::string line;
        std::vector<std::vector<Coordinate>> paths;
        bool found_solution = false;
        
        // 查找solution行
        while (std::getline(file, line)) {
            if (line.find("solution=") == 0) {
                found_solution = true;
                break;
            }
        }
        
        if (!found_solution) {
            throw std::runtime_error("未找到solution行");
        }
        
        // 解析路径数据
        while (std::getline(file, line)) {
            size_t colon_pos = line.find(':');
            if (colon_pos != std::string::npos) {
                std::string coord_part = line.substr(colon_pos + 1);
                auto coords = parse_coordinates(coord_part);
                if (!coords.empty()) {
                    paths.push_back(std::move(coords));
                }
            }
        }
        
        return paths;
    }
    
    // 从.bin文件读取转换后的数据
    struct BinaryData {
        uint16_t steps;
        uint16_t agent_num;
        std::vector<std::vector<Coordinate>> positions;
        std::vector<std::vector<int>> actions;
    };
    
    BinaryData read_binary_file(const std::string& bin_file) {
        std::ifstream file(bin_file, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("无法打开二进制文件: " + bin_file);
        }
        
        BinaryData data;
        
        // 读取头部信息
        file.read(reinterpret_cast<char*>(&data.steps), sizeof(data.steps));
        file.read(reinterpret_cast<char*>(&data.agent_num), sizeof(data.agent_num));
        
        data.positions.resize(data.steps);
        data.actions.resize(data.steps);
        
        // 读取每个时间步的数据
        for (int t = 0; t < data.steps; ++t) {
            data.positions[t].resize(data.agent_num);
            data.actions[t].resize(data.agent_num);
            
            // 读取位置
            for (int agent = 0; agent < data.agent_num; ++agent) {
                uint8_t x, y;
                file.read(reinterpret_cast<char*>(&x), sizeof(x));
                file.read(reinterpret_cast<char*>(&y), sizeof(y));
                data.positions[t][agent] = Coordinate(x, y);
            }
            
            // 读取动作
            for (int agent = 0; agent < data.agent_num; ++agent) {
                uint8_t action;
                file.read(reinterpret_cast<char*>(&action), sizeof(action));
                data.actions[t][agent] = action;
            }
        }
        
        return data;
    }
    
    // 测试单个文件的转换正确性
    TestResult test_single_file(const std::string& path_file, const std::string& bin_file) {
        try {
            // 读取原始路径数据
            auto original_paths = read_path_file(path_file);
            
            // 检查二进制文件是否存在
            if (!fs::exists(bin_file)) {
                return TestResult(false, "二进制文件不存在: " + bin_file);
            }
            
            // 读取转换后的数据
            auto binary_data = read_binary_file(bin_file);
            
            // 验证基本信息
            if (binary_data.steps != original_paths.size()) {
                return TestResult(false, 
                    "步骤数不匹配: 原始=" + std::to_string(original_paths.size()) + 
                    ", 转换=" + std::to_string(binary_data.steps));
            }
            
            if (original_paths.empty()) {
                return TestResult(false, "原始路径为空");
            }
            
            if (binary_data.agent_num != original_paths[0].size()) {
                return TestResult(false, 
                    "智能体数量不匹配: 原始=" + std::to_string(original_paths[0].size()) + 
                    ", 转换=" + std::to_string(binary_data.agent_num));
            }
            
            // 验证每个时间步的位置
            for (size_t t = 0; t < original_paths.size(); ++t) {
                for (size_t agent = 0; agent < original_paths[t].size(); ++agent) {
                    if (!(original_paths[t][agent] == binary_data.positions[t][agent])) {
                        return TestResult(false, 
                            "位置不匹配 t=" + std::to_string(t) + ", agent=" + std::to_string(agent) +
                            ": 原始=(" + std::to_string(original_paths[t][agent].x) + "," + 
                            std::to_string(original_paths[t][agent].y) + "), 转换=(" +
                            std::to_string(binary_data.positions[t][agent].x) + "," +
                            std::to_string(binary_data.positions[t][agent].y) + ")");
                    }
                }
            }
            
            // 验证动作
            for (size_t t = 0; t < original_paths.size(); ++t) {
                for (size_t agent = 0; agent < original_paths[t].size(); ++agent) {
                    Coordinate cur_pos = original_paths[t][agent];
                    Coordinate next_pos = (t + 1 < original_paths.size()) ? 
                                        original_paths[t + 1][agent] : cur_pos;
                    
                    int expected_action = get_action(cur_pos, next_pos);
                    int actual_action = binary_data.actions[t][agent];
                    
                    if (expected_action != actual_action) {
                        return TestResult(false, 
                            "动作不匹配 t=" + std::to_string(t) + ", agent=" + std::to_string(agent) +
                            ": 期望=" + std::to_string(expected_action) + 
                            ", 实际=" + std::to_string(actual_action));
                    }
                }
            }
            
            return TestResult(true, "转换正确");
            
        } catch (const std::exception& e) {
            return TestResult(false, "异常: " + std::string(e.what()));
        }
    }
    
    // 测试目录中的所有文件
    void test_directory(const std::string& input_dir) {
        std::cout << "开始测试目录: " << input_dir << std::endl;
        
        // 查找所有.path文件
        std::vector<std::string> path_files;
        for (const auto& entry : fs::recursive_directory_iterator(input_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".path") {
                path_files.push_back(entry.path().string());
            }
        }
        
        if (path_files.empty()) {
            std::cout << "未找到.path文件" << std::endl;
            return;
        }
        
        std::cout << "找到 " << path_files.size() << " 个.path文件" << std::endl;
        
        int detailed_tests = std::min(5, (int)path_files.size());
        
        for (size_t i = 0; i < path_files.size(); ++i) {
            const auto& path_file = path_files[i];
            
            // 生成对应的二进制文件路径
            std::string bin_file = path_file;
            size_t pos = bin_file.find("path_files");
            if (pos != std::string::npos) {
                bin_file.replace(pos, 10, "input_data");
            }
            pos = bin_file.find(".path");
            if (pos != std::string::npos) {
                bin_file.replace(pos, 5, ".bin");
            }
            
            tests_run++;
            auto result = test_single_file(path_file, bin_file);
            
            if (result.passed) {
                tests_passed++;
                if ((int)i < detailed_tests) {
                    std::cout << "✓ " << fs::path(path_file).filename() << std::endl;
                }
            } else {
                std::cout << "✗ " << fs::path(path_file).filename() 
                         << " - " << result.message << std::endl;
            }
            
            // 显示进度
            if (i > 0 && (i + 1) % 100 == 0) {
                std::cout << "已测试 " << (i + 1) << "/" << path_files.size() 
                         << " 个文件 (通过率: " << std::fixed << std::setprecision(1)
                         << (100.0 * tests_passed / tests_run) << "%)" << std::endl;
            }
        }
        
        // 输出最终结果
        std::cout << "\n==== 测试结果 ====" << std::endl;
        std::cout << "总计测试: " << tests_run << std::endl;
        std::cout << "测试通过: " << tests_passed << std::endl;
        std::cout << "测试失败: " << (tests_run - tests_passed) << std::endl;
        std::cout << "通过率: " << std::fixed << std::setprecision(2) 
                 << (100.0 * tests_passed / tests_run) << "%" << std::endl;
        
        if (tests_passed == tests_run) {
            std::cout << "🎉 所有测试都通过了！" << std::endl;
        } else {
            std::cout << "⚠️  有测试失败，请检查转换逻辑" << std::endl;
        }
    }
    
    // 运行基本功能测试
    void run_unit_tests() {
        std::cout << "==== 运行单元测试 ====" << std::endl;
        
        // 测试坐标解析
        {
            std::string coord_str = "(10,20) (30,40) (50,60)";
            auto coords = parse_coordinates(coord_str);
            assert(coords.size() == 3);
            assert(coords[0].x == 10 && coords[0].y == 20);
            assert(coords[1].x == 30 && coords[1].y == 40);
            assert(coords[2].x == 50 && coords[2].y == 60);
            std::cout << "✓ 坐标解析测试通过" << std::endl;
        }
        
        // 测试动作计算
        {
            assert(get_action(Coordinate(0, 0), Coordinate(0, 1)) == 1);  // 右
            assert(get_action(Coordinate(0, 1), Coordinate(0, 0)) == 2);  // 左
            assert(get_action(Coordinate(1, 0), Coordinate(0, 0)) == 3);  // 上
            assert(get_action(Coordinate(0, 0), Coordinate(1, 0)) == 4);  // 下
            assert(get_action(Coordinate(0, 0), Coordinate(0, 0)) == 0);  // 静止
            std::cout << "✓ 动作计算测试通过" << std::endl;
        }
        
        std::cout << "✓ 所有单元测试通过" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    PathConverterTester tester;
    
    if (argc == 1) {
        // 运行单元测试
        tester.run_unit_tests();
        std::cout << "\n用法: " << argv[0] << " <包含.path文件的目录>" << std::endl;
        std::cout << "      " << argv[0] << " <单个.path文件>" << std::endl;
        return 0;
    }
    
    std::string input_path = argv[1];
    
    if (!fs::exists(input_path)) {
        std::cerr << "错误: 路径不存在: " << input_path << std::endl;
        return 1;
    }
    
    // 先运行单元测试
    tester.run_unit_tests();
    std::cout << std::endl;
    
    if (fs::is_directory(input_path)) {
        // 测试整个目录
        tester.test_directory(input_path);
    } else if (input_path.size() >= 5 && input_path.substr(input_path.size() - 5) == ".path") {
        // 测试单个文件
        std::string bin_file = input_path;
        size_t pos = bin_file.find("path_files");
        if (pos != std::string::npos) {
            bin_file.replace(pos, 10, "input_data");
        }
        pos = bin_file.find(".path");
        if (pos != std::string::npos) {
            bin_file.replace(pos, 5, ".bin");
        }
        
        auto result = tester.test_single_file(input_path, bin_file);
        std::cout << "测试结果: " << (result.passed ? "通过" : "失败") << std::endl;
        if (!result.passed) {
            std::cout << "原因: " << result.message << std::endl;
        }
    } else {
        std::cerr << "错误: 不支持的文件类型" << std::endl;
        return 1;
    }
    
    return 0;
} 