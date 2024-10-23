#include <iostream>
#include <yaml-cpp/yaml.h>
#include "H5Cpp.h"
#include <vector>
#include <string>
#include <filesystem> // C++17 文件系统库

using namespace H5;
namespace fs = std::filesystem;

void processYAMLtoHDF5(const std::string& yamlFilePath, const std::string& h5FilePath) {
    // 加载 YAML 文件
    YAML::Node config = YAML::LoadFile(yamlFilePath);

    // 打开或创建 HDF5 文件
    H5File file(h5FilePath, H5F_ACC_TRUNC);

    // 存储 statistics 数据
    YAML::Node stats = config["statistics"];

    // 创建 /statistics 组
    Group statsGroup = file.createGroup("/statistics");

    double cost = stats["cost"].as<double>();
    double newnode_runtime = stats["newnode_runtime"].as<double>();
    double focal_score_time = stats["focal_score_time"].as<double>();
    double firstconflict_runtime = stats["firstconflict_runtime"].as<double>();
    double runtime = stats["runtime"].as<double>();
    double lowlevel_search_time = stats["lowlevel_search_time"].as<double>();
    int total_lowlevel_node = stats["total_lowlevel_node"].as<int>();
    int lowLevelExpanded = stats["lowLevelExpanded"].as<int>();
    int numTaskAssignments = stats["numTaskAssignments"].as<int>();
    std::string map = stats["map"].as<std::string>();

    // 定义标量数据空间
    DataSpace scalarSpace(H5S_SCALAR);

    // 写入 double 类型的属性
    statsGroup.createAttribute("cost", PredType::NATIVE_DOUBLE, scalarSpace).write(PredType::NATIVE_DOUBLE, &cost);
    statsGroup.createAttribute("newnode_runtime", PredType::NATIVE_DOUBLE, scalarSpace).write(PredType::NATIVE_DOUBLE, &newnode_runtime);
    statsGroup.createAttribute("focal_score_time", PredType::NATIVE_DOUBLE, scalarSpace).write(PredType::NATIVE_DOUBLE, &focal_score_time);
    statsGroup.createAttribute("firstconflict_runtime", PredType::NATIVE_DOUBLE, scalarSpace).write(PredType::NATIVE_DOUBLE, &firstconflict_runtime);
    statsGroup.createAttribute("runtime", PredType::NATIVE_DOUBLE, scalarSpace).write(PredType::NATIVE_DOUBLE, &runtime);
    statsGroup.createAttribute("lowlevel_search_time", PredType::NATIVE_DOUBLE, scalarSpace).write(PredType::NATIVE_DOUBLE, &lowlevel_search_time);

    // 写入 int 类型的属性
    statsGroup.createAttribute("total_lowlevel_node", PredType::NATIVE_INT, scalarSpace).write(PredType::NATIVE_INT, &total_lowlevel_node);
    statsGroup.createAttribute("lowLevelExpanded", PredType::NATIVE_INT, scalarSpace).write(PredType::NATIVE_INT, &lowLevelExpanded);
    statsGroup.createAttribute("numTaskAssignments", PredType::NATIVE_INT, scalarSpace).write(PredType::NATIVE_INT, &numTaskAssignments);

    // 写入 map 字符串属性
    StrType strType(PredType::C_S1, H5T_VARIABLE);  // 可变长度字符串类型
    statsGroup.createAttribute("map", strType, scalarSpace).write(strType, map);

    // 存储 schedule 数据
    YAML::Node schedule = config["schedule"];

    // 创建 /schedule 组
    Group scheduleGroup = file.createGroup("/schedule");

    for (YAML::const_iterator it = schedule.begin(); it != schedule.end(); ++it) {
        std::string agentName = it->first.as<std::string>();
        YAML::Node agentData = it->second;

        // 准备保存 schedule 数据
        hsize_t dims[2] = {agentData.size(), 3}; // 行数等于时间步，列数为 x, y, t
        DataSpace dataspace(2, dims);

        // 创建 agent 组
        Group agentGroup = scheduleGroup.createGroup(agentName);

        // 准备坐标数据
        std::vector<int> coordinates(agentData.size() * 3);
        for (size_t i = 0; i < agentData.size(); ++i) {
            coordinates[i * 3 + 0] = agentData[i]["x"].as<int>();
            coordinates[i * 3 + 1] = agentData[i]["y"].as<int>();
            coordinates[i * 3 + 2] = agentData[i]["t"].as<int>();
        }

        // 保存数据集
        DataSet dataset = agentGroup.createDataSet("trajectory", PredType::NATIVE_INT, dataspace);
        dataset.write(coordinates.data(), PredType::NATIVE_INT);
    }

    file.close();
}
int main(int argc, char* argv[]) {
    // 检查命令行参数
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <yaml_folder> <output_folder>" << std::endl;
        return 1;
    }

    std::string yamlFolder = argv[1];  // 第一个参数是 YAML 文件夹路径
    std::string outputFolder = argv[2];  // 第二个参数是 HDF5 输出文件夹路径

    // 检查输出文件夹是否存在，如果不存在则创建
    if (!fs::exists(outputFolder)) {
        fs::create_directory(outputFolder);
    }

    // 遍历 YAML 文件夹
    for (const auto& entry : fs::directory_iterator(yamlFolder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".yaml") {
            std::string yamlFilePath = entry.path().string();
            std::string h5FilePath = outputFolder + "/" + entry.path().stem().string() + ".h5";  // 根据 YAML 文件名生成 HDF5 文件名

            // 处理每个 YAML 文件并生成 HDF5 文件
            std::cout << "Processing: " << yamlFilePath << " -> " << h5FilePath << std::endl;
            processYAMLtoHDF5(yamlFilePath, h5FilePath);
        }
    }

    return 0;
}
