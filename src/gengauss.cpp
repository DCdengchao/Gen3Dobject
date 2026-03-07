#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <string>
#include <map>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/bounding_box.h>
#include <filesystem>
namespace fs = std::filesystem;
using namespace std;


#define M_PI 3.14159265358979323846

// 使用 CGAL 的内核
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3 Point_3;
typedef K::Iso_cuboid_3 Iso_cuboid_3;


// 定义三维点结构
struct Point3D {
    double x, y, z;

    // 构造函数
    Point3D(double x = 0.0, double y = 0.0, double z = 0.0) : x(x), y(y), z(z) {}

    // 重载加法运算符
    Point3D operator+(const Point3D& other) const {
        return Point3D(x + other.x, y + other.y, z + other.z);
    }

    // 重载减法运算符
    Point3D operator-(const Point3D& other) const {
        return Point3D(x - other.x, y - other.y, z - other.z);
    }

    // 重载标量乘法
    Point3D operator*(double scalar) const {
        return Point3D(x * scalar, y * scalar, z * scalar);
    }
};

// 3D模型结构体（存储顶点和面）
struct Model3D {
    std::vector<Point3D> vertices;
    std::vector<std::vector<int>> faces;
};


// 生成三维高斯分布的点
std::vector<Point_3> generate_gaussian_points(int num_points, double mean, double stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(mean, stddev);

    std::vector<Point_3> points;
    points.reserve(num_points);
    
    for (int i = 0; i < num_points; ++i) {
        // 直接使用正态分布生成三个独立的高斯随机变量
        double x = dist(gen);
        double y = dist(gen);
        double z = dist(gen);
        
        points.emplace_back(x, y, z);
    }
    return points;
}

// 将 CGAL Point_3 转换为 Point3D
Point3D convert_point(const Point_3& p) {
    return Point3D(p.x(), p.y(), p.z());
}

// 将 std::vector<Point_3> 转换为 std::vector<Point3D>
std::vector<Point3D> convert_points(const std::vector<Point_3>& points) {
    std::vector<Point3D> result;
    result.reserve(points.size());
    for (const auto& p : points) {
        result.push_back(convert_point(p));
    }
    return result;
}

// 去除字符串末尾的空白字符（包括\r\n）
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

// 读取OFF文件
Model3D read_off_model(const std::string& filepath) {
    Model3D model;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return model;
    }

    std::string line;
    std::getline(file, line); // 读取 "OFF"
    line = trim(line); // 去除行尾的\r\n等空白字符
    if (line != "OFF") {
        std::cerr << "Error: Invalid OFF file format (got '" << line << "'): " << filepath << std::endl;
        file.close();
        return model;
    }

    // 读取顶点数和面数
    int num_vertices, num_faces, num_edges;
    file >> num_vertices >> num_faces >> num_edges;
    
    // 检查读取是否成功
    if (file.fail() || num_vertices < 0 || num_faces < 0) {
        std::cerr << "Error: Invalid OFF file header in " << filepath << std::endl;
        file.close();
        return model;
    }

    // 读取顶点
    model.vertices.reserve(num_vertices);
    for (int i = 0; i < num_vertices; ++i) {
        double x, y, z;
        file >> x >> y >> z;
        if (file.fail()) {
            std::cerr << "Error: Failed to read vertex " << i << " in " << filepath << std::endl;
            file.close();
            return model;
        }
        model.vertices.emplace_back(x, y, z);
    }

    // 读取面
    model.faces.reserve(num_faces);
    for (int i = 0; i < num_faces; ++i) {
        int face_size;
        file >> face_size;
        if (file.fail() || face_size < 0) {
            std::cerr << "Error: Failed to read face " << i << " size in " << filepath << std::endl;
            file.close();
            return model;
        }
        std::vector<int> face;
        face.reserve(face_size);
        for (int j = 0; j < face_size; ++j) {
            int vertex_idx;
            file >> vertex_idx;
            if (file.fail()) {
                std::cerr << "Error: Failed to read face " << i << " vertex " << j << " in " << filepath << std::endl;
                file.close();
                return model;
            }
            if (vertex_idx < 0 || vertex_idx >= num_vertices) {
                std::cerr << "Warning: Invalid vertex index " << vertex_idx << " in face " << i << " of " << filepath << std::endl;
            }
            face.push_back(vertex_idx);
        }
        model.faces.push_back(face);
    }

    file.close();
    return model;
}

// 计算模型的质心
Point3D compute_centroid(const std::vector<Point3D>& vertices) {
    if (vertices.empty()) {
        return Point3D(0, 0, 0);
    }
    
    double sum_x = 0, sum_y = 0, sum_z = 0;
    for (const auto& v : vertices) {
        sum_x += v.x;
        sum_y += v.y;
        sum_z += v.z;
    }
    size_t n = vertices.size();
    return Point3D(sum_x / n, sum_y / n, sum_z / n);
}

// 归一化模型（使模型中心在原点，并缩放到合适大小）
void normalize_model(Model3D& model, double target_size = 1.0) {
    if (model.vertices.empty()) return;

    // 计算质心
    Point3D centroid = compute_centroid(model.vertices);

    // 计算包围盒以确定缩放因子
    double min_x = model.vertices[0].x, max_x = model.vertices[0].x;
    double min_y = model.vertices[0].y, max_y = model.vertices[0].y;
    double min_z = model.vertices[0].z, max_z = model.vertices[0].z;

    for (const auto& v : model.vertices) {
        min_x = std::min(min_x, v.x);
        max_x = std::max(max_x, v.x);
        min_y = std::min(min_y, v.y);
        max_y = std::max(max_y, v.y);
        min_z = std::min(min_z, v.z);
        max_z = std::max(max_z, v.z);
    }

    double width = max_x - min_x;
    double height = max_y - min_y;
    double depth = max_z - min_z;
    double max_dim = std::max({width, height, depth});

    // 中心化并归一化
    if (max_dim > 0) {
        double scale = target_size / max_dim;
        for (auto& v : model.vertices) {
            v = (v - centroid) * scale;
        }
    }
}

// 绕X轴旋转点
Point3D rotate_x(const Point3D& p, double angle) {
    double cos_a = std::cos(angle);
    double sin_a = std::sin(angle);
    return Point3D(
        p.x,
        p.y * cos_a - p.z * sin_a,
        p.y * sin_a + p.z * cos_a
    );
}

// 绕Y轴旋转点
Point3D rotate_y(const Point3D& p, double angle) {
    double cos_a = std::cos(angle);
    double sin_a = std::sin(angle);
    return Point3D(
        p.x * cos_a + p.z * sin_a,
        p.y,
        -p.x * sin_a + p.z * cos_a
    );
}

// 绕Z轴旋转点
Point3D rotate_z(const Point3D& p, double angle) {
    double cos_a = std::cos(angle);
    double sin_a = std::sin(angle);
    return Point3D(
        p.x * cos_a - p.y * sin_a,
        p.x * sin_a + p.y * cos_a,
        p.z
    );
}

// 旋转模型（使用欧拉角：绕X、Y、Z轴旋转）
// 注意：模型应该已经中心化到原点，旋转后保持在原点以便后续平移
void rotate_model(Model3D& model, double angle_x, double angle_y, double angle_z) {
    if (model.vertices.empty()) return;
    
    // 确保模型中心在原点（计算质心并平移）
    Point3D centroid = compute_centroid(model.vertices);
    
    // 先平移到原点
    for (auto& v : model.vertices) {
        v = v - centroid;
    }
    
    // 应用旋转（XYZ顺序：先绕X轴，再绕Y轴，最后绕Z轴）
    for (auto& v : model.vertices) {
        // 先绕X轴旋转
        v = rotate_x(v, angle_x);
        // 再绕Y轴旋转
        v = rotate_y(v, angle_y);
        // 最后绕Z轴旋转
        v = rotate_z(v, angle_z);
    }
    
    // 不平移回原位置，保持在原点，以便后续translate_model正确工作
    // 这样模型旋转后仍然以原点为中心，translate_model可以正确将其移动到目标位置
}

// 将模型放置在指定位置
void translate_model(Model3D& model, const Point3D& position) {
    for (auto& v : model.vertices) {
        v = v + position;
    }
}

// // 计算面的中心点
// Point3D compute_face_center(const Model3D& model, const std::vector<int>& face) {
//     if (face.empty()) return Point3D(0, 0, 0);
    
//     double sum_x = 0, sum_y = 0, sum_z = 0;
//     for (int vertex_idx : face) {
//         if (vertex_idx >= 0 && vertex_idx < static_cast<int>(model.vertices.size())) {
//             sum_x += model.vertices[vertex_idx].x;
//             sum_y += model.vertices[vertex_idx].y;
//             sum_z += model.vertices[vertex_idx].z;
//         }
//     }
//     size_t n = face.size();
//     return Point3D(sum_x / n, sum_y / n, sum_z / n);
// }


// 获取airplane模型文件列表
std::vector<std::string> get_models(const std::string& models_dir) {
    std::vector<std::string> model_files;

    // fs::path airplane_dir = fs::path(models_dir) / "bed";
    
    // if (!fs::exists(airplane_dir) || !fs::is_directory(airplane_dir)) {
    //     std::cerr << "Error: Model directory not found: " << airplane_dir << std::endl;
    //     return model_files;
    // }

    // for (const auto& entry : fs::directory_iterator(airplane_dir)) {
    //     if (entry.is_regular_file() && entry.path().extension() == ".off") {
    //         model_files.push_back(entry.path().string());
    //     }
    // }
     // 遍历 models_dir 下的每一个条目
    for (const auto& category_entry : fs::directory_iterator(models_dir)) {
        // 只处理子目录（跳过文件）
        if (category_entry.is_directory()) {
            // 遍历该子目录中的所有文件
            for (const auto& model_entry : fs::directory_iterator(category_entry.path())) {
                if (model_entry.is_regular_file() && model_entry.path().extension() == ".off") {
                    model_files.push_back(model_entry.path().string());
                }
            }
        }
    }

    std::sort(model_files.begin(), model_files.end());
    return model_files;
}

// 将airplane模型填充到高斯分布的点上
void fill_gaussian_points_with_airplanes(
    const std::vector<Point3D>& gaussian_points,
    const std::string& models_dir,
    const std::string& output_file,
    double model_scale = 0.1,
    bool random_rotation = true) {
    
    // 获取所有模型文件
    std::vector<std::string> airplane_models = get_models(models_dir);
    if (airplane_models.empty()) {
        std::cerr << "Error: No models found!" << std::endl;
        return;
    }

    std::cout << "Found " << airplane_models.size() << "models." << std::endl;
    std::cout << "Placing models at " << gaussian_points.size() << " Gaussian points..." << std::endl;
    
    if (random_rotation) {
        std::cout << "Random rotation: enabled (each model will have random orientation)" << std::endl;
    } else {
        std::cout << "Random rotation: disabled (models keep original orientation)" << std::endl;
    }

    // 合并后的所有顶点和面
    std::vector<Point3D> all_vertices;
    std::vector<std::vector<int>> all_faces;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> model_dist(0, airplane_models.size() - 1);
    
    // 用于生成随机旋转角度的分布（0到2π）
    // 注意：使用全范围旋转，确保每个模型都有不同的朝向
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);

    size_t current_vertex_offset = 0;

    // 对于每个高斯分布的点，放置一个airplane模型
    for (size_t i = 0; i < gaussian_points.size(); ++i) {
        // 随机选择一个airplane模型
        size_t model_idx = model_dist(gen);
        const std::string& model_path = airplane_models[model_idx];

        Model3D model;
        
        // 直接读取模型
        model = read_off_model(model_path);
        if (model.vertices.empty()) {
            std::cerr << "Warning: Failed to load model " << model_path << ", skipping..." << std::endl;
            continue;
        }

        // 归一化模型
        normalize_model(model, model_scale);


        // 应用随机旋转（如果需要）
        if (random_rotation) {
            double angle_x = angle_dist(gen);
            double angle_y = angle_dist(gen);
            double angle_z = angle_dist(gen);
            
            // // 调试：输出前几个模型的旋转角度（仅前5个）
            // if (i < 5) {
            //     std::cout << "Model " << (i + 1) << " rotation angles (degrees): "
            //               << "X=" << (angle_x * 180.0 / M_PI) << ", "
            //               << "Y=" << (angle_y * 180.0 / M_PI) << ", "
            //               << "Z=" << (angle_z * 180.0 / M_PI) << std::endl;
            // }
            
            rotate_model(model, angle_x, angle_y, angle_z);
        }

        // 将模型平移到高斯分布的点位置
        translate_model(model, gaussian_points[i]);

        // 添加到合并后的顶点列表
        all_vertices.insert(all_vertices.end(), model.vertices.begin(), model.vertices.end());

        // 添加面（需要调整顶点索引）
        for (const auto& face : model.faces) {
            std::vector<int> adjusted_face;
            for (int vertex_idx : face) {
                adjusted_face.push_back(vertex_idx + current_vertex_offset);
            }
            all_faces.push_back(adjusted_face);
        }

        current_vertex_offset += model.vertices.size();

        if ((i + 1) % 100 == 0) {
            std::cout << "Processed " << (i + 1) << " / " << gaussian_points.size() << " points..." << std::endl;
        }
    }

    // 写入合并后的OFF文件
    std::ofstream out_file(output_file);
    if (!out_file) {
        std::cerr << "Error: Could not open output file " << output_file << std::endl;
        return;
    }

    out_file << "OFF\n";
    out_file << all_vertices.size() << " " << all_faces.size() << " 0\n";

    // 写入顶点
    for (const auto& v : all_vertices) {
        out_file << v.x << " " << v.y << " " << v.z << "\n";
    }

    // 写入面
    for (const auto& face : all_faces) {
        out_file << face.size();
        for (int idx : face) {
            out_file << " " << idx;
        }
        out_file << "\n";
    }

    out_file.close();
    std::cout << "Successfully created " << output_file << " with " 
              << all_vertices.size() << " vertices and " << all_faces.size() << " faces." << std::endl;
}

int main(int argc, char* argv[]) {
    // 默认参数
    int num_points = 500;
    double mean = 0.0;
    double stddev = 1.0;
    std::string models_dir = "/home/dcc/code/MODELS";
    std::string output_file = "Gen3Dproject.off";
    double model_scale = 0.1;
    bool random_rotation = false;  // 是否随机旋转模型

    // 解析命令行参数
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-n" || arg == "--num-points") {
                if (i + 1 < argc) {
                    num_points = std::stoi(argv[++i]);
                }
            } else if (arg == "-m" || arg == "--mean") {
                if (i + 1 < argc) {
                    mean = std::stod(argv[++i]);
                }
            } else if (arg == "-s" || arg == "--stddev") {
                if (i + 1 < argc) {
                    stddev = std::stod(argv[++i]);
                }
            } else if (arg == "-d" || arg == "--models-dir") {
                if (i + 1 < argc) {
                    models_dir = argv[++i];
                }
            } else if (arg == "-o" || arg == "--output") {
                if (i + 1 < argc) {
                    output_file = argv[++i];
                }
            } else if (arg == "--scale") {
                if (i + 1 < argc) {
                    model_scale = std::stod(argv[++i]);
                }
            } else if (arg == "--no-rotation") {
                random_rotation = false;
            } else if (arg == "-h" || arg == "--help") {
                std::cout << "Usage: " << argv[0] << " [options]\n"
                          << "Options:\n"
                          << "  -n, --num-points N        Number of Gaussian points (default: 1000)\n"
                          << "  -m, --mean VALUE          Mean of Gaussian distribution (default: 0.0)\n"
                          << "  -s, --stddev VALUE        Standard deviation of Gaussian distribution (default: 1.0)\n"
                          << "  -d, --models-dir PATH     Path to MODELS directory (default: MODELS)\n"
                          << "  -o, --output FILE         Output OFF file (default: gaussian_airplanes.off)\n"
                          << "  --scale VALUE             Model scale factor (default: 0.1)\n"
                          << "  --no-rotation             Disable random rotation (keep original orientation)\n"
                          << "  -h, --help                Show this help message\n";
                return 0;
            }
        }
    }

    std::cout << "Generating " << num_points << " Gaussian points (mean=" << mean 
              << ", stddev=" << stddev << ")..." << std::endl;

    // 生成三维高斯分布的点
    auto gaussian_points_cgal = generate_gaussian_points(num_points, mean, stddev);

    std::cout << "Generated " << gaussian_points_cgal.size() << " Gaussian points." << std::endl;

    // 将 CGAL Point_3 转换为 Point3D
    auto gaussian_points = convert_points(gaussian_points_cgal);

    // 将airplane模型填充到高斯分布的点上
    fill_gaussian_points_with_airplanes(gaussian_points, models_dir, output_file, 
                                        model_scale, random_rotation);

    return 0;
}
