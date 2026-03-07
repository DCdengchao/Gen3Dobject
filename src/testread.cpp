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
#include<CGAL/Surface_mesh_simplification/edge_collapse.h>
#include<CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_count_ratio_stop_predicate.h>
#include<CGAL/Surface_mesh.h>
#include<CGAL/Simple_cartesian.h>
#include<chrono>
namespace fs = std::filesystem;
using namespace std;
namespace SMS = CGAL::Surface_mesh_simplification;

#define M_PI 3.14159265358979323846

// 使用 CGAL 的内核
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3 Point_3;
typedef K::Iso_cuboid_3 Iso_cuboid_3;

typedef CGAL::Simple_cartesian<double> Kernel;
typedef CGAL::Surface_mesh<Point_3> Surface_mesh;

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

std::vector<std::string> get_models(const std::string& models_dir) {
    std::vector<std::string> model_files;
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

// 将模型放置在指定位置
void translate_model(Model3D& model, const Point3D& position) {
    for (auto& v : model.vertices) {
        v = v + position;
    }
}

//简化模型面片
// std::string simplify_model_faces(const std::string& filepath,double stop_ratio,const std::string& output_file="simp.off") {
//     Surface_mesh surface_mesh;
//     const std::string filename = CGAL::data_file_path(filepath);

//     if(!CGAL::IO::read_polygon_mesh(filename,surface_mesh)){
//         std::cerr<<"Cannot open file "<<filename<<std::endl;
//         return "";
//     }
//     if(!CGAL::is_triangle_mesh(surface_mesh)){
//         std::cerr<<"Input geometry is not triangulated."<<std::endl;
//         return "";
//     }
//     std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
//     // std::cout << num_vertices(surface_mesh) << " vertices, "
//     //           << num_faces(surface_mesh) << " faces before simplification." << std::endl;   
    
//     SMS::Edge_count_ratio_stop_predicate<Surface_mesh> stop(stop_ratio);

//     int r = SMS::edge_collapse(surface_mesh, stop);
//     std::chrono::steady_clock::time_point end=std::chrono::steady_clock::now();

//     // std::cout << "\nFinished!\n" << r << " edges removed.\n" << surface_mesh.number_of_edges() << " final edges.\n";
//     // std::cout << "Time elapsed:" << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()<<"ms\n";

//     CGAL::IO::write_polygon_mesh(output_file,surface_mesh);
//     return output_file;
// }

Model3D simplify_model_faces(const std::string& filepath, double stop_ratio) {
    Surface_mesh surface_mesh;
    
    // const std::string filename = CGAL::data_file_path(filepath);
    // 直接使用传入的filepath，不使用CGAL::data_file_path
    if(!CGAL::IO::read_polygon_mesh(filepath, surface_mesh)){
        std::cerr << "Cannot open file " << filepath << std::endl;
        return Model3D{}; // 返回空模型
    }
    
    if(!CGAL::is_triangle_mesh(surface_mesh)){
        std::cerr << "Input geometry is not triangulated." << std::endl;
        return Model3D{};
    }
    
    std::cout << num_vertices(surface_mesh) << " vertices, "
              << num_faces(surface_mesh) << " faces before simplification." << std::endl;   
    
    // std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    
    SMS::Edge_count_ratio_stop_predicate<Surface_mesh> stop(stop_ratio);

    int r = SMS::edge_collapse(surface_mesh, stop);
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    // std::cout << "\nFinished!\n" << r << " edges removed.\n";
    // std::cout << num_vertices(surface_mesh) << " vertices, "
    //           << num_faces(surface_mesh) << " faces after simplification." << std::endl;
    // std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms\n";

    // 将Surface_mesh转换为Model3D结构
    Model3D simplified_model;
    
    // 提取顶点
    for(auto v_it = surface_mesh.vertices_begin(); v_it != surface_mesh.vertices_end(); ++v_it) {
        Point_3 p = surface_mesh.point(*v_it);
        simplified_model.vertices.push_back(Point3D(p.x(), p.y(), p.z()));
    }
    
    // 提取面
    for(auto f_it = surface_mesh.faces_begin(); f_it != surface_mesh.faces_end(); ++f_it) {
        std::vector<int> face;
        auto halfedge_range = surface_mesh.halfedges_around_face(surface_mesh.halfedge(*f_it));
        for(auto he_it = halfedge_range.first; he_it != halfedge_range.second; ++he_it) {
            auto v = surface_mesh.target(*he_it);
            int vertex_index = std::distance(surface_mesh.vertices_begin(), 
                                           std::find(surface_mesh.vertices_begin(), 
                                                   surface_mesh.vertices_end(), v));
            face.push_back(vertex_index);
        }
        simplified_model.faces.push_back(face);
    }
    
    return simplified_model;
}

void fill_points_with_simplified_models(
    const std::vector<Point3D>& model_points,
    const std::string&models_dir,
    const std::string& output_file,
    double model_scale = 0.1,
    double simplify_ratio=0.5,
    bool simplifification=true) {

    // 获取所有模型文件
    std::vector<std::string> all_models = get_models(models_dir);
    if (all_models.empty()) {
        std::cerr << "Error: No models found!" << std::endl;
        return;
    }

    std::cout << "Found " << all_models.size() << "models." << std::endl;
    std::cout << "Placing models at " << model_points.size() << " Gaussian points..." << std::endl;
    
    if (simplifification) {
        std::cout << "Simplification: enabled (each model will have less faces)" << std::endl;
    } else {
        std::cout << "Simplification: disabled (models keep original geometry)" << std::endl;
    }

    // 合并后的所有顶点和面
    std::vector<Point3D> all_vertices;
    std::vector<std::vector<int>> all_faces;   

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> model_dist(0, all_models.size() - 1);

    size_t current_vertex_offset = 0;

    //对于每个点，放置一个简化后的模型
    for (size_t i = 0; i < model_points.size(); ++i) {
        size_t model_idx = model_dist(gen);
        const std::string& model_path = all_models[model_idx];

        // cout << "Model path: " << model_path << endl;

        Model3D model;

        //简化模型
        if (simplifification) {
            model = simplify_model_faces(model_path, simplify_ratio);
            if (model.vertices.empty()) {
                std::cerr << "Warning: Simplification failed for " << model_path << ", skipping..." << std::endl;
                continue;   
            }
        }else {
            model = read_off_model(model_path);
            if (model.vertices.empty()) {
                std::cerr << "Warning: Failed to load model " << model_path << ", skipping..." << std::endl;
                continue;   
            }
        }

        //归一化模型
        normalize_model(model, model_scale);

        // 将模型平移到高斯分布的点位置
        translate_model(model, model_points[i]);

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
            std::cout << "Processed " << (i + 1) << " / " << model_points.size() << " points..." << std::endl;
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

int main(){
    std::string models_dir = "/home/dcc/code/MODELS";
    std::string output_file = "simp.off";
    double model_scale = 0.1;
    int num_points = 10;
    double mean = 0.0;
    double stddev = 1.0;
    double simplify_ratio=0.5;
    bool simplification = true;  // 是否简化模型

    std::cout << "Generating " << num_points << " Gaussian points (mean=" << mean 
              << ", stddev=" << stddev << ")..." << std::endl;

    // 生成三维高斯分布的点
    auto gaussian_points_cgal = generate_gaussian_points(num_points, mean, stddev);
    auto gaussian_points = convert_points(gaussian_points_cgal);

    std::cout << "Generated " << gaussian_points.size() << " Gaussian points." << std::endl;

    // 将模型填充到高斯分布的点上
    fill_points_with_simplified_models(gaussian_points, models_dir, output_file, model_scale,simplify_ratio,simplification);

    return 0;
}

