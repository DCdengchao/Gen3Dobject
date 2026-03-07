#include<bits/stdc++.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/bounding_box.h>
#include<CGAL/Surface_mesh_simplification/edge_collapse.h>
#include<CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_ratio_stop_predicate.h>
#include<CGAL/Surface_mesh.h>
#include<CGAL/Simple_cartesian.h>
#include<CGAL/Aff_transformation_3.h>
#include<CGAL/boost/graph/helpers.h>
#include<CGAL/Named_function_parameters.h>
#include<CGAL/boost/graph/named_params_helper.h>

namespace fs = std::filesystem;
using namespace std;
namespace SMS = CGAL::Surface_mesh_simplification;


// 使用 CGAL 的内核
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3 Point_3;
typedef K::Vector_3 Vector_3;
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
        double x = dist(gen);
        double y = dist(gen);
        double z = dist(gen);
        points.emplace_back(x, y, z);
    }
    return points;
}

// 计算网格的质心
Point_3 compute_mesh_centroid(Surface_mesh& mesh) {
    if(num_vertices(mesh) == 0) return Point_3(0, 0, 0);
    
    Point_3 sum(0, 0, 0);
    for(auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
        sum = sum + (mesh.point(*v_it) - Point_3(0, 0, 0));  // 将点转换为向量相加
    }
    
    double count = num_vertices(mesh);
    return Point_3(sum.x()/count, sum.y()/count, sum.z()/count);
}

// 归一化网格大小
void normalize_mesh(Surface_mesh& mesh, double target_size = 1.0) {
    if(num_vertices(mesh) == 0) return;
    
    // 计算包围盒
    std::vector<Point_3> points;
    for(auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
        points.push_back(mesh.point(*v_it));
    }
    
    if(points.empty()) return;
    
    // 计算边界
    double min_x = points[0].x(), max_x = points[0].x();
    double min_y = points[0].y(), max_y = points[0].y();
    double min_z = points[0].z(), max_z = points[0].z();
    
    for(const auto& p : points) {
        min_x = std::min(min_x, p.x());
        max_x = std::max(max_x, p.x());
        min_y = std::min(min_y, p.y());
        max_y = std::max(max_y, p.y());
        min_z = std::min(min_z, p.z());
        max_z = std::max(max_z, p.z());
    }
    
    double width = max_x - min_x;
    double height = max_y - min_y;
    double depth = max_z - min_z;
    double max_dim = std::max({width, height, depth});
    
    if(max_dim > 0) {
        double scale = target_size / max_dim;
        
        // 先移动到原点
        Point_3 centroid = compute_mesh_centroid(mesh);
        for(auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
            Point_3 old_p = mesh.point(*v_it);
            Point_3 centered = Point_3(old_p.x() - centroid.x(), 
                                     old_p.y() - centroid.y(), 
                                     old_p.z() - centroid.z());
            Point_3 scaled = Point_3(centered.x() * scale, 
                                   centered.y() * scale, 
                                   centered.z() * scale);
            mesh.point(*v_it) = scaled;
        }
    }
}

// 将网格移动到指定位置
void translate_mesh(Surface_mesh& mesh, const Point_3& translation) {
    for(auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
        Point_3 old_p = mesh.point(*v_it);
        mesh.point(*v_it) = Point_3(old_p.x() + translation.x(),
                                  old_p.y() + translation.y(),
                                  old_p.z() + translation.z());
    }
}

// 获取目录下所有OFF模型文件
std::vector<std::string> get_models(const std::string& models_dir) {
    std::vector<std::string> model_files;
    for (const auto& category_entry : fs::directory_iterator(models_dir)) {
        if (category_entry.is_directory()) {
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

// 去除字符串末尾的空白字符（包括\r\n）
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

// 自定义的OFF转Surface_mesh函数
Surface_mesh read_off_to_surface_mesh(const std::string& filepath) {
    Surface_mesh mesh;
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return mesh;
    }

    std::string line;
    std::getline(file, line); // 读取 "OFF"
    line = trim(line); // 去除行尾的\r\n等空白字符
    if (line != "OFF") {
        std::cerr << "Error: Invalid OFF file format (got '" << line << "'): " << filepath << std::endl;
        file.close();
        return mesh;
    }

    // 读取顶点数和面数
    int num_vertices, num_faces, num_edges;
    file >> num_vertices >> num_faces >> num_edges;
    
    if (file.fail() || num_vertices < 0 || num_faces < 0) {
        std::cerr << "Error: Invalid OFF file header in " << filepath << std::endl;
        file.close();
        return mesh;
    }

    // 读取顶点并添加到mesh
    std::vector<Surface_mesh::Vertex_index> vertices;
    for (int i = 0; i < num_vertices; ++i) {
        double x, y, z;
        file >> x >> y >> z;
        if (file.fail()) {
            std::cerr << "Error: Failed to read vertex " << i << " in " << filepath << std::endl;
            file.close();
            return mesh;
        }
        Point_3 point(x, y, z);
        vertices.push_back(mesh.add_vertex(point));
    }

    // 读取面
    for (int i = 0; i < num_faces; ++i) {
        int face_size;
        file >> face_size;
        if (file.fail() || face_size < 0) {
            std::cerr << "Error: Failed to read face " << i << " size in " << filepath << std::endl;
            file.close();
            return mesh;
        }
        
        if (face_size == 3) { // 仅处理三角面
            int v0, v1, v2;
            file >> v0 >> v1 >> v2;
            if (file.fail()) {
                std::cerr << "Error: Failed to read triangle face " << i << " in " << filepath << std::endl;
                file.close();
                return mesh;
            }
            
            if (v0 < 0 || v0 >= num_vertices || v1 < 0 || v1 >= num_vertices || v2 < 0 || v2 >= num_vertices) {
                std::cerr << "Error: Invalid vertex indices in face " << i << " of " << filepath << std::endl;
                file.close();
                return mesh;
            }
            
            mesh.add_face(vertices[v0], vertices[v1], vertices[v2]);
        } 
    }

    file.close();
    return mesh;
}


// 简化模型面片
Surface_mesh simplify_model_faces(const Surface_mesh& input_mesh, double stop_ratio) {
    Surface_mesh output_mesh = input_mesh;
    
    // 首先验证网格是否为三角网格
    if(!CGAL::is_triangle_mesh(output_mesh)){
        std::cerr << "Input geometry is not triangulated." << std::endl;
        return input_mesh; // 返回原始网格
    }

    // 进行额外的网格验证
    if(num_edges(output_mesh) == 0 || num_faces(output_mesh) <= 1) {
        std::cerr << "Mesh has too few edges/faces to simplify." << std::endl;
        return input_mesh; // 返回原始网格
    }

    try {
        // 创建副本用于简化，避免原网格被破坏
        Surface_mesh temp_mesh = output_mesh;

        // 设置停止条件
        SMS::Edge_count_ratio_stop_predicate<Surface_mesh> stop(stop_ratio);

        // 执行边坍缩简化，使用默认参数
        int r = SMS::edge_collapse(temp_mesh, stop);

        if(r >= 0) {
            std::cout << "Simplification successful. Removed " << r << " edges." << std::endl;
            return temp_mesh; // 返回简化后的网格
        } else {
            std::cout << "Simplification failed for this mesh, keeping original." << std::endl;
            return input_mesh; // 返回原始网格
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception during simplification: " << e.what() << std::endl;
        std::cout << "Keeping original mesh due to simplification error." << std::endl;
        return input_mesh; // 返回原始网格
    } catch (...) {
        std::cerr << "Unknown exception during simplification." << std::endl;
        std::cout << "Keeping original mesh due to simplification error." << std::endl;
        return input_mesh; // 返回原始网格
    }
}

void place_simplified_meshes_at_gaussian_points(
    const std::vector<Point_3>& gaussian_points,
    const std::string& models_dir,
    const std::string& output_file,
    double model_scale = 0.1,
    double simplify_ratio = 0.5,
    bool simplification = true) {

    // 获取所有模型文件
    std::vector<std::string> all_models = get_models(models_dir);
    if (all_models.empty()) {
        std::cerr << "Error: No models found!" << std::endl;
        return;
    }

    std::cout << "Found " << all_models.size() << " models." << std::endl;
    std::cout << "Placing models at " << gaussian_points.size() << " Gaussian points..." << std::endl;
    
    if (simplification) {
        std::cout << "Simplification: enabled (each model will have less faces)" << std::endl;
    } else {
        std::cout << "Simplification: disabled (models keep original geometry)" << std::endl;
    }

    // 创建最终合并的网格
    Surface_mesh combined_mesh;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> model_dist(0, all_models.size() - 1);

     // 用于生成随机旋转角度的分布（0到2π）
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);


    // 对于每个高斯点，放置一个模型
    for (size_t i = 0; i < gaussian_points.size(); ++i) {
        size_t model_idx = model_dist(gen);
        std::string& model_path = all_models[model_idx];

        // 测试打印模型
        cout<<"Processing model "<<(i+1)<<"/"<<gaussian_points.size()
            <<": "<<model_path<<endl;

        // 直接使用自定义的读取函数，这样能保证与前面代码的一致性
        Surface_mesh current_mesh = read_off_to_surface_mesh(model_path);

        if(num_vertices(current_mesh) == 0 || num_faces(current_mesh) == 0) {
            std::cerr << "Warning: Failed to load model " << model_path << ", skipping..." << std::endl;
            continue;
        }

        // 是否简化面片
        if(simplification) {
            current_mesh = simplify_model_faces(current_mesh, simplify_ratio);
        }
        
        
        // 归一化模型大小
        normalize_mesh(current_mesh, model_scale);

        
        // 将模型移动到对应的高斯点位置
        translate_mesh(current_mesh, gaussian_points[i]);

        // 将当前网格合并到总网格中
        // 由于Surface_mesh没有直接的合并方法，我们需要手动复制顶点和面
        std::map<typename Surface_mesh::Vertex_index, typename Surface_mesh::Vertex_index> vertex_map;
        
        // 复制顶点
        for(auto v_it = current_mesh.vertices_begin(); v_it != current_mesh.vertices_end(); ++v_it) {
            Point_3 p = current_mesh.point(*v_it);
            auto new_v = combined_mesh.add_vertex(p);
            vertex_map[*v_it] = new_v;
        }
        
        // 复制面
        for(auto f_it = current_mesh.faces_begin(); f_it != current_mesh.faces_end(); ++f_it) {
            std::vector<typename Surface_mesh::Vertex_index> face_vertices;
            auto halfedge_range = current_mesh.halfedges_around_face(current_mesh.halfedge(*f_it));
            for(auto he_it = halfedge_range.first; he_it != halfedge_range.second; ++he_it) {
                auto source_v = current_mesh.source(*he_it);
                face_vertices.push_back(vertex_map[source_v]);
            }
            combined_mesh.add_face(face_vertices);
        }


        if ((i + 1) % 100 == 0) {
            std::cout << "Processed " << (i + 1) << " / " << gaussian_points.size() 
                      << " points " << std::endl; 
        }
    }

    // 保存合并后的网格
    if(CGAL::IO::write_polygon_mesh(output_file, combined_mesh)) {
        std::cout << "Successfully created " << output_file << " with " 
                  << num_vertices(combined_mesh) << " vertices and " 
                  << num_faces(combined_mesh) << " faces." << std::endl;
    } else {
        std::cerr << "Error: Failed to write output file " << output_file << std::endl;
    }
}

int main(){
    std::string models_dir = "/home/dcc/code/MODELS";
    std::string output_file = "combined_mesh.off";
    double model_scale = 0.1;
    int num_points = 25;
    double mean = 0.0;
    double stddev = 1.0;
    double simplify_ratio = 0.5;
    bool simplification = true;  // 是否简化模型

    std::cout << "Generating " << num_points << " Gaussian points (mean=" << mean 
              << ", stddev=" << stddev << ")..." << std::endl;

    // 生成三维高斯分布的点
    auto gaussian_points = generate_gaussian_points(num_points, mean, stddev);

    std::cout << "Generated " << gaussian_points.size() << " Gaussian points." << std::endl;

    // 直接将简化后的网格放置到高斯点上
    place_simplified_meshes_at_gaussian_points(
        gaussian_points, models_dir, output_file, 
        model_scale, simplify_ratio, simplification);

    return 0;
}
