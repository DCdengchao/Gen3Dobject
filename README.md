# Gen3Dobject

Gen3Dobject 是一个基于 C++ 和 CGAL 的三维对象生成工具。项目主要用于根据不同的三维空间分布生成采样点，并将已有的 `.off` 三维模型放置到这些采样点位置，最终合成为一个新的三维场景模型文件。

当前项目支持的空间分布包括：

- Gaussian distribution，高斯分布
- Zipf distribution，齐夫分布
- Poisson distribution，泊松分布
- Uniform distribution，均匀分布

生成结果以 `.off` 文件保存，可用于三维模型实验、空间数据生成、三维对象分布模拟、LOD/网格简化实验等场景。

---

## 1. 项目功能

本项目主要完成以下工作：

1. 从指定模型目录中读取 `.off` 格式的三维模型；
2. 根据指定分布生成三维空间点；
3. 对输入模型进行归一化、缩放和随机旋转；
4. 将模型复制并放置到生成的三维点位置；
5. 将所有模型合并为一个新的三维网格；
6. 输出为 `.off` 三维模型文件。

部分程序还支持基于 CGAL 的网格简化功能，可在生成大量对象时减少最终模型的面片数量。

---

## 2. 项目结构

```text
Gen3Dobject/
├── README.md
└── src/
    ├── c.txt
    ├── gengauss.cpp
    ├── genzipf.cpp
    ├── newtest.cpp
    ├── task1.cpp
    ├── task2.cpp
    ├── temp.cpp
    └── testread.cpp
