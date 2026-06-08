# 图像向量检索实践实验报告

## 1. 实验流程与设置

本实验完成一个小型图像向量检索流程：首先下载并整理 Oxford-IIIT Pet 图像数据集，然后使用 CLIP ViT-B/32 图像编码器将图片转换为 768 维向量；接着将全量向量随机划分为 base 向量集合和 query 向量集合；再使用 FAISS `IndexFlatL2` 精确检索生成 ground truth top10；最后使用 `IndexIVFFlat` 和 `IndexHNSWFlat` 两种近似最近邻索引完成检索，并统计索引构建时间、索引大小、QPS 和 Recall@10。

### 1.1 数据集

- 数据集名称：Oxford-IIIT Pet
- 数据集来源：https://www.robots.ox.ac.uk/~vgg/data/pets/
- 图片数量：填写实际使用数量
- 标注信息：包含 37 个猫狗品种类别，并包含 species、breed id 等标注
- 子集抽取方式：如果使用完整数据集，写“未抽取子集”；如果使用子集，写明随机种子和抽取数量

### 1.2 Embedding 模型

- 模型：CLIP ViT-B/32
- 输入类型：RGB 图像
- 输出向量维度：768
- 向量数据类型：float32
- 是否归一化：是，使用 L2 normalization
- 是否支持文本搜图：CLIP 支持图文统一向量空间，本实验主流程只使用图像到图像检索

### 1.3 Base/Query 划分与 Ground Truth

- 随机种子：42
- Query 数量：填写实际 query 数量
- Base 数量：填写实际 base 数量
- 划分方式：随机选取若干图片作为 query，其余图片作为 base，query 图片不出现在 base 中
- Ground truth 生成方式：使用 FAISS `IndexFlatL2` 对每条 query 在 base 集合上进行精确 L2 检索，保存 top10 结果

### 1.4 ANNS 索引方法

本实验选择两种 FAISS ANNS 方法：

- `IndexIVFFlat`：主要参数为 `nlist=100, nprobe=10`
- `IndexHNSWFlat`：主要参数为 `M=32, efConstruction=80, efSearch=64`

`IndexFlatL2` 只用于生成 ground truth，不作为近似检索方法参与对比。

## 2. 实验结果统计

将 `outputs/metrics.csv` 的结果填入下表。

| 指标 | 第一组 | 第二组 |
|---|---:|---:|
| 数据集名称 | Oxford-IIIT Pet | Oxford-IIIT Pet |
| Embedding 模型 | CLIP ViT-B/32 | CLIP ViT-B/32 |
| 向量维度 | 768 | 768 |
| 向量数据类型 | float32 | float32 |
| Base 向量数量 | 填写 | 填写 |
| Base 向量大小 MB | 填写 | 填写 |
| Query 向量数量 | 填写 | 填写 |
| 索引方法 | IndexIVFFlat | IndexHNSWFlat |
| 索引主要参数 | nlist=100, nprobe=10 | M=32, efConstruction=80, efSearch=64 |
| 索引构建时间 ms | 填写 | 填写 |
| 索引占用空间 MB | 填写 | 填写 |
| QPS | 填写 | 填写 |
| Recall@10 | 填写 | 填写 |

## 3. 检索结果可视化

插入 `outputs/query0_visualization.png`。

可视化图包含：

- 第 1 行：Query 图片
- 第 2 行：Ground Truth Top-10
- 第 3 行：IVFFlat Top-10
- 第 4 行：HNSW Top-10

对该 query 的观察：

1. 填写哪种 ANNS 方法与 ground truth 更接近。
2. 填写是否存在明显不符合人眼直觉的结果。
3. 如果存在不符合直觉的结果，分析可能来自 embedding 表征，还是 ANNS 近似搜索误差。

## 4. 结果分析

### 4.1 ANNS 方法的速度与召回率对比

从 Recall@10 看，填写哪种方法更高；从 QPS 看，填写哪种方法更快。一般来说，参数越保守、搜索范围越大，Recall@10 越高，但 QPS 可能下降。

### 4.2 索引大小与构建时间

比较两种索引的索引文件大小和构建时间。HNSW 会额外保存图结构，因此索引大小可能高于 IVFFlat；IVFFlat 需要训练聚类中心，因此其构建过程包含训练和添加向量两个步骤。

### 4.3 Embedding 表征对结果的影响

CLIP 的图像向量更偏语义相似，而不仅是像素级相似。因此检索结果可能更关注动物类别、姿态、主体位置、颜色和场景语义。有些返回图片在人眼看来并非最像，可能是因为 CLIP embedding 的相似度定义与人眼直觉不完全一致。

### 4.4 当前流程的不足与改进方向

本实验只使用一种 embedding 模型和两种 ANNS 索引。后续可以尝试更多 embedding 模型，例如 ResNet50、DINOv2、SigLIP；也可以调节 IVF 的 `nprobe`、HNSW 的 `efSearch`，观察 Recall@10 与 QPS 的变化趋势。

## 5. 遇到的问题与解决方案

可填写：

- 环境配置问题：例如 Python 版本过高导致 FAISS 或 PyTorch 安装困难，因此使用 Python 3.10 conda 环境。
- 模型下载问题：CLIP 权重首次运行需要联网下载。
- 索引参数问题：IVF 的 `nlist` 不能相对 base 数量过大，否则训练效果不稳定。
- 运行速度问题：如果 CPU 生成 embedding 较慢，可以减小 batch size 或使用 GPU。

## 6. Bonus

如果未完成 Bonus，填写：无。

如果完成 Bonus，可补充对应实验设置、结果图和分析。
