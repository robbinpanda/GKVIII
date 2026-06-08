# 图像向量检索实践作业执行流程

## 1. 作业目标拆解

这份作业最终只要求提交实验报告 PDF，但报告里必须体现完整实验过程：

1. 选择并下载一个图像数据集。
2. 使用一种 embedding 模型把图片转为向量。
3. 将全部向量划分为 base 和 query，且 query 图片不能出现在 base 中。
4. 用精确检索生成 ground truth top10。
5. 至少使用两种 FAISS ANNS 索引进行检索，不能把 `IndexFlatL2` 算作 ANNS 方法。
6. 统计索引构建时间、索引大小、QPS、Recall@10。
7. 任选一条 query，展示 query、ground truth top10、两种 ANNS 的 top10 结果。
8. 在报告中分析结果、误差来源和改进方向。

本目录已经准备好一套主线方案：

- 数据集：Oxford-IIIT Pet，约 7390 张猫狗图片，带品种和物种标注。
- Embedding 模型：CLIP ViT-B/32 图像编码器，输出 768 维 `float32` 向量。
- 相似度设置：对 embedding 做 L2 normalization，然后用 FAISS L2 距离检索。
- Ground truth：`IndexFlatL2` 精确检索 top10。
- ANNS 方法：`IndexIVFFlat` 和 `IndexHNSWFlat`。

## 2. 环境配置

推荐使用 conda，新建 Python 3.10 环境。当前本机默认 Python 是 3.13，部分深度学习和 FAISS 包在 3.13 上可能安装不稳，所以建议单独建环境。

```powershell
conda env create -f environment.yml
conda activate image-vector-search
```

如果 conda 速度慢，也可以先创建环境，再分别安装：

```powershell
conda create -n image-vector-search python=3.10 -y
conda activate image-vector-search
conda install -c pytorch -c conda-forge pytorch torchvision faiss-cpu numpy pillow matplotlib pandas tqdm -y
pip install transformers
```

如果有 NVIDIA GPU，并且希望加速 CLIP embedding，需要按本机 CUDA 版本安装对应 PyTorch GPU 版本。FAISS 仍可使用 `faiss-cpu`，因为本作业数据规模不大。

## 3. 一键运行

在作业目录执行：

```powershell
python run_pipeline.py --query-size 100 --batch-size 32 --device auto --query-id 0
```

第一次运行会下载：

- Oxford-IIIT Pet 图片压缩包：`data/oxford_pet/archives/images.tar.gz`
- Oxford-IIIT Pet 标注压缩包：`data/oxford_pet/archives/annotations.tar.gz`
- CLIP 模型权重：默认从 Hugging Face 下载 `openai/clip-vit-base-patch32`

如果网络或显存有限，可以先用较小子集快速验证流程：

```powershell
python run_pipeline.py --limit 1000 --query-size 50 --batch-size 16 --device cpu --query-id 0
```

正式报告建议不要少于 5000 张图片；如果机器性能允许，直接使用完整 Oxford-IIIT Pet。

## 4. 分步运行

### 4.1 下载并整理数据集

```powershell
python scripts/01_prepare_oxford_pet.py
```

输出：

- `outputs/oxford_pet_metadata.csv`

该文件记录每张图片的路径、类别 id、类别名称、猫狗物种标记等。

### 4.2 生成 CLIP 图像向量

```powershell
python scripts/02_embed_clip.py --batch-size 32 --device auto
```

输出：

- `outputs/oxford_pet_all.fvecs`
- `outputs/oxford_pet_all_path.csv`

报告中可写：

- Embedding 模型：CLIP ViT-B/32
- 输入类型：RGB 图像
- 输出维度：768
- 数据类型：float32
- 是否归一化：是，L2 normalization
- 是否支持文本搜图：CLIP 支持图文统一向量空间，但本主实验只做图搜图

### 4.3 划分 base/query 并生成 ground truth

```powershell
python scripts/03_split_and_groundtruth.py --query-size 100
```

输出：

- `outputs/oxford_pet_base.fvecs`
- `outputs/oxford_pet_query.fvecs`
- `outputs/oxford_pet_base_path.csv`
- `outputs/oxford_pet_query_path.csv`
- `outputs/oxford_pet_groundtruth.ivecs`

报告中说明：

- 随机种子：42
- query 数量：100
- base 数量：总图片数 - 100
- query 不出现在 base 中
- ground truth 使用 `IndexFlatL2` 精确检索 top10 生成

### 4.4 运行两种 FAISS ANNS 索引

```powershell
python scripts/04_run_faiss_experiments.py
```

输出：

- `outputs/metrics.csv`
- `outputs/metrics.json`
- `outputs/ivfflat_results.ivecs`
- `outputs/hnsw_results.ivecs`
- `indices/ivfflat.index`
- `indices/hnsw.index`

默认参数：

- IVFFlat：`nlist=100, nprobe=10`
- HNSW：`M=32, efConstruction=80, efSearch=64`

如果想调参，例如提高 IVF 的 Recall：

```powershell
python scripts/04_run_faiss_experiments.py --ivf-nprobe 20
```

如果想提高 HNSW 的 Recall：

```powershell
python scripts/04_run_faiss_experiments.py --hnsw-ef-search 128
```

### 4.5 生成检索结果可视化图

```powershell
python scripts/05_visualize_results.py --query-id 0 --output outputs/query0_visualization.png
```

输出：

- `outputs/query0_visualization.png`

报告中插入这张图，并简要分析：

- IVFFlat 和 HNSW 哪个更接近 ground truth；
- top10 中是否有明显不符合直觉的图片；
- 不符合直觉的原因更可能来自 embedding 表征，还是 ANNS 近似误差。

## 5. 输出文件说明

| 文件 | 用途 |
|---|---|
| `outputs/oxford_pet_all.fvecs` | 全量图片向量 |
| `outputs/oxford_pet_all_path.csv` | 全量向量 id 到图片路径的映射 |
| `outputs/oxford_pet_base.fvecs` | base 向量集合 |
| `outputs/oxford_pet_query.fvecs` | query 向量集合 |
| `outputs/oxford_pet_base_path.csv` | base 向量 id 到图片路径的映射 |
| `outputs/oxford_pet_query_path.csv` | query 向量 id 到图片路径的映射 |
| `outputs/oxford_pet_groundtruth.ivecs` | 每条 query 的精确 top10 base id |
| `outputs/metrics.csv` | 报告结果表格数据 |
| `outputs/query0_visualization.png` | 一条 query 的检索结果展示图 |

## 6. 报告建议结构

1. 实验设置：数据集、embedding 模型、base/query 划分、ground truth 生成方式、ANNS 方法。
2. 结果表格：直接参考 `outputs/metrics.csv`。
3. 检索结果可视化：插入 `outputs/query0_visualization.png`。
4. 结果分析：至少写 3 个角度。
5. 遇到的问题与解决方案：可写环境、模型下载、参数调节、索引训练等。

可分析的角度：

- HNSW 通常 Recall 较高，但索引大小和构建时间可能更高。
- IVFFlat 的 `nprobe` 越大，Recall 通常越高，但 QPS 可能下降。
- CLIP embedding 更偏语义相似，可能把不同品种但姿态、颜色或语义相近的宠物排在一起。
- ANNS 返回结果与 ground truth 不一致时，要区分是“近似搜索误差”还是“embedding 本身的相似度定义不符合人眼直觉”。

## 7. 可选 Bonus 思路

时间充足时，最容易扩展的是 Bonus 3：

- 保留同一批图片和 query；
- 运行一次 `02_embed_clip.py --normalize`；
- 再运行一次 `02_embed_clip.py --no-normalize`，输出到另一组文件；
- 比较同一条 query 的 top10 是否变化。

这能很好回答“embedding 归一化是否会改变相似图片排序”的问题。
