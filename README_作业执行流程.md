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
- Embedding 模型：CLIP ViT-B/32 图像编码器，主实验输出 768 维 `float32` 向量。
- 相似度设置：对 embedding 做 L2 normalization，然后用 FAISS L2 距离检索。
- Ground truth：`IndexFlatL2` 精确检索 top10。
- ANNS 方法：`IndexIVFFlat` 和 `IndexHNSWFlat`。
- GPU 加速范围：本流程用本机 NVIDIA GPU/CUDA 加速 CLIP embedding；FAISS 继续使用 `faiss-cpu`，因为本数据集规模不大，CPU 检索已足够完成作业指标。

## 2. 环境配置：优先启用本机 GPU/CUDA

推荐使用 conda，新建 Python 3.10 环境。当前本机默认 Python 可能较新，部分深度学习和 FAISS 包在过新的 Python 上安装不稳，所以建议单独建环境。

### 2.1 检查 NVIDIA 驱动

在 PowerShell 中运行：

```powershell
nvidia-smi
```

如果能看到显卡名称、驱动版本和 CUDA Version，说明系统驱动可被识别。PyTorch 的 CUDA 安装命令建议以官方安装页为准：[PyTorch Start Locally](https://pytorch.org/get-started/locally/)。

### 2.2 创建环境

先创建基础环境：

```powershell
conda create -n image-vector-search python=3.10 -y
conda activate image-vector-search
```

安装 FAISS、数据处理和可视化依赖：

```powershell
conda install -c conda-forge faiss-cpu numpy pillow matplotlib pandas tqdm -y
pip install transformers
```

安装 PyTorch GPU 版。若本机驱动支持 CUDA 12.1，可用：

```powershell
conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1 -y
```

如果 CUDA 12.1 不合适，到 PyTorch 官方安装页选择 Windows、Conda、Python、本机 CUDA 版本后替换上面的命令。

### 2.3 验证 CUDA 是否可用

```powershell
python -c "import torch; print(torch.__version__); print('cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

若输出 `cuda available: True` 和显卡名称，后续命令使用 `--device cuda` 即可强制走 GPU。也可以用 `--device auto`，脚本会在检测到 CUDA 时自动使用 GPU。

## 3. 一键运行主实验

在作业目录执行 GPU 版主流程：

```powershell
python run_pipeline.py --query-size 100 --batch-size 64 --device cuda --query-id 0
```

如果显存较小，把 `--batch-size` 改成 `16` 或 `32`：

```powershell
python run_pipeline.py --query-size 100 --batch-size 32 --device cuda --query-id 0
```

第一次运行会下载：

- Oxford-IIIT Pet 图片压缩包：`data/oxford_pet/archives/images.tar.gz`
- Oxford-IIIT Pet 标注压缩包：`data/oxford_pet/archives/annotations.tar.gz`
- CLIP 模型权重：默认从 Hugging Face 下载 `openai/clip-vit-base-patch32`

如果网络或显存有限，可以先用较小子集快速验证流程：

```powershell
python run_pipeline.py --limit 1000 --query-size 50 --batch-size 16 --device cuda --query-id 0
```

正式报告建议不要少于 5000 张图片；如果机器性能允许，直接使用完整 Oxford-IIIT Pet。

## 4. 分步运行主实验

### 4.1 下载并整理数据集

```powershell
python scripts/01_prepare_oxford_pet.py
```

输出：

- `outputs/oxford_pet_metadata.csv`

该文件记录每张图片的路径、类别 id、类别名称、猫狗物种标记等。

### 4.2 使用 CUDA 生成 CLIP 图像向量

```powershell
python scripts/02_embed_clip.py --batch-size 64 --device cuda
```

如果 CUDA 不可用但仍想自动兼容 CPU：

```powershell
python scripts/02_embed_clip.py --batch-size 32 --device auto
```

输出：

- `outputs/oxford_pet_all.fvecs`
- `outputs/oxford_pet_all_path.csv`

报告中可写：

- Embedding 模型：CLIP ViT-B/32 图像编码器。
- 输入类型：RGB 图像。
- 输出维度：768。
- 数据类型：float32。
- 是否归一化：是，L2 normalization。
- 加速方式：使用本机 NVIDIA GPU/CUDA 进行 CLIP 前向推理；若 `torch.cuda.is_available()` 为 `False`，则回退 CPU。

核心代码位置：`scripts/02_embed_clip.py` 中的设备选择与推理部分。

```python
device = args.device
if device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPVisionModel.from_pretrained(args.model_name).to(device)
pixel_values = inputs["pixel_values"].to(device)

with torch.no_grad():
    outputs = model(pixel_values=pixel_values)
    emb = outputs.pooler_output
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
```

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

- 随机种子：42。
- query 数量：100。
- base 数量：总图片数 - 100。
- query 不出现在 base 中。
- ground truth 使用 `IndexFlatL2` 精确检索 top10 生成。

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

如果想提高 IVF 的 Recall：

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

- IVFFlat 和 HNSW 哪个更接近 ground truth。
- top10 中是否有明显不符合直觉的图片。
- 不符合直觉的原因更可能来自 embedding 表征，还是 ANNS 近似误差。

## 5. Bonus 2：将图搜图扩展为文本搜图

Bonus 2 的目标是把自然语言 query 映射到和图片相同的向量空间，然后检索最相似图片。

注意：主实验的 `scripts/02_embed_clip.py` 使用 `CLIPVisionModel.pooler_output`，输出 768 维图像特征，适合图搜图；文本搜图需要图文共享投影空间。因此 Bonus 2 使用 `scripts/06_text_search_clip.py`，通过 `CLIPModel.get_image_features()` 和 `CLIPModel.get_text_features()` 重新生成 512 维共享向量。

### 5.1 先完成主实验到 base/query 划分

至少先运行到 4.3，确保存在：

- `outputs/oxford_pet_base_path.csv`

### 5.2 运行文本搜图

```powershell
python scripts/06_text_search_clip.py --device cuda --batch-size 64 --topk 10 --queries "a black dog" "a cat lying on grass" "a dog running on grass"
```

如果显存不足：

```powershell
python scripts/06_text_search_clip.py --device cuda --batch-size 16 --topk 10 --queries "a black dog" "a cat lying on grass"
```

输出：

- `outputs/bonus2_clip_base_text_image.fvecs`：Bonus 2 专用的 512 维 base 图片向量缓存。
- `outputs/bonus2_text_search_results.csv`：每条文本 query 的 top10 图片路径、类别和距离。
- `outputs/bonus2_text_search.png`：文本搜图可视化结果。

如果修改了数据集或想重新生成 Bonus 2 图片向量缓存：

```powershell
python scripts/06_text_search_clip.py --device cuda --batch-size 64 --rebuild-image-vectors --queries "a black dog" "a cat lying on grass"
```

### 5.3 Bonus 2 核心代码

核心代码在 `scripts/06_text_search_clip.py`：

```python
processor = CLIPProcessor.from_pretrained(args.model_name)
model = CLIPModel.from_pretrained(args.model_name).to(device)

with torch.no_grad():
    image_features = model.get_image_features(pixel_values=pixel_values)
    text_features = model.get_text_features(**text_inputs)
    image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
    text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)

index = faiss.IndexFlatL2(image_vectors.shape[1])
index.add(image_vectors)
distances, indices = index.search(text_vectors, args.topk)
```

报告中可分析：

- 文本描述是否能召回语义相符的图片。
- 中文或英文 query 的效果差异。
- CLIP 文本搜图更偏语义匹配，结果不一定严格对应品种标签。

## 6. Bonus 3：比较 embedding 归一化对结果的影响

Bonus 3 的目标是在相同数据集、相同 query 划分下，对比：

- 对 embedding 做 L2 normalization。
- 不对 embedding 做 normalization。

### 6.1 生成归一化向量

主实验 4.2 默认已经生成归一化向量：

```powershell
python scripts/02_embed_clip.py --batch-size 64 --device cuda --normalize --output-fvecs outputs/oxford_pet_all.fvecs --output-paths outputs/oxford_pet_all_path.csv
```

### 6.2 生成未归一化向量

```powershell
python scripts/02_embed_clip.py --batch-size 64 --device cuda --no-normalize --output-fvecs outputs/bonus3_oxford_pet_all_no_norm.fvecs --output-paths outputs/bonus3_oxford_pet_all_no_norm_path.csv
```

### 6.3 对比同一批 query 的 top10

```powershell
python scripts/07_compare_normalization.py --query-size 100 --seed 42 --query-id 0 --topk 10
```

输出：

- `outputs/bonus3_normalization_compare.csv`：每条 query 在两种设置下的 top10 及 overlap@10。
- `outputs/bonus3_query0_compare.png`：第 0 条 query 的归一化/未归一化检索结果对比图。

### 6.4 Bonus 3 核心代码

核心代码在 `scripts/07_compare_normalization.py`：

```python
query_ids, base_ids = split_ids(len(normalized), query_size, seed)

index = faiss.IndexFlatL2(base_vectors.shape[1])
index.add(base_vectors)
distances, indices = index.search(query_vectors, topk)

overlap = len(set(normalized_top10) & set(no_normalize_top10)) / 10.0
```

报告中可分析：

- 同一 query 的 top10 是否完全相同。
- 排名顺序是否变化。
- `outputs/bonus3_normalization_compare.csv` 中平均 overlap@10 高不高。
- 若结果变化明显，可能说明向量模长参与了 L2 距离计算；归一化后检索更接近余弦相似度。

## 7. 输出文件说明

| 文件 | 用途 |
|---|---|
| `outputs/oxford_pet_all.fvecs` | 主实验全量图片向量 |
| `outputs/oxford_pet_all_path.csv` | 主实验全量向量 id 到图片路径的映射 |
| `outputs/oxford_pet_base.fvecs` | base 向量集合 |
| `outputs/oxford_pet_query.fvecs` | query 向量集合 |
| `outputs/oxford_pet_base_path.csv` | base 向量 id 到图片路径的映射 |
| `outputs/oxford_pet_query_path.csv` | query 向量 id 到图片路径的映射 |
| `outputs/oxford_pet_groundtruth.ivecs` | 每条 query 的精确 top10 base id |
| `outputs/metrics.csv` | 主实验结果表格数据 |
| `outputs/query0_visualization.png` | 一条 query 的检索结果展示图 |
| `outputs/bonus2_text_search_results.csv` | Bonus 2 文本搜图 top10 结果 |
| `outputs/bonus2_text_search.png` | Bonus 2 文本搜图可视化 |
| `outputs/bonus3_normalization_compare.csv` | Bonus 3 归一化对比结果 |
| `outputs/bonus3_query0_compare.png` | Bonus 3 单条 query 对比图 |

## 8. 报告建议结构

1. 实验设置：数据集、embedding 模型、base/query 划分、ground truth 生成方式、ANNS 方法、CUDA 加速方式。
2. 主实验结果表格：直接参考 `outputs/metrics.csv`。
3. 主实验检索结果可视化：插入 `outputs/query0_visualization.png`。
4. Bonus 2：插入 `outputs/bonus2_text_search.png`，并说明文本 query 与 top10 结果。
5. Bonus 3：插入 `outputs/bonus3_query0_compare.png`，并引用 `outputs/bonus3_normalization_compare.csv` 的 overlap@10。
6. 结果分析：至少写 3 个角度。
7. 遇到的问题与解决方案：可写 CUDA 环境、模型下载、显存不足、参数调节、索引训练等。

可分析的角度：

- HNSW 通常 Recall 较高，但索引大小和构建时间可能更高。
- IVFFlat 的 `nprobe` 越大，Recall 通常越高，但 QPS 可能下降。
- CLIP embedding 更偏语义相似，可能把不同品种但姿态、颜色或语义相近的宠物排在一起。
- ANNS 返回结果与 ground truth 不一致时，要区分是“近似搜索误差”还是“embedding 本身的相似度定义不符合人眼直觉”。
- Bonus 2 可讨论文本描述与图片内容之间的语义匹配是否稳定。
- Bonus 3 可讨论归一化使 L2 检索更接近余弦相似度，而未归一化时向量模长也会影响排序。
