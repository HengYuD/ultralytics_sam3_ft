# SAM3 + LoveDA 遥感实例分割（含完整训练/测试与类别选择说明）

> 目标：在 LoveDA 数据集上做 SAM3 微调，支持“只提升第五类 forest”与“多类别联合微调”两种模式。

## 1. LoveDA 类别映射（语义标注 -> 训练类别）

LoveDA 常见语义标签 id（原始 mask）：

- 0: background（背景，训练时忽略）
- 1: building
- 2: road
- 3: water
- 4: barren
- 5: forest（第5类，本文重点）
- 6: agriculture

去背景后的连续映射（YOLO/SAM3 配置常用）：

- 1 -> 0 (building)
- 2 -> 1 (road)
- 3 -> 2 (water)
- 4 -> 3 (barren)
- 5 -> 4 (forest)
- 6 -> 5 (agriculture)

对应配置文件：`examples/sam3_loveda/loveda_sam3.yaml`

---

## 2. 完整数据准备代码

脚本：`examples/sam3_loveda/prepare_loveda_instance_labels.py`

功能：

1. 读取 `tif/tiff/png/jpg` 图像与语义 mask。
2. 按类别提取连通域，自动拆成“实例”。
3. 导出 YOLO polygon 分割标签（每行：`class x1 y1 x2 y2 ...`）。

### 2.1 转换训练集

```bash
python examples/sam3_loveda/prepare_loveda_instance_labels.py \
  --images /data/LoveDA/Train/images_png \
  --masks /data/LoveDA/Train/masks_png \
  --out /data/LoveDA-Instance \
  --split train
```

### 2.2 转换验证集

```bash
python examples/sam3_loveda/prepare_loveda_instance_labels.py \
  --images /data/LoveDA/Val/images_png \
  --masks /data/LoveDA/Val/masks_png \
  --out /data/LoveDA-Instance \
  --split val
```

### 2.3 常用参数

- `--min-area 30`：过滤噪声小连通域。
- `--epsilon 1.5`：轮廓简化强度。

---

## 3. 完整训练代码（支持单类/多类选择）

脚本：`examples/sam3_loveda/train_sam3_loveda.py`

### 3.1 训练机制

- 读取遥感图像 + 语义 mask，动态生成 **每个目标类的二值监督通道**。
- 用 SAM3 的文本类别提示：`model.set_classes([...])`。
- 对每个类别分别前向，计算 `BCE + Dice`，最后对类别损失求均值。
- 输出每类 IoU 和 mean IoU。
- 分阶段训练：先冻结 backbone，再解冻全量微调。

### 3.2 仅训练第5类（forest）

```bash
python examples/sam3_loveda/train_sam3_loveda.py \
  --sam3-ckpt /models/sam3_b.pt \
  --train-images /data/LoveDA/Train/images_png \
  --train-masks /data/LoveDA/Train/masks_png \
  --val-images /data/LoveDA/Val/images_png \
  --val-masks /data/LoveDA/Val/masks_png \
  --focus-class-names forest \
  --focus-semantic-ids 5 \
  --epochs 40 --batch 2 --imgsz 1024 --lr 2e-5 --amp
```

### 3.3 联合训练多类别（示例：forest + water + building）

```bash
python examples/sam3_loveda/train_sam3_loveda.py \
  --sam3-ckpt /models/sam3_b.pt \
  --train-images /data/LoveDA/Train/images_png \
  --train-masks /data/LoveDA/Train/masks_png \
  --val-images /data/LoveDA/Val/images_png \
  --val-masks /data/LoveDA/Val/masks_png \
  --focus-class-names forest,water,building \
  --focus-semantic-ids 5,3,1 \
  --epochs 50 --batch 2 --imgsz 1024 --lr 2e-5 --amp
```

> 规则：`--focus-class-names` 与 `--focus-semantic-ids` 必须一一对应、长度相同。

### 3.4 训练输出

- `runs/sam3_loveda/best.pt`：最优权重（保存 class_names、semantic_ids、imgsz）。
- `runs/sam3_loveda/history.json`：每轮 train/val loss、mean IoU、per-class IoU。

---

## 4. 完整测试/评估代码

脚本：`examples/sam3_loveda/test_sam3_loveda.py`

功能：

- 加载 `best.pt` 与 SAM3 基础权重。
- 按 checkpoint 中的类别配置（或命令行覆盖）做评估。
- 输出每类 `IoU/Dice` 和整体 `mean_iou/mean_dice`。
- 可选保存可视化结果（原图 vs 叠加结果并排图）。
- 支持 `png/jpg/tif` 多格式；若输入为多通道 `tif`，自动使用前三通道推理并保留地理信息输出 GeoTIFF 结果。

### 4.1 使用 checkpoint 自带类别配置测试

```bash
python examples/sam3_loveda/test_sam3_loveda.py \
  --sam3-ckpt /models/sam3_b.pt \
  --weights runs/sam3_loveda/best.pt \
  --images /data/LoveDA/Val/images_png \
  --masks /data/LoveDA/Val/masks_png \
  --out-json runs/sam3_loveda/test_metrics.json
```

### 4.2 覆盖类别配置测试（例如只测 forest）

```bash
python examples/sam3_loveda/test_sam3_loveda.py \
  --sam3-ckpt /models/sam3_b.pt \
  --weights runs/sam3_loveda/best.pt \
  --images /data/LoveDA/Val/images_png \
  --masks /data/LoveDA/Val/masks_png \
  --focus-class-names forest \
  --focus-semantic-ids 5 \
  --save-vis --vis-dir runs/sam3_loveda/vis_forest --save-geotiff --tif-out-dir runs/sam3_loveda/tif_preds
```

---

## 5. 不同类别选择建议

### 5.1 场景 A：只提升第5类 forest（推荐起步）

- 参数：`--focus-class-names forest --focus-semantic-ids 5`
- 优点：目标明确，收敛快。
- 风险：可能牺牲其他类别表现，建议额外监控整体 mIoU。

### 5.2 场景 B：forest + 相邻易混类联合训练

建议加入：`barren(4)`、`agriculture(6)`，减少林地与周边地物混淆。

示例：

- `--focus-class-names forest,barren,agriculture`
- `--focus-semantic-ids 5,4,6`

### 5.3 场景 C：全类别联合训练

- `--focus-class-names building,road,water,barren,forest,agriculture`
- `--focus-semantic-ids 1,2,3,4,5,6`

适合需要整体泛化且防止“单类过拟合”的场景。

---

## 6. 训练增强与参数建议（遥感）

已内置：

- 水平/垂直翻转
- 90°/180°/270°旋转
- 亮度对比度扰动

推荐参数：

- `imgsz=1024`
- `batch=2~4`（按显存）
- `lr=2e-5`
- `freeze_backbone_epochs=3~5`
- `epochs=30~80`

---

## 7. 指标与验收标准

- 重点看：`forest` 的 IoU / Dice。
- 同时看：多类 mean IoU，避免单类提升但整体退化。
- 建议分场景（城市/乡村）统计指标。
