# CoLaR / Coconut / iCoT 复现与医学迁移分阶段计划

## 0. 结论先行：这个仓库能否满足你的需求？

**可以，能满足大部分核心需求**，尤其是你要做的 `CoLaR + Coconut + iCoT` baseline 对比：

- 已内置三种方法的模型配置：`colar.yaml`、`coconut.yaml`、`icot.yaml`。
- 默认支持 `Llama-3.2-1B-Instruct`，非常契合你第①阶段的 1B 规模对比。
- 训练/测试入口统一是 `run.py`，便于做可控的公平对比。

**但医疗任务需要你补齐数据与评测适配**：

- 当前数据模块主要是数学/推理导向（`QSADataModule` + `gsm` 等）。
- 你计划的医疗问答（并使用 Llama 8B）可以在该框架中做，但需要新增医疗数据预处理、数据配置、以及医疗评测脚本。

---

## 1. 总体实验矩阵（与你当前目标对齐）

### 1.1 通用数据集（阶段①）

- **模型**：Llama 1B（建议沿用仓库默认 `Llama-3.2-1B-Instruct`）
- **方法**：CoLaR / Coconut / iCoT（可加 COT 作为显式推理参照）
- **数据集**：
  - GSM8K（仓库已覆盖）
  - Commonsense（需新增数据加载与格式适配）

### 1.2 医疗问答（阶段②）

- **模型**：Llama 8B
- **方法**：CoLaR / Coconut / iCoT（并与你已复现的 CODI、SimCOT 做横向比较）
- **目标**：验证“隐式推理迁移到医疗领域”的有效性与稳定性

---

## 2. 分阶段里程碑计划（可直接执行）

## Phase A：环境与基线打通（1~2 天）

### A1. 环境搭建

```bash
conda create -n colar python=3.10
conda activate colar
pip install -r requirements.txt
```

### A2. 单卡/小样本冒烟测试

```bash
python run.py --model=colar --dataset=qsa --no_log tiny_dataset=True
python run.py --model=coconut --dataset=qsa --no_log tiny_dataset=True
python run.py --model=icot --dataset=qsa --no_log tiny_dataset=True
```

### A3. 输出检查

- 确认三种方法都能正常 forward/backward。
- 确认日志/ckpt 目录生成正常。

**验收标准**：三种 baseline 都能在 tiny_dataset 结束 1 次 train+val 流程，无 NaN / OOM。

---

## Phase B：通用数据集对比（GSM8K + Commonsense，Llama1B）（3~7 天）

### B1. 固定公平对比设置

- 统一 `model_id`、batch size、epoch、学习率、LoRA 参数。
- 建议先固定：`lr=1e-4, batch_size=256, max_epochs=50`（资源不足可按比例缩放）。

### B2. GSM8K 对比（先跑）

示例（CoLaR）：

```bash
python run.py \
  --model=colar \
  --dataset=qsa \
  --do_test \
  dataset_name=gsm \
  model_id=Llama-3.2-1B-Instruct \
  log_suffix=phaseB_gsm_colar_1b
```

Coconut / iCoT 只替换 `--model`。

### B3. Commonsense 数据接入

建议新增：

- `data_preprocessing/commonsense_xxx.py`（统一转成现有样本格式）
- `src/configs/datasets/commonsense.yaml`
- 必要时扩展 `src/datasets/qsa.py` 使其支持 `dataset_name=commonsense`

### B4. 指标

- 主指标：Accuracy / Exact Match
- 附加指标：推理长度（token 或 latent steps）、训练吞吐、推理时延

**验收标准**：得到 3 种方法在 GSM8K 与 Commonsense 上的可复现实验表格。

---

## Phase C：医疗问答迁移（Llama8B）（5~10 天）

### C1. 医疗数据选择与规范化

建议选 1~2 个公开 QA 数据集（如 MedMCQA/PubMedQA 方向），统一成：

- `question`
- `reasoning`（若无显式 rationale，可先留空或构造 teacher rationale）
- `answer`

### C2. 新增数据配置

- 新建 `src/configs/datasets/medical.yaml`
- 在 DataModule 中增加 `dataset_name=medical_xxx` 分支

### C3. 切换到 Llama 8B

训练命令核心改动：

- `model_id=你的Llama8B模型`
- 降低 batch size（例如 16/32）
- 配置梯度累积、bf16、checkpointing（按显存调）

### C4. 与 CODI / SimCOT 对齐协议

你已经复现 CODI 和 SimCOT，建议统一：

- 同训练集划分
- 同最大输入长度
- 同评测脚本
- 同采样策略（temperature/top_p）

**验收标准**：在医疗集上输出 `CoLaR/Coconut/iCoT/CODI/SimCOT` 的统一对比表。

---

## Phase D：分析与论文图表产出（2~4 天）

### D1. 主结果表

- 通用数据集表（1B）
- 医疗迁移表（8B）

### D2. 消融建议

- CoLaR 的 compression_factor（如 2/3/4/5）
- 是否 RL 增强（do_rl 开关）
- Coconut/iCoT 的 stage 数影响

### D3. 关键结论

- 隐式推理是否在医疗领域仍保留效率-性能优势
- 何种方法在“低资源/高难推理”下更稳

---

## 3. 你当前场景的最小可执行清单（先做这 6 件事）

1. 跑通 CoLaR/Coconut/iCoT 的 tiny_dataset 冒烟。
2. 用统一参数完成 GSM8K 三方法首轮结果。
3. 接入一个 Commonsense 数据集并跑首轮对比。
4. 新建 medical dataset config，先在 1B 做流程验证。
5. 切换 Llama8B，在医疗集跑 CoLaR/Coconut/iCoT。
6. 合并 CODI/SimCOT 结果，出第一版总表。

---

## 4. 风险点与规避

- **风险1：8B 显存不足** → 降 batch + 梯度累积 + LoRA + 混精。
- **风险2：医疗数据格式不统一** → 先写统一预处理脚本，固定 JSON schema。
- **风险3：对比不公平** → 先写“实验协议表”，每次训练前核对。
- **风险4：隐式推理在医疗任务不稳定** → 先小规模网格搜索 compression/stage，再放大全量训练。

---

## 5. 建议的目录与产物（便于团队协作）

- `reproduction_plan.md`（本文件）
- `docs/exp_protocol.md`（统一实验协议）
- `docs/result_table_template.md`（结果模板）
- `data_preprocessing/medical_*.py`（医疗预处理）
- `src/configs/datasets/medical.yaml`

---

## 6. 一句话建议

你现在的路线是合理的：**先在 1B 通用数据集上完成 CoLaR/Coconut/iCoT 的可复现对比，再迁移到 8B 医疗问答并对齐 CODI/SimCOT 协议**。这个仓库足够做 baseline 主干，医疗部分需要你补数据与评测层。
