# A、E：无需教师缓存的冻结表示对照

入口为 `models/kai/run_frozen_ae.py`。A 是原始 Qwen3-0.6B＋MLP；E 是 Qwen3-Embedding-0.6B＋MLP。仅训练 CPU 上的独立 MLP，两个 GPU 表示模型始终冻结，并依次加载。没有 LoRA、教师 API、解释或反事实生成。此入口不执行、不改写统一 notebook。

## 数据和比较协议

读取本地课程 CSV，按 notebook 的清单重建数据快照，检查原始 SHA-256，沿用相同清洗规则和 `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`。默认重建 473 train、118 validation、397 test；这是固定规则重建，不能声称恢复了已经丢失的教师缓存。若持有原 `split_manifest.csv`，可通过 `--split-manifest /path/to/split_manifest.csv` 指定并核验。每次保存划分清单和指纹。

A 使用与原五组入口一致的分类 chat 提示，关闭 thinking，取生成答案前最后有效 token；E 使用官方 Instruct/Query 模板及最后有效 token。两者均为 1024 维并 L2 归一化。模板、预训练目标不同，因此是系统比较，不是单变量权重消融。

两组共用 `embedding-mlp/pipeline.py` 的缓存、冻结提取和 MLP 实现。StandardScaler 只拟合训练向量；MLP 为 1024→128 ReLU→4，Adam、学习率 0.001、alpha=0.01、batch 32、最多 200 epoch、验证 macro-F1 patience 20。只用原始训练行，最终恢复最佳验证轮次。先固定所有 head，再读取测试标签。测试文本可提前提取向量，不用于拟合 scaler 或分类器。

## 新租的 5090：安装和运行

使用支持 RTX 5090 的 CUDA 12.8/PyTorch 2.7 或更新模板，推荐 Python 3.11。以下命令在远程 Linux 执行。已有仓库时进入其目录执行 `git pull --ff-only`；如提示本地冲突，先保留改动，不执行强制重置。新机器先克隆：

```bash
cd /workspace
git clone https://github.com/once123456/DSAI4205.git
cd '/workspace/DSAI4205/group project'
python -m venv --system-site-packages models/kai/.ae-env
source models/kai/.ae-env/bin/activate
python -m pip install -r models/kai/frozen-ae/requirements.txt
python -m pip check
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); x=torch.ones(4,device='cuda'); print((x+x).cpu())"
python models/kai/run_frozen_ae.py --check-data
python -u models/kai/run_frozen_ae.py --arms A E --seeds 42
```

如果基础模板没有适配 5090 的 PyTorch，先在环境中按 PyTorch 官方安装说明安装 CUDA 构建；一个明确版本的选择是 `python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128`。不要对已正常运行的 GPU 环境随意降级。

输出固定在 `runs/kai/qwen_frozen_ae_runs/<UTC时间戳>/`，不会写入外层 README、notebooks、其他模型或 reports。Hugging Face 模型下载使用其默认缓存，也可设置 `HF_HOME` 指向实例的数据盘。没有需要迁移的教师缓存或密钥。

只运行一组可用 `--arms A` 或 `--arms E`。正式多 seed 比较使用 `--seeds 42 43 44`，不要依据第一轮测试分数选择种子。

中断后使用相同代码、环境和参数续跑：

```bash
python -u models/kai/run_frozen_ae.py --arms A E --seeds 42 \
  --resume-dir runs/kai/qwen_frozen_ae_runs/实际运行目录
```

完成的向量和 head 会复用；未完成 head 从头训练。续跑默认使用协议中已经解析的模型 commit，数据、代码、版本和配置不匹配时拒绝混用。完整运行不会重复评估。显存不足时可在新运行加 `--batch-size 4`；这会改变协议，不应混入原运行。

## 结果与 E 盘报告

主要文件：`comparison.csv`、`comparison_aggregate.csv`、`paired_bootstrap.csv`（E−A）、`per_class_metrics.csv`、`mlp/*/test_predictions.csv`、`mlp/*/test_metrics.json`（含混淆矩阵）、`experiment_protocol.json`、`split_manifest.csv`、`embeddings/*.npz` 和 `completed.json`。单 seed 的标准差留空；bootstrap 仅反映测试样本不确定性。

将整个结果目录下载到本地 `group project/runs/kai/qwen_frozen_ae_runs/`。在本地 PowerShell 中生成独立 HTML 报告，默认约定报告目标为 E 盘的 `DSAI4205/Kai_AE`：

```powershell
python 'G:/E-learn/dsai4205/group project/models/kai/frozen-ae/write_report.py' `
  --run-dir 'G:/E-learn/dsai4205/group project/runs/kai/qwen_frozen_ae_runs/实际运行目录' `
  --output-dir 'E:/DSAI4205/Kai_AE'
```

报告只接受带 `completed.json` 的完整结果，不生成模拟成绩。租用实例应在结果下载并校验后释放；停止训练进程不等于停止云端计费。

离线检查：`python -B models/kai/frozen-ae/test_ae.py`。它检查旧划分一致性、缓存输入隔离和复用、冻结无梯度提取、仅训练集拟合 scaler；不下载预训练权重、不调用教师，也不训练正式数据集。

参考：[Qwen3-Embedding 模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)、[PyTorch CUDA 安装](https://pytorch.org/get-started/previous-versions/)、[Vast RTX 5 系列说明](https://github.com/vast-ai/docs/blob/main/rtx-5-series.mdx)。

## 首轮正式运行：2026-11-21

运行 ID：`20261121T101025_214748Z`，代码提交 `863cd2e`，单张 RTX 5090，PyTorch 2.7.1+cu128，seed 42。模型全程冻结、教师调用 0 次；远程进程正常退出，27 个结果文件已下载并逐个校验 SHA-256。

| 组别 | 验证 macro-F1 | 最佳 MLP epoch | 测试 macro-F1 | 测试 accuracy |
|---|---:|---:|---:|---:|
| A：Qwen3-0.6B＋MLP | 0.4157 | 1 | 0.4678 | 0.5290 |
| E：Qwen3-Embedding-0.6B＋MLP | 0.5292 | 3 | 0.4783 | 0.5063 |

E−A 的测试 macro-F1 差值为 +0.0105，1000 次配对 bootstrap 的 95% 区间为 [-0.0472, 0.0644]，跨过零，不能据此断言 E 稳定更好。E 的 Irrelevant F1 从 0.1556 升至 0.3492，Neutral 从 0.5143 升至 0.5545；Positive 则从 0.5628 降至 0.3784。平均指标掩盖了类别间的取舍，下一步应先按预定配置验证多种子稳定性，再设计监督适配模块，不能根据这次测试结果反复选择参数。

划分指纹为 `e7fc1c5525dc8d9316e755c67b47dbe574115ef68959f0f33ece2e732f1f5bfb`。完整实验数据保存在本地 `runs/kai/qwen_frozen_ae_runs/20261121T101025_214748Z/`；HTML 报告保存在 `E:/DSAI4205/Kai_AE/`。本次租用实例已释放。
