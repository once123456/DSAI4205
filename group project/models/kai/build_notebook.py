"""Build Kai's single, offline-contained Colab notebook from the model folders."""
import io
import json
import runpy
import tokenize
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "notebooks/kai/Kai_Sentiment_Experiments.ipynb"
LABEL, RAT, CF = "label-only", "rationale-distillation", "counterfactual"
RM, CM = "rationale-matched", "counterfactual-matched"
EM = "embedding-mlp"


def recipes():
    label = runpy.run_path(str(HERE / LABEL / "notebook.py"))["notebook"]
    rationale = runpy.run_path(str(HERE / RAT / "notebook.py"), {"original": label})["cells"]
    counterfactual = runpy.run_path(str(HERE / CF / "notebook.py"), {"prior": {"cells": rationale}})["cells"]
    return {LABEL: label["cells"], RAT: rationale, CF: counterfactual}


def indent_code(source):
    # Indent statements while preserving every byte inside multiline prompt literals.
    interior = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.STRING:
            interior.update(range(token.start[0] + 1, token.end[0] + 1))
    return "\n".join(line if n in interior or not line else "    " + line
                     for n, line in enumerate(source.splitlines(), 1))


def build():
    parts = recipes()
    source = {mode: {cell["id"]: cell["source"] for cell in part} for mode, part in parts.items()}
    cells = []

    def add(kind, key, body, modes=()):
        if modes:
            body = f"if RUN_MODE in {tuple(modes)!r}:\n" + indent_code(body)
        cell = {"cell_type": kind, "id": key, "metadata": {}, "source": body.strip() + "\n"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    def code(key, body, modes=()):
        add("code", key, body, modes)

    def step(mode, key, modes=None):
        code(f"{mode}-{key}", source[mode][key], modes or [mode])

    add("markdown", "intro", """# Kai 的 Twitter 情感分类实验

选择一种运行模式，然后在新的 Colab 运行时中点击「全部运行」。更换模式后从第一格重新开始。
将 DSAI4205_DATA_DIR 指向课程清洗 CSV 所在目录，或在项目目录中运行。默认使用 473 条训练样本、118 条验证样本和 397 条测试样本。

| RUN_MODE | 本次执行 | 所需资源 |
|---|---|---|
| label-only | 原始标签 LoRA 微调、早停与测试评估 | GPU |
| rationale-distillation | 原始标签、未筛选解释、筛选解释对照 | GPU 和教师缓存或 API 密钥 |
| counterfactual | 筛选解释组与加入反事实改写组的配对比较 | GPU 和教师缓存或 API 密钥 |
| rationale-matched | 解释条数和各类条数相同，随机解释 vs 严格解释 | GPU 和上次蒸馏导出 |
| counterfactual-matched | 改写总数相同，原方向比例 vs 两个方向各半 | GPU 和上次反事实导出 |
| embedding-mlp | 四种冻结 Qwen 表示＋独立 MLP，再对比 Qwen3-Embedding＋MLP | GPU 和上次完整反事实导出，无教师 API |

从零生成解释，每条训练推文需要两次教师判断，共 946 个请求，重试另计。
反事实模式默认复用解释缓存，最多为 80 个训练样本生成并核验改写，最多 160 个不同的新请求。
如需教师 API，请在 Colab Secrets 设置 TEACHER_API_KEY；本地请使用 models/kai/.env。""")
    code("mode", '''# @title 选择本次实验
RUN_MODE = "label-only" # @param ["label-only", "rationale-distillation", "counterfactual", "rationale-matched", "counterfactual-matched", "embedding-mlp"]
if RUN_MODE not in {"label-only", "rationale-distillation", "counterfactual", "rationale-matched", "counterfactual-matched", "embedding-mlp"}:
    raise ValueError("请选择表中的运行模式。")
print("本次运行", RUN_MODE)''')
    packages = source[CF]["install"].strip().removeprefix("%pip ")
    code("install", f'get_ipython().run_line_magic("pip", {packages!r})')
    add("markdown", "settings", """## 实验配置与数据

神经模型请选择 GPU。解释蒸馏默认生成缺少的教师记录，反事实模式默认只读上一版解释缓存。
在反事实模式中，可将 DISTILLATION_EXPORT 指向上一版 ZIP 或已解压的结果文件夹。
自定义缓存位置请填写 TEACHER_CACHE_ROOT，反事实缓存使用 CF_CACHE_ROOT。
两个 matched 模式会尝试读取 5090 本机的上次运行记录。也可填写 PRIOR_EXPORT 指向相应 ZIP 或已解压目录。
MATCH_SEED 控制抽样；SEEDS 控制学生训练。两种 matched 模式都只读取既有教师结果。
embedding-mlp 同样用 PRIOR_EXPORT 读取完整反事实导出，自动复用其划分、教师解释和已接受改写。
该模式从共同基础权重重训三个 adapter，随后冻结各模型，只用原始训练行训练五组独立 MLP。
Colab 上 SAVE_TO_DRIVE 可将结果写入 Google Drive；本地结果写入 group project/runs/kai/。""")
    cfg = source[CF]["config"]
    cfg = cfg.replace("SAVE_TO_DRIVE = True", "SAVE_TO_DRIVE = False")
    cfg = cfg.replace('ALLOW_TEACHER_REQUESTS = False', 'ALLOW_TEACHER_REQUESTS = RUN_MODE == "rationale-distillation"')
    cfg = cfg.replace('SEEDS = [42, 43, 44]', 'SEEDS = [42] if RUN_MODE == "embedding-mlp" else [42, 43, 44]\nRUN_UNFILTERED = True')
    cfg = cfg.replace('PRIOR_EXTRA_FILES = ()', '''PRIOR_EXPORT = ""
MATCH_SEED = 42
PRIOR_EXTRA_FILES = ()
if RUN_MODE in {"rationale-matched", "counterfactual-matched", "embedding-mlp"}:
    PRIOR_EXTRA_FILES = ("teacher_responses.json", "accepted_counterfactuals.csv")''')
    cfg = cfg.replace('MATCH_SEED = 42', '''MATCH_SEED = 42
# 冻结表示＋独立 MLP：SEEDS 同时控制 adapter 和 MLP，建议正式报告改为 [42, 43, 44]。
EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_REVISION = "main"
EMBED_BATCH_SIZE = 8
MLP_HIDDEN = 128
MLP_MAX_EPOCHS = 200
MLP_PATIENCE = 20
MLP_BATCH_SIZE = 32
MLP_ALPHA = 0.01
MLP_LEARNING_RATE = 0.001
EMBED_BOOTSTRAP_SAMPLES = 1000''')
    cfg = cfg.replace('ALLOW_CF_REQUESTS = True', 'ALLOW_CF_REQUESTS = RUN_MODE == "counterfactual"')
    start = cfg.index("if SAVE_TO_DRIVE and IN_COLAB:")
    end = cfg.index("RUN_ID =")
    cfg = cfg[:start] + '''run_folder = {"label-only": "qwen_lora",
              "rationale-distillation": "qwen_distillation", "counterfactual": "qwen_counterfactual",
              "rationale-matched": "qwen_rationale_matched", "counterfactual-matched": "qwen_cf_matched",
              "embedding-mlp": "qwen_embedding_mlp"}[RUN_MODE]
SAVE_TO_DRIVE = IN_COLAB and RUN_MODE != "label-only"
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_ROOT = Path("/content/drive/MyDrive") / ("dsai4205_" + run_folder)
elif IN_COLAB:
    OUTPUT_ROOT = Path("/content") / (run_folder + "_runs")
else:
    OUTPUT_ROOT = local_runs_root() / (run_folder + "_runs")
''' + cfg[end:]
    start = cfg.index("import gc\n")
    cfg = cfg[:start] + 'if RUN_MODE != "label-only":\n' + indent_code(cfg[start:])
    code("config", cfg)
    code("prior-default", "prior_protocol = None")
    code("matched-functions", (HERE / "matched_ablation.py").read_text(encoding="utf-8"), [RM, CM, EM])
    code("matched-input", '''origin_mode = "counterfactual-matched" if RUN_MODE == "embedding-mlp" else RUN_MODE
DISTILLATION_EXPORT = str(previous_export(OUTPUT_ROOT.parent, origin_mode, PRIOR_EXPORT))
print("复用上次运行", DISTILLATION_EXPORT)''', [RM, CM, EM])
    step(CF, "import-export", [CF, RM, CM, EM])
    code("embedded-data", source[LABEL]["embedded-data"])
    code("data", source[CF]["data"])
    add("markdown", "teacher-note", source[RAT]["teacher-note"])
    for key in ["teacher-functions", "teacher-run"]:
        step(RAT, key, [RAT, CF, RM, CM, EM])
    add("markdown", "token-note", source[LABEL]["tokenization-note"])
    code("label-only-tokenization", source[LABEL]["tokenization"])
    add("markdown", "label-note", "## 原始标签微调\n\n本模式按验证 macro-F1 选取最佳 adapter，支持早停。实验完成后下载 ZIP，最后一格可试用自定义推文。")
    for key in ["model", "train", "inference", "test", "export", "demo"]:
        step(LABEL, key)
    add("markdown", "aux-note", source[RAT]["training-note"])
    common, extra = source[RAT]["multitask"].split("\nEXPERIMENTS =", 1)
    assert source[CF]["multitask"].startswith(common.rstrip())
    code("shared-multitask", common, [RAT, CF, RM, CM, EM])
    code("rationale-datasets", 'CONTROL = "label_only"\nEXPERIMENTS =' + extra, [RAT])
    code("rationale-matched-datasets", '''paired_datasets, selection = rationale_arms(train_df, teacher_table, PairedDataset, MATCH_SEED)
CONTROL = "rationale_random_matched"
EXPERIMENTS = list(paired_datasets)
display(record_selection(RUN_DIR, RUN_MODE, MATCH_SEED, SPLIT_HASH, selection, "row_id", "Sentiment"))''', [RM])
    code("cf-originals", source[CF]["multitask"][len(common.rstrip()):], [CF, CM])
    add("markdown", "cf-note", source[CF]["generation-note"])
    for key in ["counterfactual-functions", "generate"]:
        step(CF, key)
    cf_classes, cf_arms = source[CF]["paired-training"].split("\nCONTROL, AUGMENTED =", 1)
    code("cf-trainer", cf_classes, [CF, CM, EM])
    code("cf-datasets", "CONTROL, AUGMENTED =" + cf_arms, [CF])
    code("cf-matched-datasets", '''accepted_file = imported / "accepted_counterfactuals.csv"
if not accepted_file.is_file() or "counterfactual" not in prior_protocol:
    raise ValueError("请用原 counterfactual 实验的完整导出，需含 accepted_counterfactuals.csv。")
accepted_cf = pd.read_csv(accepted_file, keep_default_na=False)
prior_cf = prior_protocol["counterfactual"]
CF_WEIGHT = prior_cf["weight"]
arms = counterfactual_arms(accepted_cf, train_df, original_dataset.used_ids, MATCH_SEED, prior_cf["accepted"])
CONTROL = "cf_observed_ratio"
EXPERIMENTS = list(arms)
paired_datasets = {name: CounterfactualDataset(original_dataset, train_df.row_id, rows) for name, rows in arms.items()}
selection = pd.concat([rows.assign(experiment=name) for name, rows in arms.items()], ignore_index=True)
display(record_selection(RUN_DIR, RUN_MODE, MATCH_SEED, SPLIT_HASH, selection, "parent_row_id", "source_label"))
cf_run_info = {"weight": CF_WEIGHT, "prior_accepted": len(accepted_cf), "api_calls_this_run": 0}''', [CM])
    add("markdown", "compare-note", """## 训练对照组与导出

解释蒸馏模式以原始标签组为参照。反事实模式以仅筛选解释组为参照。
两种模式均从相同的基础模型分别训练本次对照组，按验证 macro-F1 选模型。
查看 comparison.csv 的 delta_macro_f1，配对区间保存在 paired_bootstrap.csv。""")
    add("markdown", "matched-note", """matched 模式分别报告严格解释减随机解释、平衡方向减原方向比例的差值。
matched_counts.csv 给出实际各类数量，matched_selection.csv 保存两组入选的训练行。""")
    for key in ["metrics", "inference", "model-factory"]:
        assert source[RAT][key] == source[CF][key]
        step(RAT, key, [RAT, CF, RM, CM, EM])
    for mode, matched in [(RAT, RM), (CF, CM)]:
        experiments = source[mode]["experiments"].replace('write_json(RUN_DIR / "experiment_protocol.json", protocol)', '''
if RUN_MODE in {"rationale-matched", "counterfactual-matched"}:
    protocol["matched_design"] = json.loads((RUN_DIR / "matched_design.json").read_text(encoding="utf-8"))
    protocol["control"] = CONTROL
    protocol["prior_export"] = DISTILLATION_EXPORT
    previous = RUN_DIR / "experiment_protocol.json"
    if previous.is_file() and json.loads(previous.read_text(encoding="utf-8")) != protocol:
        raise ValueError("续跑配置与已有记录不同，请创建新的运行目录。")
write_json(RUN_DIR / "experiment_protocol.json", protocol)''')
        code(f"{mode}-experiments", experiments, [mode, matched])
        test = source[mode]["test"].replace('"label_only"', 'CONTROL')
        code(f"{mode}-test", test, [mode, matched])
        step(mode, "export")
    export_head, export_tail = source[RAT]["export"].split('(EXPORT_DIR / "README.txt")', 1)
    code("matched-export", export_head + '''
(EXPORT_DIR / "README.txt").write_text(
    f"{RUN_MODE}, control: {CONTROL}. See matched_counts.csv, matched_selection.csv, "
    "comparison.csv and paired_bootstrap.csv.", encoding="utf-8")
archive = shutil.make_archive(str(RUN_DIR / RUN_MODE), "zip", EXPORT_DIR)
''' + export_tail[export_tail.index("print(archive)"):], [RM, CM])
    embedding_cells = runpy.run_path(str(HERE / EM / "notebook.py"),
                                    {"sources": source, "helper_path": HERE / EM / "pipeline.py"})["cells"]
    for cell in embedding_cells:
        add(cell["cell_type"], cell["id"], cell["source"], [EM] if cell["cell_type"] == "code" else ())
    metadata = {"accelerator": "GPU", "colab": {"name": OUT.name, "gpuType": "T4"},
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    out = {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    build()
