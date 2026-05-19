# GR00T N1.7 微调：Franka 双臂翻牛排（flip the steak）

复用 openpi 训练 `pi05_franka_flip_steak` 用的同一份数据集
`lerobot_flipsteak_500_224`，把 GR00T N1.7-3B 微调到这个具身上。

## 1. 数据集

- 路径：`/public/hz_nas/tong/data/lerobot_flipsteak_500_224`
- 格式：LeRobot v2.1
- 规模：500 个 episode，1,074,876 帧，30 fps
- 机器人：双臂 Franka（左 7 关节 + 右 7 关节 + 左夹爪 + 右夹爪 = **16 维**）
- 摄像头（3 路 224×224）：
  - `observation.images.left_side`（第三人称）
  - `observation.images.left_wrist`（左腕）
  - `observation.images.right_wrist`（右腕）
- 任务：单一 task — `"flip the steak"`

state / action 名字对照（与 `meta/info.json` 一致）：

| 维度 | 0–6 | 7–13 | 14 | 15 |
|------|-----|------|-----|-----|
| 含义 | left_arm 7 关节 | right_arm 7 关节 | left_gripper | right_gripper |

## 2. 我对仓库做的三处修改

### (a) 新增 `meta/modality.json`

GR00T 在 LeRobot v2 之外要求一份额外的 `meta/modality.json`，
用来把 16 维的 state / action 拆名、把视频键重命名。
新文件：`/public/hz_nas/tong/data/lerobot_flipsteak_500_224/meta/modality.json`

```json
{
  "state":  { "left_arm":{"start":0,"end":7}, "right_arm":{"start":7,"end":14},
              "left_gripper":{"start":14,"end":15}, "right_gripper":{"start":15,"end":16} },
  "action": { 同上 },
  "video":  { "left_side":  {"original_key":"observation.images.left_side"},
              "left_wrist": {"original_key":"observation.images.left_wrist"},
              "right_wrist":{"original_key":"observation.images.right_wrist"} },
  "annotation": { "human.task_description": {"original_key":"task_index"} }
}
```

> 这个文件只是补充，不影响 openpi。

### (b) 新增 modality 配置脚本

`examples/franka_steak/franka_steak_config.py`：
向 GR00T 注册 `NEW_EMBODIMENT` 标签下的 modality 配置，
和 openpi 的 `LeRobotFrankaBimanualDataConfig`
（`extra_delta_transform=True`, mask `[True]*14 + [False, False]`）保持一致：

- 双臂关节：`RELATIVE` + `NON_EEF`（预测相对当前 state 的 delta）
- 双夹爪：`ABSOLUTE` + `NON_EEF`（直接预测目标位置）
- action_horizon = 16

### (c) 新增训练启动脚本

`Isaac-GR00T/training_sh/start_franka_steak_gr00t.sh`（仿 openpi
`training_sh/start_franka_steak_pi05.sh` 的风格，新建在 GR00T 仓库下），
关键超参与 openpi 对齐：

| 项 | openpi pi0.5 | GR00T N1.7 |
|----|-------------|------------|
| GPU | 0–7 (8 张 L20Z) | 同 |
| global batch | 128 | 128 |
| steps | 30,000 | 30,000 |
| 学习率 | 默认 | 1e-4 |
| 输出目录 | OPENPI_DATA_HOME 下 | `/public/hz_nas/tong/model/gr00t_franka_flip_steak` |

启动方式：

```bash
bash /public/hz_nas/tong/code/Isaac-GR00T/training_sh/start_franka_steak_gr00t.sh
```

## 3. 环境

`conda activate gr00t` (Python 3.10, /root/miniconda3/envs/gr00t)
- torch 2.7.1+cu128 / torchvision 0.22.1+cu128
- flash-attn 2.7.4.post1
- transformers 4.57.3, diffusers 0.35.1, peft 0.17.1, deepspeed 0.17.6
- torchcodec 0.4.0 + ffmpeg 7（来自 conda-forge）
- tensorrt-cu12 10.16.1.11
- gr00t 0.1.0（editable，源码在 `Isaac-GR00T/`）

## 4. 注意事项

1. **首次启动要生成 stats**
   GR00T 训练前需要 `meta/stats.json` 与 `meta/relative_stats.json`，
   `launch_finetune.py` 内部会在 rank0 自动调用 `generate_stats` /
   `generate_rel_stats` 生成。本次配置我已经提前跑过一次，
   两份文件已经存在；以后切换 dataset 第一次跑时这一步要花几分钟。

2. **基模 + VLM backbone 已经预下载，并需要 4 处源码改动才能离线跑**

   集群当前出不去 HuggingFace（连 hf-mirror.com 也 timeout），
   所以以下两份权重都从 ModelScope 拉到 NAS：

   | HF id | ModelScope 镜像 | 本地路径 | 大小 |
   |-------|------|------|------|
   | `nvidia/GR00T-N1.7-3B` | `nv-community/GR00T-N1.7-3B` | `/public/hz_nas/tong/model/GR00T/GR00T-N1.7-3B` | 6.5 GB |
   | `nvidia/Cosmos-Reason2-2B`（VLM backbone） | `nv-community/Cosmos-Reason2-2B` | `/public/hz_nas/tong/model/GR00T/Cosmos-Reason2-2B`（symlink 至 `/public/hz_nas/tong/model/GR00T/nvidia/Cosmos-Reason2-2B`，让路径里包含 `nvidia/Cosmos-Reason2` 子串以通过源码里的判断）| 4.6 GB |

   **重要**：所有权重一律放 `/public/hz_nas/tong/model/` 下（NAS），绝不要落到 `/root` —— root 目录会被清理。`HF_HOME` 也指到 `/public/hz_nas/tong/model/hf_cache`，防止 transformers 偷偷写 `~/.cache`。

   GR00T 训练入口在多个地方硬编码了 `nvidia/Cosmos-Reason2-2B` 的 HF id；为了让训练完全走本地路径，做了如下 **4 处修改**：

   | 路径 | 修改内容 |
   |------|---------|
   | `Isaac-GR00T/gr00t/configs/model/gr00t_n1d7.py:40` | `model_name` 默认值改为本地路径 |
   | `Isaac-GR00T/gr00t/experiment/launch_finetune.py:90` | `config.model.model_name` 强制赋值改为本地路径（这里会覆盖上面的默认值） |
   | `GR00T-N1.7-3B/config.json` | `model_name` 字段改为本地路径（备份在 `config.json.bak`） |
   | `GR00T-N1.7-3B/processor_config.json` | `processor_kwargs.model_name` 增加并指向本地路径（备份在 `processor_config.json.bak`） |

   训练脚本默认开启 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，避免任何隐式网络访问（比如 transformers 4.57 内部的 `_patch_mistral_regex` 会偷偷调 `huggingface_hub.model_info`）。

   重新下载（万一权重丢了）：

   ```bash
   pip install modelscope
   modelscope download --model nv-community/GR00T-N1.7-3B  \
       --local_dir /public/hz_nas/tong/model/GR00T/GR00T-N1.7-3B
   modelscope download --model nv-community/Cosmos-Reason2-2B \
       --local_dir /public/hz_nas/tong/model/GR00T/Cosmos-Reason2-2B
   # symlink 让路径里出现 nvidia/Cosmos-Reason2 子串
   mkdir -p /public/hz_nas/tong/model/GR00T/nvidia
   ln -sf /public/hz_nas/tong/model/GR00T/Cosmos-Reason2-2B \
          /public/hz_nas/tong/model/GR00T/nvidia/Cosmos-Reason2-2B
   ```

   其他 finetune 后的 checkpoint（LIBERO/DROID/SimplerEnv）也在 ModelScope `nv-community` 组织下，按需下载。

3. **Smoke test 已通过**

   单卡 L20Z + batch=8，跑 20 步 ≈ 113 s，最终 `train_loss ≈ 1.143`，模型加载、数据加载、forward/backward 全链路都正常。脚本：参考 [smoke test 命令](#smoke-test-命令)。

3. **action 表示与 openpi 对齐**
   openpi 用 `delta_action_mask = [True]*14 + [False, False]`，
   即 14 维关节做 delta，2 维夹爪保持绝对值。
   `franka_steak_config.py` 用 `RELATIVE`/`ABSOLUTE` 对应起来；
   如果以后换成 EEF 训练，要把 `type` 改成 `ActionType.EEF`、
   `format` 改成 `ActionFormat.XYZ_ROT6D`，并补 `state_key`。

4. **batch / 显存**
   `--global-batch-size 128` × 8 GPU = 每卡 16。L20Z 80 GB 可以放下；
   如果某次 OOM，先把 `--global-batch-size` 降到 64
   或加 `--gradient-accumulation-steps 2`。

5. **NEW_EMBODIMENT 标签**
   自定义具身一律走 `--embodiment-tag NEW_EMBODIMENT`，
   并通过 `--modality-config-path` 指向 Python 配置文件，
   `register_modality_config(..., embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)`
   会在导入时把配置写入全局 `MODALITY_CONFIGS`。

6. **`uv` 与 `conda` 的差别**
   GR00T 官方文档全部用 `uv run`。我们这套环境是用 conda 装的，
   `pyproject.toml` 中的 `[tool.uv.sources]`（torch/cu128 索引、
   flash-attn 的固定 wheel URL）不会走，
   所以 torch、flash-attn、ffmpeg 在本次配置时是手动装的：
   - torch/torchvision：`mirrors.aliyun.com/pytorch-wheels/cu128/`
   - flash-attn：通过 `gh-proxy.com` 下载 GitHub release wheel
   - ffmpeg：`conda install -c conda-forge ffmpeg=7.*`
   一次装好后跑训练直接 `python` / `torchrun` 即可，不需要 `uv run`。

7. **每次 `uv run` 警告**
   官方 README 提到 `uv run` 每次会重新校验 flash-attn URL；
   我们用 conda 没有这个问题，可以忽略相关说明。

8. **wandb**
   脚本内嵌了 openpi 用的同一个 `WANDB_API_KEY`，
   project 设为 `finetune-gr00t-franka-steak`。
   如果想关掉 wandb，把 `--use-wandb` 这一行删掉即可。

9. **评估**
   微调完成后可用 open-loop 评估对照 GT 轨迹：

   ```bash
   python gr00t/eval/open_loop_eval.py \
       --dataset-path /public/hz_nas/tong/data/lerobot_flipsteak_500_224 \
       --embodiment-tag NEW_EMBODIMENT \
       --model-path /public/hz_nas/tong/model/gr00t_franka_flip_steak/checkpoint-30000 \
       --traj-ids 0 1 2 \
       --action-horizon 16 \
       --modality-keys left_arm right_arm left_gripper right_gripper
   ```
   注意 `open_loop_eval.py` 也会触发 modality config 注册流程，
   建议运行前 `import` 一次 `examples/franka_steak/franka_steak_config.py`
   或把它放到 `PYTHONPATH` 里。

## Smoke test 命令

下面这段直接拷过去能跑（单卡 ~2 分钟）：

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate gr00t
export HF_HOME=/public/hz_nas/tong/model/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /public/hz_nas/tong/code/Isaac-GR00T
CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
    --base-model-path /public/hz_nas/tong/model/GR00T/GR00T-N1.7-3B \
    --dataset-path /public/hz_nas/tong/data/lerobot_flipsteak_500_224 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path /public/hz_nas/tong/code/Isaac-GR00T/examples/franka_steak/franka_steak_config.py \
    --num-gpus 1 --output-dir /tmp/gr00t_smoke_test \
    --max-steps 20 --global-batch-size 8 \
    --no-use-wandb --save-only-model --dataloader-num-workers 2
```

最后两行应当看到类似：
```
{'loss': 1.1485, 'grad_norm': 0.5955, ...}
{'train_runtime': 62.4, 'train_samples_per_second': 2.56, 'train_loss': 1.143}
```

## 5. 文件清单

| 路径 | 说明 |
|------|------|
| `/public/hz_nas/tong/data/lerobot_flipsteak_500_224/meta/modality.json` | 新增，GR00T modality 描述 |
| `/public/hz_nas/tong/data/lerobot_flipsteak_500_224/meta/stats.json` | `generate_stats` 自动生成 |
| `/public/hz_nas/tong/data/lerobot_flipsteak_500_224/meta/relative_stats.json` | `generate_rel_stats` 自动生成 |
| `Isaac-GR00T/examples/franka_steak/franka_steak_config.py` | 新增，NEW_EMBODIMENT 配置 |
| `Isaac-GR00T/examples/franka_steak/README.md` | 本文件 |
| `Isaac-GR00T/training_sh/start_franka_steak_gr00t.sh` | 新增，训练启动脚本 |
