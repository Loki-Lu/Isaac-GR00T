# GR00T N1.7 翻牛排微调 — 配置优化报告 (2026-05-18)

## 1. 最大 Batch Size 测试结果

硬件：8× NVIDIA L20Z 80GB

| Per-GPU BS | Global BS | 显存/卡 | 状态 |
|-----------|-----------|---------|------|
| 320 | 2560 | 72.6 GB (89%) | **推荐** |
| 360 | 2880 | 80 GB (98%) | 能跑但危险 |
| 400 | 3200 | OOM | 超了 |

## 2. GPU 利用率抖动排查

- 现象：GPU 在 0% 和 100% 之间周期性跳动（~3.5s 一个 step）
- 排除了 NAS I/O（RAM disk 无改善）和 dataloader workers（8 比 4 更慢）
- **根因：DDP all-reduce 同步 + CPU 数据预处理的固有开销**，有效利用率 ~70%，正常

## 3. 最终训练配置

```bash
Global BS = 2560 (per-GPU=320)
max_steps = 2100 (≈5 epochs)
LR = 5e-4 (sqrt缩放, 原1e-4@BS=128)
save_steps = 420 (每epoch存一次)
dataloader_num_workers = 4
输出: /public/hz_nas/tong/ckpt
预计时长: ~2小时
```

启动脚本: `training_sh/start_franka_steak_gr00t.sh`

tmux 进入查看训练:
```bash
tmux attach -t gr00t_train
```

## 4. 关键发现

- **微调策略**：非全量、非 LoRA，是部分模块微调（51.5% 参数可训练）
  - 冻结：VLM backbone（视觉编码器 + LLM）
  - 训练：Action Head（projector + DiT + VLLN）
- **大 BS 能跑的原因**：冻结层不存梯度和优化器状态，省 ~15 GB/卡
- **Loss 收敛**：~1.5 epoch 后 loss 降到 0.06-0.07 并趋平
- **与 openpi 的 loss 差异**：openpi 能到 0.00x，主要因为小 BS(128) × 30000 步有更多优化器更新机会，大 BS 的 loss floor 更高

## 5. 可调微调开关

| 开关 | 默认 | 作用 |
|------|------|------|
| `tune_llm` | False | 解冻 LLM 全部层 (~1.5B) |
| `tune_visual` | False | 解冻视觉编码器 (~300M) |
| `tune_top_llm_layers` | 0 | 只解冻 LLM 顶部 N 层 (每层~50M) |
| `tune_projector` | True | state/action encoder/decoder (~330M) |
| `tune_diffusion_model` | True | DiT 扩散网络 (~1.09B) |
| `tune_vlln` | True | VL LayerNorm + Self-Attention (~200M) |

## 6. 模型架构

```
Gr00tN1d7Model (3.14B total, 1.62B trainable)
├── Backbone (Cosmos-Reason2-2B) [冻结]
│   ├── visual (视觉编码器)        ← tune_visual
│   └── language_model (LLM)      ← tune_llm / tune_top_llm_layers
└── Action Head [训练]
    ├── vlln + vl_self_attention   ← tune_vlln
    ├── state_encoder              ← tune_projector
    ├── action_encoder             ← tune_projector
    ├── action_decoder             ← tune_projector
    ├── position_embedding         ← tune_projector
    └── DiT (diffusion model)      ← tune_diffusion_model
```

## 7. 如果要降低 loss（接近 openpi 的 0.00x）

- 方案 A：降 BS 到 256, steps=21000, ~8-9h
- 方案 B：保持大 BS 训 20-30 epoch, ~8-12h
- 方案 C：gradient_accumulation_steps=10, 有效 BS=256, per-GPU 仍满载, steps=21000, ~8-9h
