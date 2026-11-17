# 春日野穹 GPT-SoVITS 表演向语音合成（AI 导演 + 贴耳轻语控制）

基于 GPT-SoVITS v2ProPlus 的角色表演向日语语音合成项目，支持 **DeepSeek / OpenAI 兼容 / Ollama** 自动情绪导演、句内多情绪分段控制、自适应声学限幅、以及 Demucs 洗净训练的 **ASMR 贴耳轻语模式**。

## 核心特性

1. **AI 配音导演规划（`performance.py`）**
   - 自动将长台词切分并结合上下文（`context`）逐段分配主/次情绪、权重、强度、停顿（`pause_s`）、发声方式（`delivery`）与独立拟声事件（`<breath>`、`<gasp>`、`<sob>`、`<laugh>` 等）。
   - 支持**超长台词自动分批规划与无缝拼接**，无 300 字 / 8 分段限制。
   - 支持 **DeepSeek 自动联动控制**：识别到贴耳/睡前低语场景时自动开启轻语模式与轻度去刺耳；规划出局部颤音（`tremble_strength`）或局部重音/音高变化时自动启用局部声学后处理。

2. **自适应声学控制引擎与防劈音限幅（`synthesize.py`）**
   - 基于参考音频轻柔度（`REF_SOFTNESS`）对负向能量/音高偏置做**连续衰减**，并按维度（情绪、音高、能量、韵律静音概率）独立限幅，防止“轻语参考 + 负向控制向量”双重叠加压坏声带。
   - **贴耳轻语模式（ASMR）**：自动切换 Demucs 洗净后的 ASMR SoVITS 权重（`epoch5`）与低气泡音参考（`ASMR_0290`），普通模式则使用 `epoch25` + `SR011455` 保持稳定张力。

3. **后处理与自动质检（`acoustic_delivery.py`）**
   - 可选**轻度 De-clicker + 动态 De-esser**（`apply_light_declick_deess`）：柔和压平清辅音高频尖刺，保留元音与气声。
   - `check_take` 自动检测长静音、音素对齐度及轻语模式下的异常气泡音/卡顿，配合候选队列自动切换下一条更稳的版本。

4. **WebUI 交互界面（`webui_sora.py`）**
   - 支持中文自动转穹口吻日语台词、导演对话讨论定稿、多轮试听评审闭环（`audio_goal.py`）、历史归档（`director_archive.py`）及端口被占用自动顺延。

## 目录结构（源码）

- `webui_sora.py` — Gradio WebUI 主程序
- `synthesize.py` — 推理引擎、自适应控制限幅与 SoVITS 权重热切换
- `performance.py` — 情绪特征编码、DeepSeek / Ollama 导演规划与参考排序
- `acoustic_delivery.py` — 局部音高/响度/颤声、轻度 De-click/De-ess 与 `check_take` 质检
- `audio_goal.py` / `director_archive.py` — 多模态音频目标评审与对话归档
- `colab_bundle/` — 训练脚本（`train_condition.py`、`train_sovits.py`、`prosody_director.py`、`sovits_emotion.py`、`vocal_tokens.py`、`cross_reference.py` 等）

## 快速使用

1. 安装好 GPT-SoVITS v2ProPlus 运行环境（`F:\GPT-SoVITS\app` 及对应 Python 环境），并准备好微调权重：
   - GPT 条件适配器：`v3_checkpoints/gpt_eval/phrase-adapter-epoch15.pt`
   - 常规 SoVITS：`v3_bs16_checkpoints/sovits_eval/sora-sovits-v2proplus-epoch25.pth`
   - ASMR 轻语 SoVITS（可选）：`v3_asmr_clean_checkpoints/sora-sovits-v2proplus-epoch5.pth` 与参考 `asmr_refs/ASMR_0290.wav`（及 `.wav.pt`）
2. 启动 WebUI：
   ```powershell
   python webui_sora.py
   ```
   或双击 `启动穹语音生成.cmd`（按本机环境路径调整）。
3. 在网页底部填写 DeepSeek（或其他 OpenAI 兼容接口）API Key，输入中文或日语台词即可生成试听。

> 注：本仓库仅托管源码与训练脚本；完整绿色免安装整合包（约 7.9 GB）及模型权重因超出 GitHub 单文件限制未直接放入仓库。
