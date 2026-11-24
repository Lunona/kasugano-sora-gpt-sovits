# 春日野穹 GPT-SoVITS 表演向语音合成（AI 导演 + 贴耳轻语 + 多模态自迭代）

基于 **GPT-SoVITS v2ProPlus** 的春日野穹（《缘之空》）表演向日语语音合成项目，集成了 **DeepSeek / 本地 Ollama AI 配音导演**、**阿里云千问 / Gemini / OpenAI 音频审听闭环**、**48kHz 双声道 ASMR 贴耳音景** 以及 **本地自优化微调引擎**。

---

## 核心特性

1. **AI 配音导演工作流（`performance.py` / `webui_sora.py`）**
   - **实时流式导演讨论**：支持在对话区与导演实时推敲台词、语气与潜台词，支持流式输出与一键定稿；搭配 `sora_persona.md` 角色表演档案，保持穹的性格、语感与口吻一致。
   - **中文台词自动地道化**：自动将中文输入改写为贴合角色习惯的日语口语，拒绝生硬机翻。
   - **句内多情绪分段规划**：自动按停顿切分台词并分配主/次情绪、权重、强度、停顿时长（`pause_s`）、发声方式（`delivery`）与局部韵律提示；支持超长独白分批规划与稳态拼接，纯喘息短段自动并入实词上下文。
   - **63 档声学画像选角**：结合语速（字/秒）与 F0 音高跨度筛选最匹配的参考音频（支持本地规则、DeepSeek 与 Ollama 语义排序）。

2. **导演自动声效与 48kHz 双声道音景（`synthesize.py` / `acoustic_delivery.py`）**
   - **贴耳轻语模式（ASMR）**：支持自动或手动切换 Demucs 洗净训练的 ASMR SoVITS 权重（`epoch5`）与低气泡音参考（`ASMR_0290`）；常规模式使用高清晰度 `epoch25` + 连续防劈音限幅（`REF_SOFTNESS`）。
   - **自动轻度去刺耳（De-click + De-ess）**：导演在贴耳轻语或密集喘息/气口场景下自动启用，柔化高频气沙与清辅音尖刺，保留元音自然质感。
   - **48kHz 双声道音景混音**：支持人头麦贴耳方位（正中 / 左耳 / 右耳 / 左右耳交替）与窗外微雨背景音，导演可根据语境自动调节雨声音量比例（`0.08 ~ 0.32`）。

3. **多模态音频目标审听闭环（`audio_goal.py`）**
   - 开启「目标模式」后，可将生成的 `.wav` 直接送给 **阿里云百炼 `qwen3.8-omni-flash`**（内置 SSE 流式支持）、Google Gemini 或 OpenAI Audio 进行真人级听感打分与逐轮改进，直到达到目标要求。

4. **质检与本地自优化升级（`self_optimize.py` / `director_archive.py`）**
   - `check_take` 自动检测台词对齐度、情绪张力与异常静音；
   - 支持标记「满意样本 / 错题反馈」，一键在本地微调韵律导演与 GPT 表现力适配器，并保留出厂备份随时回滚。

---

## 目录结构（源码）

- `webui_sora.py` — Gradio WebUI 主程序（浅/深色清透主题、独立生成进度条、对话归档）
- `synthesize.py` — 推理引擎、自适应控制限幅、分段合并与 SoVITS 权重热切换
- `performance.py` — 情绪特征编码、DeepSeek / Ollama 导演规划与参考排序
- `acoustic_delivery.py` — 局部音高/响度/颤声后处理、轻度 De-click/De-ess、48kHz 音景与 `check_take` 质检
- `audio_goal.py` — 多模态音频目标评审（支持阿里云百炼 Qwen-Omni SSE / Gemini / OpenAI）
- `self_optimize.py` / `director_archive.py` — 历史记录数据库、错题统计与本地权重微调/回滚
- `sora_persona.md` — 春日野穹角色性格与日语配音表演基准档案
- `colab_bundle/` — 训练与对齐脚本（`train_condition.py`、`train_sovits.py`、`prosody_director.py`、`sovits_emotion.py`、`vocal_tokens.py`、`cross_reference.py` 等）

---

## 绿色免安装整合包（推荐普通用户）

绿色版内置 Python、GPT-SoVITS 环境、合成权重、63 档声学参考和音素对齐模型。Windows 用户解压后双击 `启动穹语音生成.cmd`。

- **首次安装**：下载 [v2.0 完整版](https://github.com/Lunona/kasugano-sora-gpt-sovits/releases/tag/v2.0) 的六个分卷。放在同一目录，用 7-Zip 或 Bandizip 打开 `.001` 解压，直接启动。
- **已有 v1.0 或 v1.1**：下载 [v1.2 累计更新补丁](https://github.com/Lunona/kasugano-sora-gpt-sovits/releases/tag/v1.2)，解压覆盖到包含启动脚本、`app`、`sora` 和 `data` 的目录。
- [v1.0 完整包](https://github.com/Lunona/kasugano-sora-gpt-sovits/releases/tag/v1.0) 仍可下载。其 [百度网盘备份](https://pan.baidu.com/s/1oHZiATkQdyCzA1DN9_sJXg) 提取码为 `s2ns`。

### 本地 Ollama 导演

1. 安装并打开 [Ollama](https://ollama.com/download/windows)，下载对话模型。可用 `ollama pull qwen2.5:3b` 获取已测试的模型。
2. 在「API 设置 → 导演模型」选择「本地 Ollama」，从检测到的列表选一个模型，点「保存配置」。
3. 左侧「情绪控制」选择「完整导演（使用已配置模型）」。右侧可讨论演法、定稿、下达制作任务。采用本地 Ollama 时，导演可以读取完整角色档案和 63 条参考资料。

同一位置也可配置 DeepSeek 或其他 OpenAI 兼容接口。目标模式的 WAV 评审在「音频评审模型」独立配置。局部变调使用随包提供的 [Praat-Parselmouth](https://github.com/YannickJadoul/Parselmouth)。

---

## 快速使用（源码部署）

1. 准备 GPT-SoVITS v2ProPlus 运行环境（`F:\GPT-SoVITS\app` 及对应 Python 环境），并放置微调权重与辅助文件：
   - GPT 条件适配器：`v3_checkpoints/gpt_eval/phrase-adapter-epoch15-ft1.pt`
   - 常规 SoVITS：`v3_bs16_checkpoints/sovits_eval/sora-sovits-v2proplus-epoch25.pth`
   - ASMR 轻语 SoVITS（可选）：`v3_asmr_clean_checkpoints/sora-sovits-v2proplus-epoch5.pth` 与参考 `asmr_refs/ASMR_0290.wav`（及 `.wav.pt`）
   - 音素对齐模型：`tmp/japanese-hubert-base-phoneme-ctc-v4/`
2. 启动 WebUI：
   ```powershell
   python webui_sora.py
   ```
   或按本机路径修改后运行 `启动穹语音生成.cmd`（支持通过环境变量 `SORAVOICE_PYTHON`、`SORAVOICE_APP`、`SORAVOICE_DATA`、`SORAVOICE_ASMR_DATA` 自定义路径）。

> 注：本仓库仅托管源码、角色档案与训练脚本；模型权重、音频素材及约 9.3 GB 绿色免安装整合包未直接提交到 Git 仓库。
