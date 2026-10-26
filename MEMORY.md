# RVC 项目长期记忆

**这是本项目唯一的长期记忆文件。** 取代原来的 `.workbuddy/memory/`（含按日流水）与
`.workbuddy/tmp/`（临时脚本）。以后所有跨会话仍然成立的事实、约定、硬约束、被否决的
方向，都只写进这里——不再新建按日文件，不再建临时笔记目录。

写之前先读，避免重复踩坑。只写「现在为什么这样」和「别改成什么、否则会怎样」，
不写过程叙事（见 §5）。条目被推翻时**就地改写**，不要追加"更正：…"。

---

## 1. 环境与命令

- venv：`.venv/Scripts/python.exe`（Python 3.13.7，torch 2.11.0+cu128，
  transformers 5.9.0，PySide6 6.11.1，numpy/onnxruntime 见 requirements.txt）
- 测试：`.venv/Scripts/python.exe -m unittest discover -s tests`
  **stdlib unittest，不用 pytest，不写 conftest**。当前 187 例 / ~1.9s，无需 GPU 与模型
- torch 必须单独装：
  `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`。
  仓库 `requirements.txt` 顶部有大段说明，别把 `--index-url` 塞进去（会替换整个索引）
- 临时脚本必须设 `$env:PYTHONPATH='E:\Projects\Python\RVC'`，否则 `import rvc` 失败
- 真实模型冒烟：`assets/models/xd80+ay20.pth` + `chinese` 变体；`load_model` 冷启
  23–26s（transformers 冷启），热启 ~7s
- **GUI 启动不拉 torch**：`import gui.view.window` 0.41s。链路上别往 gui/controller
  的模块顶层加会 import torch 的东西（cache/graph/engine 一律惰性）
- GUI 不能离屏构造：`QT_QPA_PLATFORM=offscreen` 构造 MainWindow 会因 PortAudio 在沙箱
  内挂死（SIGTERM 零输出）。启动类验证只能靠用户实机；其余用 `MainWindow.__new__` 桩

## 2. 分层与边界

八层单向依赖（数字 = `tests/test_layering.py` 的 `RANKS`，依赖只准指向更小的）：
core=0，runtime=1，dsp/nn/io=2，models=3，pipeline=4，streaming/train=5，gui=6

- `rvc/` 不依赖 `gui/`，可作为库独立使用
- **view 层独占 widget**（window / lifecycle / tabs）；controller 层零 widget，只经窄协议
  `InferWindowHost` / `OfflineViewHost` / `DeviceViewHost` 通信
- `test_layering.py` 用 AST 静态守卫（含变异验证）。**已知盲区**：`gui/state/bindings.py`
  在模块级 import `rvc.core.config`，AST 看不到这条传递依赖
- 新包必须登记进 `RANKS`
- **推论（踩过）**：参数 dataclass 只能活在 `rvc/core/config.py`。`rvc.core` 不得 import
  `rvc.dsp`，所以 DSP 需要参数时反向 `from rvc.core.config import XxxParams`。
  这也正是参数能被 `PARAM_GROUPS` 序列化/拷贝自动带上的唯一位置
- `PARAM_GROUPS` 驱动的四处自动机：`update_from`、`params_from_dict`、`params_to_dict`、
  GUI `BINDINGS` 都遍历 `__dataclass_fields__`。新增分组只要登记进 `PARAM_GROUPS` 就自动
  获得持久化与热读；**property 不是 field**，所以"单滑条 + 四个内部系数"用 property
  折算即可，不会污染存档

## 3. 热路径硬约束（音频回调线程）

回调每秒 ~4–10 块，每块 20–40ms 预算。**禁止**：无界分配、强制 GPU 同步、阻塞等待、
在主线程 `wait()` 线程。

- CUDA Graph 捕获期间**禁止数据依赖形状操作**：布尔掩码索引、`.item()`、`if tensor`
  都会让捕获失败
- 标量进图闭包必须进 `graph_key`；张量走参数（地址静态）
- `formant_factor` 运行中可变：它经 `return_length2` 决定合成器图形状，量程内 ~10 个档位
  的图在 `prewarm_formant` 里启动前整段预捕获（实测 2.5s / +79MB），运行中切档是纯回放
- 调 formant **不能动 pitchf**：decoder 里 `n_res` 的 interpolate 已把时长拉伸的降频抵掉
- F0 side stream 的图回放**不能与主流图回放并发**（各持私有显存池，PyTorch 要求同流同序）
- 同步点位置有讲究：`.item()` 要放在同步 D2H 之后（队列已排空），GUI 侧一律读缓存
- 实测（RTX 5060 Laptop）：合成器图**首张** capture 585.5ms（含 cuDNN 冷启），
  **后续每张 190–300ms**（10 档合计 2.5s）；replay 5.4ms、eager 132.6ms（回放 24.46x）。
  所以 formant 走「启动预捕获」而不是「运行中懒捕获」——懒捕获会在回调线程里卡 0.2–0.3s
- 新增 DSP 阶段必须提供 `setup(sr, block_samples, device)` / `reset()` / `process()` 三件套，
  且 `setup` 在 `warmup()` 前、`reset()` 在 `warmup()` 后（预热静音会污染滤波器尾部状态）

## 4. 术语

- **work_sr** = 流实际跑的采样率（设备率或模型率之一）；**model_sr** = 模型原生采样率。
  缓冲区/重采样器按 `work_sr` 命名，禁用 `48k` 字样（device 模式下是谎言）
- **主输出 = `main_*`，副输出 = `out2_*`**（与 `audio.output2_device` / `enable_out2` 同词表）
- 时长参数统一 `block_time` / `crossfade_time` / `extra_time`（与 `BufferParams` 同名）
- 音频缓冲量字段统一 `_samples` 后缀
- 缓冲区几何：`zc = work_sr // 100`；`skip_head = extra_samples // zc`；
  `return_length = (block_samples + sola_buffer_samples + sola_search_samples) // zc`；
  消费区 = `feats[:, skip_head : skip_head + return_length]`
- HuBERT 帧率 50fps，**帧跳 = 320 个 16k 样本 = 20ms**

## 5. 注释与文档写作规范

**不写历史叙事。** 后来者没见过"旧实现/曾经/早期/当年/此前/现已改成"，写了只会让人困惑。
要保留的是「为什么现在这样」和「别改成什么，否则会怎样」，不是"以前怎样"。
唯一例外：时间顺序（"同步之前"、"过去帧"）与运行时对象（"旧流"、"旧模型实例"）。
项目内也不要留计划项编号（`A2`/`A6`/`D4` 这类）——外部无人能解码。
注释用中文。README 的设计决策段会随行为演进过期，**改行为时记得同步**。

## 6. git 与工作区安全（血泪经验，务必遵守）

本仓库多次发生 **git 操作导致 gitignore 的大文件从磁盘消失**（`assets/` 下模型、ffmpeg
全丢过三次）。

- **禁止** `git add -A`（曾把用户自建未跟踪的 `tools/` 扫进提交）。提交只 add 指定路径
- **禁止** `git mv`，移动文件用文件系统移动
- 恢复 tracked 文件用 `git show HEAD:<path> > <path>`（纯读 object、不碰 index、不触发
  工作区重建）。父目录整个消失时先 `mkdir`，重定向不会建目录
- 每次动工作区后必须跑完整性核对：
  `git ls-files | ForEach-Object { if (-not (Test-Path -LiteralPath $_)) { "MISSING $_" } }`
- push 报 `schannel: failed to receive handshake` = 代理不通，不是仓库/凭据问题。
  用户机器 Clash 端口会变（7897 / 56375 都出现过通与不通），先探测再单条命令
  `-c http.proxy=...` 覆盖，**不动全局 config**
- **不主动 push**，等用户明确说
- 大段删除用 python 按字节 `split(b'\n')` 切片后 `wb` 写回（本仓 Python 源是 LF）；
  改代码一律用 Edit 工具，脚本补丁只用于机械批量替换且改后必须语法校验
- 删函数时 Edit 的 old/new 弄反会把函数复制一份——删后必须 grep 验证"零残留 + 无重复定义"

## 7. 协作偏好

- 不要"调参数"式建议（用户已试遍参数组合，要的是定位并修代码）
- 小改动不做截图/实机验证，编译过 + 说清改了什么即可；**别把验证本身做成项目**
- 说 review 就是只读检查 + 报告，别顺手改代码
- 清理类需求 = 删死代码与死资源，不是趁机重构；删完报告清单
- 用户会直接推翻结论。被推翻时先找自己证据的等级问题（n=1？指标失明？），
  别辩解，也别把推翻前的说法留在记忆里

## 8. 已知设计取舍（别当 bug 修）

- 无 GPU 直接报中文错误，**不做 CPU 回退**（刻意行为）
- 实时链路延迟显示默认只算算法延迟，驱动补偿默认 0（宁可少报不虚报）
- 缓存失效语义只有 LRU 容量淘汰，**无文件 mtime 校验**（同名 pth 被外部覆盖会返回旧权重，已接受）
- F0 每块重提取，RMVPE 走整段、FCPE 只算尾部 0.6s 窗口
- `crossfade_time=0.05` 保持 dataclass 默认值，不具名化（SOLA 缓冲实际取
  `min(crossfade_samples, 50ms)`）
- **RMVPE 的清浊判决在阈值边缘不可稳定**：conf 落在 `rmvpe_threshold ± 0.005` 的帧，
  输入差 1e-8（重采样归约顺序的 ULP）就能翻成另一个判决，pitch 整 bin 跳。这是模型输出
  落在判决边界上的固有性质，不是实现差异。逐位比对遇到这种帧应记录并豁免，别去"修"它
- 停机直通恒启用（用户裁定不要勾选框）：底部三档 = 原声(蓝)/停止(红)/变声(绿)，
  分段 toggle 单选，托盘三色图标一致
- **出错严禁回落原声**（露馅），一律停止档。`start_passthrough` 全项目唯一调用点 =
  用户点原声档。任何新错误路径都必须落到停止档
- 合成器三处随机量的形状校验**不符就静默退回随机、不报错**：`sine_rand_ini` 要
  `(1,1,dim)`、`sine_noise` 要 `(1, return_length*upp, dim)`、`z_sample` 要与 `m_p` 完全同形，
  `upp = prod(upsample_rates)`（本模型 400）。改随机源相关代码时先确认形状真被接受，
  否则「注入了但没生效」极难反查

## 9. 已封死的方向（做过、被否决，别重做）

- **protect / 特征检索**：两次加回两次被删。本仓不提供 protect，检索链路也不恢复。
  旧版「100%女声」体验依赖 9/5 已整体移除的 torch-GPU 检索链路；无检索时特征侧 protect
  是恒等式，输出侧 protect 会漏原声（不可接受）
- **特征时间反锐化**（`f' = f + α(f − 前 k 帧均值)`，α=0.3/0.6）：用户裁定"不如不加"。
  封死结论：**特征下游补救（protect / unsharp 同病）勿再做**；弥散干预只剩注意力本体
  （温度/局部窗）或换编码器重训
- **气息感 `MicAirflowEffect`**（9334cbd 整体删除，-493 行）：用户裁定连续调标定/结构都
  无法解决「不像真实风噪 + 尾音沙哑」。**v10 教训：连续嘶声/沙沙、拖尾保持一律否决。**
  根因记录（对任何噪声类功能都适用）：合成噪声床是**绝对电平**、与音节无相关性，
  且 floor 层完全不受语音门控 → 尾音 SNR 从 +9.5dB 掉到 +1.7dB（噪声≈人声）= 沙哑。
  结构性放大器是「floor 不随门控退让」，不是强度本身
- **HuBERT 右端补零**（384594c → 95d63c4 回退）：HuBERT 是**全局注意力**，补进右端的
  静音经注意力作用于窗口内**所有**帧，「补零 = 因果化」的前提本身不成立。0.3/0.5/1.0s
  三档都不能消除漂移，1.0s 更差。别再走"补零长度"方向
- **把 extra_time 当解法**：结论随素材切换而反转（1.2s 单元平铺测 5.0 在 2s 就漂；
  凤凰.wav 平铺测 5.0 与 1.2 都全程稳定）。漂移与素材内容/平铺单元长度强相关，
  单纯调 extra_time 不是可控解法
- **重叠区特征缝合**（沿用上一块特征）：对数梅尔余弦 0.7965 → 0.8425，收益有限且有
  拼接不连续风险，**未采用**
- **F0 接缝融合**（重叠帧沿用旧判决 + 中值）：A/B 反而把 F0 轨道弄坏，已完整回退。
  教训：写者间分歧 ≠ 消费轨道缺陷，每帧只被 owner 块消费一次，「不一致」不等于「错误」
- **`rand_ini`（SineGen 初相）固定**：更差。正弦相位链按窗内起点累加，窗口滑动时绝对
  相位本来就会漂，与初相无关
- **ε=0（彻底确定性噪声）**：可消"会话抽签"，但 OOD 尾段更糟（狒睡），未采用
- **C++ 重写**：2026-10-26 用户决定放弃，`native/`（331MB）已整体删除且**从未纳入 git**。
  不要再提议或恢复 C++ 移植，也不要去找 `native/`
- **CPU 回退**、**`git add -A`**、**`git mv`**：见 §8 / §6

## 10. 咬字漂移（objective 3）—— 证据链与现状

用户症状：连续"啊啊啊"/"凤凰凤凰"约 10s 后段变怪，怀疑 HuBERT 非因果 + 滑窗自我相似。

**已确证的事实**

1. **不存在渐进漂移 bug**：把输入造成严格周期且块对齐的信号，第 k 块与第 k−P 块滚动窗口
   逐位相同，特征/合成/输出收敛到**逐位相同**稳态（feat cos → 1.000000），前后半段无趋势
2. **窗口相位依赖是真的**：同一段 20ms 音频被滑窗放在不同位置看到时，HuBERT 特征余弦
   只有 0.52（extra 5.0）。用户"特征收敛到另一个吸引域"的直觉在特征层成立
3. **窗口越长相位依赖越强**，lookahead 显著改善。实测（xd80+ay20 / hubert base / rmvpe / 40k，
   凤凰.wav 平铺）：

   | extra_time | lookahead | 相位余弦(中位) | 相位余弦(最差) | ms/块 |
   |---|---|---|---|---|
   | 0.6 | 0 | 0.6646 | 0.4470 | 24.5 |
   | 0.6 | 1 | 0.8259 | 0.5251 | 25.1 |
   | 1.2 | 0 | 0.5996 | 0.4409 | 27.9 |
   | 1.2 | 1 | 0.8429 | 0.5750 | 28.4 |
   | 2.5 | 0 | 0.5431 | 0.4344 | 29.9 |
   | 2.5 | 1 | 0.8182 | 0.5758 | 30.3 |
   | 3.4 | 0 | 0.5318 | 0.4177 | 31.0 |
   | 3.4 | 1 | 0.8149 | 0.5914 | 32.1 |
   | 5.0 | 0 | 0.5194 | 0.4289 | 39.0 |
   | 5.0 | 1 | 0.8036 | 0.6437 | 38.7 |

4. **`HUBERT_LOOKAHEAD_BLOCKS` 默认 0（关闭）**：在窗口右端放 N 个真实未来输入块、
   输出整体延后同样时长，离线路径裁掉 `pad + lookahead`。实测耗时只 +0–1ms/块，
   **代价是 +N 块算法延迟（100ms 块下 = +100ms）**。
   曾默认开 1，用户立刻反馈"延迟多了 100ms"——收益只在特征层，听感层没兑现（见第 5 条），
   拿 100ms 换不划算，已改回默认 0。要自己听 A/B 就设 `RVC_HUBERT_LOOKAHEAD_BLOCKS=1`，
   **不要再改默认值**。这条仍是 objective 3 唯一"不改模型/不改音色/不要素材/不让用户调参"
   的正解方向，只是它买不到听感
5. 与 10-25「真前瞻无效」不矛盾：那次的裁判是 ASR 只看最终字；相位一致性是机制层指标。
   但反过来也成立——**机制层指标变好 ≠ 症状变好**。判 objective 3 的成败只能看
   "同一段内容反复出现时字会不会变"，而这项 ASR 测试当时明确显示前瞻不改变漂移时刻
   （0/300/1000ms 都在约 8s 起漂，只是漂到不同的字）。以后拿数值指标宣称修好了咬字，
   必须先说明它和听感判据的对应关系
6. 残余症状更可能是：(a) 窗口在 `extra_time ≈ 5s` 填满那一刻特征统计突变（之前是短有效
   窗口，之后是 5s 长窗）；(b) 持续发声时嗓音自身退化成气泡音（HNR 从 24dB 跌到 −1.7dB，
   已复现），特征落入训练分布外模糊区，流水线把它放大。两者都指向"让特征更稳"，不是找累积 bug
7. 独立发现（与时间无关）：输入 F0 > 570Hz（音域映射顶 src640→dst720）输出也会崩

**其他已排除的嫌疑**（别再查）：16k 缓冲接缝（48k/44.1k/40k/16k 整除率严格对齐）、
SOLA 打分（对严格连续信号 62/62 块选 offset=0、score 0.91）、拼接与交叉淡化
（边界前后 50ms 谱差 7.08dB vs 块内部 7.29dB = 0.97x，不比内部更突变）、
rms_mix 增益（只在衰减，中位 0.16x）、设备率重采样链、`_stage_preprocess` 实时错位
（帧数不匹配会当场抛异常，不可能静默渐进错位）

## 11. 真实感搬运链（objective 4，已落地）

`rvc/dsp/realism.py` 的 `RealismChain`：从**参考输入块**搬录音痕迹到输出，**不合成噪声**。
四级：`residual`（底噪）/ `breath`（>4kHz 气息齿音）/ `rumble`（近讲低频搁架 + tanh 软饱和）/
`distance`（OU 随机游走电平 + 谱倾斜）。

**参数在 `rvc/core/config.py`**（`RealismParams` + `REALISM_MIX`），理由见 §2。面板只给一根
滑条，走 `realism.strength` property（getter = `max(级/权重)`，setter 按 `s·w` 分摊）。

**接入点固定**：`runner.py` stage_output 里 `effects.process_output` **之后**、volume **之前**。
SOLA 之前注入会被块间拼接切碎；volume 之后注入则调小音量时底噪相对变大。

实测（RTX 5060 Laptop / 40kHz）：强度 0 旁路 0.053ms/块（0.1%）；0.35 为 1.92ms/块
（100ms 块 = 1.9%）；0.7 为 1.85ms/块；250ms 块下 0.8%。

端到端（生产离线路径，输入 = 111.wav + −50dB 底噪 + >4kHz 空气感 + 0.4s 静音段）：
强度 0 → 0.35 → 0.70 时，>4kHz 占比 0.0329 → 0.0424（×1.29）→ 0.0570（×1.73）；
<200Hz 占比 0.0038 → 0.0102（×2.67）→ 0.0112（×2.95）；最静 200ms 的 RMS
0.00063 → 0.00246（×3.93）→ 0.00328（×5.24）。

关键读法：**输入**底噪 0.00354，输出静音段抬到 0.0033 就停，不会超过输入——这正是它
不像"加了噪声"的原因。反之干净素材（111.wav 原样：HF 占比 0.0075、无静音段）上
residual/breath 几乎不动，**不是 bug，是"没有东西可搬"**。验证实现必须用带痕迹的素材。

**默认四级全 0**（逐位旁路）。首次试听建议 0.3–0.4；`distance` 上限建议 0.3（它让电平
自己动，容易被当成 bug）。副作用：极低电平输入会作为"双重声"漏一点（约 −25dB），
嫌有双声就把 residual 降到 0.2 以内，别改增益公式。

**风险（与 §9 气息感教训直接相关）**：本链搬的是**输入自己的**底噪，不是绝对电平噪声床，
也不受语音门控——所以强度开大时尾音 SNR 仍会向输入靠拢。用户历史上正是因"尾音沙哑"
否决过气息感。若再听到拖尾变沙：**先降 residual**，不要动 breath/distance，也不要加门控层。

## 12. 性能（objective 2）

**分阶段剖析（当前口径）**：用户实机配置 40kHz / 100ms 块 / crossfade 50ms / extra 5.0 /
rmvpe / fp16，整块 **38.23ms**：

| 阶段 | ms/块 | 占整块 |
|---|---|---|
| `postprocess_features`（F0 后处理 + 特征上采样） | 15.65 | **40.9%** |
| `extract_features`（HuBERT） | 9.29 | 24.3% |
| `synthesize_audio` | 7.05 | 18.4% |
| `effects.process_output`（RMS + SOLA） | 2.85 | 7.5% |
| `_stage_preprocess`（重采样） | 1.39 | 3.6% |
| `realism.process` | 0.48 | 1.2% |
| `_stage_input` | 0.29 | 0.8% |
| `_stage_formant` | 0.00 | 0.0% |

**最大头是 F0 后处理 + 特征上采样，不是 HuBERT**——早期记的"HuBERT 17.8ms 占七成"是用
别的仪器（D2H 同步钩子）和别的素材测的，不可直接比。优化优先看 `postprocess_features`。

**剖析方法（脚本随 .workbuddy 删除，按此重建）**：用 `torch.cuda.Event` 成对包住各阶段
（事件在同 stream 顺序记录，主机端只在最后同步一次，不像 `.cpu().numpy()` 钩子那样把耗时
抬高一大截）。宿主对象映射：`_stage_input`/`_stage_preprocess`/`_stage_formant` → runner；
`extract_features`/`postprocess_features`/`synthesize_audio` → `runner.pipeline`；
`process_output` → `runner.effects`；`process` → `runner.realism`。
runner 要自己造：`engine._make_runner(work_sr, 1, block, crossfade, extra, model_sr, reset_buffers=True,
prewarm_formant=False)`（`load_model` 不建 runner；`model_sr` 传 0 会让 torchaudio 的 sinc
核宽算成 0 → ZeroDivisionError）。

**其他已测**：`extra_time` 5.0 → 39.0ms/块 vs 0.6 → 24.5ms（早期口径）；
块耗时基线（早期口径）HuBERT 17.8 + F0 后处理 1.2 + 合成器 5.6 + RMS/SOLA 1.0 ≈ 26.5ms；
首块 2041ms 是图捕获一次性开销。

**设备层已调**：AudioStreams 用 `latency="low"` + 显式 `blocksize`。WASAPI 共享模式下
设备缓冲只有 10–30ms（实测 1056 帧 @48k），**远小于块长**——任何"整块塞得下才写"的判据
在 Windows 上永远不成立，必须用待写队列（`pending` + `pending_pos`，上界 3 块，超出丢最旧）。
这条来自已删除的 C++ 实现，但结论对 Python 侧同样成立。

## 13. 训练模块（objective 6）

- DataLoader 选项由 `rvc/train/trainer.py::training_dataloader_options` 统一产出：
  `RVC_TRAIN_WORKERS`（默认 2，设 0 关闭）、`RVC_TRAIN_CACHE_MB`（进程总预算，按 worker 均分，
  0 关闭样本内存缓存）、`RVC_TRAIN_PREFETCH`（默认 1）。`persistent_workers` 避免每 epoch
  重新 spawn；`cleanup()` 显式 `iterator._shutdown_workers()`，否则 Windows 上进程被 worker 持有
- **`atomic_write` 对 `os.replace` 的 `PermissionError` 重试 5 次**（退避 0.05/0.10/0.15/0.20s）：
  POSIX rename 只改目录项，Windows MoveFileEx 在目标被以"不共享删除"方式打开时失败；
  `.pth`/`.pt` 正是杀毒软件、Everything 索引器、资源管理器预览最常扫的文件。只重试
  PermissionError，ENOSPC/EROFS 必须立刻抛（否则"磁盘满"被拖成半分钟卡顿再报错）。
  契约锁在 `tests/test_atomic.py`：序列化中途抛异常时目标文件仍是旧内容（不留截断文件）
- 所有子进程 spawn 必须带 `rvc.io.CREATE_NO_WINDOW`（含 `stdin=subprocess.DEVNULL`），
  否则 GUI 拉起训练时 Windows 会闪黑框
- 两个入口都有 `if __name__ == "__main__"` 守卫（`app.py`、`rvc/train/cli/__main__.py`），
  Windows spawn 必需，别删
- ffmpeg 路径统一经 `cfg["ffmpeg"]` 下发（PreProcessor / TrainF0Extractor / HuBERTExtractor /
  Trainer），不在各模块里各自找
- 续跑判据是"文件存在即跳过"，所以**所有中间产物必须原子写**，否则中途强杀留下截断文件
  会被静默复用

## 14. 测量方法论与踩坑（做实验前必读）

**可用的判据**
- **周期输入 ⇒ 块等价**：把输入造成严格周期且周期 = 整数个块，第 k 块与第 k−P 块滚动窗口
  逐位相同，则特征/合成/输出都该逐位相同。用来抓"时间累积型 bug"，最灵敏
- **相位散度**：周期素材下按"绝对帧号 mod 周期"分桶，同桶 = 同内容被不同窗口位置看到，
  桶内两两余弦 = 窗口相位依赖强度
- **重叠一致性**：相邻两块对同一段 60ms 内容各渲染一次，比较两次渲染
- **扰动放大率**：同一输入 + −60dB 白噪，比较输出变化
- **分阶段 CUDA Event**：见 §12

**失灵的判据（别再用）**
- 波形余弦/波形相关性：谐波相位不同，最佳滞后互相关只有 0.18–0.29，毫无意义
- `out` 块余弦：SOLA 会重新对齐，块间比较无意义
- mel/特征 MSE 对**音位翻转失明**：噪声地板就高于信号差。判"咬字有没有变"只有 ASR 有效，
  但本环境已无 ASR 工具（AudioTranscribe 不可用）
- 对数梅尔比较要小心 `clamp(min=1e-8)` 会放大静音段的差异
- FFT 峰值在语音上常抓到共振峰而非 F0（系统性偏高 5–10%），只能看相对轨迹不能读绝对值

**坑**
- HuBERT 输入 N 帧 → 输出 **N−1 帧**（不是 N）
- 改缓冲区长度/`extra_time` 后**必须重建 `state.noise_z`**：形状不匹配时 `_z_sample`
  静默退回 `randn_like`，逐块随机抽取会让 A/B 结论完全失真
- 计时前必须 `warmup()`，否则首块 2s 图捕获污染均值
- 拦截 `enc_p`/中间量前必须先 `configure_cuda_graph(..., use_cuda_graph=False)`，
  图回放会绕过 Python 调用
- 分析输出采样率 ≠ 输入采样率（输出 40k、按输入 48k 切段会漏高频尾部）
- SOLA 的 NCC 归一化要除 `√BUF`
- `sd.default.device` 读回是 `sounddevice._InputOutputPair`（不继承 list），
  `== [1,6]` 是身份比较恒 False，必须 `list(x) == [1,6]`；赋 None 读回会展开成具体设备索引
- `QPixmap` 需先建 `QGuiApplication`，否则 Qt 静默 abort（exit 127 无输出）
- `build_*_tab` 返回的顶层 widget 不保留引用会被 GC，`findChildren` 返回空
- radio/控件包容器入 grid 必须钉垂直 Fixed，否则富余高度被该行独吞
- grid 无列 stretch 时富余宽度分给外边距（内容居中），左右边缘会莫名其妙很大
- tab 构建期引用的 `win` 属性必须在 `_build_ui` 前就绪

**本机环境坑**
- 控制台中文在 pwsh 里会乱码：跑脚本前设
  `$env:PYTHONIOENCODING='utf-8'; [Console]::OutputEncoding=[Text.Encoding]::UTF8`
- `load_audio(path, None)` 会让 ffmpeg 报 `Expected number for ar but found: None`，
  探针必须显式传目标采样率
- 探针用相对路径 + 非仓库根 workdir 会写出嵌套目录，输出目录一律绝对路径或从仓库根运行
- 工作区 ACL 曾缺 `WRITE_OWNER`，导致 DSH 沙箱无法 provision、**所有 pwsh 命令失败**而读
  工具正常。修复脚本见 `diagnose-windows-sandbox-acl` skill，备份在
  `E:\Projects\Python\dsh-acl-recovery\`
- 递归 grep 整个仓库会因 `.venv` 挂住，用 grep 工具并限定路径

## 15. 素材与用户配置

- 素材：`E:\Download\audio\{凤凰,111,aaa,ah_sweep_step1}.wav`；环境音 `E:\Download\环境音素材`；
  KTUM 应用 `E:\Download\ktum\KtUltraMorph.exe`
- 用户存档 `assets/configs/save_state.json`：block 0.1，crossfade 0.05，extra 5.0，
  rmvpe 0.03，sr_mode device，WASAPI，输出 "CABLE Input (VB-Audio Virtual Cable)"，
  volume 1.5，rms_mix 0.25，模型 `assets/models/xd80+ay20.pth`，hubert `base`，formant −0.25
- A/B 试听要先把输入峰值归一化到 0.5 并把 `volume` 置 1.0，否则 111.wav（峰值 1.098）×
  volume 1.5 直接削顶到 1.000，比较失效
