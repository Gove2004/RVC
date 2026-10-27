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
- 真实模型冒烟：`assets/models/xd80+ay20.pth` + 默认 `base` 变体；`load_model` 冷启
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
- `test_no_undefined_names.py` 用 `symtable` 守卫「函数体引用了模块中未绑定的全局名」。
  **这类缺陷 import 阶段无症状、只在调用时抛 NameError，靠测试用例撞见纯属运气**——
  已踩到两次：`rvc/train/cli/defaults.py` 漏 `from rvc.io.atomic import load_json_or_default`
  （训练向导预填默认值必崩）、`gui/configs/config.py` 漏 `import json`
  （`save_config` 每次必崩，被 `_tray_quit` 的 try/except 吞掉 → **用户设置静默全丢**）。
  当前基线 0 处，新增一处的成本远低于线上炸一次
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
  封死结论：**特征下游补救（protect / unsharp 同病）勿再做**；弥散干预只剩注意力温度
  或换编码器重训（局部窗已实测封死，见下条）
- **气息感 `MicAirflowEffect`**（9334cbd 整体删除，-493 行）：用户裁定连续调标定/结构都
  无法解决「不像真实风噪 + 尾音沙哑」。**v10 教训：连续嘶声/沙沙、拖尾保持一律否决。**
  根因记录（对任何噪声类功能都适用）：合成噪声床是**绝对电平**、与音节无相关性，
  且 floor 层完全不受语音门控 → 尾音 SNR 从 +9.5dB 掉到 +1.7dB（噪声≈人声）= 沙哑。
  结构性放大器是「floor 不随门控退让」，不是强度本身
- **给窗口右侧加未来上下文**（右端补零 / 真前瞻留未来音频）：两种做法都被用户实听否决。
  补零的前提「补零 = 因果化」本身不成立——HuBERT 是**全局注意力**，补进去的静音作用于
  窗口内**所有**帧，0.3/0.5/1.0s 三档都不能消除漂移（1.0s 更差）。真前瞻机制确实有效
  （相位余弦 0.52→0.80）但听感无可辨变化，白付 100ms 算法延迟，已整体移除。
  详见 §10 第 4 条。**别再走这条路**
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
- **局部注意力窗口**（12 层自注意力各加一条带状加性偏置：±W 帧内不动、窗外线性衰减到
  切断；推理侧**零权重变更**，音色在结构上不可能变）：**实测不成立，已整体回退**。
  单层偏置是精确的（改一帧，距离 300 帧处输出变化恰好 `0.000000`），但**模型级感受野
  不受约束**——12 层各 ±(W+过渡带) 叠加后本就很大，外加 `pos_conv_embed`（kernel 128）
  的 ±64 帧。宽度扫描（chinese 特征器，10s 凤凰.wav，fp16）：

  | W+过渡带 | 可达帧数 | 远距离影响力 vs 全注意力 | 正常语音特征扰动 |
  |---|---|---|---|
  | 4+4 | 96 | **184%** | 89.6% |
  | 8+4 | 144 | 134% | 64.1% |
  | 12+6 | 216 | 114% | 49.6% |
  | 25+10 | 420 | 102% | 37.1% |
  | 50+25 | 900 | 93% | 29.2% |

  **收紧窗口时远距离影响力不降反升**，而正常语音的特征扰动始终巨大——没有一个档位可用。
  延迟无影响（11.03 → 11.19 ms / 10s 输入），所以不是性能问题，是机制本身不成立。
  封死结论：**靠局部窗口约束 HuBERT 上下文这条路走不通，别再试**。要动注意力只剩温度。
- **CPU 回退**、**`git add -A`**、**`git mv`**：见 §8 / §6

## 10. 咬字漂移（objective 3）—— 证据链与现状

用户症状：连续"啊啊啊"/"凤凰凤凰"约 10s 后段变怪，怀疑 HuBERT 非因果 + 滑窗自我相似。

**已确证的事实**

1. **不存在渐进漂移 bug**：把输入造成严格周期且块对齐的信号，第 k 块与第 k−P 块滚动窗口
   逐位相同，特征/合成/输出收敛到**逐位相同**稳态（feat cos → 1.000000），前后半段无趋势
2. **窗口相位依赖是真的**：同一段 20ms 音频被滑窗放在不同位置看到时，HuBERT 特征余弦
   只有 0.52（extra 5.0）。用户"特征收敛到另一个吸引域"的直觉在特征层成立
3. **窗口越长相位依赖越强**，加真实未来上下文能显著改善。实测（xd80+ay20 / hubert base /
   rmvpe / 40k，凤凰.wav 平铺）。表中 lookahead 列指"窗口右端多留 N 个完整输入块的真实
   未来音频"这个**实验**——机制有效但没进生产，原因见第 4 条：

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

4. **前瞻已整体移除（代码层面不存在）**。曾经实现过"窗口右端留 N 个完整输入块的真实
   未来音频、输出整体延后 N 块、离线裁掉 `pad + lookahead`"，耗时只 +0–1ms/块，代价是
   +N 块算法延迟（100ms 块下 = +100ms）。用户实机听感结论：**咬字没有可辨变化，100ms
   延迟完全不值得**。相位余弦 0.52→0.80 是真的，但买不到听感，所以连同
   `lookahead_samples` 状态字段、延迟遥测、离线裁切、`RVC_HUBERT_LOOKAHEAD_BLOCKS`
   环境变量一起删干净了。**别再实现"用未来音频/补零改善咬字"这条路**，两种做法
   （补零见 §9、真前瞻见本条）都已被用户实听否决
5. 机制层指标变好 ≠ 症状变好。判 objective 3 的成败只能看"同一段内容反复出现时字会不会
   变"，而这项 ASR 测试明确显示前瞻不改变漂移时刻（0/300/1000ms 都在约 8s 起漂，只是
   漂到不同的字）。以后拿数值指标宣称修好了咬字，必须先说明它和听感判据的对应关系
6. 残余症状更可能是：(a) 窗口在 `extra_time ≈ 5s` 填满那一刻特征统计突变（之前是短有效
   窗口，之后是 5s 长窗）；(b) 持续发声时嗓音自身退化成气泡音（HNR 从 24dB 跌到 −1.7dB，
   已复现），特征落入训练分布外模糊区，流水线把它放大。两者都指向"让特征更稳"，不是找累积 bug
7. 独立发现（与时间无关）：输入 F0 > 570Hz（音域映射顶 src640→dst720）输出也会崩

**其他已排除的嫌疑**（别再查）：16k 缓冲接缝（48k/44.1k/40k/16k 整除率严格对齐）、
SOLA 打分（对严格连续信号 62/62 块选 offset=0、score 0.91）、拼接与交叉淡化
（边界前后 50ms 谱差 7.08dB vs 块内部 7.29dB = 0.97x，不比内部更突变）、
rms_mix 增益（只在衰减，中位 0.16x）、设备率重采样链、`_stage_preprocess` 实时错位
（帧数不匹配会当场抛异常，不可能静默渐进错位）

**剩余路径（2026-10-27 已全部封死，别再提）**

- **换编码器重训**：用户明确"不能重新，我现在就是哪个80+20的基于base的效果"。而且这条路
  本身也不成立——现有 .pth 里**没有 `enc_q`**（v2 导出时丢掉，续训没有 KL 项可用）、
  **没有 D**（对抗损失只能从 `f0D40k.pth` 重新起步，前几千步是噪声梯度）、`xd80+ay20`
  是 80/20 **权重混合**（插值点不是任何训练的收敛点，开训即被拉向某个父模型），而漂移在
  HuBERT 侧，下游重新加权无法逆信息损失。音色又联合编码在 `enc_p`+`dec`+`emb_g` 里，
  不存在"只训咬字不动音色"的损失函数。**"重训 + 保证音色不变"在结构上互斥**；
  唯一能字面保证音色的做法是零权重变更
- **局部注意力窗口**：已实测不成立（单层精确但模型级感受野不受约束，收紧反而更差），见 §9
- **真前瞻 / 右端补零**：用户实听否决，见本条第 4 与 §9
- **特征下游补救（protect / unsharp / 时间反锐化）**：用户实听否决，见 §9
- 只剩**注意力温度**未试。它同样是弥散干预，按本条第 5 条的标准，在没有听感判据之前不该盲做

**特征器变体曾选错（已修，2026-10-27）**：`assets/configs/save_state.json` 里没有 `hubert` 键，
于是全链路落到默认 `HUBERT_DEFAULT = "chinese"`；而这几个 .pth 都是从官方 v2 预训练
（`pretrained/f0G40k.pth`）微调来的，官方预训练用的是 **base** 特征器。用户已确认
"这批 .pth 都是基于 base 的"。特征器选错的表现正是"音色发闷、咬字怪"，且**全程无告警**
（两套特征器同为 768 维，形状校验查不出来）。

修法：`HUBERT_DEFAULT` 改为 `"base"`，`HUBERT_VARIANTS` 提到 `rvc.core.config` 作单一事实源
（torch-free，GUI 状态层与训练向导不再为拿白名单而导入 torch）；`hubert_path` /
`load_hubert` / `ModelSessions.load` / `InferencePipeline.__init__` / `VoiceEngine.load_model` /
`steps.step_feature` 的签名缺省全部引用常量；`params_from_dict` 对白名单外的值（缺键/空串/
手改错值）回退到 `HUBERT_DEFAULT`，避免打不开软件。`tests/test_hubert_variant.py` 7 条锁死。
`chinese` 仍是合法可选变体，只不再是缺省。

旁证：`assets/hubert/base/pytorch_model.bin` 有 `final_proj.weight/bias`（213 键），
`chinese` 没有（211 键）——前者正是 RVC 标准 `hubert_base.pt` 的签名。另外 `final_proj`
本身是死代码：`rvc/pipeline/features.py` 只取 `last_hidden_state`（feature_projection 经
encoder 的输出），`final_proj` 只为给 base 权重里的键找落点，不参与计算。

**重要更正（用户当场指出，2026-10-27）**：上面这段曾把"缺省从 chinese 改为 base"
写成"修好了用户咬字发闷的根因"——**这是错的**。`assets/configs/save_state.json` 里
`gui.hubert` 本来就是 `"base"`，该值在 `HUBERT_VARIANTS` 白名单内，`params_from_dict`
会直接用存档值，**用户的运行时路径从头到尾没经过 chinese 缺省**。所以：

- `9648f61` 只影响"存档里缺这个键 / 值非法 / 全新安装"的场合，属于**防御性纠错**
- 它**不是**用户"咬字怪/发假"的成因，不能拿它解释用户的实际听感
- 教训：改缺省值 ≠ 改用户路径。判断"某修复是否影响用户"必须先读用户存档/配置的
  实际取值，不能只看到代码缺省就推断生效。这次是先看到 `HUBERT_DEFAULT="chinese"`
  就下结论，跳过了核对存档这一步

（`final_proj` 是死代码、v2 用第 12 层 `last_hidden_state` 这两条结论仍然成立，与上面
的更正无关。）

### 10.1 固定噪声场随窗口滚动（已试 → 用户裁定无效 → 已整体删除）

`state.noise_z` 是 z_p 采样用的"固定噪声场"，按**缓冲区索引**存放，形状 = m_p
（100fps）。而 `input_wav_work` / `input_wav_16k` 每块左移 `block_samples`，
`noise_z` 原来**从不滚动**——同一段绝对音频在相邻块落到不同索引、取到不同噪声。

实测（真实 base 变体、0.25s 块、凤凰.wav、关图取中间量）：

| 量 | 值 |
|---|---|
| 噪声项 / m_p 范数比 | 2.83（剔除尖峰通道后 **1.11**，见下方更正） |
| 仅"噪声错位"造成的 z_p 相对差 | **112%** |
| 换一份独立噪声时 z_p 相对差（上界） | 127% |

即固定噪声场在滑动窗口下几乎没起作用，等价于逐块 `randn_like`。于是实现了
`_roll_noise_field()`：与输入缓冲区同步左移 `block_samples // samples_per_centis`
帧、尾部补新标准正态（经预分配中转，因源/目标显存重叠、GPU `copy_` 无 memmove 语义）。

实现本身验证通过：相邻块逐对 `noise[N+1][:L-adv] == noise[N][adv:]` **80/80 逐位一致**；
图开启（生产默认）下滚动 81 次、输出 11.51s / rms 0.02612 / 无 NaN；开销 58.79 µs/块
= 实时预算的 0.386%。A/B（同音频同模型、固定种子）：RMS 0.02614 vs 0.02610、
块 RMS 差中位 +0.020 dB、对数梅尔帧余弦中位 **0.9947**、2kHz+ 占比 +2.02%。

**用户实听裁定：没用。** 按"没用就删"已整体回退（代码 + 测试），只保留本记录。

教训（比这条改动本身重要）：
- **"实现不正确"与"能改善听感"是两件事**。这里确实是一处真实的不一致（输入缓冲
  滚动、噪声场不滚动），修完在数值层面完全成立，但对听感无影响。以后遇到同类
  "实现与自我描述不符"的问题，可以修，但不要预期它会改变用户听到的东西
- 噪声项占比要用**稳健口径**：整体范数比 2.83 是被通道 86 的极少数尖峰拉高的，
  剔除后只有 1.11（逐元素比值中位 1.32）。别再引用 2.5 倍这类说法
- **通道 86**：`exp(logs_p)` 的尖峰全部落在通道 86，三个独立模型一致
  （xd80+ay20 / xiaodan0319xin / Ayuan 2-1 均占噪声总能量 ~84%、信号 ~2%），
  4536 帧里每帧恰好 1 个尖峰且全在该通道。但运行时把该通道 `logs_p` 压到 0 后，
  RMS 差 −0.001 dB、梅尔余弦 **0.9975**、波形相关 −0.006 → **对输出无实质影响，
  不要再动**。latent 里 SNR 低 ≠ 信息丢失（flow 可逆，且该方差本就是模型应有的）

## 11. 真实感（objective 4，已实现→已整体删除）

**结论先行：这条链做不出来，用户裁定"啥也不是"，已于 2026-10-27 整条删除。不要重做。**

删掉的东西（都在 git 历史里，`82ea914` 是最后一版）：`rvc/dsp/realism.py`（`RealismChain`）、
`rvc/core/config.py` 的 `RealismParams` / `REALISM_MIX` / `PARAM_GROUPS` 里的 `realism`、
`runner.py` stage_output 的接入点、GUI 参数页滑条与 `realism.strength` 绑定、
`tests/test_realism.py` 与两处相关断言。`tests/test_gui_smoke.py` 留了防回退断言
（两页都不该再有 `realism_slider`）。用户存档 JSON 里从来没有 `realism` 字段，无迁移问题。

**为什么失败（这才是要记住的）**：四级（residual 底噪 / breath 气息 / rumble 近讲低频 /
distance 远近起伏）全部走"从参考输入搬统计量"的路子，思路本身没错——
不凭空合成噪声、只搬输入自己的痕迹。内部指标也确实一路变好：注入量与原声余弦 0.75 → 0.02、
安静段漏音 −1.4dB → −53.8dB、底噪估计 3.56× → 0.87×、尾音超额注入 +5.6dB → −0.4dB。
**但用户听完的判词是"啥也不是"。** 这印证了 §9 早就写下的那条：机制层指标变好 ≠ 症状变好。

根本障碍是**我无法评估听感**：没有 ASR、没有耳朵，只能量 SNR / 频谱统计 / 余弦，
而这些量对"像不像真人录音"没有预测力。在这个前提下继续调参就是瞎猜，
用户的原话是"不要为了完成任务而完成任务"。**所以任何"听感类"功能（噪声、气息、
距离感、空间感）都不该再由我盲做**——§9 已否决过气息感、protect、unsharp，这是第三次。

**两条真实缺陷仍值得记住**（将来若有人真要做，别再踩）：

1. **电平换算两侧必须同一条跟踪曲线。** 原实现 `lv = clamp(_out_lvl / ref_rms, max=2.0)`，
   输出侧慢放（`_LEVEL_DECAY=0.85`）而输入侧取当前块瞬时 RMS。尾音里输入塌到底噪、
   输出侧还停在高位，比值被顶到上限 → 注入比应有多 6dB，噪声盖过人声。
   这正是 §9 记的 v10 形状「floor 不随门控退让」，换了个马甲又出现一次。
   修法：两侧同用 `_LEVEL_DECAY` 跟踪，比值稳定在浊音段测出的真实换算关系上。

2. **频率方向中值滤波估底噪之前必须加分析窗。** 直接对未加窗的块做 `rfft`，矩形窗 sinc
   旁瓣只按 1/f 衰减，谐波之间的"谷底"被泄漏抬到真实底噪之上，中值滤出来的是**谐波包络**
   而不是底噪。实测 12 次谐波浊音块上底噪高估 **3.56×（+11dB）**，重建出的噪声床带着
   谐波形状（随机相位）= 一份"耳语副本"，听感发虚、像双重声。加 Hann 窗 + 除掉窗 RMS 增益
   后落到 0.87×。（`_FLOOR_WIN=41` 还要求窗宽盖住 ≥2 个谐波周期：40kHz/9.8Hz-per-bin 下
   约 400Hz，即 F0 ≤ 200Hz。）

3. **测这一级有两个坑**：注入量必须量 `y - b`（旁路返回原块），直接量 `y` 会把人声本体
   算进去；尾音必须写成"谐波·衰减 + 底噪不变"，若整块乘衰减系数则谐波与底噪比值恒定、
   `noise_like` 一直停在浊音值，判据形同虚设（旧公式也能"通过"）。

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

**最大头是 F0，不是 HuBERT**——早期记的"HuBERT 17.8ms 占七成"是用别的仪器（D2H 同步钩子）
和别的素材测的，不可直接比。优化优先看 F0。

**`postprocess_features` 内部拆分（图启用，5160ms 缓冲 / 514 帧）**：
特征上采样 0.05ms（可忽略）+ F0 后处理 0.56ms + **RMVPE 11.3ms**。也就是说这个阶段
99% 就是 RMVPE 本体。514 帧的 F0 后处理要 0.56ms 本身偏高（十几个 514 元素的小算子，
主要吃的是 kernel launch 的 ~25µs/个），但省下来也只有 0.3ms 量级，不值得动。

**RMVPE 成本随窗口长度的曲线（图启用，这是唯一有效的口径）**：

| 窗口 ms | 帧数 | mel | network | decode | 整条 |
|---|---|---|---|---|---|
| 100 | 11 | 0.28 | 5.88 | 0.32 | 6.17 |
| 300 | 31 | 0.30 | 5.90 | 0.31 | 6.13 |
| 600 | 61 | 0.25 | 6.07 | 0.31 | 6.34 |
| 1000 | 101 | 0.25 | 6.60 | 0.30 | 6.91 |
| 2000 | 201 | 0.26 | 7.69 | 0.32 | 8.32 |
| 5160 | 517 | 0.25 | 11.87 | 0.31 | **12.31** |

读法：**有约 6ms 的固定地板**（图回放 + 算子启动），长度相关部分只有 ~6ms。
所以"缩短 F0 窗口"最多省 5–6ms（5160→1000 只省 5.4ms），而下面的等价性实测证明
这条路不安全 —— 投入产出比不成立，别做。

**RMVPE 尾窗不等价（已否决的方向）**：把 F0 提取的输入截成尾巴（像 FCPE 那样），
在被消费的尾部 16 帧上 `torch.equal` **全部为假**；尾部浊音帧占比 整段 0.125 vs
300ms 尾窗 **0.000**、500/600ms 0.062、800/1000ms 才回到 0.125；末 60 帧最大差
短窗 194–317Hz、800/1000ms 仍有 ~1Hz。**RMVPE 的感受野很大（和 HuBERT 一样是非因果的），
截窗口会改变真正被合成器消费的 F0**。FCPE 能截是因为它感受野小、且当时逐帧验过
（见 `FCPE_TAIL_WINDOW_SEC`）；不能想当然套到 RMVPE 上。

**合成器 7.05ms 已经是裁剪过的，别再想去"只合成尾部"**：`SynthesizerTrnMsNSFsid.infer`
拿到 `skip_head`/`return_length` 后自己就切了——`enc_p` 从 `flow_head = max(skip_head −
FLOW_CONTEXT_COMPENSATION_FRAMES, 0)` 起算、`flow` 只跑 `flow_head` 之后、`dec` 只出
`return_length` 帧、`nsff0` 取 `[:, head:head+length]`。用户配置下 p_len≈514、skip_head≈500，
所以真正进网络的只有 enc_p/flow ~38 帧、dec 16 帧。`FLOW_CONTEXT_COMPENSATION_FRAMES = 24`
是 flow 分支（WaveNet 感受野长）相对 decoder 分支必须多看的上下文帧数，动它会让 `dec_head`
处的隐变量与整段推理不一致。

**所以整条链路里与窗口长度成正比的只有 HuBERT 和 F0**（两者都必须看全窗才能给出正确特征/
音高），合成器只吃尾部。窗口长度 = 用户的 `extra_time`，这是设计约束不是浪费。

**`effects.process_output` 的 2.85ms 绝大部分不是算术，是同步气泡**：脱离流水线单测
（40k/块 4000/buf 2000/search 400/infer 6400）只有 **0.70ms** = `apply_rms_mix` 0.185 +
`apply_sola` 0.482。而 sola 内部两个 conv1d 各 0.084/0.086、argmax 0.008、切片 0.011，
剩下 ~0.29ms 是三处 `isnan().any()` + 两处 `torch.where` 的启动开销。剖析表里多出来的
~2.1ms 来自 `infer[offset:]` 的**整机同步**（它要等 HuBERT/F0/合成器全部排空），
Event 口径会把它算进 process_output。所以别去优化 SOLA 的算术——那里没有 2.85ms。
`apply_rms_mix` 在 `rms_mix >= 1.0` 时直接返回 infer，用户配 0.25 才走实际计算。

**剖析方法（脚本随 .workbuddy 删除，按此重建）**：用 `torch.cuda.Event` 成对包住各阶段
（事件在同 stream 顺序记录，主机端只在最后同步一次，不像 `.cpu().numpy()` 钩子那样把耗时
抬高一大截）。宿主对象映射：`_stage_input`/`_stage_preprocess`/`_stage_formant` → runner；
`extract_features`/`postprocess_features`/`synthesize_audio` → `runner.pipeline`；
`process_output` → `runner.effects`。（`process` → `runner.realism` 已随目标 4 整条删除，
该属性不存在了，照旧笔记写会 AttributeError。）
runner 要自己造：`engine._make_runner(work_sr, 1, block, crossfade, extra, model_sr, reset_buffers=True,
prewarm_formant=False)`（`load_model` 不建 runner；`model_sr` 传 0 会让 torchaudio 的 sinc
核宽算成 0 → ZeroDivisionError）。造 `VoiceEngine` 会顺带构造 `RuntimeDeviceConfig`，
图才会启用 —— 见 §14 的"图未启用"坑。

**其他已测**：`extra_time` 5.0 → 39.0ms/块 vs 0.6 → 24.5ms（早期口径）；
块耗时基线（早期口径）HuBERT 17.8 + F0 后处理 1.2 + 合成器 5.6 + RMS/SOLA 1.0 ≈ 26.5ms；
首块 2041ms 是图捕获一次性开销。

**复测（2026-10-27，base 特征器，用户实际参数 block 0.1 / crossfade 0.05 / extra 5.0，
`xd80+ay20.pth`，RTX 5060 Laptop，fp16）**：沿真实 `process_file` 路径包三个阶段方法累计，
凤凰.wav 11.51s 跑 186 块：

| 阶段 | 每块 | 占比 |
|---|---|---|
| HuBERT（含 F0 入队） | 10.58 ms | 69.5% |
| F0 后处理 + 上采样 | 1.53 ms | 10.0% |
| 合成 | 3.11 ms | 20.4% |
| 合计 | **15.22 ms** | / 100 ms 音频 → **6.57x 余量** |

  → 实时路径**不是计算瓶颈**。6.57 倍余量下，用户感知的延迟来自缓冲设置
  （block + crossfade + 设备缓冲），不是算子速度。既然 HuBERT 占近七成且它的开销由
  「上下文长度 × 双向注意力」决定，在不动上下文长度的前提下**没有安全余量可挖**：
  - 不能跨块缓存上下文特征（双向模型里每个远帧特征都会随新样本变化，§9/§10 已实测封死）
    ——**2026-10-27 补上了量化证据**（此前这条只是断言，引用的是§9 局部窗口实验，
    与缓存不是同一件事）：真实 `base` 变体 + 5.16s 窗口 + 块前移 100ms，
    同一绝对时刻的特征余弦（fp16）：
    - 按距新音频侧分桶：0.896（0.1s）→ 0.964（0.4s）→ 0.971（1.6s）→
      0.975（2.5s）→ 0.964（4.6s）。**不随距离单调改善**，两端都更差
    - 多块累积：1 块 0.969、2 块 0.958、5 块 0.950、10 块 0.906、20 块 0.851，
      相对 L2 始终 0.22–0.26。误差随时间跨度单调恶化，**不存在"足够远就稳定"的安全区**
    - 内建基准：相邻帧（20ms）余弦 0.843、间隔 5 帧 0.141 —— 所以缓存误差虽然
      远小于"用错 100ms 音频"，但它并不随距离衰减，无法构造安全缓存策略
    - **踩坑记录**：第一版比对写成 `A[i] vs B[i+5]`，符号反了（B 起点晚 100ms，
      应比 `B[i-5]`），于是拿 200ms 后的内容在比，得到"处处 1.3 相对变化"的荒谬值。
      自检抓住了它：相邻帧余弦 0.84 却"平移 100ms 只有 0.11"，且与"窗口内 vs 整段"
      的 0.89 自相矛盾。**做特征比对必须先建恒等自检（必须恰为 0）和帧间基准**，
      否则单看一个数无法判断对错
  - 顺带确认：v2 模型用 `last_hidden_state`（第 12 层）**是正确的**。上游
    `infer/hubert.py:71-73` 注释写明"RVC v1 uses layer 9 followed by final_proj;
    RVC v2 uses the final (12th) encoder layer directly"，所以 `final_proj` 对 v2
    确实是死代码，本项目 `hubert_forward` 与上游 v2 路径一致。层选择不是咬字成因
  - fp16、CUDA Graph、pin_memory、non_blocking 全部已用
  - F0 与 HuBERT 真并发会让两个 CUDA Graph 跨流回放，实测会挂死（见 pipeline.py 注释）
  结论：**目标 2 在代码侧维持封死**。要更快只能减 `extra_time`，那是听感换延迟的
  用户取舍，不是本项目该替他决定的事。

**切换模型不泄漏显存（2026-10-27 实测）**：合成器缓存容量 2，HuBERT 2 / RMVPE 1 /
FCPE 1。`LRUCache.insert` 把被淘汰的值作为返回值交给调用方作为**回收钩子**
（其注释明确：值可能持有 CUDA Graph 资源，必须在被遗忘那一刻显式回收，不能指望
GC 时机）。实测 21 次连续切换（3 个模型轮转，每次 `gc.collect()` 后采样）：

| 次数 | allocated | reserved |
|---|---|---|
| 1（首个模型） | 240.4 MB | 306.0 MB |
| 2（缓存已满） | 294.3 MB | 346.0 MB |
| 21（轮转 7 圈后） | 294.0 MB | 382.0 MB |

- allocated 稳在 ~294 MB（= HuBERT fp16 189MB + 2 个合成器各 ~54MB），
  漂移 −0.2 MB → **无泄漏，回收钩子生效**
- reserved 在 346–422 MB 间来回波动，20 次里仅 10 次上升，**非单调** → 是
  PyTorch 分配器复用空闲块（空闲池 51–127 MB 有界），不是碎片累积
- 结论：**别去加"切换模型后 empty_cache"之类的清理**，`engine.py:86-88` 已明确
  记录"不做周期性 empty_cache：稳态下分配模式固定、缓存池不再增长"

**设备层已调**：AudioStreams 用 `latency="low"` + 显式 `blocksize`。WASAPI 共享模式下
设备缓冲只有 10–30ms（实测 1056 帧 @48k），**远小于块长**——任何"整块塞得下才写"的判据
在 Windows 上永远不成立，必须用待写队列（`pending` + `pending_pos`，上界 3 块，超出丢最旧）。
这条来自已删除的 C++ 实现，但结论对 Python 侧同样成立。

**VB-CABLE 真实往返实测（2026-10-27，一次否掉我自己推断的实验）**：主/副输出流都已显式
`latency="low"`（`stream.py` 注释记录了"不传会多约 180ms"的实测），且
`blocksize = state.block_samples`（engine.py:204），即回调块长 == 业务推理块长。
我据此推断"100ms 块长会把端口缓冲底座抬到 200ms，解耦回调与推理就能省下来"，
**实测证明这个推断是错的**：

| 块长 | 实测真实往返 | PortAudio 报告合计 |
|---|---|---|
| 100ms | **58.1ms** | 200.0ms |
| 250ms | **65.4ms** | 500.0ms |

方法：向 CABLE Input 播周期脉冲、从对端 CABLE Output 录回、互相关求实际偏移。
结论：`st.latency` 报的是**理论缓冲环大小**，不是实际听到的延迟；VB-CABLE 是纯软件
内存传输、没有物理硬件节拍，实际缓冲极浅，且几乎不随 `blocksize` 变化（100→250ms
只多 7ms）。**所以设备端没有可压缩的 200ms，不要去做"回调与推理解耦"那类改造**——
收益在噪声里，代价是把热路径改成生产者/消费者。真实延迟底座来自"每次回调要攒满
一个块长"这个业务设计本身，那是 `block_time` 的取舍，不属于设备缓冲优化空间。

## 13. 训练模块（objective 6）

- **特征器变体有两处独立存档，改代码默认值不会覆盖它们**（2026-10-27 踩到）：
  `assets/configs/train_cli_defaults.json` 是训练命令行向导的「上次答案」，向导按它预填
  默认值，回车即采用——当时里面还写着 `"hubert": "chinese"`，等于把错特征器又带回来；
  `assets/configs/save_state.json` 的 `train.hubert` 同样是 `chinese`。两个文件都**不在版本库**
  （gitignore），所以改 `HUBERT_DEFAULT` 之后必须手工同步一遍，两处都已改为 `base`。
  注意 `save_state.json` 的顶层是 `{gui, models, train}`，**当前代码只读写 `gui` 一个块**
  （`gui/view/lifecycle.py` 的 `params_from_dict(config.get("gui", {}))`），
  `models` 与 `train` 是旧架构遗留、无人读取的死状态——改它们只是防止将来被人接上时踩坑
- DataLoader 选项由 `rvc/train/trainer.py::training_dataloader_options` 统一产出：
  `RVC_TRAIN_WORKERS`（默认 2，设 0 关闭）、`RVC_TRAIN_CACHE_MB`（进程总预算，按 worker 均分，
  0 关闭样本内存缓存）、`RVC_TRAIN_PREFETCH`（默认 1）。`persistent_workers` 避免每 epoch
  重新 spawn；`cleanup()` 显式 `iterator._shutdown_workers()`，否则 Windows 上进程被 worker 持有
- **worker > 0 时必须显式 `multiprocessing_context="spawn"`**（已写进 `training_dataloader_options`）：
  `setup()` 先建模型搬上 CUDA、再建 DataLoader，Linux 默认 fork，worker 会继承父进程
  半初始化的 CUDA 上下文 → 训练卡死或 "Cannot re-initialize CUDA in forked subprocess"。
  **Windows 只有 spawn，所以这个坑在 Windows 上永远看不出来**，只在 Linux 复现。两个平台
  统一走 spawn（也就是 Windows 早就验证过的那条路径），行为一致。锁在
  `tests/test_train_dataset.py::test_workers_always_use_spawn`；`RVC_TRAIN_WORKERS=0` 时
  不许出现这个键（该分支有字典全等断言）。端到端回归：`$env:TEMP\spawn_loader_probe.py`
  （2 个 worker 真的拉起、6 批、`phone(2,298,768)`）
- **GradScaler 初始倍率必须放低（已修，2026-10-27）**：torch 默认 `init_scale=65536`，
  而本模型 `loss_mel` 量级约 1e2，梯度偏大，前若干 batch 恒溢出、每批折半。
  实测（batch_size=2、每 epoch 6 batch、加载 `f0G40k.pth` 底模、12 片合成素材）：

  | | 旧 65536 | 新 256 |
  |---|---|---|
  | D 首次真正 `optimizer.step()` | batch 3 | batch 0 |
  | G 首次真正 `optimizer.step()` | **batch 8** | **batch 1** |
  | 稳态倍率 | 128 | 128 |
  | torch 的 `lr_scheduler.step() 早于 optimizer.step()` 警告 | 1 条 | 0 条 |

  稳态倍率两边都是 128 → **起点放低不改变数值行为，只省掉折半过程**。旧值下
  第 1 个 epoch 里 G 的 `optimizer.state` 到 epoch 结束仍是空的，也就是生成器
  整个 epoch 一次参数都没更新（这条是确凿证据，不是估计）。修后同一份素材
  跑 2 epoch：Mel 113.5→**64.1**（旧 94.6）、D 8.9→3.4（旧 4.1）。
  常量 `TRAIN_GRAD_SCALER_INIT_SCALE_DEFAULT = 2**8` 在 `rvc/core/constants.py`，
  可经 `RVC_TRAIN_INIT_SCALE` 覆盖；读取函数 `trainer.grad_scaler_init_scale()`
  保证 ≥1（0 会让整个 AMP 失效）。契约锁在
  `tests/test_train_pipeline_contracts.py::TestGradScalerInitScale`
- **那条 LR 警告的成因已查透，别再重新怀疑顺序写错**：torch 的触发条件是
  `lr_scheduler.py:255` 首次 `step()` 时 `_step_count==1`，第一分支被
  `patch_track_step_called` 装的 `_wrapped_by_lr_sched` 挡掉，落到
  `elif not getattr(optimizer, "_opt_called", False)` —— 也就是**只有当该优化器
  一次 `optimizer.step()` 都没真执行过才警告**。`_opt_called` 由 `scaler.step()`
  在未跳过时才设置；`GradScaler` 跳过则永不设置。所以警告 ≠ 调用顺序写错，
  而是「这轮 AMP 把所有 step 都跳过了」。trainer 的调用顺序本来就是
  `optimizer.step()` 在前、`scheduler.step()` 在 epoch 末，无需调整
- **`last_epoch = start_epoch - 2` 是对的，别当 off-by-one 改**：`ExponentialLR.get_lr()`
  是纯乘性（在当前 lr 上再乘 gamma），看着像续训会多衰减一次；实测（2026-10-27）
  全新训练每个 epoch 精确等于 `gamma^(N-1)`，第 5 epoch 末存盘后续训，epoch 6 仍
  恰为 `gamma^5`，偏差 0.000000%。原因是 `last_epoch >= 0` 时构造走 closed-form
  路径（`initial_lr * gamma^last_epoch` 重算），不做累乘；`initial_lr` 由
  `LRScheduler.__init__` 在 `last_epoch == -1` 时 `setdefault` 进 param_groups，
  并随 optimizer state_dict 一起落盘。契约锁在
  `tests/test_train_pipeline_contracts.py::TestLrScheduleAcrossResume`
- 训练侧特征器变体的接线已确认闭合：`step_feature` 写 `3_feature768/.variant`，
  `run.py` 先判 `_features_ready(exp_dir, hubert)`，为 False 且目录存在时
  **整目录 rmtree 重提**并打出 WARN——不会静默混用 base/chinese 的旧特征。
  探针 `train_e2e.py` 因为绕过了 CLI 才需要自己手工写这个标记
- **离散 pitch 的 bin 划分与上游逐值一致（已核对源码 + 测试钉死，别再"修"）**：
  网上流传上游用硬编码 `f0_mel_min=112.0 / f0_mel_max=1975.0`，我一度也信了并准备
  按"bug"去改。实际上游源码（本地
  `E:\Projects\Git\Retrieval-based-Voice-Conversion-WebUI\train\dataset\extract_f0.py:61-65`
  与 `infer\rtrvc.py:79-82,148-159`）写的是
  `f0_mel_min = 1127*ln(1+50/700)`、`f0_mel_max = 1127*ln(1+1100/700)`，与本项目
  `rvc/dsp/mel.py` 的公式完全一致；`(bins-2)/(max-min)+1`、钳制 `[1,255]`、
  UV 帧落 1 也都等价。对照测试
  `tests/test_pitch_map.py::TestCoarseF0MatchesUpstream` 用独立重写的上游公式
  跑 1200 个频点逐值比对。
  为什么值得钉死：模型是从上游 `f0G40k.pth` 微调的，bin 划分错位**不会**让音高
  明显跑调（输出音高由连续 `nsff0` 决定），只会让 `enc_p` 收到训练时没见过的
  音高条件，表现为咬字/音色发闷——极难归因，正是最容易"顺手改坏"的地方
- **固定噪声场（`state.noise_z`）已实测确认生效（2026-10-27）**：`_z_sample` 的形状
  校验是**静默退回** `randn_like`——差一帧就退回随机，不报错不告警，固定噪声场等于
  没接，表现为持续/重复内容的咬字逐块随机漂移。代码注释说这特性「从上线起就一直
  没生效」（长度曾按 `skip_head + return_length` 估，与真实 m_p 恒差一帧），但修完
  后从没有测试证明过：`tests/test_noise_z_shape.py` 原先只测 `hubert_frame_count`
  对**假** conv 配置的帧数。实测（真实 `xd80+ay20.pth` + 凤凰.wav，走离线
  `process_file`，与实时共用同一 `runner.process_block`）：4/4 块全部命中，
  `m_p=(1,192,56)` == `noise_z=(1,192,56)`，且 `_z_sample` 返回的就是传入张量本身。
  现在补了两类测试：`TestZSampleContract`（命中必须返回同一对象、不匹配必须退回）
  与 `TestRealHubertConfigFrames`（用真实 `assets/hubert/base/config.json` 校验
  `2*(frames+1)` == `upsample_features` 实际输出长度、且 runner 的 `length` 公式
  == `p_len - flow_head`）
- Linux 侧已确认无需改动的几处：`CREATE_NO_WINDOW` 在非 Windows 恒为 0；`locate_ffmpeg()`
  第三级（`assets/ffmpeg`）**有意限定 Windows**（只随包提供 `ffmpeg.exe`），Linux 走
  `RVC_FFMPEG`/系统 PATH；`_system_memory_gb()` 有 `os.sysconf` 分支；设备枚举走
  `sd.query_hostapis()` 而非硬编码 WASAPI，本身跨平台。GUI 的 `recycle_file` 非 Windows 抛
  `NotImplementedError`（仅 `offline_controller` 用），不在训练模块范围内
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
- **训练链路在 2026-10 才第一次被真正端到端跑通**，一次暴露四处必崩缺陷（已修，`c061d60`）：
  - `HuBERTExtractor` 调了 `LRUCache.set`，而真实方法名是 `insert`（返回被淘汰值，作回收钩子）
  - `checkpoint.py` 用 `os.replace` / `os.path.join` 却没 `import os`：setup 找 checkpoint
    和每次保存两条路径都崩
  - collate 给 `TrainBatch` 多传了第 9 个字段 `wav_lengths`，全仓无人消费 → 删死代码，
    不要加宽 NamedTuple（`pin_memory` 的钉页只对 namedtuple 生效）
  - `SineGen.forward` 无条件 `torch.inference_mode()`，而它的输出要喂给可训练的
    `SourceModuleHnNSF.l_linear`：fp32 直接抛 `Inference tensors cannot be saved for
    backward`，fp16 被 autocast 掩盖成静默断链。改为 `not torch.is_grad_enabled()` 时才进
    （推理侧调用方本就在 `inference_mode` 里，见 `rvc/streaming/runner.py:188/223/287`，
    行为与速度不变）
- **`Detected call of lr_scheduler.step() before optimizer.step()` 不是 scheduler 用法错**：
  `GradScaler` 发现 inf 就跳过 `optimizer.step()`，而 `_opt_called` 只在
  `lr_scheduler.py:165` 的包装器里置位，于是首个 `scheduler.step()` 报警。inf 本身是 fp16
  动态标定的正常预热：默认 `init_scale=65536` 对 fp16 上限 65504 没有余量，本模型梯度量级
  O(1)~O(10²)，实测每 batch 折半一次（65536→512），**第 8 个 batch 起 `opt_called=True`、
  scale 稳定在 256、loss 243→119**，自行收敛，不是 bug。代价是小数据集（RVC 典型 5~15 分钟
  素材 → 每 epoch 只有几十个 batch）会白烧掉首 epoch 的前几个 batch
- 验训练要跑 `$env:TEMP\train_e2e.py`（合成 4 个 6.5s 共振峰音频 → 预处理 → F0 → HuBERT →
  filelist → 2 epoch → 导出），单测发现不了上面四处：187 项全过时它们全都还在。
  逐 batch 观察 scaler 用 `$env:TEMP\grad_trace.py 1|0 [批数]`
- **续训链路已验证**（`$env:TEMP\train_resume_probe.py`）：有 `G_2/D_2` 时 `setup()` 认出
  `start_epoch=3`，权重逐张量等于 checkpoint（不是随机初始化），scheduler 按
  `last_epoch=start_epoch-2` 续接（lr 已衰减到 0.00009998），只跑 epoch 3，
  `keep_ckpts=1` 正确淘汰到只剩 `G_3/D_3`。**GradScaler 状态现在会进 checkpoint**：
  `save_checkpoint`/`save_checkpoint_pair`/`load_checkpoint` 都有可选 `scaler` 参数，
  `setup()` 里 scaler 必须在恢复逻辑**之前**创建（原来在后，存了也接不上）。
  续训不再重新预热，省掉那 7 个白烧的 batch。旧 checkpoint 没有 `scaler` 键，
  用 `checkpoint.get("scaler")` 读，缺省就保持默认 65536，不会 KeyError。
  验证：`$env:TEMP\scaler_wiring_probe.py`（起训 65536 → 续训 256 → 旧格式 65536）
- **测 `GradScaler` 必须用 `enabled=True`**：禁用态 `state_dict()` 恒为 `{}`、
  `get_scale()` 恒为 1.0，设 `_scale` 也不生效，拿它测往返等于什么都没测。
  只有 `enabled=True` 才有 `{scale, growth_factor, backoff_factor, growth_interval,
  _growth_tracker}`，且 `_scale` 是 cuda 张量——单测要加 `skipUnless(torch.cuda.is_available())`
- checkpoint 目录名是 `4_checkpoints`，取路径用 `checkpoints_dir()`，别手拼字符串
  （e2e 脚本曾按 `exp/checkpoints` 数文件，于是"checkpoint 0 个"是假警报）
- **weight_norm 有两套键名，但加载侧不需要改写**（`rvc/models/state_dict_keys.py`）：
  上游模型与官方预训练资产用旧版 `torch.nn.utils.weight_norm` 命名
  （`<layer>.weight_g`/`weight_v`），本项目模块用 `parametrizations.weight_norm`
  （`<layer>.parametrizations.weight.original0/1`）。**`ParametrizationList` 自带
  旧键名兼容层**：直接 `load_state_dict({"weight_g":..., "weight_v":...}, strict=False)`
  得到 `missing=[] unexpected=[]`，`original0/1` 与物化 `weight` 全部正确落地
  （最小复现 `$env:TEMP\shim_probe.py`，回归锁在 `tests/test_state_dict_keys.py`
  的 `TestLegacyKeysAcceptedOnLoad`）。**别手写加载侧键名改写，也别据此判断"权重没加载"**
  - 真正需要改写的是**两份字典互相比对的场合**——那里没有 torch 兼容层：
    `merge_models` 按裸键名取交集加权平均，一个旧命名文件 + 一个本流程导出的文件
    会让同一层变成两个键，权重不参与平均，结果还同时带两套键名。
    现在两个输入先 `to_parametrized()`、输出 `to_legacy()` 落盘
  - `export_model` 落盘统一用旧版命名，与本仓库已有模型文件、上游工具一致
- **踩过的坑（代价很大，务必记住）：`state_dict()` 的键名比对 ≠ 加载结果**。
  曾用"模型 state_dict 里有 208 个 `parametrizations.*` 键不在文件里"推断
  "用户模型解码器全是随机初始化、音质一直错"，并把这条错误结论写进本文件、
  提交说明和给用户的汇报。实际上 torch 的兼容层早把 `weight_g/v` 认了。
  两点教训：
  - 判断"权重是否落地"要比**值**（`torch.equal` 逐张量），或直接看
    `load_state_dict` 返回的 `missing_keys`/`unexpected_keys`，不要比键名
  - 只要涉及"输出是否改变"，就做 A/B 音频比对兜底。这次正是 A/B 测出
    修复前后**输出完全相同**（相关性 +1.0000、差异 −93.8 dB）才翻案；
    若只停在权重层面就会一直错下去

### 13.1 训练是 GPU-bound，数据管道参数不是杠杆（2026-10-27 实测）

首次实测「等取数」与「算梯度」的时间比（此前 workers=2 / prefetch=1 / 持久 worker
都只是按经验设的）。用 `Temp\rvc_pretrain`（12 切片，特征/F0/spec 缓存就绪）、
batch=2、真实 `Trainer.setup()`：

| 项 | 每 batch |
|---|---|
| 纯取数（workers=2，热态） | **20.1 ms** |
| 完整训练步（H2D + 前反向 + 优化器） | **11 000–28 800 ms** |

取数占 0.07%–0.2%。**训练是压倒性 GPU-bound**，结论：
- 调 `num_workers` / `prefetch_factor` / 样本缓存**换不来速度**，别再往这个方向想
- 要更快只能动 batch / 精度 / segment 长度 / 模型本身（都会影响训练质量，不属于
  「优化」范畴）

worker 数真正影响的是**内存**（同一探针，峰值 RSS 含子进程树）：

| RVC_TRAIN_WORKERS | 峰值 RSS | 取数（热） |
|---|---|---|
| 2（默认） | **2009 MB** | 20.1 ms |
| 1 | **1498 MB** | 18.8 ms |
| 0 | **960 MB** | 24.0 ms |

Windows 上 spawn worker 会重新 import 全部模块，每个 worker 数百 MB。
`workers=0` 已验证可正常训练（loss 全部有限且下行：mel 98.5→72.2、KL 427→183，
checkpoint `D_1.pth`/`G_1.pth` 正常落盘）。

**但没有改默认值**，理由：
- 本测量只用了 12 个切片的玩具集、batch=2、步长 11–29s。真实数据集步长可能降到
  0.3–1s，那时 24ms 的无重叠取数就变成 2%–8%，`workers=0` 会真的慢一点
- `workers=1` 在本次测量里两个轴都优于 2（省 511MB 且 18.8ms 略快），是备选，
  但同样缺乏真实规模数据支撑
- `RVC_TRAIN_WORKERS` 环境变量已存在，需要省内存的用户可以自己设；按玩具集调
  默认值属于过拟合
- 结论记在这里，避免以后有人（包括我）再去"优化" DataLoader

## 14. 测量方法论与踩坑（做实验前必读）

**可用的判据**
- **周期输入 ⇒ 块等价**：把输入造成严格周期且周期 = 整数个块，第 k 块与第 k−P 块滚动窗口
  逐位相同，则特征/合成/输出都该逐位相同。用来抓"时间累积型 bug"，最灵敏
- **相位散度**：周期素材下按"绝对帧号 mod 周期"分桶，同桶 = 同内容被不同窗口位置看到，
  桶内两两余弦 = 窗口相位依赖强度
- **重叠一致性**：相邻两块对同一段 60ms 内容各渲染一次，比较两次渲染
- **扰动放大率**：同一输入 + −60dB 白噪，比较输出变化
- **注入类功能必须测「与原声的相关性」**：`injected = process(out, ref) - out` 是精确注入量
  （加性链路），把它和原声做余弦、再做投影，就知道有没有把本音漏进输出。**只看电平不够**：
  真实感链早期 residual 注入 −19.5dB 看着无害，余弦却是 0.75，等于原声按 −22dB 混进去
- **注入量的相对电平要按「本块」算，不能只看聚合**：输出电平跟踪是慢放的，静音/安静段里
  跟踪值仍停在响段电平上，聚合指标会把问题平均掉（同一份素材聚合 −19.5dB，最坏一块 −1.4dB）
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
- **脱离 `VoiceEngine` 的探针脚本默认跑的是 eager，不是生产路径**：`run_cuda_graph` 在
  `configure_cuda_graph` 从未被调用时静默走 `return function(*inputs)`，而唯一的调用点是
  `RuntimeDeviceConfig._init` ← `VoiceEngine.__init__`。所以单纯 import 模型自己跑的基准，
  数字会比实机**大一倍多**：同一个 RMVPE 整条 5160ms 缓冲，eager 31.7ms vs 图 12.3ms。
  探针里必须显式 `configure_cuda_graph(torch.device('cuda'))` 并断言返回 True，
  否则"哪个阶段最慢"的结论整个作废（这条坑踩过两次）
- **笔记本 GPU 会降频，顺序计时会骗人**：同一基准按固定顺序跑，后测的配置会莫名变慢
  （实测 residual 1.5 → breath 3.8 → rumble 5.6ms 的假趋势）。必须**交叉轮转 + 每轮取最小**，
  否则得出的"哪个阶段最慢"可能是顺序造成的
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
- **HuBERT 权重存盘精度**：项目全程 fp16（推理侧 `rvc/runtime/device.py:32` 把 `is_half`
  硬编码为 `True`，模型总是 `.to(dtype=float16)`）。原始 `base/` 本来就是 fp16 存盘
  （189,206,711 B / 213 张量），`chinese/` 原本是 fp32（377,552,987 B / 211 张量）——
  那份 fp32 是**假的精度**：加载时立刻被 `.to(float16)` 降下来，白白多占 180 MB 磁盘，
  `torch.load` 期间还多占 180 MB 宿主内存。已按用户要求转成 fp16
  （188,813,420 B，省 180.0 MB / 50.0%）。转换经两道关卡实测：211 个张量
  `orig.half()` 逐位相等；端到端真实音频取特征 `(1,399,768)` **逐位相等、最大绝对差
  0.000e+00**；替换后复核一致；`chinese(fp16)` 端到端冒烟通过（94.6M 参数全 fp16、
  输出 11.51s、rms 0.0261、peak 0.2057、无 NaN）。两个变体现在格式统一
- 由此得出的判据：**任何"用 fp32 权重换精度"的想法在项目里都无效**——全链路 fp16，
  fp32 存盘只会在加载瞬间丢掉低 13 位。实测 `is_half=False` 时参数 dtype 是 float32，
  但低 13 位全零占比 1.0000（就是 fp16 值零填充），所以那条路径不存在真实 fp32 精度
- 与训练侧 fp32 回退的关系：`rvc/train/cli/envcheck.py:133-136` 在算力 < 7.0
  （P40/1080Ti）时把 `use_fp16` 置 False。**不要因为这个就删掉它**——合成器权重本来
  是 fp32，`fp16_run=False` 对它确实提供真实 fp32 精度，那条路径不是死代码。只是
  HuBERT 侧现在两个变体都是 fp16 存盘，因此老卡上 HuBERT 特征精度不会再高于 fp16。
  本机算力 12.0，恒走 fp16，不受影响
