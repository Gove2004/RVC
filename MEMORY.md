# RVC 项目长期记忆

**这是本项目唯一的长期记忆文件。** 所有跨会话仍然成立的事实、约定、硬约束、被否决的
方向，都只写进这里——不建按日流水、不建临时笔记目录。

写之前先读，避免重复踩坑。只写「现在为什么这样」和「别改成什么、否则会怎样」，
不写过程叙事（见 §5）。条目被推翻时**就地改写**，不要追加"更正：…"。

---

## 1. 环境与命令

- venv：`.venv/Scripts/python.exe`（Python 3.13.7，torch 2.11.0+cu128，
  transformers 5.9.0，PySide6 6.11.1，numpy/onnxruntime 见 requirements.txt）
- 测试：`.venv/Scripts/python.exe -m unittest discover -s tests`
  **stdlib unittest，不用 pytest，不写 conftest**。当前 241 例，全部无需 GPU 与模型；
  耗时大头是真实 checkpoint 用例
  （`test_state_dict_keys`、`test_train_pipeline_contracts`），纯函数契约部分仍是秒级
- torch 必须单独装：
  `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`。
  仓库 `requirements.txt` 顶部有大段说明，别把 `--index-url` 塞进去（会替换整个索引）
- 临时脚本必须设 `$env:PYTHONPATH='E:\Projects\Python\RVC'`，否则 `import rvc` 失败
- 真实模型冒烟：`assets/models/songyue/xd80+ay20.pth` + `base` 特征器（2026-11-07 用户确认
  为现役配置）；**一切"当前实际用什么"以 `assets/configs/save_state.json` 为准，文档不存值**。
  换特征器变体必须与模型训练时一致，配错**不报错只是变差**（三个变体同为 768 维）。
  `load_model` 冷启大头是 transformers 冷启，热启明显更快（具体秒数不记录）
- **GUI 启动不拉 torch**：`import gui.view.window` 不 import torch（具体秒数不记录，
  环境变过就不要拿旧数字当真）。链路上别往 gui/controller
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
- `formant_factor` 运行中可变：它经 `return_length2` 决定合成器图形状，量程内多个档位
  各持一份私有显存池。图**首张捕获明显贵**（含 cuDNN 冷启）、后续每张仍是百毫秒级，
  远大于回调预算。取舍：只预捕获**当前档位**（省常驻显存），拖滑条到未预捕获的新档位
  会在回调线程现场补捕获一次（该块一次性顿一下），之后常驻——用一次性顿换取显存下降
- 调 formant **不能动 pitchf**：decoder 里 `n_res` 的 interpolate 已把时长拉伸的降频抵掉
- F0 side stream 的图回放**不能与主流图回放并发**（各持私有显存池，PyTorch 要求同流同序）
- 同步点位置有讲究：`.item()` 要放在同步 D2H 之后（队列已排空），GUI 侧一律读缓存
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

## 8. 已知设计取舍（别当 bug 修）

- 无 GPU 直接报中文错误，**不做 CPU 回退**（刻意行为）
- 延迟显示含「驱动层补偿」：PortAudio 拿不到虚拟声卡/系统混音器那段，只能按实测校准
  （见 `engine._DRIVER_LATENCY_COMPENSATION_MS`），换输出设备该数不成立，别当固定常数
- 缓存失效语义只有 LRU 容量淘汰，**无文件 mtime 校验**（同名 pth 被外部覆盖会返回旧权重，已接受）
- F0 每块重提取，RMVPE 走整段、FCPE 只算尾部 0.6s 窗口
- `crossfade_time=0.10` 是 dataclass 字段默认值，不具名化（SOLA 缓冲实际取
  `min(crossfade_samples, 100ms)`，上限 100ms 系用户实听裁定）
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

- **特征索引/检索（index）不做了**（2026-11-07 用户裁定）：protect、特征反锐化、
  窗口右侧未来上下文（补零/真前瞻）、合成噪声床五条死路等围绕它们的实验记录
  一并从本文档移除。以后有人再提检索类功能，直接引这条
- **C++ 重写**：2026-10-26 用户决定放弃，`native/`（331MB）已整体删除且**从未纳入 git**。
  不要再提议或恢复 C++ 移植，也不要去找 `native/`
- **CPU 回退**、**`git add -A`**、**`git mv`**：见 §8 / §6

## 10. 性能

**瓶颈结构（结论，数字随环境过期一律不留）**
- 整块推理里 **HuBERT 占大头（约七成），实时路径不是计算瓶颈**——用户感知延迟来自
  缓冲设置（block + crossfade + 设备缓冲），不是算子速度；在不动上下文长度的前提下
  没有安全余量可挖，**性能方向在代码侧维持封死**
- F0 后处理阶段 ≈ RMVPE 本体 + kernel launch 开销，可挖空间不到 1ms 量级，不值得动
- 合成器成本已是裁剪后的：`SynthesizerTrnMsNSFsid.infer` 自己按 `skip_head`/`return_length`
  切尾，真正进网络的只有几十帧。`FLOW_CONTEXT_COMPENSATION_FRAMES` 是 flow 分支相对
  decoder 必须多看的上下文帧数，动它会让 `dec_head` 处隐变量与整段推理不一致
- RMS/SOLA 的算术量很小；剖析表里它"贵"是 `infer[offset:]` 整机同步的气泡被记进了
  该阶段（Event 口径会把排队等待算进来）——**别去优化 SOLA 的算术**
- `extra_time` 决定窗口长度 ⇒ 与窗口成正比的只有 HuBERT 和 F0（都必须看全窗），
  这是设计约束不是浪费；减 `extra_time` 是听感换延迟的用户取舍，不是项目该替他决定的

**已实测否决的方向（结论级封死）**
- **RMVPE 截尾窗**：RMVPE 感受野大且非因果，截窗口会改变真正被合成器消费的 F0
  （尾部帧 `torch.equal` 全假、浊音占比统计在短窗下失真）。FCPE 能截尾是逐帧验过的
  （`FCPE_TAIL_WINDOW_SEC`），不能想当然套到 RMVPE 上
- **跨块缓存上下文特征**：双向模型里每个远帧特征都会随新样本变化；实测余弦**不随距离
  单调改善**、误差随时间跨度单调恶化——不存在"足够远就稳定"的安全区
- v2 模型用 `last_hidden_state`（第 12 层）**是正确的**（上游注释明确 v1 用第 9 层 +
  final_proj、v2 直接取第 12 层）；`final_proj` 对 v2 是死代码，层选择不是音质成因
- fp16、CUDA Graph、pin_memory、non_blocking 全部已用；F0 与 HuBERT 真并发会让两个
  CUDA Graph 跨流回放，实测会挂死（见 pipeline.py 注释）
- **切换模型不泄漏显存**：`LRUCache` 淘汰时把被淘汰值回交给调用方做显式回收（模型清图、
  提取器 `release()`）；轮转实测 allocated 稳定非单调。**别加"切换后 empty_cache"类
  清理**，`engine.py` 已记录不做周期性 empty_cache 的理由
- **设备层已调**：WASAPI 共享模式设备缓冲远小于块长，"整块塞得下才写"在 Windows
  永远不成立，必须用待写队列（`pending` + `pending_pos`，上界 3 块超出丢最旧）
- **VB-CABLE 真实往返 = 与块长无关的固定管道 + 约 1×块长**（`tools/vb_loopback_latency.py`
  脉冲互相关实测；驱动面板报告的缓冲环深度是理论值，不可信，别拿它推延迟或配对规则）。
  固定那段在代码里压缩不了，**不要做"回调与推理解耦"类改造**；真实延迟底座是
  "每回调攒满一个块长"的业务设计，属 `block_time` 取舍

**剖析方法（重建脚本时的口径）**
- `torch.cuda.Event` 成对包住各阶段、主机端只在最后同步一次（D2H 钩子会把耗时抬高
  一大截，与钩子口径的旧数字不可互比）；阶段宿主映射见 runner/pipeline/effects 源码
- runner 要自己造：`engine._make_runner(work_sr, 1, block, crossfade, extra, model_sr,
  reset_buffers=True, prewarm_formant=False)`（`load_model` 不建 runner；`model_sr`
  传 0 会让 torchaudio sinc 核宽算成 0 → ZeroDivisionError）。造 `VoiceEngine` 才会
  构造 `RuntimeDeviceConfig` 开图——见 §12 的"图未启用"坑
- **特征比对必须先建恒等自检（必须恰为 0）和帧间基准**，否则单看一个数无法判断对错
  （曾因 `A[i] vs B[i+5]` 符号反了写出"处处 1.3 相对变化"的荒谬值）

## 11. 训练模块

**存档与变体（两处独立存档改代码默认值覆盖不掉）**
- `train_cli_defaults.json`（向导「上次答案」预填源）与 `save_state.json` 的 `train.hubert`
  各存一份特征器变体；两个文件都在 gitignore 里。**换机器/改 `HUBERT_DEFAULT` 后必须手工
  同步这两处**，否则向导回车就把旧变体带回来。变体合法性一律以
  `rvc.core.config.HUBERT_VARIANTS` 为白名单（向导/体检/存档清洗三处共用）
- `save_state.json` 顶层 `{gui, models, train}`，**只有 `gui` 块被代码读取**
  （`gui/view/lifecycle.py`）；`models`/`train` 是旧架构遗留死状态
- 特征器变体接线闭合：`step_feature` 写 `3_feature768/.variant` 标记，特征目录变体不匹配
  时**整目录重提并 WARN**，不会静默混用旧特征
- 续跑判据 = 文件存在即跳过 ⇒ **所有中间产物必须原子写**；`atomic_write` 对
  `os.replace` 的 `PermissionError` 退避重试 5 次（`.pth`/`.pt` 常被杀软/索引器短暂占用），
  ENOSPC/EROFS 必须立刻抛。锁在 `tests/test_atomic.py`
- checkpoint 目录名是 `4_checkpoints`，取路径一律 `checkpoints_dir()`，别手拼字符串

**数值契约（都被单测锁死，别再"修"）**
- **GradScaler 初始倍率低置**：torch 默认 65536 对 fp16 上限 65504 没有余量，本模型梯度
  量级下前若干 batch 恒溢出、step 全被跳过——第一个 epoch 等于白跑。现默认
  `TRAIN_GRAD_SCALER_INIT_SCALE_DEFAULT = 2**8`（`RVC_TRAIN_INIT_SCALE` 可覆盖，
  `grad_scaler_init_scale()` 保证 ≥1），稳态倍率与旧值相同 ⇒ **不改变数值行为，只省掉
  折半预热**。锁在 `TestGradScalerInitScale`
- **`Detected call of lr_scheduler.step() before optimizer.step()` 警告 ≠ 顺序写错**：
  触发条件是"该优化器一次真 step 都没执行过"（`_opt_called` 只在未跳过时置位）。
  调用顺序本来就是 optimizer 在前、scheduler 在 epoch 末，勿动
- **`last_epoch = start_epoch - 2` 是对的，别当 off-by-one 改**：`ExponentialLR` 在
  `last_epoch >= 0` 走 closed-form 重算而非累乘，续训不会多衰减一次。
  锁在 `TestLrScheduleAcrossResume`
- **GradScaler 状态进 checkpoint**：save/load 都带可选 `scaler` 参数；`setup()` 里 scaler
  必须在恢复逻辑**之前**创建；旧格式缺 `scaler` 键走默认值不 KeyError。
  锁在 `TestGradScalerStateSurvivesResume`。测往返必须用 `enabled=True` 的 scaler
  （禁用态 `state_dict()` 恒空，拿它测等于什么都没测）
- **离散 pitch bin 划分与上游逐值一致**：上游用的是 mel 公式版 `f0_mel_min/max`
  （不是网传的硬编码 1127/1998 常数），钳制、UV 落 1 也都等价；对照测试用独立重写的
  上游公式跑 1200 频点逐值比对（`tests/test_pitch_map.py::TestCoarseF0MatchesUpstream`）。
  bin 错位不会明显跑调，只会让 `enc_p` 收到训练时没见过的条件 ⇒ 咬字/音色发闷、极难归因
- **固定噪声场 `state.noise_z`**：`_z_sample` 形状不匹配时**静默退回** `randn_like`、
  不报错——等于没接上，表现为持续/重复内容逐块随机漂移。契约（命中必须返回同一对象、
  不匹配必须退回）+ 真实 Hubert 配置的形状校验都锁在 `tests/test_noise_z_shape.py`
- **`SineGen.forward` 用 `not torch.is_grad_enabled()` 而非无条件 `inference_mode()`**：
  否则 fp32 直接抛、fp16 被 autocast 掩盖成静默断链。锁在 `TestSineGenStaysTrainable`
- **weight_norm 两套键名**：加载侧有 `ParametrizationList` 兼容层，旧键直接认，
  **别手写键名改写、更别拿键名比对判断"权重没加载"**（曾据此误判"解码器全是随机初始化"
  并写进文档——判权重落地要比**值**或看 `missing_keys`/`unexpected_keys`，输出类结论要
  A/B 音频兜底）。需要改写的是**两份字典互比**的场合：`merge_models` 先
  `to_parametrized()`、输出 `to_legacy()`；`export_model` 统一旧版命名

**平台与进程**
- worker > 0 必须显式 `multiprocessing_context="spawn"`（Linux 默认 fork 会让 worker 继承
  半初始化 CUDA 上下文；**这个坑 Windows 上永远看不出来**）。锁在
  `test_workers_always_use_spawn`，`RVC_TRAIN_WORKERS=0` 分支有字典全等断言
- DataLoader 选项统一出自 `trainer.py::training_dataloader_options`：`RVC_TRAIN_WORKERS`
  （默认 2）、`RVC_TRAIN_CACHE_MB`（进程总预算按 worker 均分，0 关缓存）、
  `RVC_TRAIN_PREFETCH`（默认 1）；`persistent_workers` + `cleanup()` 显式关 worker
  （否则 Windows 进程被 worker 持有）
- Linux 无需改动的几处：`CREATE_NO_WINDOW` 非 Windows 恒 0；ffmpeg 第三级查找
  （`assets/ffmpeg`）**有意仅 Windows**，Linux 走 `RVC_FFMPEG`/PATH；内存探测有
  `os.sysconf` 分支；设备枚举跨平台。GUI `recycle_file` 非 Windows 抛
  `NotImplementedError`（不在训练范围）
- 所有子进程 spawn 带 `CREATE_NO_WINDOW` + `stdin=DEVNULL`（否则 GUI 拉训练闪黑框）；
  两个入口的 `if __name__ == "__main__"` 守卫是 Windows spawn 必需，别删
- ffmpeg 路径统一经 `cfg["ffmpeg"]` 下发（PreProcessor / F0 / 特征 / Trainer），不在
  各模块各自查找

**协作口径**
- **训练在哪跑 = 每轮现场问**：本机 8GB 显存 batch 上限 4，云端慢但解放本机。素材按
  **截取**用（"不需要全部用上，分别截取就行"）。试听交付 = 走**生产离线链路** +
  用户存档参数渲染指定文件到桌面，不是 loss 数字
- **端到端验证脚本没有存档**：验训练按本节口径重建最小链路（合成素材 → 预处理 → F0 →
  特征 → filelist → 2 epoch → 导出 → 续训）。241 例单测覆盖不到"必崩缺陷"那一层

### 11.1 训练是 GPU-bound，数据管道参数不是杠杆

真实 `Trainer.setup()` 实测：取数占单步的 **远低于 1%**。结论：
- 调 `num_workers` / `prefetch_factor` / 样本缓存**换不来速度**，别再往这个方向想
- 要更快只能动 batch / 精度 / segment 长度 / 模型本身（都影响训练质量，不属于"优化"）
- worker 数真正影响的是**内存**（Windows 上每个 spawn worker 重新 import 全部模块，
  数百 MB）。`workers=0` 已验证可正常训练。**默认值不动**：测量用的是玩具数据集，
  按它调默认属于过拟合；要省内存自己设 `RVC_TRAIN_WORKERS`

## 12. 测量方法论与踩坑（做实验前必读）

**可用的判据**
- **周期输入 ⇒ 块等价**：把输入造成严格周期且周期 = 整数个块，第 k 块与第 k−P 块滚动窗口
  逐位相同，则特征/合成/输出都该逐位相同。用来抓"时间累积型 bug"，最灵敏
- **相位散度**：周期素材下按"绝对帧号 mod 周期"分桶，同桶 = 同内容被不同窗口位置看到，
  桶内两两余弦 = 窗口相位依赖强度
- **重叠一致性**：相邻两块对同一段 60ms 内容各渲染一次，比较两次渲染
- **扰动放大率**：同一输入 + −60dB 白噪，比较输出变化
- **注入类功能必须测「与原声的相关性」**：`injected = process(out, ref) - out` 是精确注入量
  （加性链路），把它和原声做余弦、再做投影，就知道有没有把本音漏进输出。**只看电平不够**：
  真实感功能早期有一版 residual 注入看着电平无害（−19.5dB），余弦却是 0.75，等于原声按 −22dB 混进去
- **注入量的相对电平要按「本块」算，不能只看聚合**：输出电平跟踪是慢放的，静音/安静段里
  跟踪值仍停在响段电平上，聚合指标会把问题平均掉（同一份素材聚合 −19.5dB，最坏一块 −1.4dB）
- **分阶段 CUDA Event**：见 §10

**失灵的判据（别再用）**
- 波形余弦/波形相关性：谐波相位不同，最佳滞后互相关只有 0.18–0.29，毫无意义
- `out` 块余弦：SOLA 会重新对齐，块间比较无意义
- mel/特征 MSE 对**音位翻转失明**：噪声地板就高于信号差。判"咬字有没有变"只有 ASR 有效，
  但本环境已无 ASR 工具（AudioTranscribe 不可用）
- 对数梅尔比较要小心 `clamp(min=1e-8)` 会放大静音段的差异
- FFT 峰值在语音上常抓到共振峰而非 F0（系统性偏高 5–10%），只能看相对轨迹不能读绝对值

**特征侧探针的失灵判据与陷阱（2026-11-07）**
- 特征空间余弦类代理**分辨不了用户听感裁定**：一天内写过的五类代理（窗长扫描、重复占比
  扫描、上下文修复变体、流内出现次数自一致、MFCC 漂移/斜率）全部复现不了实听听感排序。
  唯一能定案的测量 = **带时间戳实听渲染音频**；无参考代理只有周期度（80–660Hz lag 的
  自相关峰/能量）+ >3kHz 谱倾斜，能看嗡/毛刺/破音，**看不到咬字糊**
- **凤凰.wav 不是有效的特征探针**：源 F0 中位 118Hz，落在 §14 低端跳八度带（渲染出来
  就是 238Hz 的八度跳）——它测的是音域越界（§14），不是咬字/发声方式。有效的在域重复素材：
  `绕口令2 四四四（录音室版）`（映射后 ≈330Hz，5 onsets/s）、`f3 干笑哈 哈 哈`（≈319）、
  `f5 咳咳`（≈400）。音域越界与发声方式两类案例必须分开测，否则特征修复会显得"失败"
- 逐条陷阱（每条都翻转过结论）：拿**第一个 onset** 会取到弱起音（低 19dB）量到静音；
  比较**不同长度**窗口把几何和内容混淆；`np.tile(syl)[:lead]` 截在音节中间，把爆点正好
  放在被测帧上；`per = d // 320` 整除漂移累计 ~7 帧能伪造"崩塌"；ridge 在 float32 解出
  0.641、float64 是 0.797；**静默帧占缓存 43%** 且余弦接近 1，任何特征拟合数字必须
  有声/静默分开报

**坑**
- HuBERT 输入 N 帧 → 输出 **N−1 帧**（不是 N）
- 改缓冲区长度/`extra_time` 后**必须重建 `state.noise_z`**：形状不匹配时 `_z_sample`
  静默退回 `randn_like`，逐块随机抽取会让 A/B 结论完全失真
- 计时前必须 `warmup()`，否则首块 2s 图捕获污染均值
- **脱离 `VoiceEngine` 的探针脚本默认跑的是 eager，不是生产路径**：`run_cuda_graph` 在
  `configure_cuda_graph` 从未被调用时静默走 `return function(*inputs)`，而唯一的调用点是
  `RuntimeDeviceConfig._init` ← `VoiceEngine.__init__`。所以单纯 import 模型自己跑的基准，
  数字会比实机**大一倍多**。探针里必须显式
  `configure_cuda_graph(torch.device('cuda'))` 并断言返回 True，
  否则"哪个阶段最慢"的结论整个作废
- **笔记本 GPU 会降频，顺序计时会骗人**：同一基准按固定顺序跑，后测的配置会莫名变慢
  （曾按顺序测出过一条单调变慢的假趋势，实为降频造成）。必须**交叉轮转 + 每轮取最小**，
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

## 13. 素材与用户配置

- **素材与探针脚本不在仓库里**：`tools/` 只剩 `pitch_map_demo.html`（2026-11-07 用户裁定），
  实验/测量脚本与 ~9.4GB 未跟踪数据已删除且不可恢复。§14 提到的 `probe_pitch.py` 等
  **现在不存在**，需要时按 §14 记录的判据重写。
- 素材：`E:\Download\audio\{凤凰,111,aaa,ah_sweep_step1}.wav`；环境音 `E:\Download\环境音素材`；
  KTUM 应用 `E:\Download\ktum\KtUltraMorph.exe`
- **存档参数不进文档**（2026-11-07 用户裁定）：`save_state.json` /
  `train_cli_defaults.json` 的具体值只存在于文件本身，文档记录**结构与规矩**
  （顶层 `{gui, models, train}`、只有 `gui` 块被读取等见 §11），不拄参数值。
  需要用户实际路径时直接读文件
- A/B 试听要先把输入峰值归一化到 0.5 并把 `volume` 置 1.0，否则 111.wav（峰值 1.098）×
  volume 1.5 直接削顶到 1.000，比较失效
- **HuBERT 权重存盘精度**：三个变体（base / chinese / ktum）现在都是 fp16 存盘。推理侧
  `rvc/runtime/device.py:32` 把 `is_half` 硬编码 `True`、模型总是 `.to(float16)`，所以
  fp32 存盘只是**假的精度**：加载瞬间降下来，白白多占一倍磁盘和 `torch.load` 期间的宿主内存。
  `chinese` 已由 fp32 转成 fp16，转换经张量级与端到端特征级两道逐位相等复核。
  判据：**任何"用 fp32 权重换精度"的想法在项目里都无效**——`is_half=False` 路径读回的
  dtype 虽是 float32，低 13 位全零占比 1.0000（fp16 值零填充），不存在真实 fp32 精度
- 与训练侧 fp32 回退的关系：`rvc/train/cli/envcheck.py` 在算力 < 7.0（P40/1080Ti）时把
  `use_fp16` 置 False。**不要因为这个就删掉它**——合成器权重本来就是 fp32，`fp16_run=False`
  对它确实提供真实 fp32 精度，那条路径不是死代码。本机算力 12.0，恒走 fp16，不受影响

---

## 14. xd80+ay20 的真实可用音域（2026-11-06 实测，2026-11-07 复核：现役配置即被测配置）

**结论：这个模型唱得出来的只有 155–660Hz（MIDI 51–76，coarse bin 39–174），跨度约 2 个八度。
低端不是"电平低"，是系统性跳高八度；高端从 ~698Hz 起塌成噪声。** 所以"音域不全"的成因是
模型本来没在那儿训过，不是映射把音高压平。
用户存档的 `pitch_map_dst_min/max = 40–800Hz` 两端都在可用区间之外，那个数字描述的是映射的
写法，不是模型的能力。

八度折叠（把越界帧搬进 155–660）只是**止血**，不是修法：它能把 ≥698Hz 的噪声和 ≤87Hz 的
嗡声换成一个干净但差八度的音（92–123Hz 那段模型本来就在自己跳八度，折叠过去等于什么都不改），
但它造不出低音区和高音区本身——那只能补素材。

测量脚本 `probe_pitch.py`（A 常数音高、B 移调真实轮廓）已随 tools/ 清理删除（见 §13），
下表数字是当时实测留存。判据：谐波比、跟随误差都**逐帧按该帧请求值**计算，并修了
"±6Hz 容差比 15.6Hz 的谱线间距还窄、196Hz 凭空读成 0.06"这个假阴性。素材无关性已验：
`抒情慢歌`、`高音歌 2`、`对话聊天 2`（录音室与非录音室两版）三条素材给出同一条边界。

| 请求 Hz（MIDI / coarse bin） | 实测症状 | 判据 |
|---|---|---|
| ≤87（≤41 / ≤15） | 又轻又浑，不成音 | RMS 0.05–0.13（中段 0.15–0.45）、周期相关 0.15–0.68、谐波比低至 0.02 |
| **92–123（42–47 / 17–28）** | **跳高八度**，电平正常 | 请求周期的归一化自相关 0.90–0.97 **且** 半周期也 0.86–0.97、基线比 2 倍线弱 23–46dB |
| 131–147（48–50 / 31–36） | 过渡带，偶发翻倍 | 基线−2 倍线 −13.4…+0.2dB，半周期相关 +0.1…+0.4（正常应为负） |
| **155–659（51–76 / 39–174）** | **干净** | 跟随误差 ≤0.05st、周期相关 ≥0.93、半周期相关为负、谐波比 0.5–0.96 |
| 698–740（77–78 / 182–190） | 塌陷起点 | 周期相关 0.55→−0.23、>4kHz 杂 0.04→0.11、RMS 0.09→0.056 |
| ≥831（≥80 / ≥199） | 噪声 | 谐波比 0.02–0.27、RMS ≈0.05；≳1100Hz 后 coarse bin 触顶 255，映射本身失去分辨率 |

**"跳八度"这件事不能拿跟随误差单独定案**（这是本项目第二次栽在"尺子自己会犯被测的错"上）：
自相关在低次谐波占优时会把周期认成半周期，正好报 +12st；HPS 复核更不可信，它的搜索窗只有
0.6–1.7 倍请求值，真跳到 2 倍时压根不在窗内，加权和还会把请求值的整条谐波族记成"请求值
这个候选"的得分，于是永远报 0。两把会犯错的尺子互相打架时，定案要用**不经过任何周期估计**
的量：`lag = 1/请求频率` 上的归一化自相关（真跳了八度它才 1.0，且半周期也 1.0）+ 基频谱线
相对 2 倍谱线的 dB（窗必须 Blackman-Harris，Hann 的 −31dB 旁瓣会在 −26dB 这个量级上伪造出
一个假基频峰）。上表用的就是这两条。

**补素材的量化目标**：下探到 ~82Hz 需要把干净素材**保共振峰**
下移 8–12 半音；上探到 ~930Hz 需要上移 5–7 半音。**共振峰必须锁住**——`librosa.pitch_shift`
（含 `st`/`rm`）连同 formant 一起搬，伪造出来的"低音"是另一个音色，训进去会把模型的音色锚
打散；用 rubberband 的 formant 保留路径，并在训练前用同一套谱线/周期指标复量伪造素材本身，
而不是等训完再听。

**当前阻断（2026-11-07）**：用户**手里没有可用素材**（"我手里没素材啊"），补素材只剩伪造
一条路；但本机既无 `rubberband` CLI 也无 `pyrubberband` 模块，`librosa.pitch_shift` 退化到
sm/hs/st/rm（连共振峰一起搬，即上面说的禁忌）。装 RubberBand 还是本地实现保包络移调，
**用户未选择，别擅自开工**。
