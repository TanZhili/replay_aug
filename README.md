# replay_opus

把 TTS 合成的语音渲染成两类训练数据，用来训练全双工语音对话系统的**拒识**能力：

- **真人（live）**：一个人在客厅、卧室、厨房或车里，对着设备说出这句话；
- **外放（device）**：电视、音箱、手机、平板、笔记本、智能音箱、收音机或车机，把同一句话播放出来。

两类放在**同一个场景**里生成：同一个房间、同一组噪声、同一个麦克风、同一条采集链。它们之间唯一的区别是声音是怎么发出来的。

![同一句 TTS 在同一场景里的真人版与外放版](demo/spectrograms.jpg)

## 为什么推倒重来

这个仓库原来的代码是给**深伪 / 重放攻击检测**写的（ASVspoof PA、ReplayDF 那一套）。拿它给全双工拒识造数据，有四个根本问题：

1. **只有一类。** 它只把音频变成「重放音」，没有「真人在同一个房间里说话」的对照。模型能学到的最省事的规则就是「有混响、有噪声、音质差 = 外放」。你已经指出这不够用：真人在客厅里说话，同样有混响和噪声。
2. **链路的位置不对。** 编解码器放在了麦克风**之后**，这对应的是「偷录的文件又被压缩」。拒识场景里，编解码发生在扬声器**之前**：电视节目是 AAC/AC-3 播出的，短视频是 Opus/AAC，电话是 AMR/G.711。麦克风之后的处理两类都要经过，不该成为区分依据。
3. **房间不是同一个房间。** 旧代码每条音频独立抽一个 T60/DRR，真人和电视没法放进同一个客厅，也就无从体现「电视在墙边、人在沙发上」这种几何差异。
4. **没有全双工需要的输出形式。** 全双工模型听到的是一条连续的流：电视一直在说，用户时不时插话，经常和电视重叠。它需要这样的流，并且需要带时间戳的标签。

新架构按「共享场景 + 两条发声链」重写，只保留了原来的 DSP 基础件、编解码往返、扫频测量和路径列表这些可复用的部分。

## 架构

```text
                ┌─ 真人：嗓音电平 · Lombard 效应 · 嘴部指向性 ───────────────────────────────┐
TTS 语音 ──────┤                                                                           ├─→ 同一房间 ─→ 同一噪声 ─→ 同一采集链 ─→ 输出
                └─ 外放：节目链（广播/流媒体/通话/电台）─→ 设备 DSP ─→ 扬声器非线性 ─→ 多单元频响 ┘
                   （编解码、响度、压缩、配乐、原录音房间）   （对白 EQ、虚拟低音、AVL、智能功放限幅）
```

**共享的部分**（`scene.py`、`acoustics.py`、`noise.py`、`capture.py`）：

- **房间**：鞋盒模型。早期反射用镜像源法，每个镜像按倍频程计算墙面吸声、空气吸收（ISO 9613-1）和声源指向性；混合时间之后接扩散尾，尾部电平由同一房间按 Barron 修正理论算出。家具和座椅的散射用散射系数建模：每次反射把一部分镜面能量交给扩散场。所以同一场景里的真人和电视共享 RT60，但各自的直达声、早期反射和直混比取决于自己的位置和朝向。
- **环境预设**（`environment.py`）：客厅、卧室、厨房、书房、车舱。每种预设给出房间尺寸、混响、麦克风位置、说话人位置、设备类型的比例和噪声工况。车舱有固定座位（主驾、副驾、后排）和车机扬声器布局（门板低音、A 柱高音、后门同轴，可选中置）。
- **噪声**：以麦克风处的 A 计权声压级给出。家居有安静底噪、空调风扇、电器嗡声、窗外交通；车内有怠速、城市、高速三种工况，由胎噪、发动机阶次、风噪和鼓风机合成。同一场景的噪声对两类完全相同，逐样本一致。
- **采集链**：MEMS 麦克风频响和自噪声、声压到数字满度的映射、前端高通、AGC、ADC 削波，以及可选的上行编解码。

**不同的部分**，也就是模型真正该学的东西：

| | 真人（`talker.py`） | 外放（`content.py` + `speaker.py`） |
| --- | --- | --- |
| 电平 | 1 m 处 55–66 dB(A)，噪声大时按 Lombard 效应抬高并变亮 | 按设备听音音量设定，噪声大时用户会调大音量 |
| 频带 | 无限制 | 小喇叭高通：手机 400–900 Hz、平板 250–550 Hz、笔记本 180–450 Hz、电视 90–220 Hz；外加谐振峰、腔体峰和箱体起伏 |
| 节目处理 | 无 | 响度归一（-23/-24 LUFS 或 -12~-18 LUFS）、母带压缩、限幅、配乐、原录音房间 |
| 编解码 | 无 | AAC / AC-3 / MP2（广播），AAC / Opus / MP3（流媒体），AMR-NB / G.711 / G.722 / Opus（通话） |
| 非线性 | 无 | 由音量决定：接近最大音量时进入智能功放限幅和振膜非线性 |
| 辐射 | 一个嘴，高频向前 | 多个单元分处不同位置，可以朝下、朝上、朝墙发声 |
| 时间结构 | 零星几句 | 连续节目，可能有多个角色轮流说话 |

所有信号都在物理单位（Pa）下渲染，所以真人、设备和噪声之间的相对电平是物理上自洽的。

## 安装

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
python -m replay_opus selftest
```

`av`（PyAV）自带 ffmpeg，上面所有编解码器都**不需要**另装系统 ffmpeg。`selftest` 用答案已知的系统逐项验证：BS.1770 响度、A 计权、房间 RT60 与直混比、所有编解码器往返、扫频测量、成对渲染的共享性质、场景标签的位置。

## 输入：一行一条语音

输入是一个文本文件，一行一条，不是文件夹：

```text
tts/zh_000.wav	帮我把客厅的灯关掉。
tts/spk03_0012.wav	spk03	今天北京的天气怎么样？
tts/zh_009.wav
```

每行可以是 `路径`、`路径<TAB>文本` 或 `路径<TAB>说话人<TAB>文本`，也可以用 `.jsonl`，每行写 `{"audio": ..., "text": ..., "speaker": ...}`。相对路径相对于列表文件所在的目录。文本和说话人会原样写进标签。编码接受 UTF-8、带 BOM 的 UTF-8 和 UTF-16。

`demo/inputs.txt` 是 12 条真实 TTS（Windows 自带的 Huihui / Zira 语音，由 `scripts/make_demo_tts.ps1` 生成），其中既有助手指令也有新闻、剧集、广告式的句子。两类数据来自同一个语音池，所以句子内容本身不会决定标签。

## 用法一：`render`，逐句成对

```bash
python -m replay_opus render --input demo/inputs.txt --output out/pairs --variants 4 --workers 8
```

每条语音、每个 variant 抽一个场景，把同一句话往里渲染两遍：

```text
out/pairs/live/zh_000__v0.wav      真人版
out/pairs/device/zh_000__v0.wav    外放版（同一房间、同一噪声、同一麦克风）
out/pairs/manifest.jsonl           每个文件一行
```

manifest 的主要字段：

| 字段 | 内容 |
| --- | --- |
| `label` | `live` 或 `device` |
| `pair` | 成对 ID：`key#variant`，两类共用 |
| `text` / `speaker` | 从输入列表带过来 |
| `at_mic` | 麦克风处的语音 A 计权电平、信噪比、直达声延迟 |
| `scene` | 环境、房间尺寸与分频带 RT60、麦克风位置、噪声工况与电平、采集链参数 |
| `source` | 真人：电平、Lombard、朝向、距离；设备：类型、各单元位置与频响、DSP 设置、节目链（响度、压缩、编解码、配乐） |

`--mode` 可以是 `paired`（默认，两类都出）、`mixed`（每个 variant 按 `render.device_prob` 只出一类）、`live` 或 `device`。

吞吐：单进程成对模式约 18 倍实时。这台机器上 8 进程渲染 480 条（约 29 分钟音频）用时 19 秒，约 90 倍实时。

## 用法二：`scenes`，全双工流加时间标签

```bash
python -m replay_opus scenes --input demo/inputs.txt --output out/scenes --count 1000 --workers 8
```

每个场景是一条 10–24 秒的流。可能有一台设备在播节目（「电视本来就开着」的连续节目，或一段短视频式的突发），也可能有用户零星说一到三句，并且经常和设备的声音重叠：

![全双工场景：智能音箱在播一通窄带电话，用户从 11 秒起插话](demo/scene.jpg)

manifest 每行一个场景，标签有两层，时间都是语音**到达麦克风**的时刻：

- `utterances`：每条放进场景的语音一项，带角色、声源、设备、节目类型、文本和说话人；
- `segments`：这些语音内部的语音活动段。一句 TTS 如果中间停顿较长，会分成几段，每段用 `utterance` 指回它所属的那一句，不重复文本。

下面是 `demo/scenes` 里上图那个场景的实际输出（节选）：

```json
{"audio": "scenes/scene_000002.wav", "scene_type": "both", "duration_s": 19.049,
 "utterances": [
   {"start": 7.08, "end": 9.39, "role": "device", "source": "device0", "device": "smart_speaker",
    "content": "phone_call", "utt": "zh_004.wav", "text": "把空调温度调到二十六度。", "speaker": null},
   {"start": 10.929, "end": 17.289, "role": "live", "source": "live0",
    "utt": "zh_007.wav", "text": "据气象部门预报，明天白天有小到中雨，请大家出行注意安全。", "speaker": null}
 ],
 "segments": [
   {"start": 10.929, "end": 12.459, "role": "live", "source": "live0", "utterance": 2},
   {"start": 10.984, "end": 13.294, "role": "device", "source": "device0", "utterance": 3},
   {"start": 12.879, "end": 14.879, "role": "live", "source": "live0", "utterance": 2}
 ],
 "sources": {"device0": {...}, "live0": {...}}, "scene": {...}}
```

场景类型按权重抽取：`both` 0.5、`device_only` 0.25、`live_only` 0.2、`empty` 0.05，可以在 `scene.types` 里改。如果输入列表带了说话人，同一场景里用户的声音不会和节目里任何一个角色相同，但每个说话人在整个数据集里两种角色都会出现。`--stems` 会额外写出每个声源在麦克风处的分轨（按混音同样的增益，但在 AGC、削波和上行编解码之前）。要把 `segments` 转成帧级标签：

```python
from replay_opus import frame_labels
labels = frame_labels(record["segments"], record["duration_s"], frame_s=0.02)
# 0 = 无语音，1 = 仅真人，2 = 仅外放，3 = 重叠
```

## 防捷径设计

拒识模型会学训练集里最便宜的区分特征。这里刻意让下面这些**不能**用来区分两类：

- **内容和音色**：两类来自同一个 TTS 语音池；成对模式下连句子都相同。
- **房间、噪声、麦克风、前端**：同一场景共享，噪声逐样本一致。
- **电平**：设备音量和人声电平的分布是按麦克风处的实际声压校准过的。车舱会额外带来约 7 dB 的混响增益，车机的电平已经把这一点算进去了。
- **时长和起始静音**：成对的两个文件等长。

同时保留了难例：条形音箱和智能音箱近距离播放（难负例）、远处背对设备的用户、70 dB(A) 车噪下的用户（难正例）。

`audit` 命令逐个特征检查有没有漏网的捷径。它计算每个简单特征单独区分两类的 AUC（0.5 表示没有信息）：

```bash
python -m replay_opus audit --manifest out/pairs/manifest.jsonl
```

默认配置下 12 条语音 × 20 个场景（480 个文件）的实测结果：

```text
feature             kind          AUC        live      device
level_dbfs          nuisance    0.553     -33.740     -32.304
peak_dbfs           nuisance    0.466     -14.669     -15.244
duration_s          nuisance    0.500       3.625       3.625
noise_floor_dbfs    nuisance    0.510     -54.864     -52.953
crest_db            acoustic    0.257      19.093      16.393
...
level AUC by environment: bedroom 0.56, car_cabin 0.56, kitchen 0.55, living_room 0.54, study 0.56
paired level difference device - live: mean +1.0 dB, sd 4.8, 10-90% [-4.6, +7.9]
```

电平、峰值、时长、底噪这几个「无关特征」都在 0.5 附近，每个环境内部也是。波峰因数偏低（AUC 0.26）是物理上应有的差异：节目压缩和功放限幅会压低峰值。简单的全局频带比几乎分不开两类，因为噪声、麦克风低频滚降和前端高通在两类上一视同仁地抹掉了低频。模型要学的线索比全局频带比细得多。改了预设之后要重新跑一次 `audit`；`scripts/plot_manifest.py` 可以把两类的电平、信噪比、距离分布叠起来画出来。

## 配置

`configs/default.yaml` 就是全部内置默认值，每一项都有注释。环境、设备、节目链这三张预设表写在代码里，用下面的命令打印当前生效的版本：

```bash
python -m replay_opus presets --section devices
```

YAML 里只需要写要改的东西，预设会做深合并；任何一项也都可以在命令行用 `--set` 临时覆盖：

```yaml
environments: {car_cabin: 1.0}                 # 只要车舱
environment_presets:
  car_cabin: {drive_side: right}               # 右舵
devices:
  presets:
    tv: {level_dba: [60.0, 72.0]}              # 电视开得更响
content:
  profiles:
    streaming: {bed_prob: 0.8}                 # 流媒体更常带配乐
```

```bash
python -m replay_opus render --input in.txt --output out --config my.yaml --set capture.agc_prob=0
```

采样率：内部默认 32 kHz 处理，输出 16 kHz。如果模型吃 24 kHz（例如 Mimi 系编解码器），设 `sr_internal: 48000` 和 `output.sr: 24000`。数据量大时可以设 `output.format: flac`，无损，磁盘占用约为 WAV 的一半。

## 接入真实素材（强烈建议）

合成素材让流程开箱即用，真实素材会明显提高逼真度。每一类都用路径列表接入：

| 素材 | 配置项 | 说明 |
| --- | --- | --- |
| 环境噪声 | `noise.lists: {home: ..., car: ...}` | 家居、车内的真实底噪录音。电平仍按场景抽取 |
| 配乐 / 音效 | `content.bed_list` | 电视剧、短视频里的人声常压在音乐下面。可用 MUSAN music 之类的音乐库 |
| 实测房间 IR | `room.measured: {living_room: ...}` | 每个子目录算一个房间；同一场景的真人和各个扬声器单元会从同一个房间的不同位置里抽 IR |
| 麦克风频响 | `capture.mic_ir_list` | 你的目标设备或 POLIPHONE 之类的手机麦克风 IR |
| 实测设备频响 | `devices.measured: {tv: ...}` | 用 `sweep` / `extract` 在设备前 10–30 cm 测得，加载时自动截取前几毫秒以去掉测量房间 |

测自己的电视或手机：

```bash
python -m replay_opus sweep --output sweep.wav          # 用目标设备播放，在正前方 10-30 cm 录音
python -m replay_opus extract --recording rec.wav --output responses/ --name living_tv
```

`extract` 会写出 `living_tv.npz`，把它写进一个列表交给 `devices.measured.tv` 即可。报告里的 `thd_at_sweep_level` 可以拿来核对对应预设的 `thd_full_scale` 范围。

## 验证

- `python -m replay_opus selftest`：安装后的快速自检。
- `python -m pytest tests -q`：101 项测试，检查的都是物理规律和两类之间的不变量，而不是快照。主要包括：各倍频程 RT60 符合 Eyring 公式（误差在 12% 以内，实测约 5%），直混比随距离的变化符合扩散场理论（2 dB 以内），指向性只改变直达声不改变混响尾，设备输出电平校准到目标值（0.05 dB 以内），失真随音量增大，窄带通话 4.5 kHz 以上能量低于 -35 dB，成对渲染在语音到来之前逐样本一致，场景标签内外的能量差大于 10 dB，`configs/default.yaml` 与代码默认值一致。
- **下游验证只有一条路**：留一小份**真实录制**的评测集，在你目标的真实家庭和车里，分别录真人说话和真实电视、手机的外放。仿真训出的模型在这份集合上的表现，才是唯一能说明问题的指标。公开数据里，ReMASC 录了室内、户外和车内环境下的真人与重放语音，可以作为参照。

## 已知局限

- **单通道。** 输出单个麦克风的信号。如果前端是麦克风阵列加波束形成，空间线索（声源方向稳定、多单元的空间扩展）只能部分体现。
- **鞋盒近似。** 车舱和房间都按长方体计算，低频模态、桌面反射、座椅遮挡没有单独建模，只由散射系数和吸声分布近似。有实测房间 IR 时优先用实测。
- **设备是参数模型。** 频响、失真、DSP 都按设备类型的典型范围随机抽取，指向性用心形族近似。要逼近某款具体产品，就测它的频响。
- **没有建模回声。** 全双工系统自己播放的声音（AEC 残留）没有包含在内，那需要另外的数据。
- **没有其他真人。** 家里其他人之间的对话同样是「真人」，从声学上无法拒识，需要语义或说话对象判断。
- **只有 TTS 声音。** 模型在训练里没听过真人嗓音。如果有真实语音语料，建议把它加进同一个语音池，两类一起用。
- **Lombard 效应只建模了电平和频谱倾斜**，没有建模基频和语速的变化。

## 主要参考

- J. B. Allen, D. A. Berkley, *Image method for efficiently simulating small-room acoustics*, JASA 65(4), 1979.
- M. Barron, L.-J. Lee, *Energy relations in concert auditoriums. I*, JASA 84(2), 1988.
- A. Lindau, L. Kosanke, S. Weinzierl, *Perceptual evaluation of model- and signal-based predictors of the mixing time in binaural room impulse responses*, JAES 60(11), 2012.
- ISO 9613-1, *Attenuation of sound during propagation outdoors — Part 1: Calculation of the absorption of sound by the atmosphere*.
- ITU-R BS.1770-4, *Algorithms to measure audio programme loudness and true-peak audio level*；EBU R 128；ATSC A/85.
- ANSI S3.5-1997 / ISO 9921：不同发声力度下的语音声压级。
- H. Lane, B. Tranel, *The Lombard sign and the role of hearing in speech*, J. Speech Hear. Res. 14, 1971.
- W. T. Chu, A. C. C. Warnock, *Detailed directivity of sound fields around human talkers*, NRC IRC-RR-104, 2002.
- W. Klippel, *Tutorial: Loudspeaker nonlinearities — causes, parameters, symptoms*, JAES 54(10), 2006.
- E. Larsen, R. M. Aarts, *Reproducing low-pitched signals through small loudspeakers*, JAES 50(3), 2002.
- A. Novak, P. Lotton, L. Simon, *Synchronized Swept-Sine: Theory, Application and Implementation*, JAES 63(10), 2015.
- M. E. Ahmed et al., *Void: A fast and light voice liveness detection system*, USENIX Security 2020.
- Y. Gong et al., *ReMASC: Realistic Replay Attack Corpus for Voice Controlled Systems*, Interspeech 2019.
