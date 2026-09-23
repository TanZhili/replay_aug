# replay_opus

批量把干净音频（尤其是 TTS 合成的）转换成**「用扬声器在某个环境里播放、再被麦克风重新录下来」**的效果。

## 为什么不是「加混响」就够了

这个项目的结构来自一个反直觉的实验结果。Müller 等人（Interspeech 2025，ReplayDF）把 TTS 音频在
109 组真实「扬声器 × 麦克风」组合上播放并重录，发现：

- 给干净 TTS 音频加 15–40 dB 的高斯 / 白 / 粉噪声，检测器准确率几乎不变（−2.3% ~ +3.1%）；
- 但真实重录一遍，伪造样本的检出率掉 **28–42 个百分点**；
- 即使用真实 RIR 做卷积增广来训练，EER 也只能从 18.2% 拉回 11.09%（干净基线 4.7%）。

也就是说，真实链路里起作用的成分**既不是噪声，也不只是房间混响**。所以这里实现的是整条物理链路，
而不是单次卷积：

```
输入 ──▶ ① 回放侧 ──▶ ② 房间 ──▶ ③ 环境噪声 ──▶ ④ 采集侧 ──▶ ⑤ 传输 ──▶ 输出
          扬声器 IR      RIR 卷积     底噪 / 自噪声     麦克风 IR      重采样
          Hammerstein    T60 / DRR    目标 SNR         ADC 削波       时钟漂移
          非线性 (1–5阶) 控制                          AGC            Opus/AAC/MP3
          带限 + 软削波
```

每一级的建模依据、参数范围的文献出处，见 `configs/default.yaml` 里的注释和各模块的 docstring。

## 安装

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
```

`av`（PyAV）自带 ffmpeg，所以 Opus / AAC / MP3 / AMR 编解码**不需要系统装 ffmpeg**。
`pyroomacoustics` 和 `matplotlib` 是可选的，只有 `room.mode: simulated` 和绘图脚本会用到。

装完先验证一遍：

```bash
python -m replay_opus selftest
```

这条命令会拿解析解已知的系统去验证 IR 合成、扫频测量、编解码往返和整条链路。

## 输入是路径列表，不是文件夹

`simulate` 读到的每一类素材都是一个文本文件，一行一个路径：

| 材料 | 命令行 | 配置项 | 允许的后缀 |
| --- | --- | --- | --- |
| 干净音频 / TTS | `--input` | — | `.wav .flac .ogg .mp3 .m4a .opus` |
| 房间 IR | `--room-ir-list` | `room.ir_list` | `.wav .flac .ogg .aiff .aif` |
| 噪声 / 房间静音 | `--noise-list` | `noise.noise_list` | `.wav .flac .ogg` |
| 麦克风 IR | `--mic-ir-list` | `capture.mic_ir_list` | 同房间 IR |
| 实测扬声器模型 | `--speaker-model-list` | `playback.model_list` | `.npz` |

空行和 `#` 注释会跳过，两端的引号会去掉。相对路径相对于**列表文件所在的目录**，重复路径只保留第一次。文件不存在，或者后缀不在上表里，会直接报错而不是静默跳过。编码接受 UTF-8、带 BOM 的 UTF-8，以及 PowerShell 重定向常见的 UTF-16。

输出文件名和抽样用的 key，都是列表里各路径相对于它们公共父目录的相对路径。列表最好落在同一棵目录下：加进一个公共父目录之外的文件，会把已有条目的相对路径一起改掉，同一条语音的抽样就对不上了。

仓库里的 `demo/` 只留了 10 个音频，避免把示例目录当成语料库。分别是 `source.wav`、`source_batch/` 里的 5 条、`sweep.wav`、`recorded_sweep.wav`、`irs/livingroom_test.wav` 和一条重放结果 `replayed_v0.wav`。对应的列表是 `demo/inputs.txt`、`demo/irs.txt`、`demo/speakers.txt`。

## 三种用法

### 路线 A：零硬件，直接跑

不需要任何下载，房间用参数化合成 IR：

```bash
python -m replay_opus simulate \
  --input  demo/inputs.txt \
  --output path/to/replayed \
  --config configs/default.yaml \
  --variants 3 --workers 8
```

`--output` 仍然是目录。里面会有一份 `manifest.jsonl`，逐条记录该文件实际抽到的每一个条件
（扬声器等级、驱动电平、实测 T60/DRR/C50、SNR、麦克风参数、AGC 增益范围、削波比例、时钟漂移、
编解码与码率）。这份元数据是后续做分层评测和排查问题的唯一依据。

吞吐参考：8 进程下约 **40× 实时**（在一台普通开发机上实测 300 s 音频耗时 7.6 s），
所以 1000 小时素材约 25 小时可以跑完，进程数再加还能线性提升。

### 路线 A+：接上公开 IR 库

有了真实 IR 之后保真度会明显提升。先把要用的 IR 写进列表，再用 `ir-stats` 看这批 IR 的分布：

```bash
python -m replay_opus ir-stats --ir-list demo/irs.txt --output real.jsonl
python -m replay_opus simulate \
  --input demo/inputs.txt --output out \
  --room-ir-list demo/irs.txt
```

`--room-ir-list` 默认启用 `reshaped` 模式：每次卷积前把 IR 重定向到抽样得到的 T60/DRR，
这样 N 条实测 IR 能派生出远多于 N 种声学条件（Luong 等人 SLT 2024 用这个方法把 270 条 MIT IR
扩成了 500 倍）。要原样使用 IR 就加 `--room-mode measured`。

推荐的 IR 来源：

| 来源 | 内容 | 许可 |
| --- | --- | --- |
| [ReplayDF](https://huggingface.co/datasets/mueller91/ReplayDF) | 109 条**端到端重放链路** IR（已含扬声器+房间+麦克风） | 仅非商用研究 |
| [OpenSLR SLR28](https://openslr.org/28) | 实测 + 仿真 RIR，附 MUSAN 点源噪声 | Apache 2.0 |
| [MIT IR Survey](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html) | 271 条日常真实环境 IR | 免费 |
| [BUT ReverbDB](http://merlin.fit.vutbr.cz/ReverbDB/) | RIR **加上同条件的底噪录音** | 开放 |
| [POLIPHONE](https://github.com/polimi-ispl/POLIPHONE) | 20 款手机的**麦克风** IR | 研究用 |

把 BUT ReverbDB 的 silence 或 MUSAN 噪声写进一个列表，用 `--noise-list`（或 `noise.noise_list`）配上；
把 POLIPHONE 的麦克风 IR 写进另一个列表，用 `--mic-ir-list`（或 `capture.mic_ir_list`）配上。
这比内置的合成底噪和合成麦克风频响真实得多。

### 路线 B：测自己的链路（保真度最高）

每个「扬声器 × 房间 × 距离 × 麦克风」配置只需要放**一次扫频**，就能同时得到整条链路的线性 IR
和它的谐波失真剖面。

```bash
# 1. 生成扫频信号（12 s，50 Hz–20 kHz，前后各留 2 s 静音）
python -m replay_opus sweep --output sweep.wav

# 2. 用目标扫声器播放 sweep.wav，用目标麦克风录下来，两端增益保持固定不变。
#    录音要保留开头的静音段。

# 3. 提取
python -m replay_opus extract --recording recorded.wav --output irs/ --name livingroom_jbl_iphone
```

这会产出三个文件：`.wav`（可直接拿去卷积的链路 IR）、`.npz`（扬声器模型）、`.json`（T60 / DRR /
C50 与谐波电平报告）。把 `.wav` 写进一个列表交给 `--room-ir-list`，把 `.npz` 写进另一个列表交给
`--speaker-model-list`（或 `playback.model_list`），就是在用你自己测的链路做批量生成：

```bash
python -m replay_opus simulate \
  --input demo/inputs.txt --output out \
  --room-ir-list demo/irs.txt --room-mode measured \
  --speaker-model-list demo/speakers.txt
```

20–50 个配置足以建出一个可用的库，每个配置大约 2 分钟。这也是 ASVspoof 2019 PA 和 RIRplay
（TIFS 2026）采用的做法。

### 路线 C：真实播放 + 重录

这个项目不做这件事，但**建议你留一小份**真实重录的评测集。ASVspoof 2021 PA 已经证明，
只在仿真数据上训练的模型对真实重放的泛化会明显掉一档。

## 读 ir-stats

`ir-stats` 按 `metrics.py` 里和 ASVspoof PA、ACE Challenge 对齐的定义，给列表中每一条 IR 算四个数。
读入时会切掉直达声之前的空白，并把尾巴截到 1500 ms（和链路里的 `room.ir_max_ms` 相同）。

| 字段 | 含义 |
| --- | --- |
| `t60_s` | T60，单位秒。取 Schroeder 能量衰减曲线上 −5 dB 到 −35 dB 这一段的斜率，外推到 −60 dB（也就是 T30）。不是 Sabine 公式，也不是分频带 T60。 |
| `drr_db` | DRR，单位 dB。直达声取峰值前后各 **2.5 ms**，其余算晚到能量，再取两者能量比。窗很窄：车内几毫秒就到的早反射会算进直达，所以车内的 DRR 偏高。 |
| `c50_db` | C50，单位 dB。直达之后 **50 ms** 以内的能量，比上 50 ms 以后的能量。语音清晰度常用这个窗。它和 DRR 相关，但不是同一个数。 |
| `length_ms` | 对齐、截断之后的 IR 长度。顶在 1500 附近说明尾巴被切掉了，厅堂的 T60 会被低估。 |

标准输出先报总数，再报「可用」子集的最小值、中位数、最大值。可用的判定和 Luong 等人（SLT 2024）一样：T60 ≤ 2 s，并且 DRR 在 −10 到 30 dB。超出这个范围的 IR、以及 T60/DRR 算出来是 NaN 的 IR，不进这三行汇总。`--output` 写成 JSONL 时仍然逐条保留全部记录；读失败的文件只有 `ir` 和 `error`。

`demo/irs.txt` 只有一条客厅 IR，实际输出是：

```text
1 impulse responses, 1 within usable T60/DRR bounds
  t60_s    min=  0.491 median=  0.491 max=  0.491
  drr_db   min=  2.210 median=  2.210 max=  2.210
  c50_db   min= 11.910 median= 11.910 max= 11.910
```

```json
{"ir": "livingroom_test.wav", "t60_s": 0.491, "drr_db": 2.21, "c50_db": 11.91, "length_ms": 826.4}
```

`ir` 是该文件相对于列表公共父目录的路径。同一条 IR 的 `extract` 报告写的是 DRR 2.7 dB、长度 1500 ms：
`ir-stats` 读入时会切掉直达声之前的空白再做能量归一，所以 DRR 和 `length_ms` 会和那份报告差一截。
这条的 `length_ms` 是 826，没有顶到 1500，说明尾巴没有被截断。

### 不同场景下这三个量大概落在哪

下表是按上面这套定义、做宽带估计时的典型区间，不是某一个语料的官方标注。同一个房间里把麦克风拉远，T60 差不多，DRR 和 C50 会掉。距离按「人声到麦克风」理解。

| 场景 | T60 (s) | DRR (dB) | C50 (dB) |
| --- | --- | --- | --- |
| 车内座舱，约 0.4–0.8 m | 0.05–0.15 | +6 – +18 | +10 – +25 |
| 卧室、书房，有软装，约 1 m | 0.20–0.50 | +2 – +12 | +5 – +16 |
| 客厅，约 1–2 m | 0.35–0.80 | −2 – +10 | 0 – +12 |
| 厨房、卫生间等硬表面小房间 | 0.60–1.50 | −6 – +6 | −3 – +8 |
| 办公室、小会议室，1–3 m | 0.40–1.00 | −5 – +8 | −2 – +10 |
| 教室、走廊 | 0.60–1.50 | −8 – +4 | −5 – +6 |
| 礼堂、体育馆、楼梯间 | 1.5–4 以上 | −12 – 0 | −8 – +2 |
| 户外空旷，近处对话 | 通常 < 0.10，或测不出来 | +15 – +30 | +15 – +30 |
| 户外半封闭（街巷、廊下、庭院） | 0.15–0.80 | 0 – +15 | +2 – +18 |

读的时候有三件事：

- 默认配置 `room.t60_s: [0.08, 0.90]`、`room.drr_db: [-3, 20]` 覆盖的是客厅和办公室的主体。车内极短混响的下沿、户外极高的 DRR、厅堂 1 s 以上的长尾都不在这个区间里。要做这些场景就改这两个范围。`reshaped` 只能把一条 IR 的 T60 **缩短**，拉不长，长尾要用 `room.mode: synthetic`。
- 户外空旷没有一段指数衰减的混响尾。T60 经常很小，或者因为衰减曲线不成立而变成 NaN，这种记录不会进汇总。这时 DRR、C50 很高，只说明几乎没有晚到的反射，不要把它读成「一个 T60 = 0.05 s 的小房间」。
- `ir-stats` 不报告信噪比。SNR 是噪声级的参数，`noise.snr_db` 默认在 12–40 dB 之间抽，和 IR 无关。

## 造训练数据时的一个关键注意点

如果你只对伪造音频做重放处理、真实音频保持干净，模型会直接学到「音质差 = 假」这个捷径。
ReplayDF 明确观察到：在真假录音条件本身就有差异的数据上训练的模型，会把整个 ReplayDF
（含真实样本）都判成伪造。

**所以真假两侧要过同一条链**，用同一个配置、不同的 `--seed` 跑两遍即可：

```bash
python -m replay_opus simulate --input bonafide.txt --output out/bonafide --seed 11
python -m replay_opus simulate --input spoof.txt    --output out/spoof    --seed 22
```

## 验证仿真是否够真

三层，从便宜到贵：

1. **声学层** —— `scripts/plot_manifest.py` 把生成批次的 (T60, DRR) 覆盖、SNR 分布和编解码构成
   画出来，可以叠加一份参考分布对比：

   ```bash
   python -m replay_opus ir-stats --ir-list real_irs.txt --output real.jsonl
   python scripts/plot_manifest.py out/manifest.jsonl --reference real.jsonl
   ```

   仿真集的 (T60, DRR) 散点应该覆盖并大致重合真实集，而不是缩在角落里。

2. **感知层** —— 对照原始音频算 PESQ，再做小规模 MOS 听测。ReplayDF 的真实重录覆盖 MOS/PESQ
   从 <2（很脏）到 >4（很干净）的全谱；你的仿真集也应该铺满这个区间。

3. **下游层** —— 用仿真集训练的检测器在**真实重录留出集**上的 EER。这是唯一真正说明问题的指标。
   参照系：ASVspoof 2019 训练 → 2021 PA 评测 EER 36.89%；RIRplay 训练 → 28.04%。

## 用重放检测器验收：只看一个分数没有用

先说清楚这件事为什么不能省事。上面这些检测器的训练目标几乎都是「干净近讲」对「真实重录」。
只加了混响的音频同样不是近讲，所以也会被判成重放。**拿一段生成音频去跑检测器、得到一个高分，
只说明它不像近讲，不能说明整条链路仿真得像。** 也没有哪个公开模型是专门拿「真实重录」对「只加混响」
训练的，所以没有单个分数能直接回答「是不是只有混响」。

能回答这个问题的是对照实验：同一批源音频过三个条件，看检测器被推动了多远，再和真实重录比。
`configs/ablation_*.yaml` 就是这组对照，三个文件除了开哪些阶段之外完全一致
（同 seed、同输出采样率、同归一化、`tail_ms: 0` 让三者时长严格相等——否则按固定窗长切片的检测器
会把长度差异当成条件差异）：

| 条件 | 配置 | 开了什么 |
| --- | --- | --- |
| A 干净 | `configs/ablation_clean.yaml` | 只重采样和归一化，作对照基线 |
| B 只加混响 | `configs/ablation_room_only.yaml` | 只开房间卷积，等价于普通 RIR 增广 |
| C 整条链路 | `configs/ablation_full_chain.yaml` | 回放、房间、噪声、采集、传输全开 |

```powershell
foreach ($c in 'ablation_clean','ablation_room_only','ablation_full_chain') {
  python -m replay_opus simulate --input eval.txt --output out/$c --config configs/$c.yaml
}
```

判据是 **C 要比 B 明显更靠近真实重录**。如果 C 和 B 打成平手，那多出来的四个阶段没起作用，
这条链路确实就只是混响。

### 用哪个检测器

| 模型 | 训的什么 | 权重 | 适用 |
| --- | --- | --- | --- |
| [W2V2-AASIST](https://huggingface.co/SpeechAntiSpoofingBenchmarks/W2V2-AASIST) | ASVspoof 2019 LA 深伪 | 有，[HF 上自带推理封装](https://huggingface.co/SpeechAntiSpoofingBenchmarks/W2V2-AASIST)，另需 `xlsr2_300m.pt` | **首选**：ReplayDF 用的就是它，有可对照的公开数字 |
| [ASVspoof 2021 PA LFCC-LCNN](https://github.com/asvspoof-challenge/2021/tree/main/PA/Baseline-LFCC-LCNN) | ASVspoof 2019 **PA**，真·重放 | 官方预训练，仓库里是 `__pretrained/` 占位，按 README 下载 | 唯一现成的重放专用预训练权重 |
| [AASIST](https://github.com/clovaai/aasist) / [RawNet2](https://github.com/TakHemlata/SSL_Anti-spoofing) | ASVspoof 2019 LA | 有 | 轻量对照，`AASIST.pth` 是 LA 不是 PA |
| [EchoFake](https://github.com/EchoFake/EchoFake) 基线 | EchoFake（含真实重放，ICASSP 2026） | 无，需自己训 | 最贴近这个任务，但要先训 |
| [M-ALRAD](https://arxiv.org/abs/2502.13473) | ReMASC 多通道 | 见论文 | ReMASC 上 SOTA，但要麦克风阵列，本仓库输出单通道用不上 |

首选 W2V2-AASIST 的原因是它给了**可对照的目标数字**。ReplayDF（Interspeech 2025）把同一批音频
真实重录一遍，量到：公开权重的 EER 从 10.6% 涨到 24.8%；在 ODSS 上重训后从 4.7% 涨到 18.2%，
即使训练时用 ReplayDF 的 RIR 做增广也只能压回 11.09%。还有一个更细的特征：air-gap **只影响伪造样本**，
伪造侧准确率掉 28–42 个百分点，真实样本那一侧基本不动（98.3% → 97.7%）。
你的 C 条件如果也呈现这个不对称，比总 EER 涨了多少更能说明问题。

### 把分数变成结论

每个条件导出一个打分文件，一行一条 `路径<TAB>标签<TAB>分数`，标签是 `bonafide` 或 `spoof`，
分数默认越大越像真实（上面几个权重都是这个约定，反过来就加 `--higher-is-spoof`）：

```bash
python scripts/detector_report.py \
  --scores clean=scores_A.txt \
  --scores room=scores_B.txt \
  --scores chain=scores_C.txt \
  --scores real=scores_replaydf.txt \
  --baseline clean --reference real --threshold 0.0
```

输出每个条件的 EER、相对基线的变化，以及 **gap closed**——这个条件复现了多少「干净 → 真实重录」
的 EER 差距。B 通常只能吃掉一部分，C 应该明显更高。加上 `--threshold` 还会分别报真实侧和伪造侧的
准确率，用来核对上面那个不对称特征。

真实重录那一列（`--reference`）建议直接用 [ReplayDF](https://huggingface.co/datasets/mueller91/ReplayDF)：
它公开了重录音频，而且原始未重录的音频来自 MLAAD v5 和 M-AILABS，可以拿同一批源音频过你的链路，
构成严格配对的对照。没有这一列时脚本仍然能跑，只是只能看 B 和 C 的相对关系，没有上界参照。

## 已验证的行为

`python -m replay_opus selftest` 和 `python -m pytest tests -q`（37 项）覆盖：

| 检查项 | 判据 | 实测 |
| --- | --- | --- |
| 合成 IR 命中目标 T60 / DRR | T60 误差 <10%，DRR <0.5 dB | T60 <2%，DRR <0.01 dB |
| IR 重定向（Bryan 方法） | 同上 | T60 <2%，DRR <0.01 dB |
| 扫频恢复房间 IR | T60 误差 <0.02 s | 小数点后三位一致 |
| 用测得 IR 卷积复现系统 | 残差 < −30 dB | −57.9 dB |
| 已知多项式的谐波剖面 | 误差 <0.2 dB | <0.01 dB |
| 编解码往返 | 保长度、增益 0.9–1.1、延迟已补偿 | 全部通过 |
| 同种子可复现 | 逐比特一致 | 通过 |
| AGC 不把静音段抬到语音之上 | 静音低于语音 >20 dB | 通过 |
| 输出时长受控 | 输入时长 + `tail_ms` | 通过 |

## 已知局限

- **高阶 Hammerstein 核的频率响应没有精确辨识。** `measure` 会精确测出各阶谐波**相对电平**
  （对已知多项式系统误差 <0.01 dB），并用这个斜率去标定合成核；但每一阶自己的频响需要完整的
  Novak 核反演，这里没有做。实践上影响不大——ReplayDF 和 Luong 的结果都表明卷积成分占主导。
- **测得的 T60 / DRR 是带限量。** 扫频测量得到的是真 IR 与带限冲激的卷积，所以 DRR 的 ±2.5 ms
  直达窗在低频处会被涂抹。ACE 之类的语料用分频带统计正是因为这个。
- **`reshape_ir` 只能缩短 T60**，不能拉长——拉长等于凭空编造一段没被录下来的混响尾巴。
  需要长尾就用 `room.mode: synthetic`。
- **AGC / 降噪是简化模型。** 真实手机和会议软件跑的是 WebRTC APM（高通 → AEC → 降噪 → AGC2）。
  需要逐比特一致就换 `pywebrtc-audio` 绑定。
- **没有建模 AEC**、也没有建模二次重放（「偷录再重放」需要两段串联卷积）。

## 主要参考

- N. Müller et al., *Replay Attacks Against Audio Deepfake Detection*, Interspeech 2025.
  [arXiv:2505.14862](https://arxiv.org/abs/2505.14862)
- H.-T. Luong, D.-T. Truong, K. A. Lee, E. S. Chng, *Room Impulse Responses Help Attackers to
  Evade Deep Fake Detection*, IEEE SLT 2024. [arXiv:2409.14712](https://arxiv.org/abs/2409.14712)
- J. C. Sanchez-Valera et al., *RIRplay: Generation of a Replay Stereo Corpus for Voice Biometrics
  Anti-Spoofing*, IEEE TIFS 21:4106–4118, 2026.
- X. Wang et al., *ASVspoof 2019: A Large-Scale Public Database of Synthesized, Converted and
  Replayed Speech*, Computer Speech & Language, 2020. [arXiv:1911.01601](https://arxiv.org/abs/1911.01601)
- J. Yamagishi et al., *ASVspoof 2021: Towards Spoofed and Deepfake Speech Detection in the Wild*,
  IEEE/ACM TASLP, 2023. [arXiv:2210.02437](https://arxiv.org/abs/2210.02437)
- A. Novak, P. Lotton, L. Simon, *Synchronized Swept-Sine: Theory, Application and Implementation*,
  JAES 63(10):786–798, 2015.
- N. J. Bryan, *Impulse Response Data Augmentation and Deep Neural Networks for Blind Room
  Acoustic Parameter Estimation*, ICASSP 2020.
- H. Tak et al., *RawBoost: A Raw Data Boosting and Augmentation Method Applied to ASV
  Anti-Spoofing*, ICASSP 2022.
