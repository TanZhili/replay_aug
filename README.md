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

## 三种用法

### 路线 A：零硬件，直接跑

不需要任何下载，房间用参数化合成 IR：

```bash
python -m replay_opus simulate \
  --input  path/to/tts_wavs \
  --output path/to/replayed \
  --config configs/default.yaml \
  --variants 3 --workers 8
```

输出目录里会有一份 `manifest.jsonl`，逐条记录该文件实际抽到的每一个条件
（扬声器等级、驱动电平、实测 T60/DRR/C50、SNR、麦克风参数、AGC 增益范围、削波比例、时钟漂移、
编解码与码率）。这份元数据是后续做分层评测和排查问题的唯一依据。

吞吐参考：8 进程下约 **40× 实时**（在一台普通开发机上实测 300 s 音频耗时 7.6 s），
所以 1000 小时素材约 25 小时可以跑完，进程数再加还能线性提升。

### 路线 A+：接上公开 IR 库

有了真实 IR 之后保真度会明显提升。把 IR 放在一个目录里（会递归扫描 `.wav/.flac/.ogg/.aiff`）：

```bash
python -m replay_opus ir-stats --ir-dir path/to/irs     # 先看看这批 IR 的 T60/DRR 分布
python -m replay_opus simulate --input in --output out --room-ir-dir path/to/irs
```

`--room-ir-dir` 默认启用 `reshaped` 模式：每次卷积前把 IR 重定向到抽样得到的 T60/DRR，
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

把 BUT ReverbDB 的 silence 或 MUSAN 噪声配到 `noise.noise_dir`，把 POLIPHONE 的 IR 配到
`capture.mic_ir_dir`，比内置的合成版本真实得多。

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
C50 与谐波电平报告）。把 `irs/` 指给 `--room-ir-dir`，把 `.npz` 所在目录指给
`playback.model_dir`，就是在用你自己测的链路做批量生成。

20–50 个配置足以建出一个可用的库，每个配置大约 2 分钟。这也是 ASVspoof 2019 PA 和 RIRplay
（TIFS 2026）采用的做法。

### 路线 C：真实播放 + 重录

这个项目不做这件事，但**建议你留一小份**真实重录的评测集。ASVspoof 2021 PA 已经证明，
只在仿真数据上训练的模型对真实重放的泛化会明显掉一档。

## 造训练数据时的一个关键注意点

如果你只对伪造音频做重放处理、真实音频保持干净，模型会直接学到「音质差 = 假」这个捷径。
ReplayDF 明确观察到：在真假录音条件本身就有差异的数据上训练的模型，会把整个 ReplayDF
（含真实样本）都判成伪造。

**所以真假两侧要过同一条链**，用同一个配置、不同的 `--seed` 跑两遍即可：

```bash
python -m replay_opus simulate --input bonafide --output out/bonafide --seed 11
python -m replay_opus simulate --input spoof    --output out/spoof    --seed 22
```

## 验证仿真是否够真

三层，从便宜到贵：

1. **声学层** —— `scripts/plot_manifest.py` 把生成批次的 (T60, DRR) 覆盖、SNR 分布和编解码构成
   画出来，可以叠加一份参考分布对比：

   ```bash
   python -m replay_opus ir-stats --ir-dir real_irs --output real.jsonl
   python scripts/plot_manifest.py out/manifest.jsonl --reference real.jsonl
   ```

   仿真集的 (T60, DRR) 散点应该覆盖并大致重合真实集，而不是缩在角落里。

2. **感知层** —— 对照原始音频算 PESQ，再做小规模 MOS 听测。ReplayDF 的真实重录覆盖 MOS/PESQ
   从 <2（很脏）到 >4（很干净）的全谱；你的仿真集也应该铺满这个区间。

3. **下游层** —— 用仿真集训练的检测器在**真实重录留出集**上的 EER。这是唯一真正说明问题的指标。
   参照系：ASVspoof 2019 训练 → 2021 PA 评测 EER 36.89%；RIRplay 训练 → 28.04%。

## 已验证的行为

`python -m replay_opus selftest` 和 `python -m pytest tests -q`（29 项）覆盖：

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
