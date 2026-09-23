# 把 TTS 或真人录音变成「真人麦克风收音」和「外放重录」

这份文档说明本仓库要解决的问题、怎么解决、依据是什么。目标是：

- **输入**：TTS 语音，或者真人对着麦克风说话的录音；
- **输出**：同一段话的两种样子。一种是**真人麦克风收音**，也就是一个人在客厅、卧室、厨房或车里对着设备说出来、被设备麦克风收进去；另一种是**电子设备外放重录**，也就是电视、手机、音箱、车机把这段话播出来、再被同一只麦克风收进去；
- **用途**：给全双工语音对话模型造训练数据，让它学会拒识外放重录的语音，放行真人说的话。

第 1 节讲输入和输出分别是什么。第 2、3 节讲两种输出各自怎么生成、参数从哪里来。第 4 节讲为什么三种外放共用一个标签。第 5 节讲两类样本必须共享什么、怎么防止模型学到捷径。第 6 节对照代码和参数。第 7 节讲怎么验收，附上默认配置下的实测数字。附录汇总公开文献对「TTS、不同档次麦克风收录的真人、电子设备外放」三类信号的区分依据，前面各项设计的理由大多来自这里。

材料来自本仓库 `README.md` 引用的 2020 年及以后的工作，并补了同一问题上直接相关的论文和数据集。年份以发表或挑战举办年为准。2019 年及更早的语料（ReMASC、ASVspoof 2019）和经典声学文献，只在被这些工作引用、或作为工具和参数来源时出现。

## 核心结论

- **训练标签只有两个：`live`（放行）和 `device`（拒识）。** 三种外放形式都标 `device`，形式本身只记在元数据里（`playback_form`），用来控制比例和分组评测。
- **输入分两种来源（`origin`）。** `dry` 是 TTS，或者近讲、安静条件下录的真人，可以当作「嘴里发出的声音」，两种输出都能生成。`recorded` 是在某个房间里被某只麦克风录下的真人，本身已经带着那个房间和那只麦克风，只能变成「录音再外放」；它的真人版就是它自己。
- **真人麦克风收音是一次声学传播**：嘴的辐射、本地房间、麦克风和采集链。麦克风档次属于采集链，两类共享。
- **外放重录有三种形式：**
  1. **单次外放（`single`）**：演播室近讲录制、后期处理过的节目，经设备在本地房间播放一次。电视、电台、播客属于这种。
  2. **录音再外放（`rerecorded`）**：有人在另一个房间对着手机说话，经远端麦克风、处理和编码，再由本地设备播放。短视频、语音消息、免提通话属于这种。远端的房间、麦克风和噪声属于节目，和本地场景无关。
  3. **同场景二次重放（`replay`）**：有人在这个房间里被附近的一部手机录下，录音再在这个房间里放出来。ASVspoof 定义的重放攻击就是这种。
- **两类共享一切，只换声源。** 同一个房间、同一段噪声、同一只麦克风和采集链。文献一再发现，检测器会抓住两类之间不该有的差异，比如静音长度、音质、带宽。
- **最后都要在真实录音上验收。**

```text
输入 ─┬─ dry（TTS、近讲真人）─┬─ 真人收音：嘴 · 本地房间 · 采集链 ───────────────────────────── live
      │                        ├─ single：节目链 · 设备 · 本地房间 · 采集链 ─────────────────── device
      │                        ├─ rerecorded：远端房间 · 远端手机 · 节目链 · 设备 · 本地房间 · 采集链 ─ device
      │                        └─ replay：本地房间 · 附近手机 · 存储编码 · 设备 · 本地房间 · 采集链 ── device
      └─ recorded（房间录音）─── rerecorded（录音本身就是远端那一段）· 节目链 · 设备 · … ───── device
                                  （它的真人版就是它自己；可选原样写出）
```

---

## 1. 输入与输出

### 1.1 两种输入来源

| `origin` | 是什么 | 例子 | 能变成的真人收音 | 能变成的外放 |
| --- | --- | --- | --- | --- |
| `dry` | 可以当作嘴里发出的声音 | TTS；消声室或安静室内的近讲录音（EARS、VCTK、AISHELL-3、录音棚、头戴麦） | 在本地场景里渲染 | 三种形式都行 |
| `recorded` | 已经在某个房间被某只麦克风录下 | 手机录的语音、智能音箱或会议设备的远场录音、Common Voice 这类众包录音 | 它自己就是，可以原样写出（`render.recorded_live: passthrough`） | 只有 `rerecorded`，录音本身就是远端那一段 |

判断一段真人录音算 `dry` 还是 `recorded`，看它还剩多少房间：直达声占绝对主导、底噪很低、听不出混响的，算 `dry`；手机离嘴半米以上、有明显混响或背景声的，算 `recorded`。把 `recorded` 录音当成 `dry` 渲染成真人，等于让同一个声音经过两个房间，而真实的人说话只经过一个。

`dry` 真人录音的价值是让模型听过真实的声带，TTS 负责覆盖文本和说话人，两者按相同比例进入两类（2.2）。`recorded` 录音的价值是给外放类提供真实的远端房间、远端手机和平台压缩（3.2）。

### 1.2 两种输出和三种外放形式

| 输出 | 标签 | 形式 | 声音的来路 | 现实里的例子 |
| --- | --- | --- | --- | --- |
| 真人麦克风收音 | `live` | — | 嘴 → 本地房间 → 本地麦克风 | 用户对设备说话 |
| 外放重录 | `device` | `single` | 近讲录制、后期处理的节目 → 节目链 → 设备 → 本地房间 → 本地麦克风 | 电视新闻、剧集、电台 |
| 外放重录 | `device` | `rerecorded` | 远端真人 → 远端房间 → 远端手机和处理 → 平台或信道编码 → 设备 → 本地房间 → 本地麦克风 | 短视频、直播、语音消息、免提通话 |
| 外放重录 | `device` | `replay` | 本地真人 → 本地房间 → 附近的手机 → 存储编码 → 设备 → 本地房间 → 本地麦克风 | 把刚录的一段语音在同一个房间放出来；ASVspoof 的重放攻击 |

还有多级外放：节目里本身含一段外放，比如谈话节目里连线的电话，或者翻拍电视的短视频。VSDC（Baumann 等，Computer Speech & Language 2021）专门录了一阶和二阶重放，可以用来估计二阶重放的量级。本仓库目前不单独生成多级外放。

---

## 2. 输出一：真人麦克风收音

### 2.1 原始 TTS 缺什么

- **房间和麦克风。**
- **自噪声。** 句首句尾和词间常有精确为零的数字静音，任何真实麦克风录音都不会有。静音模式在 ASVspoof 2017（Chettri 等 2020）、2019 LA（Müller 等 2021）、2021（Liu 等 2023）上都被检测器当成过捷径。
- **发声力度的变化。** TTS 输出响度固定，不随噪声、距离和情绪变化。
- **足够的带宽。** 带宽取决于合成器的采样率：22.05 kHz 的 TTS 在 11 kHz 以上没有能量。输出 16 kHz 时这不要紧；如果模型吃 24 kHz 以上的音频，TTS 本身的采样率就要够高，否则真人类整体缺高频，而真实的人没有这个上限。
- **真实的嗓音。** 声码器会留下过平滑、基频过稳、缺少呼吸之类的痕迹。两类来自同一个语音池，这些痕迹和标签无关；但模型从没听过真实嗓音，所以要把真实语音混进语音池（2.2）。
- **自然的口语。** TTS 多是朗读腔，声学渲染改不了，只能靠文本和合成风格的多样性。

本仓库的对照实验说明了为什么必须渲染：ASVspoof 2021 官方 LFCC-LCNN 把干 TTS 判成极像重放，在真实重放录音上微调的 wav2vec2 则把任何带混响和噪声的声音都判得更像重放（附录 A.3）。直接拿干 TTS 当真人，模型学到的会是「有没有房间」。

### 2.2 声源：在 TTS 之外混入真实的干语音

TTS 负责覆盖文本和说话人，真实语音负责让模型听过声带。候选语料要尽量干，也就是近讲、安静、少混响，这样后面加上的房间才是唯一的一次传播。

| 语料 | 语言 | 录制条件 | 备注 |
| --- | --- | --- | --- |
| EARS（Richter 等，Interspeech 2024） | 英语 | 消声室，48 kHz，107 人，含多种情绪和说话方式 | Neri 和 Virtanen 的重放仿真直接拿它当干声源 |
| VCTK | 英语 | 半消声室，96 kHz | ASVspoof 2019/2021 PA 与 RIRplay 的种子 |
| AISHELL-3（Shi 等，2020） | 普通话 | 安静室内，高保真麦克风距嘴约 20 cm，44.1 kHz，218 人，约 85 小时 | 不是消声条件，带一点近场房间；文本里有智能家居指令 |

这些语料自带录音麦克风的频响和底噪，近讲、安静时这一层很薄，可以忽略，按 `dry` 输入。真实语音和 TTS 要按相同比例进入两类，不能只进真人类。

### 2.3 发声：电平、Lombard、频谱倾斜、指向性

1. **发声电平。** ISO 9921 给出 1 m 处的 A 计权语音级：放松 54、正常 60、提高 66、大声 72、很大声 78 dB(A)。对着设备说话的人多在放松到提高之间，本仓库取 55–66 dB(A)（`talker.level_dba`）。电平按 ITU-T P.56 式的有效语音电平计算：只算离这个电平 15.9 dB 以内的帧，所以句间的底噪和配乐不会把电平拉低。设备的听音音量用同一种算法，两类的相对电平因此一致。
2. **Lombard 效应。** 噪声超过一定水平后，人会不自觉地抬高音量，声音也变亮（Lane & Tranel 1971）。本仓库从 45 dB(A) 起算，每 dB 噪声抬高 0.2–0.5 dB，最多 12 dB（`lombard_onset_dba`、`lombard_slope`、`lombard_max_db`）。车舱最需要这一项：高速工况的噪声有 62–74 dB(A)。
3. **频谱倾斜。** 提高嗓门时 1 kHz 以上的能量相对变多，放松时相反。本仓库按相对 60 dB(A) 的差值，每 dB 给 0.2–0.4 dB 的高频搁架（`effort_tilt_db_per_db`）。基频和语速随 Lombard 的变化没有建模。
4. **嘴部指向性。** 人声在低频近似全向，1 kHz 以上越来越朝前（Chu & Warnock 2002）。人背对设备时，直达声的高频变暗，直混比下降，混响尾基本不变。扬声器的指向性和嘴不同：手机喇叭常朝下或朝侧面，电视单元朝下打到台面，智能音箱近似全向。Neri 和 Virtanen 分别用实测的人声指向性和实测的喇叭指向性仿真两类，并让喇叭在心形和实测指向性之间随机选，避免模型记住某一只喇叭的方向图。
5. **近讲线索只在近讲时出现。** 爆破音的气流会在麦克风上形成低频的「喷麦」。VoicePop+（Jiang 等，IEEE TDSC 2022）让用户在 12 cm 以内说话，利用呼气声在一定距离外就衰减、重放很难录到这一点做活体检测，攻击者在 30 cm 外偷录。全双工设备通常离人 0.5 m 以上（本仓库客厅取 0.5–5 m），这类线索基本不存在，不应该合成。

### 2.4 房间：一次传播

- **直达声、早期反射和混响尾来自同一个房间。** 本仓库的早期部分用镜像源法，每个镜像按倍频程计算墙面吸声、空气吸收和声源指向性；扩散场的电平由同一房间按 Barron 修正理论算出。家具和座椅的散射用散射系数表示：每次反射把一部分镜面能量交给扩散场，所以扩散场从直达声之后就开始累积。
- **RT60 按倍频程给。** RIRplay 用 ACE Challenge 的实测倍频程 RT60 驱动镜像源法（7 个房间 × 2 种摆位 × 8 通道，共 112 个环境）。环境从 112 个减到 27 个时，ASVspoof 2021 上的 EER 从约 30.7% 升到 33.4%，房间多样性本身就有用。本仓库的中频 RT60：客厅 0.30–0.70 s，卧室 0.25–0.55 s，厨房 0.40–1.00 s，书房 0.30–0.65 s，车舱 0.04–0.12 s，各倍频程再乘一组形状系数。
- **距离和临界距离。** 声源离麦克风超过临界距离后，混响能量超过直达声。Gaubitch 和 Looney（ICASSP 2024）用的近似式是 `d_c ≈ 0.057·√(γ·V/T60)`，γ 是声源的指向因数，V 是房间体积（m³）。60 m³、RT60 0.5 s 的客厅里，全向声源的临界距离约 0.6 m。人站在 2–3 m 外说话，麦克风收到的主要是混响。
- **不规则房间和家具。** 鞋盒模型没有低频模态的偏移、桌面反射和座椅遮挡。GWA（Tang 等，SIGGRAPH 2022）把几何声学和低频的 FDTD 波动求解结合，在 6,813 套 3D-FRONT 住宅的 18,968 个带家具房间里算了约 200 万条 RIR，可以作为更真实的房间来源，数据许可是非商业。这类 RIR 和实测 RIR 一样通过 `room.measured` 接入。
- **实测房间 IR 只取混响尾。** 实测 RIR 是用一只测量扬声器放扫频录下来的，直达声和早期部分带着那只扬声器的频响、指向性和那个测量位置。直接拿它渲染真人，等于让「真人」也经过了一只喇叭。本仓库默认的 `room.measured_mode: hybrid` 保留模型算出的直达声和早期反射（带嘴或喇叭各自的指向性，直混比随场景里的距离变化），只把实测 IR 的混响尾接在后面，并把它缩放到模型在这个距离上的混响能量。`full` 模式保留旧做法，整条使用实测 IR。

### 2.5 噪声

- **同一场景的噪声对两类完全相同，逐样本一致。** 噪声按麦克风处的 A 计权声压级给，而不是按每句话的信噪比。这样，安静房间里远处的说话人自然是低信噪比；按句子设信噪比，则会让两类的信噪比分布各自由句子电平决定。
- **电平。** RIRplay 发现 25–35 dB 的信噪比泛化最好，15–25 dB 反而变差；Sanchez 等（IberSPEECH 2024）在 ASVspoof 2019 PA 上做增强，30–40 dB 的加性噪声让 2021 PA 的 EER 从 44.06% 降到 36.33%。这些是按信噪比设定的增强实验。本仓库直接取物理噪声级：家居安静 24–35 dB(A)、空调 32–46、电器 30–44、街道 30–48；车内怠速 36–48、城市 52–64、高速 62–74 dB(A)。
- **素材。** MUSAN、WHAM! 等公开噪声库，或在目标家庭和车里录的底噪（`noise.lists`）。家里电视开着时的节目声不能当「噪声」混进真人类，那等于把外放内容塞进正类；这种情形应该做成重叠场景，用带时间戳的标签处理（`scenes` 命令）。

### 2.6 麦克风与采集链：按档次建模

「不同级别的麦克风」在建模上就是采集链的一组参数。常见的几档：

| 档次 | 典型设备 | 频响 | 自噪声 | 处理 | 建模方式 |
| --- | --- | --- | --- | --- | --- |
| 测量或录音棚 | 测量麦、电容麦 | 平直到 20 kHz | 很低，十几 dB(A) 或更低 | 没有 | 近似透明，适合当参照 |
| 消费级 MEMS | 手机、平板、笔记本、智能音箱、车机顶棚麦 | 低频有排气孔高通，高频有声孔谐振 | 常见信噪比约 60–70 dB（以 94 dB SPL 为参考），即自噪声约 24–34 dB(A) | 高通、降噪、AGC | 本仓库默认模型，或 POLIPHONE 等实测 IR |
| 通话链路 | 手机通话、蓝牙耳机 | 窄带 300–3400 Hz，或宽带到 7 kHz | 同上 | 降噪、AGC、语音编码 | 带限加编码 |
| 远场阵列前端 | 智能音箱、会议设备 | 取决于阵列 | 同上 | 波束形成、回声消除、降噪、AGC | 单通道只能近似；阵列线索见附录 A.6 |

要点：

- **本仓库默认的 MEMS 模型**：排气孔高通 30–150 Hz（1–2 阶），声孔谐振 9–15 kHz、+1–6 dB，外加两处 ±1.5 dB 的缓变；自噪声 22–34 dB(A)；满量程对应 92–112 dB SPL；70% 的概率有 60–150 Hz 的前端高通；40% 的概率有 AGC（目标 −28 到 −18 dBFS，最大增益 24 dB）；ADC 削波；上行编码默认关闭（`capture.*`）。
- **降噪。** 采集链可以带一个谱减式降噪器（`capture.ns_prob`、`ns_max_db`）：按整段信号里最安静的帧估计每个频点的噪声，用 Ephraim–Malah 的判决引导法估先验信噪比，维纳增益带下限，所以压制量不超过 `ns_max_db`。目标设备默认不开；如果你的设备在模型前面跑了降噪，就打开它。远端手机和偷录手机的采集链默认按一定概率打开（3.2、3.3）。
- **实测麦克风 IR。** POLIPHONE（Salvi 等，IEEE Access 2025）在符合 ISO 3745 的消声室里，用 Genelec 8020C 在约 3 m 外播放，20 部 2018 年以后发布的手机用底部主麦绕过系统音频处理，直接录 44.1 kHz 的原始信号，另录了 5 s、20 Hz–20 kHz 的扫频用于提取线性 IR。这些 IR 可以放进 `capture.mic_ir_list`，远端手机和偷录手机也各有自己的 `mic_ir_list`。IR 里含那只 Genelec 的频响；它是绕过系统处理的原始通路，真实的通话或录音 App 还会叠加降噪、AGC 和均衡。
- **档次对检测的影响有明确的方向**（附录 A.5）：验证侧麦克风越差，真人越容易被误判成攻击；攻击侧麦克风越好，重放越难检；深伪检测器会把消费设备录的真人判成假。
- **平行多设备录音可以用来校准档次。** LRPD（Yakovlev 等，ICASSP 2022）每次用一台设备播放、多台设备同时录，共 19 台录音设备、11 台播放设备、17 个环境，超过 100 万条。LibriWASN 用九台设备（五部手机、四个阵列）同时录了 LibriCSS 的回放信号。它们的声源都是扬声器，只能用来校准麦克风，不能当真人参照。

### 2.7 时间结构与标签

真人说话零星、有停顿、可能打断电视；设备节目连续，可能有多个角色轮流说话。这是真实的差异，可以保留，但两类里都要有反例：连续说话的真人（给孩子读故事），短促的外放（手机提示、短视频片段）。`scenes` 命令的 `scene.types`、`scene.live_utterances`、`scene.device_continuous_prob`、`scene.device_voices` 控制这些比例。

### 2.8 常见错误

- 直接拿干 TTS 当真人。
- 把 `recorded` 录音当 `dry` 渲染成真人。
- 只给真人类加房间、噪声或麦克风，外放类却是干净节目。
- 渲染之后逐文件做峰值或响度归一化。
- 整条使用测量扬声器录的房间 IR 渲染真人（2.4）。
- 把扬声器播放的语料当真人参照。ASVspoof 2021 PA 的 bona fide 是高品质扬声器播放的 VCTK；LibriCSS（Chen 等，ICASSP 2020）把 LibriSpeech 用扬声器在会议室里放出来，用七通道圆形阵列录下，扬声器到麦克风 33–409 cm；Libri-adhoc40（2021）用 40 个同步的分布式麦克风录扬声器回放。按「嘴还是喇叭」来分，它们全都属于外放类。
- 为远场样本合成喷麦、近讲效应这类近讲线索。

---

## 3. 输出二：外放重录的三种形式

三种形式的区别只在喇叭之前：声音是怎么来到设备里的。喇叭之后的一切，也就是设备 DSP、单元频响和非线性、摆位、本地房间、噪声和采集链，三种形式完全相同（3.4）。本仓库在抽形式之前就先抽好设备、摆位和各单元到麦克风的房间路径，所以同一句话的三种形式从同一台设备、同一个位置放出来。

### 3.1 单次外放（`single`）

- **节目源**：近讲录制、后期处理过的节目。一部分节目是外景录的，带一个小房间（`content.profiles.*.source_room_prob`，RT60 0.15–0.5 s，直混比 3–15 dB），相当于领夹或吊杆话筒在现场录到的一点房间声。
- **节目链**：从设备自己的内容档案里挑 `playback.single.contents` 允许的那些（默认 `broadcast`、`streaming`、`radio`），设备的档案里没有就用它自己的。
- **响度**：广播按 EBU R 128（−23 LUFS）或 ATSC A/85（−24 LKFS）；流媒体和短视频更响、压缩更重。
- **动态处理**：母带压缩和限幅；FM 电台还有多段压缩、削波、50 或 75 µs 的预加重和 15 kHz 带限。
- **编码**：广播用 AAC、AC-3、MP2；流媒体用 AAC、Opus、MP3。编码发生在扬声器之前，只属于外放类；麦克风之后的上行编码两类共享。
- **配乐和音效**：电视剧和短视频的人声常压在音乐下面（`content.bed_list`）。

### 3.2 录音再外放（`rerecorded`）

对 `dry` 输入，本仓库另起一个完整的远端场景，把这段话先「录」一遍：

1. **远端房间**：从 `playback.rerecorded.environments` 里抽一个环境（默认客厅 0.3、卧室 0.25、厨房 0.15、书房 0.2、车舱 0.1），房间尺寸、混响、噪声工况和本地场景一样按环境预设抽，但用自己的随机种子，和本地场景无关。
2. **远端说话人**：离远端手机的距离按节目类型给（`playback.rerecorded.distance_m`，默认 0.3–2.5 m，通话 0.2–1.0 m），Lombard 效应按远端噪声算。车舱里按座位摆，手机在顶棚或后视镜处，相当于车载免提。
3. **远端手机**：一条独立的采集链，默认值上叠加 `playback.rerecorded.capture`（默认 AGC 概率 0.8、高通 0.9、降噪 0.5），也可以给它单独的实测麦克风 IR。
4. **节目链**：从 `playback.rerecorded.contents` 允许的档案里挑（默认 `streaming`、`phone_call`），并且不再加参数化的小房间和替身远端底噪，因为远端的房间和噪声已经真实渲染过了。通话走电话频带和 AMR-NB、G.711、G.722、Opus 编码；短视频走平台响度和 AAC、Opus、MP3。

对 `recorded` 输入，第 1–3 步跳过：录音本身就是远端那一段，直接进节目链。

真实的用户生成录音也可以直接当节目源。Common Voice（Ardila 等，LREC 2020）是众包录音，说话人用自己的设备在自己的房间里录，已压成 MP3；EchoFake 用它的 17.0 版当 bona fide。In-the-Wild（Müller 等，Interspeech 2022）从网络上收集了公众人物的真实和伪造语音。它们都按 `recorded` 输入。使用前要看许可条款，并检查说话人和文本的分布。

### 3.3 同场景二次重放（`replay`）

1. **被录的人**：在同一个房间里另抽一个说话人位置和朝向，电平和 Lombard 按本地噪声算。
2. **偷录手机**：放在这个人面前的一定距离上（`playback.replay.distance_m`，默认 0.2–1.5 m），朝向正对脸部的方向附近随机偏开（标准差 45°），高度略低于嘴，相当于手里拿着或放在桌上。
3. **偷录路径**：从人到手机的房间路径，和其他路径一样按同一个房间计算。
4. **偷录时的噪声**：同一个房间、同一种噪声工况和电平，但取另一个时刻的片段，表示同一个噪声源在录音时和回放时各响一次。RIRplay 用的也是这种做法。
5. **手机的采集链**：默认值上叠加 `playback.replay.capture`（默认 AGC 0.7、高通 0.8、降噪 0.2）。
6. **存储编码**：录音 App 或聊天软件保存时的编码（`playback.replay.codecs`，默认 AAC 0.45、Opus 0.2、AMR-NB 0.1、不编码 0.25）。
7. **回放**：直接送进设备，不再经过节目链。

`judge` 命令另有一个更简单的阳性对照，即报告里的 `control` 条件：把配对里的真人版经场景麦克风变成数字信号，在同一台设备、同一个位置再放一次。它不走形式渲染的代码，`playback` 下的参数怎么改它都不变，所以不同版本、不同形式比例下的检测器结果可以直接比较。

### 3.4 三种形式共享的播放端

**设备 DSP**：对白增强 EQ（1.5–3.5 kHz 附近提升几 dB）；虚拟低音（小喇叭放不出低频，设备生成低频的谐波，让人仍然听得到缺失的基频，Larsen & Aarts 2002，它改变了浊音的谐波结构，这是真人不会产生的）；自动音量和动态压缩；智能功放限幅。

**扬声器的线性频响**：本仓库按设备类型给出高通范围，手机 400–900 Hz，平板 250–550 Hz，笔记本 180–450 Hz，电视 90–220 Hz，条形音箱 50–110 Hz，车机 40–90 Hz；高通附近有谐振峰，手机和平板常在几 kHz 处有前腔峰，另有箱体起伏和高频滚降。这和 Void 的观察一致：廉价喇叭让累积功率谱更平、更线性，高品质音箱则向人声靠拢（附录 A.4）。也可以用同步扫频（Novak 等 2010、2015）在设备正前方 10–30 cm 实测，只保留前几毫秒以去掉测量房间（`sweep`、`extract` 命令和 `devices.measured`）。同一类设备要随机化，不能让所有电视共用一条频响。

**随音量变化的非线性**：ASVspoof 2019 PA 和 RIRplay 用广义多项式 Hammerstein 模型，`y[n] = Σ_l h_l[n] * x[n]^l`。本仓库在低通后的信号上加二次和三次项，模拟振膜位移最大的低频段，再做软削波；满幅时的总谐波失真按设备类型取，手机 3–10%，电视 1–4%，条形音箱 0.3–1.5%，车机 0.5–2%。失真由驱动电平决定：同样的听音音量，手机要开到接近最大，条形音箱还很轻松。本仓库按「目标声压级减去设备最大声压级」的余量来驱动节目，再单独校准 1 m 处的声压级，所以失真和响度可以各自变化。Klippel（2006）总结了这类非线性的来源：力因子、悬挂劲度和音圈电感都随振膜位移变化。RIRplay 的 v0.5 版本发现部分样本削波，改为在进入扬声器非线性之前先归一化输入；本仓库的驱动电平由节目的有效语音电平决定，同样不受输入电平影响。

**辐射、摆位与多单元**：电视放在墙边，60% 的单元朝下打向台面；条形音箱沿墙朝前；智能音箱靠墙，60% 朝上；笔记本放在桌面高度，喇叭朝上、朝前、朝下各有一定比例；手机和平板拿在手里，离麦克风 0.3–4 m。车机有门板低音、A 柱高音和后门同轴，分频点 2.5–4.5 kHz 两侧的频段从车内不同位置发出。指向性让直混比和早期反射随频率变化，而且和嘴的规律不同。

**本地房间、噪声、采集链**：和真人完全共享。听音音量随噪声调大，对应人在吵的时候会把电视开响：从 45 dB(A) 起，每 dB 噪声调大 0–0.4 dB（`devices.volume_slope`）。

### 3.5 两个房间叠加以后

Gaubitch 和 Looney（ICASSP 2024）把问题写成：真人只经过一条声学脉冲响应，重放经过两条的卷积。从房间声学理论出发，两条卷积有几处可测的差别：

- **频谱起伏更大。** 在 Schroeder 频率（`f_Sch ≈ 2000·√(T60/V)`）以上、声源离麦克风超过临界距离时，单条脉冲响应的对数幅度谱标准差（SSTD）约 5.6 dB，两条卷积约 8 dB。在他们比较的三个量里，这个分得最开，16 kHz 和 48 kHz 采样下都成立。
- **衰减由较长的那个混响时间主导**，表观 RT60 变长；能量包络在起始后先上升再衰减；扩散尾开始得更早；调制传递函数更低。
- **距离决定能不能看出来。** 说话人离偷录麦克风，或重放设备离收音麦克风，只要有一处在临界距离以内，那一次传播就以直达声为主，两种分布重叠。

他们用 CNN 从语音估计 SSTD 做零样本检测：ASVspoof 2019 PA 全集 EER 22.37%，大房间、短混响、近距离的子集 33.43%，混响条件有利的子集 2.24%，再加上更远的偷录距离和更差的播放设备降到 1.04%；ASVspoof 2021 PA 全集 36.28%（官方最好基线 38.07%），偷录和重放距离都最大时降到 16.39%。

本仓库在渲染时直接算每条样本声学部分的 SSTD（`source.path.sstd_db`、`source.playback.acoustic_sstd_db`，300–7000 Hz，先去掉 1/3 倍频程的包络）。对 `rerecorded` 和 `replay`，算的是第一个房间和本地房间两条路径卷积以后的结果。默认配置下的中位数和 10–90% 区间见第 7 节：真人和 `single` 都在单房间的 5 dB 左右，`replay` 6.0 dB，`rerecorded` 7.2 dB，和理论一致，而且几种形式的分布彼此重叠。

### 3.6 物理重录

仿真之外，最有说服力的是真的放、真的录：

- **EchoFake** 开源了基于 WebRTC 的自动录制工具：局域网内的设备同步播放和录音，模拟会议通话的场景。
- **ReplayDF** 为每个播放–录音组合拍了照片、测了 RIR，并用 MOS 和 PESQ 给录音质量打分。质量越低，深伪越难检（相关系数 0.42 和 0.51）。
- **Delgado 等（arXiv 2025）** 在没有声学处理的房间里，用一台监听音箱和一台便携蓝牙音箱把合成语音放进两部手机，拨入模拟的呼叫中心，得到约 7 万条只用于训练的样本；另有一套 80 名参与者、至少 16 种手机、5 种扬声器、10 个 TTS 引擎的真实电话数据集。训练里加入这些经过「呈现」的数据，比用神经声码器和编解码器做大规模增强更有效。8 kHz 下，最好的 WavLM-LLGF 在 1% 误报率时，对真实外放的检出率是 76.3%，对直接注入电话的是 88.2%。
- **LRPD、LibriWASN** 用多台设备同时录，适合检查麦克风档次（2.6）。

建议至少录一份小的真实对照集：在目标家庭和车里，同一批句子由真人说一遍，再由真实的电视、手机、音箱放一遍，覆盖几档音量和距离，并用扫频测下这些设备的频响（`sweep`、`extract`）。

### 3.7 常见错误

- 只做同场景二次重放，把它当成外放的全部。它是 ASVspoof 的第二遍录音，电视节目并不经过这一步。
- 只给外放类加麦克风之后的压缩。
- 失真量固定，不随音量变化。
- 所有设备共用一条频响，或同类设备不做随机化。
- 用 RIR 增强代替扬声器建模。ReplayDF 用它的 RIR 做增强后，重录 EER 从 18.2% 降到 11.0%，仍远高于原始文件的 4.7%；RIR 只有线性部分，不含扬声器的带限、DSP 和非线性。
- 让外放类系统性地更响或更亮。
- 让远端录音一律很远、很吵。贴着手机录的 vlog、近讲通话，远端那一层基本只是直达声。

---

## 4. 为什么三种外放用同一个标签

产品上，三种都要拒识。技术上，把它们分成三个训练类别既没有必要，也做不到可靠：

1. **`single` 和 `rerecorded` 只在远端录音有明显第二个房间和手机特征时才分得开。** 远端说话人离手机在临界距离以内（贴着手机录的 vlog、近讲通话），远端那一层几乎只有直达声，结果和单次外放差不多（3.5）。反过来，外景节目也带一点房间。
2. **`rerecorded` 和 `replay` 的差别只在于第一个房间是不是本地这间。** 从麦克风听到的信号里，看不出第一个房间是不是就是现在这间：两者都是「一个房间 + 一部手机」再加本地的播放链。
3. **文献里最难的就是这些重叠的情形。** EchoFake 的开集里，重放真人（RB）是四类中 F1 最低的一类（RawNet2 / AASIST / wav2vec2 分别只有 27.1 / 26.3 / 40.1）；ASVspoof 的重放检测器学到的是第二遍录音多出来的那一层着色，第一遍如果已经是一只好喇叭，两遍在频谱上很接近（附录 A.3）。
4. **本仓库的实测也是重叠的。** 第 7 节里，三种形式的 SSTD 中位数依次升高，但 10–90% 区间互相覆盖。

强行分成三类，模型要学的就是从信号里观察不到的东西，这些样本实际上成了标签噪声。所以三种形式共用 `device` 标签，形式只写在元数据里。要按形式评测，就用 `audit` 和 `judge` 的按形式分组结果；要调整比例，就改 `playback.forms`，或者在输入列表里逐条指定。

---

## 5. 两类共享什么，怎么防止学到捷径

| 环节 | 真人收音 | 外放重录 | 是否共享 |
| --- | --- | --- | --- |
| 声源 | 嘴：发声电平、Lombard、频谱倾斜、嘴部指向性 | 扬声器：设备 DSP、单元频响、随音量变化的非线性、指向性、多单元位置 | 不共享，标签来自这里 |
| 喇叭之前 | 没有 | 节目链；`rerecorded` 还有远端房间和远端手机，`replay` 还有偷录手机和存储编码 | 不共享 |
| 本地房间 | 同一房间，声源的位置和朝向不同 | 同一房间，各单元有自己的位置和朝向 | 共享房间，路径各自计算 |
| 噪声 | 同一段噪声，逐样本一致 | 同上 | 共享 |
| 采集链 | 麦克风频响、自噪声、高通、降噪、AGC、削波、上行编码 | 同上 | 共享 |

五条规则：

1. **只加在一类上的处理，必须有现实依据。** 播出编码只属于外放；麦克风之后的上行编码两类都有，就必须两类都加。
2. **电平用物理单位设定。** 真人按 1 m 处的 dB(A)，设备按听音音量，噪声按麦克风处的 dB(A)，然后经同一条采集链变成数字信号。不要在渲染后逐文件做峰值或响度归一化。反过来，也要随机化采集链里的 AGC 和满量程映射，防止绝对电平成为太干净的捷径。
3. **静音和时间对齐。** 数字静音要被同一噪声底填满；成对样本之间的起点和首尾长度要一致。编解码器的预读延迟按互相关对齐掉，成对渲染在语音到来之前逐样本一致（`single` 形式；录音类形式在语音之前就会放出录音里的底噪，这是它们本来的样子）。
4. **成对保存。** 每个外放样本记下它对应的真人样本和场景参数。RIRplay 称之为立体声配对；Xu 等（TASLP 2023）用成对的真人–重放数据学出设备变换，说明这种配对本身就是有用的监督。
5. **同场景二次重放不能代替其他形式。** 检测器在它上面应当比在单次外放上更容易分；如果只在它上面好，说明学到的是 ASVspoof 的第二遍。

`recorded` 输入的真人版（`passthrough`）是例外：它来自另一个房间和另一只麦克风，和它的外放版不共享场景，只按有效语音电平对齐到外放版。它的外放版里有本地的房间和噪声，而它自己没有，所以一个只由 `recorded` 输入组成的数据集会让「本地底噪多不多」变成捷径。只有在你的训练集里本来就会大量出现这类真实录音、而且同时有足够的 `dry` 配对时，才打开它。

---

## 6. 代码与参数

### 6.1 流水线

```text
输入列表（每行：路径、文本、说话人；JSONL 还可以写 origin 和 playback）
  │
  ├─ render：每条输入、每个 variant 抽一个场景，写出 live/ 和 device/ 两个文件
  ├─ scenes：多条输入拼成 10–24 秒的全双工流，带时间戳标签
  └─ mix：给定用户语音、外放素材、噪声、RIR，渲染麦克风里的一段
        │
        └─ SceneRenderer
             ├─ sample_scene：房间、麦克风、噪声、采集链
             ├─ render_talker：嘴 → 本地房间
             ├─ draw_form：按 playback.forms 或该输入自己的 playback 抽形式
             └─ render_device：设备与摆位 → [远端场景 | 偷录手机] → 节目链 → 设备 DSP → 扬声器 → 本地房间
             然后两类都经过 ambient（噪声）和 capture（采集链）
```

### 6.2 逐条控制输入

纯路径列表和 TSV（`路径<TAB>文本`、`路径<TAB>说话人<TAB>文本`）里的每一条都按全局设置处理。JSONL 的每一行可以自己说：

```json
{"audio": "tts/zh_000.wav", "text": "打开客厅的灯"}
{"audio": "tts/zh_001.wav", "text": "今天有雨", "playback": "replay"}
{"audio": "rec/user_17.wav", "text": "帮我定个闹钟", "origin": "recorded"}
{"audio": "tts/zh_002.wav", "playback": {"single": 1, "rerecorded": 2}}
```

- `origin`：`dry` 或 `recorded`，不写就用 `render.origin`（命令行 `--origin`）。
- `playback`：一个形式名、一个形式名列表，或者 `{形式: 权重}`，不写就用 `playback.forms`（命令行 `--forms`，例如 `--forms replay` 或 `--forms single=2,rerecorded=1,replay=1`）。
- `recorded` 输入只能是 `rerecorded` 形式；写了别的形式，`render` 会在开始之前报出是哪一条。
- `mix` 的用户语音列表总是真人，必须是 `dry`；外放素材列表的来源默认看 `--playback-origin`。
- `scenes` 里 `recorded` 输入永远不当用户说话，只进 `rerecorded` 形式的节目，按原样播放。

### 6.3 配置

`configs/default.yaml` 就是全部内置默认值。和本文相关的几节：

```yaml
render:
  origin: dry             # 没写来源的输入：dry | recorded
  recorded_live: skip     # recorded 输入的真人版：skip | passthrough

playback:
  forms: {single: 0.5, rerecorded: 0.3, replay: 0.2}
  single:
    contents: [broadcast, streaming, radio]
  rerecorded:
    environments: {living_room: 0.3, bedroom: 0.25, kitchen: 0.15, study: 0.2, car_cabin: 0.1}
    distance_m: {default: [0.3, 2.5], phone_call: [0.2, 1.0]}
    contents: [streaming, phone_call]
    capture: {agc_prob: 0.8, highpass_prob: 0.9, ns_prob: 0.5}
  replay:
    distance_m: [0.2, 1.5]
    capture: {agc_prob: 0.7, highpass_prob: 0.8, ns_prob: 0.2}
    codecs: {aac: 0.45, opus: 0.2, amr_nb: 0.1, none: 0.25}

room:
  measured_mode: hybrid   # 实测 IR 只提供混响尾；full 整条使用

capture:
  ns_prob: 0.0            # 你的设备在模型前面跑了降噪就打开
```

`capture` 这两个覆盖项里的键和顶层 `capture` 完全一样，写在里面的值覆盖内置默认值，不继承顶层 `capture` 的修改，因为远端手机和偷录手机不是你的设备。

### 6.4 模块

| 环节 | 实现 | 配置 |
| --- | --- | --- |
| 真人发声 | `talker.py`：1 m 处有效语音电平、Lombard、频谱倾斜、嘴部指向性 | `talker.*` |
| 房间 | `acoustics.py`：镜像源早期反射加扩散场；`environment.py`：房间尺寸、RT60、摆位、偷录手机的位置 | `environments`、`environment_presets`、`room.*` |
| 实测房间 | `irbank.py` 的 `graft_tail`、`scene.py`：模型给直达声和早期反射，实测 IR 给混响尾 | `room.measured`、`room.measured_mode` |
| 噪声 | `noise.py`：按麦克风处 dB(A)，同一场景两类逐样本相同 | `noise.*` |
| 采集链 | `capture.py`：MEMS 频响或实测 IR、自噪声、满量程、高通、降噪（`denoise.py`）、AGC、削波、上行编码 | `capture.*` |
| 外放形式 | `scene.py`：`draw_form`、远端场景、偷录手机、存储编码；设备和摆位在形式之前抽好 | `playback.*` |
| 节目链 | `content.py`：四种档案；外景小房间、替身远端底噪、通话带限、压缩、响度、FM、限幅、编码、配乐 | `content.*` |
| 设备 | `speaker.py`：对白 EQ、虚拟低音、动态压缩、限幅、振膜非线性、各单元频响；实测近场响应 | `devices.*` |
| 电平 | `dsp.py` 的 `speech_level_db`：P.56 式有效语音电平，真人和设备的校准都用它 | — |
| 输出 | `render.py`（成对片段）、`compose.py`（全双工流）、`mix.py`（整段麦克风信号）、`jobs.py`（批量执行） | `render.*`、`scene.*`、`output.*` |
| 验收 | `audit.py`、`judge.py`、`selftest.py`、`metrics.py` 的 `spectral_std` | — |

`python -m replay_opus presets --section environments`（或 `devices`、`content`）打印当前生效的预设表。

### 6.5 仍然没有覆盖的部分

- **单通道。** 麦克风阵列上的空间线索（附录 A.6）只能部分体现。要做多通道，扩散尾在不同麦克风之间的相干性需要重新建模。
- **鞋盒近似。** 家具、桌面反射、座椅遮挡由散射系数近似；有带家具的仿真 RIR 或实测 RIR 时，通过 `room.measured` 接入它们的混响尾。
- **远端噪声按环境预设取。** 录音类外放会把远端或偷录时的底噪一起放出来（第 7 节）。如果你的真实外放素材底噪更低，调低远端环境里吵的工况（`environment_presets`），或者提高远端手机的降噪概率。
- **多级外放**（节目里再套外放）没有单独生成。
- **系统自身播放的回声**（AEC 残留）不在这里。
- **家里其他人之间的对话**同样是真人，声学上无法拒识，需要语义或说话对象判断。
- **Lombard 效应只建模了电平和频谱倾斜**，没有建模基频和语速。

---

## 7. 验收

每一次转换都要回答两个问题：物理上对不对，和真实录音像不像。

1. **物理自检。** `python -m replay_opus selftest` 和 `python -m pytest tests -q` 检查的都是物理规律和两类之间的不变量：各倍频程 RT60 和 Eyring 公式相差 12% 以内，直混比随距离的变化和扩散场理论相差 2 dB 以内，设备输出电平校准误差在 0.05 dB 以内，失真随音量增大，窄带通话 4.5 kHz 以上的能量低于 −35 dB，成对样本在语音到来之前逐样本一致，三种外放形式从同一台设备、同一个位置放出，单房间和两房间串联的 SSTD 分别落在 5.6 和 8 dB 附近，降噪器按设定量压低噪声而不动语音，实测混响尾拼接后指向性和直混比仍随场景变化。
2. **捷径审计。** `python -m replay_opus audit --manifest out/pairs/manifest.jsonl` 对每个简单特征单独算区分两类的 AUC，并按环境、外放形式、输入来源分组；同时汇总每类样本声学部分的 SSTD。
3. **检测器对照。** `judge` 看同场景的配对胜率。阳性对照（二次重放比真人更像重放）必须明显高于 50%，否则检测器看不见重放；外放比真人更像重放应当显著高于 50%，但低于阳性对照；按形式分组时，带录音环节的形式应当比 `single` 更像重放。
4. **真实参照。** 用 `judge --reference real.txt` 让检测器在你自己的真实录音上打分。检测器在真实数据上分不开两类时，它对仿真数据的判断也不作数。
5. **仿真和真实的差距。** 把真实的真人录音和仿真的真人版（同类设备、同类房间）放在一起，用冻结的自监督特征加线性分类器去分「真实还是仿真」。AUC 越接近 0.5 越好；分得开时，看它依靠哪些频带或统计量，再回头调整对应的环节。外放类同样处理。
6. **质量分布。** ReplayDF 用 MOS 和 PESQ 描述录音质量；无参考的 DNSMOS（Reddy 等，ICASSP 2021）可以比较仿真和真实录音的质量分布。
7. **消融。** 像 RIRplay 那样，每次去掉一项设计（某种外放形式、虚拟低音、Lombard、AGC 等），在匹配和交叉条件下测，真实集上随之变差的才是有用的设计。
8. **按条件拆开报告。** 按设备类型、环境、距离、音量、麦克风档次、外放形式分别报告，而不是只给一个总 EER。车舱单独看。

默认配置下，12 条 demo 语音 × 20 个场景（240 对、480 个文件）的 `audit` 结果：

```text
feature             kind          AUC        live      device
level_dbfs          nuisance    0.569     -33.967     -32.126
peak_dbfs           nuisance    0.487     -15.253     -15.450
duration_s          nuisance    0.500       3.625       3.625
noise_floor_dbfs    nuisance    0.552     -53.731     -49.502
lead_s              nuisance    0.510       0.175       0.175
crest_db            acoustic    0.262      18.994      16.235
...
level AUC by playback form (vs all live): replay 0.63 (n=46), rerecorded 0.61 (n=73), single 0.52 (n=121)
acoustic spectral spread, median [10-90%] dB: device:replay 6.0 [5.2, 7.4],
    device:rerecorded 7.2 [6.1, 7.8], device:single 5.2 [4.2, 5.6], live 5.0 [4.0, 5.5]
```

电平、峰值、时长、底噪和语音起始时间这些「无关特征」都在 0.5 ± 0.2 以内。按形式看，`rerecorded` 和 `replay` 的电平 AUC 略高，原因是它们把录音时的底噪一起放了出来：麦克风处的语音电平和真人版的配对差中位数只有 −0.3 到 +1.1 dB，底噪的配对差中位数则是 `rerecorded` +1.9 dB、`replay` +1.3 dB，`single` 几乎为 0。这是物理上应有的差异，但它的大小取决于远端噪声的设定（6.5）。SSTD 的顺序和理论一致，而且几种形式的区间互相覆盖，这也是第 4 节不分开标注的依据之一。波峰因数偏低（AUC 0.26）来自节目压缩和功放限幅。

同一批样本的 `judge` 结果（完整表格见 README）：阳性对照 `control` 比真人版更像重放的同场景胜率，官方 LFCC-LCNN 是 88%，wav2vec2 是 76%；外放版比真人版更像重放的胜率分别是 70%（95% CI 64–75%）和 68%（62–74%）。按形式看，两个检测器的中位分差都是带录音环节的两种形式大于 `single`：官方 LCNN 下 `rerecorded` 和 `replay` 都是 +0.87，`single` 是 +0.59；wav2vec2 下分别是 +0.47、+0.38 和 +0.16。这和第 3 节的物理预期一致。分组后的胜率置信区间彼此重叠。

---

## 附录 A：三类信号的区分依据

这一部分讨论一个三类问题：一段语音到底是

1. **TTS 文件**（合成器直接输出，没有经过嘴，也没有经过房间里的扬声器）；
2. **真人语音**（声带发声，被不同档次的麦克风收录）；
3. **电子设备外放**（电视、手机、音箱、车机等把一段节目播出来，再被麦克风收录）。

结论先写在前面：这三类不能用同一个检测器、同一套「像不像重放」的分数分开。深伪检测器学的是合成器痕迹；ASVspoof 物理访问（PA）检测器学的是**第二次**录音；麦克风档次改变的是采集链，三类都会经过它。能分开「嘴」和「喇叭」的，是声源辐射和播放设备自己的链路，而且必须让房间、噪声和麦克风在两类之间对齐。

```text
                    ┌─ 没有房间、没有喇叭 ─→ TTS 文件（逻辑访问 / 深伪）
一段音频 ──────────┤
                    └─ 有房间、有麦克风 ─┬─ 声源是嘴 ─→ 真人（麦克风档次在这里起作用）
                                        └─ 声源是喇叭 ─→ 电子设备外放
                                              └─ 若节目本身是 TTS，合成痕迹大多已被喇叭滤掉
```

### A.1 三类信号各自留下什么

| | TTS 文件 | 真人，经麦克风 | 电子设备外放，经麦克风 |
| --- | --- | --- | --- |
| 声源 | 声码器或神经解码器 | 声门 + 声道，嘴的指向性 | 扬声器单元，常是多只、有指向、有箱体 |
| 节目处理 | 合成器内部 | 没有 | 响度、压缩、限幅、配乐、播出编解码（AAC、Opus、AMR 等） |
| 房间 | 没有，或只是合成时加的人工混响 | 一次传播：直达声 + 反射 | `single` 一次传播；`rerecorded`、`replay` 在喇叭之前还有一个房间 |
| 设备失真 | 声码器的周期性、过度平滑 | 近讲时的喷气、齿音 | 小喇叭高通、谐振峰、功放限幅、振膜非线性 |
| 麦克风 | 没有 | 有，档次决定高频、底噪、AGC | 有。若和真人用同一只麦克风，这项不能当特征 |

全双工拒识问的是后两列：同一个人机场景里，这句话是人说的，还是电视、手机放的。TTS 是造数据用的声源，不是要检出的类别。公开文献几乎都把问题收成两分类，而且两分类的定义和这张表对不齐。下面每一节都先标明论文实际在分哪两类。

### A.2 深伪检测器分的是「合成器」，过了喇叭明显变差

这一支的正类是录音（通常是演播室或众包真人），负类是 TTS / 声音转换。代表系统是 wav2vec 2.0 特征加 AASIST、RawNet2、LFCC-LCNN。它们在干净文件上可以很强，一旦声音从扬声器里放出来再录回去，合成痕迹会被喇叭的频响和房间卷积大量削弱。

**ReplayDF（Müller 等，Interspeech 2025）** 把 M-AILABS 的真人录音和 MLAAD 的合成语音，用 109 组扬声器–麦克风在六种语言上播放再录，共 132.5 小时，TTS 为 Bark、VITS、XTTS v1.1、XTTS v2。

- 公开权重里最好的 W2V2-AASIST，在原始文件上 EER 10.6%，重录后升到 24.8%。
- 在 ODSS 上重训后，原始文件 EER 4.7%，重录后 18.2%。用该数据集的房间脉冲响应做增强，重录 EER 降到 11.0%，原始文件 EER 降到约 2.2%，缺口还在。
- 掉下去的全是合成样本：各 TTS 的检出准确率下降约 28–42 个百分点。真人重录后的准确率几乎不动（约 98%）。
- 往原始文件里加高斯、白、粉红噪声（15–40 dB SNR）几乎不造成同样的下降。抹掉线索的是喇叭和房间的卷积，不是「变吵了」。
- 录音质量越差，深伪越难抓：检出准确率与 MOS 的相关系数 0.423，与 PESQ 0.509。

因此，一段音频如果已经是外放录音，深伪分数回答的是「合成痕迹还剩多少」，不是「这是不是喇叭」。

**EchoFake（Zhang、Huang、Ren，ICASSP 2026）** 把问题明确做成四类：真人（B）、重放的真人（RB）、零样本 TTS（F）、重放的 TTS（RF）。81,890 条，13,005 人，11 个 TTS，126.4 小时。闭集用 MacBook / iPad 播放、手机录，两个房间、15 cm 和 50 cm；开集换成监听音箱、小米手机和有线耳机麦，另一个房间、30 cm。全部再压到 −23 LUFS 和 64 kbps MP3。

- 只用既有深伪数据训练的模型，遇到重放音频时平均准确率掉到 59.6%。
- 在 EchoFake 上训练后，开集四分类仍然拆不开重放：RawNet2 / AASIST / wav2vec2 对 RB 的 F1 只有 27.1 / 26.3 / 40.1，对纯 TTS（F）仍有 72.8 / 79.9 / 75.6。
- 二元检测（把 RB、F、RF 都当攻击）开集 EER：wav2vec2 11.86%，AASIST 14.88%，RawNet2 21.13%。闭集可以到 0.27–3.95%，换设备就崩。
- 训练时拿掉重放样本，在常规深伪集上 EER 变化不大，加上 EchoFake 开集后平均 EER 上升 1.5–6.2 个百分点。
- 在演播室数据上训练的检测器，会把消费电子设备录的真人判成假。麦克风和房间的变化，会被当成合成痕迹。

**ASVspoof 2021 的逻辑访问与深伪轨（Yamagishi 等，2021；Liu 等，IEEE/ACM TASLP 2023）**：逻辑访问（LA）是电话信道上的 TTS / 声音转换，没有声学传播。深伪轨（DF）是媒体压缩后的合成语音。官方基线在评测集上：LA 的 LFCC-LCNN EER 9.26%；DF 的 RawNet2 22.38%、LFCC-LCNN 23.48%。DF 赛程中进度集最好可以低于 1%，评测集全部高于 15%，最好 15.64%。神经声码器比传统声码器更难抓。Liu 等对隐藏子集的分析说明，LA 上很多系统在利用句首句尾的静音。

**对三类问题的用法：** 深伪检测器只适合区分 TTS 文件和同一档麦克风、同一信道下录的真人。它不能判断外放，也不能在麦克风档次变化时保持「真人」这个标签。

### A.3 重放检测器分的是「第二遍录音」，不是「嘴对喇叭」

物理访问文献里的攻击是：攻击者先录下目标说话人，再拿到验证麦克风前播放。

**ASVspoof 2021 PA**：评测集是真实房间、真实设备。670 句 VCTK 作为种子。所谓 bona fide **不是人在说话**：种子由一只频响相当平坦的高品质扬声器播放，在 9 个房间、3 档 ASV 麦克风、6 个距离上录，共 162 个声学条件。攻击是在同样的房间里再录一次、再用另一台设备放一次。官方基线只用 ASVspoof 2019 的仿真 PA 训练：CQCC-GMM 38.07%，LFCC-GMM 39.54%，LFCC-LCNN 44.77%，RawNet2 48.60%。最好的参赛系统（T07，DKU-CMRI）EER 24.25%。Liu 等按条件拆开前十名后，房间大小和成绩没有稳定关系；更难的是攻击者的麦克风更好、扬声器更好、离说话人更近；说话人离 ASV 麦克风更远时 t-DCF 更高，RIRplay 的逐条件分析显示增加的主要是正类的误拒。也就是说，PA 检测器依赖的是第二遍多出来的那一层设备着色。

**压缩残差与声码器残差（约 22–24% EER）**：Wang 等（ASVspoof 2021 研讨班）用原频谱和声码器频谱之差做特征，后接一类异常检测，PA 评测 24.25%；Shi 等（arXiv 2023）用 WORLD 声码器重合成再经 16 kbps Opus 压缩后与原信号相减，AnoGAN 下 22.71%。残差必须在同一只麦克风上比较，否则麦克风档次会主导差值。

**RIRplay（Sanchez-Valera 等，IEEE TIFS 2026）** 用仿真把整条重放链做全：ACE Challenge 的实测倍频程 RT60（112 个环境）驱动镜像源法；攻击者和说话人在同一间房，但偷录和呈现用不同的 RIR 和距离；噪声来自同一段环境声的不同时刻，SNR 25–35 dB；扬声器用 40 只的 Hammerstein 模型；补偿直达声和扬声器谐波的延迟，VAD 去掉首尾静音；每条 spoof 指向对应的 bona fide。wav2vec 2.0 + Conformer 在扩展版上训练，ASVspoof 2021 PA 的 EER 从 36.89% 降到 28.04%，AASIST 从 37.91% 降到 29.69%，ReMASC 上两者都在 33.9% 左右。RIRplay 改善的是「仿真的二次重放 → 真实的二次重放」，它仍然不是嘴对喇叭。

**本仓库用两个现成检测器做的同场景对照**（240 对，三种外放形式按默认比例）：

| | LFCC-LCNN | wav2vec2 |
| --- | --- | --- |
| 真二次重放（`control`）比真人更像重放 | 88%，AUC 0.79 | 76%，AUC 0.69 |
| 外放比真人更像重放 | 70%，AUC 0.66 | 68%，AUC 0.63 |
| 外放−真人的中位分差，相对二次重放−真人 | +0.76 / +1.77 | +0.28 / +0.51 |
| 其中单次外放 / 录音再外放 / 同场景重放的中位分差 | +0.59 / +0.87 / +0.87 | +0.16 / +0.47 / +0.38 |

单次外放的分离约为真二次重放的三分之一，和「只经过一次喇叭」一致；带录音环节的两种形式，在 LCNN 下约为一半，在 wav2vec2 下是四分之三到九成。LCNN 在车舱最低（52%）。干净 TTS 会被 LCNN 判成极像重放（live 比 dry 更像重放的 AUC 为 0），会被 wav2vec2 判成「一带混响就像重放」（AUC 0.98）。现成 PA 检测器对外放有一点敏感，但敏感的是二次录音伪影；单次外放里它只看得到这种伪影的弱化版。

### A.4 频谱功率能分开嘴和廉价喇叭，高保真喇叭会靠拢

**Void（Ahmed 等，USENIX Security 2020）** 是轻量活体检测：97 维特征，分类器是 RBF 核 SVM，比较真人说话和扬声器重放的累积功率谱，只保留 5 kHz 以内。人声功率集中在大约 20 Hz–1 kHz，累积功率在 1 kHz 附近按指数下降；廉价喇叭的功率沿频率更均匀、更线性。高品质音箱会在很宽的频带上补功率，这时要看 2 kHz 以下的精细起伏：人声的峰又多又尖，音箱更平。自采数据上 EER 0.3%（十折交叉验证，训练和测试共享设备），ASVspoof 2017 评测集 11.6%。高品质监听音箱明显更难：Genelec 6010A 的检出率掉到 81.1%。

所以手机、平板、笔记本外放用频带边缘和低频功率形状就比较好分；电视、条形音箱居中；车机、高品质音箱和「人在同一房间里说话」会靠拢，必须靠失真、多单元、节目链。

### A.5 麦克风档次是第三轴，和声源正交

麦克风给所有经过它的声音加上同一层：高频滚降和 MEMS 谐振、底噪、AGC、高通、削波、采样率。文献里能分开的规律：攻击侧麦克风越好，二次重放越难检；验证侧麦克风越差，真人越容易被拒；深伪检测器会把「录得差」当成「假」；重放把合成痕迹洗掉的程度随录音质量下降而加重。所以要把目标设备的麦克风放进训练，并且让外放类走同一只麦克风。

### A.6 哪些线索在三类之间真正有效

**能分开 TTS 文件和任何真实采集的**：声码器痕迹（过喇叭再录后失效）、句首句尾静音模式（双方都做 VAD 后失效）、完全没有房间（合成器加了混响后失效）。

**能在同一房间、同一麦克风下分开嘴和喇叭的**（不依赖第二遍录音）：

| 线索 | 物理来源 | 对哪些设备管用 |
| --- | --- | --- |
| 低频滚降 / 高频不足 | 小口径喇叭的高通和频响 | 手机、平板、笔记本强；电视中等；车机和监听弱 |
| 低频峰的疏密 | 嘴的谐波对音箱的平滑 | 近讲、高采样率时更清楚 |
| 谐波失真、限幅 | 功放和振膜，音量越大越强 | 接近满幅的小喇叭 |
| 节目链 | 播出编解码、响度、配乐、对白处理 | 电视、流媒体、电话 |
| 多单元的空间展宽 | 低音和高音不在同一点 | 有麦克风阵列时 |
| 嘴的指向性 | 高频向前，侧向变暗 | 远场、有角度时 |

有麦克风阵列时，空间线索比频谱纹理更容易从仿真迁移到真实录音。Neri 和 Virtanen（arXiv 2025）在仿真里分别给嘴和喇叭加实测指向性，只用仿真数据训练多通道检测器，再在 ReMASC 的真实录音上测：相邻麦克风对的相位差（IPD）让户外环境的 EER 比只用波束形成输出低 7.5 个百分点（53.6% → 46.1%）；逐通道对数功率反而让真实环境变差；声源位置变化大的室内环境所有配置都在随机水平或更差，作者归因于训练里的声源只在正前方 ±10° 以内。

**只对「第二遍」有效、不能当成外放定义的**：双倍混响（偷录房间再卷积播放房间）、攻击者麦克风的着色、ASVspoof 2017 里的开头噪声和爆音。

**捷径**：静音长度；正类干净、负类有噪或反过来；训练和测试的带宽不一致。

### A.7 按产品场景怎么做

- **只判断「文件是不是 TTS」**：用 wav2vec / AASIST 这一支，训练数据要包含目标麦克风和压缩，评测时按 TTS 系统、编解码、麦克风拆开。
- **只判断「这段已经采集的声音是不是喇叭放的」**：不要用深伪检测器。配对采集或配对仿真；特征落在声源和播放链上；把麦克风档次做成共享或显式条件；阳性对照用真二次重放，但不要优化到它上面；车舱单独留真实录音。现成模型只能当弱参照：2021 PA 上最好的公开结果仍在 22–28% EER，而且正类是高品质喇叭播放的 VCTK，不是嘴。
- **三类一起要**：做成 EchoFake 那样的四格（真人内容 / TTS 内容 × 没过喇叭 / 过喇叭再录）。若产品只需要拒掉外放、放行真人，应把两格外放合成一个拒绝类，并在训练里让拒绝类包含真人节目和 TTS 节目，否则模型会用「像不像合成」去拒。

---

## 参考文献

README 中的 2020 年及以后文献：

1. M. E. Ahmed, I.-Y. Kwak, J. H. Huh, I. Kim, T. Oh, H. Kim. *Void: A fast and light voice liveness detection system*. USENIX Security, 2020.
2. J. Yamagishi et al. *ASVspoof 2021: Accelerating progress in spoofed and deepfake speech detection*. Proc. ASVspoof Workshop, 2021.
3. J. C. Sanchez-Valera, A. M. Peinado, J. M. Martin-Doñas, A. Gomez-Alanis, A. M. Gomez, M. Todisco. *RIRplay: Generation of a Replay Stereo Corpus for Voice Biometrics Anti-Spoofing*. IEEE TIFS, vol. 21, pp. 4106–4118, 2026.
4. T. Zhang, Y. Huang, Y. Ren. *EchoFake: A Replay-Aware Dataset for Practical Speech Deepfake Detection*. ICASSP 2026. arXiv:2510.19414.
5. N. Müller et al. *Replay Attacks Against Audio Deepfake Detection*. Interspeech 2025, pp. 2245–2249.

补充的直接相关工作：

6. X. Liu et al. *ASVspoof 2021: Towards spoofed and deepfake speech detection in the wild*. IEEE/ACM TASLP, vol. 31, pp. 2507–2522, 2023.
7. X. Wang et al. *The DKU-CMRI system for the ASVspoof 2021 challenge: Vocoder based replay channel response estimation*. ASVspoof Workshop, 2021.
8. X. Shi et al. *Audio compression-assisted feature extraction for voice replay attack detection*. arXiv:2310.05813, 2023.
9. J.-w. Jung et al. *AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks*. ICASSP 2022.
10. H. Tak et al. *Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation*. Odyssey 2022.
11. N. Müller, F. Dieckmann, P. Czempin, R. Canals, K. Böttinger, J. Williams. *Speech is silver, silence is golden: What do ASVspoof-trained models really learn?* ASVspoof Workshop, 2021.
12. H.-T. Luong, D.-T. Truong, K. A. Lee, E. S. Chng. *Room impulse responses help attackers to evade deepfake detection*. IEEE SLT, 2024.
13. X. Wang et al. *ASVspoof 5: Crowdsourced speech data, deepfakes, and adversarial attacks at scale*. ASVspoof Workshop, 2024.
14. M. Neri, T. Virtanen. *Acoustic Simulation Framework for Multi-channel Replay Speech Detection*. arXiv:2509.14789, 2025.
15. N. D. Gaubitch, D. Looney. *On the role of room acoustics in audio presentation attack detection*. ICASSP 2024, pp. 906–910.
16. H. Delgado, G. Ramondetti, E. Dalmasso, G. Karvitsky, D. Colibro, H. Talib. *On deepfake voice detection – It's all in the presentation*. arXiv:2509.26471, 2025.
17. L. Xu, J. Yang, C. H. You, X. Qian, D. Huang. *Device features based on linear transformation with parallel training data for replay speech detection*. IEEE/ACM TASLP, vol. 31, pp. 1574–1586, 2023.
18. X. Wang et al. *ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech*. Computer Speech & Language, vol. 64, 2020.
19. J. C. Sanchez, A. M. Peinado, A. M. Gomez. *Data augmentation techniques for Physical Access in voice anti-spoofing*. IberSPEECH 2024, pp. 1–5.
20. B. Chettri, E. Benetos, B. L. T. Sturm. *Dataset artefacts in anti-spoofing systems: A case study on the ASVspoof 2017 benchmark*. IEEE/ACM TASLP, vol. 28, pp. 3018–3028, 2020.
21. P. Jiang, Q. Wang, X. Lin, M. Zhou, W. Ding, C. Wang, C. Shen, Q. Li. *Securing liveness detection for voice authentication via pop noises*. IEEE TDSC, 2022.
22. D. Salvi, D. U. Leonzio, A. Giganti, C. Eutizi, S. Mandelli, P. Bestagini, S. Tubaro. *POLIPHONE: A dataset for smartphone model identification from audio recordings*. IEEE Access, 2025.
23. I. Yakovlev et al. *LRPD: Large replay parallel dataset*. ICASSP 2022, pp. 6612–6616.
24. R. Baumann, K. M. Malik, A. Javed, A. Ball, B. Kujawa, H. Malik. *Voice spoofing detection corpus for single and multi-order audio replays*. Computer Speech & Language, vol. 65, 2021.
25. Z. Chen et al. *Continuous speech separation: Dataset and analysis*. ICASSP 2020.
26. *Libri-adhoc40: A dataset collected from synchronized ad-hoc microphone arrays*. arXiv:2103.15118, 2021.
27. *LibriWASN*：九台设备同时录制的 LibriCSS 回放信号，https://github.com/fgnt/libriwasn 。
28. Z. Tang, R. Aralikatti, A. Ratnarajah, D. Manocha. *GWA: A large high-quality acoustic dataset for audio processing*. SIGGRAPH 2022.
29. A. Gallien, K. Prawda, S. J. Schlecht. *Matching early reflections of simulated and measured RIRs by applying sound-source directivity filters*. AES Conference, 2024.
30. J. Richter, Y. Wu, S. Krenn, S. Welker, B. Lay, S. Watanabe, A. Richard, T. Gerkmann. *EARS: An anechoic fullband speech dataset benchmarked for speech enhancement and dereverberation*. Interspeech 2024.
31. Y. Shi, H. Bu, X. Xu, S. Zhang, M. Li. *AISHELL-3: A multi-speaker Mandarin TTS corpus and the baselines*. arXiv:2010.11567, 2020.
32. R. Ardila et al. *Common Voice: A massively-multilingual speech corpus*. LREC 2020.
33. N. M. Müller, P. Czempin, F. Dieckmann, A. Froghyar, K. Böttinger. *Does audio deepfake detection generalize?* Interspeech 2022.
34. C. K. A. Reddy, V. Gopal, R. Cutler. *DNSMOS: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors*. ICASSP 2021.

经典文献、标准与工具（多数已列在 `README.md`）：ISO 9921 与 ANSI S3.5（发声力度与语音级）；ITU-T P.56（有效语音电平）；ITU-R BS.1770、EBU R 128、ATSC A/85（响度）；H. Lane, B. Tranel 1971（Lombard 效应）；W. T. Chu, A. C. C. Warnock 2002（人声指向性）；J. B. Allen, D. A. Berkley 1979（镜像源法）；M. Barron, L.-J. Lee 1988（混响能量）；Y. Ephraim, D. Malah 1984（判决引导的先验信噪比）；A. Novak, L. Simon, F. Kadlec, P. Lotton, *Nonlinear system identification using exponential swept-sine signal*, IEEE TIM, 2010，以及 Novak 等 2015（同步扫频）；W. Klippel 2006（扬声器非线性）；E. Larsen, R. M. Aarts 2002（虚拟低音）；CSTR VCTK Corpus；ACE Challenge（J. Eaton 等 2015）；MUSAN、WHAM! 噪声库。
