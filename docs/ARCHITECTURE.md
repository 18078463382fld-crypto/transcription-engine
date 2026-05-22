# 架构概览 — transcribe-engine

> 一个高扩展性、插件驱动的语音转文本引擎，支持本地 Whisper 和云端 API。

---

## 目录

1. [设计目标](#1-设计目标)
2. [组件架构图](#2-组件架构图)
3. [核心模块说明](#3-核心模块说明)
4. [数据流：批处理模式](#4-数据流批处理模式)
5. [数据流：流式/实时模式](#5-数据流流式实时模式)
6. [插件系统](#6-插件系统)
7. [事件系统](#7-事件系统)
8. [CompositeTranscriber — 多后端组合](#8-compositetranscriber--多后端组合)
9. [包目录结构](#9-包目录结构)
10. [依赖关系](#10-依赖关系)

---

## 1. 设计目标

- **后端无关**：同一套接口适用于本地 `faster-whisper` 和云端 OpenAI API（任何 OpenAI 兼容端点）。
- **流式优先**：原生支持基于 `AsyncIterator` 的实时音频流处理。
- **插件驱动**：通过 `pre_process` / `post_process` 钩子实现非侵入式扩展。
- **异步全栈**：从数据读取到转录到回调，全程 `async/await`。
- **安全回退**：`CompositeTranscriber` 支持主备切换（cloud → local）或多后端融合。

---

## 2. 组件架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         transcribe-engine                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────────┐     ┌───────────────┐   │
│  │  AudioSource  │────▶│TranscriptionEngine│────▶│  Transcriber  │   │
│  │  (输入抽象)    │     │   (编排器)        │     │  (转录后端)    │   │
│  └──────────────┘     └──────┬───────────┘     └───────────────┘   │
│        │                     │                                       │
│        │             ┌───────┴────────┐                              │
│        │             │  StreamHandler  │ (可选, 流预处理链)           │
│        │             │  [VAD/缓存/...] │                              │
│        │             └───────┬────────┘                              │
│        │                     │                                       │
│        │             ┌───────┴────────┐                              │
│        │             │TranscriptionPlugin                            │
│        │             │ (pre/post hooks)│                             │
│        │             └────────────────┘                              │
│        │                                                            │
│  ┌─────┴──────────┐    ┌───────────────┐                            │
│  │ FileSource     │    │PluginManager  │  (发现/注册/生命周期)       │
│  │ MicrophoneSrc  │    └───────────────┘                            │
│  │ BytesSource    │                                                 │
│  └────────────────┘    ┌──────────────────┐                        │
│                        │AsyncTranscriptionAPI│                     │
│                        │(批处理并发/任务管理)│                       │
│                        └──────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 顶层调用路径

```
客户端代码
    │
    ▼
AsyncTranscriptionAPI   ─── 任务队列、并发控制、进度回调
    │
    ▼
TranscriptionEngine    ─── 生命周期管理、事件调度、插件编排
    │
    ├── AudioSource     ─── 音频数据适配器（文件/麦克风/字节）
    ├── StreamHandler   ─── 流预处理链（VAD / 缓冲）
    ├── Transcriber     ─── 转录后端（本地 Whisper / 云端 API）
    └── PluginManager   ─── 插件发现、注册、hook 执行
```

---

## 3. 核心模块说明

### 3.1 `core/interfaces.py` — 抽象契约

定义了引擎的全部 **抽象基类**，所有组件都必须实现其中之一：

| 接口 | 职责 | 关键方法 |
|---|---|---|
| `AudioSource` | 音频输入适配器 | `read()`, `stream()`, `close()` |
| `Transcriber` | 转录后端 | `initialize()`, `transcribe()`, `transcribe_stream()`, `shutdown()` |
| `StreamHandler` | 流预处理单元 | `process()`, `reset()` |
| `TranscriptionPlugin` | 管道插件 | `setup()`, `pre_process()`, `post_process()`, `teardown()` |
| `CompositeTranscriber` | 多后端组合器 | 合并多个 `Transcriber` 的结果 |

### 3.2 `core/models.py` — 数据模型

| 模型 | 用途 |
|---|---|
| `TranscriptSegment` | 单段文本 + 起止时间 + 置信度 + 说话人标签 |
| `TranscriptResult` | 完整转录结果（全文 + 段列表 + 元数据） |
| `EngineConfig` | 全局配置（后端类型、模型名、设备、采样率等） |
| `TranscriberBackend` | 枚举：LOCAL / CLOUD / HYBRID |
| `AudioSourceType` | 枚举：FILE / MICROPHONE / STREAM / BYTES / URL |

### 3.3 `core/engine.py` — 引擎编排器

`TranscriptionEngine` 是所有转录任务的入口。职责：

- 生命周期管理：`initialize()` → `shutdown()`
- 批处理：`transcribe(source)` — 读取 → 预处理 → 转录 → 后处理
- 流处理：`transcribe_stream(source)` — 流式读取 → StreamHandler 链 → 流式转录
- 事件发布：`on()` / `off()` 订阅，`_emit()` 通知
- 语言覆盖：每次调用可临时覆盖 `config.language`

### 3.4 `source/` — 音频源实现

| 实现 | 输入来源 | 备注 |
|---|---|---|
| `FileSource` | 本地文件 | 使用 `soundfile` + `pydub` 解码多种格式 |
| `MicrophoneSource` | 麦克风 | 使用 `sounddevice`，线程安全，支持 `stop()` |
| `BytesSource` | 内存字节 | 纯 PCM 输入快捷方式 |

### 3.5 `transcriber/` — 转录后端

| 实现 | 技术栈 | 特点 |
|---|---|---|
| `LocalWhisperTranscriber` | `faster-whisper` + torch | 支持 CPU/CUDA/MPS，各种量化精度 |
| `CloudAPITranscriber` | `aiohttp` | 兼容 OpenAI 格式 API，自动重试，WAV 封装 |
| `BaseTranscriber` | 抽象基类 | 通用初始化、配置保存、断言检查 |

### 3.6 `stream/` — 流处理器

| 实现 | 功能 |
|---|---|
| `VADStreamHandler` | 基于 RMS 能量的语音活动检测，支持外挂检测器 |
| `BufferingStreamHandler` | 按时长/静音间隔累积缓冲区，到达阈值时释放 |

---

## 4. 数据流：批处理模式

```
用户调用 engine.transcribe(FileSource("speech.mp3"))
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ① AudioSource.read()                                              │
│    FileSource ─→ 解码文件 ─→ PCM int16 mono @ 16kHz ─→ bytes     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ audio (bytes)
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ② Plugin[0].pre_process(audio)   │ Plugin[1].pre_process(audio)   │
│    ├─ VAD 过滤 (返回 b"" 则终止)   │   ├─ 降噪                       │
│    └─ 输出修改后的音频             │   └─ 输出修改后的音频           │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ processed_audio (bytes)
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ③ Transcriber.transcribe(processed_audio)                         │
│    ├─ LocalWhisperTranscriber:                                     │
│    │   numpy → faster-whisper → 段列表                              │
│    └─ CloudAPITranscriber:                                         │
│        PCM→WAV → multipart POST → 解析 verbose_json                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ result (TranscriptResult)
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ④ Plugin[0].post_process(result)  │ Plugin[1].post_process(result) │
│    ├─ 语法纠正 / 翻译               │   ├─ 实体提取                   │
│    └─ 输出修改后的 result           │   └─ 输出修改后的 result        │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ enriched_result
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ⑤ engine._emit("final", result)  →  订阅者收到通知                  │
│ ⑥ AudioSource.close()            →  释放资源                        │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    返回 TranscriptResult
```

---

## 5. 数据流：流式/实时模式

```
用户调用 engine.transcribe_stream(MicrophoneSource())
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ① AudioSource.stream() → AsyncIterator[bytes]                     │
│    ├─ FileSource:        将 PCM 分块发送                             │
│    └─ MicrophoneSource:  实时音频回调 → asyncio.Queue               │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ chunk (bytes)
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ② StreamHandler 链 (engine._wrap_handler)                          │
│    ┌────────────────┐    ┌────────────────────┐                     │
│    │ VADHandler     │───▶│ BufferingHandler   │                    │
│    │ 丢弃静音帧      │    │ 累积到缓冲区       │                    │
│    │ 返回 b"" 或原始  │    │ 到达阈值才释放     │                    │
│    └────────────────┘    └─────────┬──────────┘                     │
└───────────────────────────┬────────┼────────────────────────────────┘
                            │        │
                            │  只有当缓冲区就绪时
                            ▼        ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ③ Transcriber.transcribe_stream(processed_stream)                 │
│    ├─ LocalWhisperTranscriber:                                     │
│    │   (默认实现：缓存全部 → 单次 transcription)                     │
│    └─ CloudAPITranscriber:                                         │
│       (通过 super() 回退到缓冲模式)                                  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ result (TranscriptResult)
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ④ Plugin.post_process(result)  (与批处理相同)                       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ enriched_result
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ⑤ engine._emit("segment", result) → 订阅者收到部分结果              │
│ ⑥ yield result 给客户端                                            │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
                 循环到 ①，直到 stream 耗尽
```

**关键差异**：

| 维度 | 批处理 | 流处理 |
|---|---|---|
| 输入 | `AudioSource.read()` → 完整 bytes | `AudioSource.stream()` → `AsyncIterator[bytes]` |
| 预处理 | `Plugin.pre_process` 仅一次 | `StreamHandler` 链逐块处理 + `Plugin.pre_process` |
| 转录 | 一次 `transcribe()` 调用 | 多次 `transcribe_stream()` yield |
| 输出 | 单一 `TranscriptResult` | 多个增量 `TranscriptResult` |
| 事件 | 仅 `"final"` | 每段 `"segment"` + 最后 `"stopped"` |

---

## 6. 插件系统

### 6.1 架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Plugin 生命周期                           │
│                                                              │
│  discovery              setup              teardown          │
│  ──────────    ──────────────────    ───────────────────     │
│  PluginManager │  Plugin.setup()      │  Plugin.teardown()  │
│  .discover()   │  加载模型/配置       │  释放资源            │
│  .register()   └─────────────────    └──────────────────     │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────────────────────────────────────────┐        │
│  │          Hook 执行链                              │        │
│  │                                                   │        │
│  │  pre_process 链 (按 priority 升序):               │        │
│  │    audio_in → Plugin[A] → Plugin[B] → audio_out  │        │
│  │                                                   │        │
│  │  post_process 链 (按 priority 升序):              │        │
│  │    result_in → Plugin[A] → Plugin[B] → result_out │        │
│  └──────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 插件接口

所有插件必须继承 `AbstractPlugin`（位于 `plugins/base.py`）：

```python
class AbstractPlugin(TranscriptionPlugin):
    name: str          # 必填，唯一名称
    priority: int = 100  # 优先级，越小越早执行
    enabled: bool = True # 是否启用

    async def setup(self, config: EngineConfig, plugin_config: dict | None = None)
    async def pre_process(self, audio: bytes) -> bytes
    async def post_process(self, result: TranscriptResult) -> TranscriptResult
    async def teardown(self)
    async def on_error(self, exception: Exception, context: dict | None = None)
```

### 6.3 插件发现机制

使用 Python 标准库 `importlib.metadata` 的 **entry points** 机制：

**声明方式**（第三方包的 `pyproject.toml`）：

```toml
[project.entry-points."transcribe.plugins"]
my_vad = "my_package.plugins:VADFilterPlugin"
my_translator = "my_package.plugins:TranslatePlugin"
```

**代码注册方式**：

```python
manager = PluginManager()
manager.register(MyCustomPlugin())
```

### 6.4 PluginManager 职责

| 方法 | 功能 |
|---|---|
| `discover()` | 扫描 entry points 并实例化所有插件 |
| `register(plugin)` | 手动注册插件实例（按名称去重） |
| `setup_all(config)` | 依次调用所有插件的 `setup()` |
| `teardown_all()` | 逆序调用所有插件的 `teardown()` |
| `run_pre_process(audio)` | 链式执行 `pre_process`（短路机制） |
| `run_post_process(result)` | 链式执行 `post_process` |
| `active_plugins` | 返回 `enabled=True` 的插件列表（按 priority 排序） |

### 6.5 优先级排序

插件按 `(priority, name)` 升序排列。建议优先级范围：

| 优先级区间 | 用途示例 |
|---|---|
| 0–49 | 音频预处理（降噪、重采样、VAD） |
| 50–99 | 特征提取（说话人分离、语速检测） |
| 100 (默认) | 通用后处理 |
| 101–200 | 高级分析（实体提取、翻译、情感分析） |

### 6.6 预处理器短路机制

若 `pre_process` 返回 `b""`，整个管道立即中止，后续插件和转录器都不会执行。适用于：

- VAD 过滤器检测到纯静音
- 内容策略阻止特定音频
- 自定义条件预检失败

### 6.7 典型的插件使用场景

```
输入 PCM 音频
    │
    ▼
┌───────────────────────────────────────────────────────────┐
│ ① NoiseReductionPlugin      priority=10                  │
│    └─ 使用 noisereduce 库降噪                              │
├───────────────────────────────────────────────────────────┤
│ ② VADFilterPlugin           priority=20                  │
│    └─ 基于 Silero VAD 丢弃静音帧                           │
├───────────────────────────────────────────────────────────┤
│ ── 转录器 ──────────────────────────────────────────────────│
├───────────────────────────────────────────────────────────┤
│ ③ GrammarCorrectionPlugin   priority=100                 │
│    └─ 用 LLM 对转录文本做语法修正                           │
├───────────────────────────────────────────────────────────┤
│ ④ TranslatePlugin           priority=150                 │
│    └─ 将英文转录结果翻译成中文                               │
├───────────────────────────────────────────────────────────┤
│ ⑤ EntityExtractionPlugin    priority=200                 │
│    └─ 从文本中提取人名/地名/时间                             │
└───────────────────────────────────────────────────────────┘
```

---

## 7. 事件系统

引擎内置一个轻量级事件总线，用于向外部观察者通知转录生命周期事件。

### 7.1 事件类型

| 事件名 | 触发时机 | data 内容 |
|---|---|---|
| `"initialized"` | `engine.initialize()` 完成 | `None` |
| `"started"` | 转录开始 | `AudioSource` 实例 |
| `"segment"` | 流式模式下每产出一个片段 | `TranscriptResult`（部分） |
| `"final"` | 批处理模式转录完成 | `TranscriptResult`（完整） |
| `"stopped"` | 引擎/流停止 | `None` |
| `"error"` | 转录过程中发生异常 | `Exception` 对象 |

### 7.2 使用示例

```python
engine = TranscriptionEngine(transcriber=...)

async def on_segment(event: TranscriptionEvent):
    result: TranscriptResult = event.data
    print(f"[{result.language}] {result.text}")

engine.on("segment", on_segment)

# 取消订阅
engine.off("segment", on_segment)
```

---

## 8. CompositeTranscriber — 多后端组合

`CompositeTranscriber` 将多个 `Transcriber` 实例包装为单一后端，支持三种合并策略：

### 策略

| 策略 | 行为 | 适用场景 |
|---|---|---|
| `"first"` (默认) | 返回第一个成功的转录结果 | 主备切换（cloud → local） |
| `"best"` | 选择置信度最高的结果 | 多模型投票/质量优先 |
| `"merge"` | 拼接所有后端的文本和段列表 | 实验性融合 |

### 示例

```python
transcribers = [
    CloudAPITranscriber(api_key="sk-..."),
    LocalWhisperTranscriber("base"),
]

engine = TranscriptionEngine(
    transcriber=CompositeTranscriber(
        transcribers=transcribers,
        merge_strategy="first",   # 先试云端，失败则用本地
    ),
)
```

---

## 9. 包目录结构

```
src/transcribe/
├── __init__.py                    # 包入口，导出主要类和函数
├── core/
│   ├── __init__.py
│   ├── engine.py                  # TranscriptionEngine 编排器
│   ├── interfaces.py              # 抽象接口定义（所有 ABC）
│   └── models.py                  # 数据模型（dataclasses + enums）
├── source/
│   ├── __init__.py                # 导出 FileSource, BytesSource, MicrophoneSource
│   ├── file_source.py             # 本地文件音频源
│   ├── bytes_source.py            # 内存字节音频源
│   └── microphone_source.py       # 麦克风音频源
├── transcriber/
│   ├── __init__.py
│   ├── base.py                    # BaseTranscriber 抽象基类
│   ├── local_whisper.py           # LocalWhisperTranscriber
│   └── cloud_api.py               # CloudAPITranscriber
├── stream/
│   ├── __init__.py
│   └── stream_processor.py        # VADStreamHandler, BufferingStreamHandler
├── plugins/
│   ├── __init__.py
│   ├── base.py                    # AbstractPlugin 基类
│   └── plugin_manager.py          # PluginManager 发现/生命周期管理
└── api/
    ├── __init__.py
    └── async_api.py               # AsyncTranscriptionAPI 高层并发 API

docs/
├── ARCHITECTURE.md                # 本文档
└── ...                            # 其他文档
```

---

## 10. 依赖关系

### 运行时依赖

| 包 | 用途 | 必需 |
|---|---|---|
| `numpy` | 音频数据处理 | ✅ |
| `soundfile` | WAV/FLAC 解码 | ✅ |
| `pydub` | MP3/M4A/OGG 解码回退 | ✅ |
| `faster-whisper` | 本地 Whisper 推理 | 可选 (`[local]`) |
| `torch` | PyTorch 后端 | 可选 (`[local]`) |
| `openai` / `httpx` | 云端 API 调用 | 可选 (`[cloud]`) |
| `sounddevice` / `pyaudio` | 麦克风输入 | 可选 (`[stream]`) |

### 安装快捷方式

```bash
pip install transcribe-engine        # 仅核心（文件转录）
pip install transcribe-engine[local] # + 本地 Whisper
pip install transcribe-engine[cloud] # + 云端 API
pip install transcribe-engine[stream]# + 实时麦克风
pip install transcribe-engine[full]  # 全部功能
```

---

> **文档维护者**：Hermes Agent  
> **项目仓库**：`D:/projects/transcription-engine`  
> **许可证**：MIT
