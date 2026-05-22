# API 参考文档

> 项目：**transcribe-engine** v0.1.0  
> 包名：`transcribe`  
> 本文档自动生成，涵盖所有公开类、方法、签名及说明。

---

## 目录

1. [核心模型 (core.models)](#1-核心模型-coremodels)
2. [抽象接口 (core.interfaces)](#2-抽象接口-coreinterfaces)
3. [引擎 (core.engine)](#3-引擎-coreengine)
4. [音频源 (source)](#4-音频源-source)
5. [转录器 (transcriber)](#5-转录器-transcriber)
6. [插件系统 (plugins)](#6-插件系统-plugins)
7. [流处理器 (stream)](#7-流处理器-stream)
8. [异步 API (api.async_api)](#8-异步-api-apiasync_api)
9. [顶层导出 (transcribe)](#9-顶层导出-transcribe)

---

## 1. 核心模型 (core.models)

**模块路径：** `transcribe.core.models`

该模块定义了引擎中使用的所有数据模型，均为纯 dataclass，字段语义清晰。

---

### `TranscriberBackend`

```python
class TranscriberBackend(str, Enum)
```

转录后端类型枚举。

| 成员 | 值 | 说明 |
|------|-----|------|
| `LOCAL` | `"local"` | 本地 Whisper 推理 |
| `CLOUD` | `"cloud"` | 云端 API 转录 |
| `HYBRID` | `"hybrid"` | 混合模式（本地回退 / 云优先） |

---

### `AudioSourceType`

```python
class AudioSourceType(str, Enum)
```

音频数据来源枚举。

| 成员 | 值 | 说明 |
|------|-----|------|
| `FILE` | `"file"` | 本地文件 |
| `MICROPHONE` | `"microphone"` | 麦克风实时输入 |
| `STREAM` | `"stream"` | 流式输入 |
| `BYTES` | `"bytes"` | 内存中的原始字节 |
| `URL` | `"url"` | 远程音频 URL |

---

### `TranscriptSegment`

```python
@dataclass
class TranscriptSegment
```

单个转录文本段，包含时序信息。

**属性：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `start` | `float` | — | 开始时间（秒） |
| `end` | `float` | — | 结束时间（秒） |
| `text` | `str` | — | 该段的转录文本 |
| `confidence` | `float` | `1.0` | 置信度分数 [0.0, 1.0] |
| `speaker` | `Optional[str]` | `None` | 说话人标签（启用说话人分离时） |
| `metadata` | `dict[str, object]` | `{}` | 插件或后端附加的任意元数据 |

**属性方法：**

```python
@property
def duration(self) -> float
```

返回该段的时长（秒），等价于 `end - start`。

---

### `TranscriptResult`

```python
@dataclass
class TranscriptResult
```

一次完整的转录操作结果。

**属性：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `text` | `str` | — | 完整的拼接转录文本 |
| `segments` | `list[TranscriptSegment]` | `[]` | 带时间戳的段列表 |
| `language` | `str` | `"en"` | 检测到或请求的语言代码 |
| `backend` | `str` | `"unknown"` | 产生该结果的后端名称 |
| `source_type` | `AudioSourceType` | `FILE` | 音频数据来源 |
| `duration_seconds` | `Optional[float]` | `None` | 总音频时长（秒） |
| `processing_time` | `Optional[float]` | `None` | 实际处理耗时（秒） |
| `metadata` | `dict[str, object]` | `{}` | 任意键值元数据 |

**方法：**

```python
def segment_count(self) -> int
```

返回段的数量，等价于 `len(self.segments)`。

---

### `EngineConfig`

```python
@dataclass
class EngineConfig
```

引擎的全局配置。

**属性：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `backend` | `TranscriberBackend` | `LOCAL` | 首选后端类型 |
| `model_name` | `str` | `"base"` | 模型名称/大小 |
| `device` | `str` | `"cpu"` | Torch 设备（`"cpu"`, `"cuda"`, `"mps"`） |
| `compute_type` | `str` | `"float32"` | 精度（`"float16"`, `"int8"`, `"float32"`） |
| `language` | `Optional[str]` | `None` | 默认语言提示（`None` = 自动检测） |
| `sample_rate` | `int` | `16000` | 目标采样率（Hz） |
| `chunk_seconds` | `float` | `0.5` | 流式分块时长（秒） |
| `max_retries` | `int` | `3` | 云 API 瞬态故障重试次数 |
| `timeout_seconds` | `int` | `60` | 云 API 请求超时（秒） |

**类方法：**

```python
@classmethod
def local_defaults(cls) -> EngineConfig
```

返回针对本地 Whisper 推理调优的配置（`LOCAL` / `cpu` / `float32`）。

```python
@classmethod
def cloud_defaults(cls) -> EngineConfig
```

返回针对云 API 使用调优的配置（`CLOUD` / `"whisper-1"` / 120秒超时）。

---

## 2. 抽象接口 (core.interfaces)

**模块路径：** `transcribe.core.interfaces`

定义了引擎的插件契约——所有转录器、音频源、流处理器和插件都必须实现其中的抽象基类。

---

### `AudioSource`

```python
class AudioSource(abc.ABC)
```

音频数据的抽象来源。实现类可以封装文件、麦克风流、内存字节或远程 URL。

**抽象属性：**

```python
@property
@abc.abstractmethod
def source_type(self) -> AudioSourceType
```

返回当前音频源的来源类型。

**抽象方法：**

```python
@abc.abstractmethod
async def read(self) -> bytes
```

返回完整的音频数据（原始 PCM 16-bit 单声道 16 kHz）。用于批量/一次性转录。

```python
@abc.abstractmethod
async def stream(self) -> AsyncIterator[bytes]
```

逐块产出原始 PCM 数据，用于实时流式转录。每块应为 `EngineConfig.chunk_seconds` 时长的音频。

```python
@abc.abstractmethod
async def close(self) -> None
```

释放资源（文件句柄、麦克风、网络连接等）。

---

### `Transcriber`

```python
class Transcriber(abc.ABC)
```

核心转录引擎插件。子类实现 `transcribe()`（批量）和/或 `transcribe_stream()`（实时）。

**抽象方法：**

```python
@abc.abstractmethod
async def initialize(self, config: EngineConfig) -> None
```

一次性初始化：加载模型权重、打开 API 会话等。由引擎在转录前调用一次。

```python
@abc.abstractmethod
async def transcribe(self, audio: bytes) -> TranscriptResult
```

转录完整的音频缓冲区（批量模式）。

- **参数：** `audio` — 原始 PCM 16-bit 单声道音频（配置的采样率）
- **返回：** 包含完整转录文本和时序段的 `TranscriptResult`

```python
@abc.abstractmethod
async def shutdown(self) -> None
```

清理：卸载模型、关闭连接、释放 GPU 内存。

**方法（可覆盖）：**

```python
async def transcribe_stream(
    self, stream: AsyncIterator[bytes]
) -> AsyncIterator[TranscriptResult]
```

转录实时音频流。默认实现缓冲所有块并调用 `transcribe()`；覆盖以实现真正的流式行为（如 Whisper 的流式模式）。

**属性：**

```python
@property
def backend_name(self) -> str
```

人类可读的后端标识符，默认返回类名。

---

### `StreamHandler`

```python
class StreamHandler(abc.ABC)
```

实时音频流水线中的处理阶段，位于音频源和转录器之间，执行 VAD、降噪、重采样或缓冲。

**抽象方法：**

```python
@abc.abstractmethod
async def process(self, chunk: bytes) -> bytes
```

处理单个音频块。返回处理后的块，或返回空的 `b""` 表示丢弃该块（如静音/非语音）。

```python
@abc.abstractmethod
async def reset(self) -> None
```

重置内部状态（例如在话语之间）。

---

### `TranscriptionPlugin`

```python
class TranscriptionPlugin(abc.ABC)
```

扩展转录流水线的基类。插件可以：

- 在 **转录前** 修改音频（预处理）
- 在 **转录后** 修改/过滤结果（后处理）
- 向结果添加自定义 **元数据**
- 使用 NLP 丰富结果（实体提取、翻译等）

**抽象属性：**

```python
@property
@abc.abstractmethod
def name(self) -> str
```

唯一插件名称（用于排序和日志记录）。

**方法（可覆盖）：**

```python
async def setup(self, config: EngineConfig, plugin_config: dict | None = None) -> None
```

插件加载时调用一次。

```python
async def pre_process(self, audio: bytes) -> bytes
```

在转录前修改原始音频。返回修改后的音频或原样返回。

```python
async def post_process(self, result: TranscriptResult) -> TranscriptResult
```

在转录后修改/丰富转录结果。典型用途：语法纠正、翻译、关键词高亮。

```python
async def teardown(self) -> None
```

引擎关闭时调用。

---

### `TranscriptionEvent`

```python
@dataclass
class TranscriptionEvent
```

转录生命周期中发出的事件。

**属性：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | `str` | 事件类型（如 `"segment"`, `"final"`, `"error"`） |
| `engine` | `TranscriptionEngine` | 发出事件的引擎引用 |
| `data` | `Optional[object]` | 可选负载（`TranscriptResult`, `Exception` 等） |

---

### `CompositeTranscriber`

```python
class CompositeTranscriber(Transcriber)
```

将多个后端组合并合并结果的转录器。支持 **回退**（先尝试云端，失败后回退到本地）、**集成**（合并多个后端的输出）。

**策略选项：**

- `"first"` — 返回第一个成功的结果
- `"best"` — 返回置信度最高的结果
- `"merge"` — 拼接所有后端的全部段

**构造方法：**

```python
def __init__(
    self,
    transcribers: list[Transcriber],
    merge_strategy: str = "first",
)
```

- `transcribers` — 要组合的转录器列表
- `merge_strategy` — 合并策略（`"first"`, `"best"`, `"merge"`）

**方法实现：**

```python
async def initialize(self, config: EngineConfig) -> None
```

```python
async def transcribe(self, audio: bytes) -> TranscriptResult
```

```python
async def transcribe_stream(
    self, stream: AsyncIterator[bytes]
) -> AsyncIterator[TranscriptResult]
```

```python
async def shutdown(self) -> None
```

```python
@property
def backend_name(self) -> str
```

返回所有后端名称用 `"+"` 拼接的结果。

---

## 3. 引擎 (core.engine)

**模块路径：** `transcribe.core.engine`

---

### `TranscriptionEngine`

```python
class TranscriptionEngine
```

转录工作负载的高级编排器。将转录器、音频源、流处理器和插件串联为完整的转录流水线。

**类型别名：**

```python
EventHandler = Callable[[TranscriptionEvent], Coroutine[object, None, None]]
```

**构造方法：**

```python
def __init__(
    self,
    transcriber: Transcriber,
    config: EngineConfig | None = None,
    stream_handlers: list[StreamHandler] | None = None,
    plugins: list[TranscriptionPlugin] | None = None,
)
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `transcriber` | `Transcriber` | — | 核心 Transcriber 实现 |
| `config` | `EngineConfig \| None` | `None` | 引擎配置（默认 `EngineConfig()`） |
| `stream_handlers` | `list[StreamHandler] \| None` | `None` | 可选的流预处理器列表 |
| `plugins` | `list[TranscriptionPlugin] \| None` | `None` | 可选的流水线插件列表 |

**生命周期方法：**

```python
async def initialize(self) -> None
```

加载转录器和设置插件。幂等——如果已初始化则跳过。

```python
async def shutdown(self) -> None
```

释放所有资源。按逆序调用插件的 `teardown()`，然后调用转录器的 `shutdown()`。

**事件系统：**

```python
def on(self, event_type: str, handler: EventHandler) -> None
```

订阅引擎事件。

- `event_type` — 可选值：`"initialized"`, `"started"`, `"segment"`, `"final"`, `"stopped"`, `"error"`
- `handler` — 异步可调用对象 `async def handler(event: TranscriptionEvent)`

```python
def off(self, event_type: str, handler: EventHandler) -> None
```

取消订阅事件处理器。

**批量转录：**

```python
async def transcribe(
    self,
    source: AudioSource,
    language: str | None = None,
) -> TranscriptResult
```

转录完整的音频源（批量/文件模式）。

- `source` — 音频源实现
- `language` — 可选的本次调用语言覆盖

处理流程：
1. 读取完整音频 → 2. 插件 `pre_process` → 3. 转录器转录 → 4. 插件 `post_process` → 5. 发射 `"final"` 事件 → 6. 关闭源

**流式转录：**

```python
async def transcribe_stream(
    self,
    source: AudioSource,
    language: str | None = None,
) -> AsyncIterator[TranscriptResult]
```

转录实时音频流。产出 `TranscriptResult` 对象作为部分段被识别。

处理流程：
1. 获取 `source.stream()` → 2. 应用 `StreamHandler` 链 → 3. 转录器逐段产出 → 4. 插件 `post_process` → 5. 发射 `"segment"` 事件 → 6. 产出

**便捷方法：**

```python
async def transcribe_bytes(
    self,
    audio_bytes: bytes,
    language: str | None = None,
) -> TranscriptResult
```

直接转录原始 PCM 音频字节。内部将字节包装为 `BytesSource`。

**属性：**

```python
@property
def config(self) -> EngineConfig
```

当前引擎配置。

```python
@property
def backend_name(self) -> str
```

活动转录后端的名称。

---

## 4. 音频源 (source)

**包路径：** `transcribe.source`

---

### `FileSource`

```python
class FileSource(AudioSource)
```

从本地文件读取的音频源。支持 WAV、FLAC、MP3、M4A、OGG 等格式。文件在首次调用 `read()` 或 `stream()` 时解码为 **原始 PCM 16-bit 单声道 16 kHz**。

**构造方法：**

```python
def __init__(
    self,
    path: str | os.PathLike[str],
    config: EngineConfig | None = None,
)
```

- `path` — 音频文件路径
- `config` — 可选的引擎配置（默认 `EngineConfig()`）

**属性：**

```python
@property
def source_type(self) -> AudioSourceType
```

返回 `AudioSourceType.FILE`。

```python
@property
def path(self) -> str
```

构造时提供的原始文件路径。

**AudioSource 接口实现：**

```python
async def read(self) -> bytes
```

```python
async def stream(self) -> AsyncIterator[bytes]
```

```python
async def close(self) -> None
```

---

### `BytesSource`

```python
class BytesSource(AudioSource)
```

包装内存中原始 PCM 字节的音频源。调用者需确保数据格式正确（16-bit 单声道 PCM），不会执行重采样或格式转换。

**构造方法：**

```python
def __init__(
    self,
    data: bytes,
    sample_rate: int = 16000,
    chunk_seconds: float = 0.5,
)
```

- `data` — 原始 PCM 音频数据（16-bit 有符号小端序单声道）
- `sample_rate` — 数据采样率（默认 16000）
- `chunk_seconds` — `stream()` 产出的每块时长（默认 0.5）

**属性：**

```python
@property
def source_type(self) -> AudioSourceType
```

返回 `AudioSourceType.BYTES`。

```python
@property
def data(self) -> bytes
```

包装的原始 PCM 字节。

```python
@property
def sample_rate(self) -> int
```

音频数据的采样率。

```python
@property
def duration_seconds(self) -> float
```

音频总时长（秒）。

**AudioSource 接口实现：**

```python
async def read(self) -> bytes
```

```python
async def stream(self) -> AsyncIterator[bytes]
```

```python
async def close(self) -> None
```

---

### `MicrophoneSource`

```python
class MicrophoneSource(AudioSource)
```

从默认麦克风捕获的音频源。使用 `sounddevice`（可选依赖，安装 `[stream]` extra）。

`read()` 方法持续捕获直到调用 `stop()` 或达到超时；`stream()` 方法无限产出块直到调用 `close()`。

**构造方法：**

```python
def __init__(
    self,
    config: EngineConfig | None = None,
    device: int | str | None = None,
    blocksize: int | None = None,
)
```

- `config` — 引擎配置（提供 `sample_rate` 和 `chunk_seconds`）
- `device` — 可选的 sounddevice 设备索引或子串；`None` 使用默认输入设备
- `blocksize` — 每音频回调块的帧数；默认 `chunk_seconds * sample_rate`

**属性：**

```python
@property
def source_type(self) -> AudioSourceType
```

返回 `AudioSourceType.MICROPHONE`。

**AudioSource 接口实现：**

```python
async def read(self) -> bytes
```

捕获所有麦克风输入直到调用 `stop()`。若无显式停止信号，将无限阻塞。

```python
async def stream(self) -> AsyncIterator[bytes]
```

实时产出麦克风音频块。迭代持续直到调用 `close()` 或流发生错误。

```python
async def close(self) -> None
```

停止麦克风流并释放资源。

**控制方法：**

```python
def stop(self) -> None
```

信号麦克风停止捕获。**线程安全的同步调用**——可在 UI 回调、信号处理器等中安全调用。

---

## 5. 转录器 (transcriber)

**包路径：** `transcribe.transcriber`

---

### `BaseTranscriber`

```python
class BaseTranscriber(Transcriber)
```

所有转录后端的共享抽象基类。实现了配置存储、生命周期日志、语言覆盖辅助和模型信息属性等样板代码。

**子类必须实现：**
- `transcribe()`
- `shutdown()`

**子类应覆盖：**
- `initialize()`（先调用 `super().initialize(config)`）
- `transcribe_stream()`（默认缓冲所有音频）

**构造方法：**

```python
def __init__(self, model_name: str = "base")
```

**生命周期：**

```python
async def initialize(self, config: EngineConfig) -> None
```

存储引擎配置并设置初始化标志。

```python
async def shutdown(self) -> None
```

标记为未初始化。

**抽象方法（由子类实现）：**

```python
@abc.abstractmethod
async def transcribe(self, audio: bytes) -> TranscriptResult
```

**流式方法（可覆盖）：**

```python
async def transcribe_stream(
    self, stream: AsyncIterator[bytes]
) -> AsyncIterator[TranscriptResult]
```

默认实现：缓冲所有块，然后调用 `transcribe()`。

**属性：**

```python
@property
def backend_name(self) -> str
```

```python
@property
def model_name(self) -> str
```

配置的模型标识符（如 `"base"`, `"whisper-1"`）。

```python
@property
def config(self) -> EngineConfig
```

当前引擎配置（仅在初始化后有效）。

**辅助方法：**

```python
def _assert_initialized(self) -> None
```

若 `initialize()` 从未被调用，抛出 `RuntimeError`。

```python
def _language_param(self) -> str | None
```

从配置中返回语言设置（若已设置）。

---

### `LocalWhisperTranscriber`

```python
class LocalWhisperTranscriber(BaseTranscriber)
```

使用本地 `faster-whisper` 模型进行转录的后端。支持批量（`transcribe`）和流式（`transcribe_stream`）模式。

**构造方法：**

```python
def __init__(self, model_name: str = "base")
```

- `model_name` — 模型大小或路径：`"tiny"`, `"base"`, `"small"`, `"medium"`, `"large-v3"` 或本地 `.bin` 路径

**方法：**

```python
async def initialize(self, config: EngineConfig) -> None
```

将 Whisper 模型加载到内存。在后台线程中运行 `faster-whisper` 的阻塞构造函数。

```python
async def transcribe(self, audio: bytes) -> TranscriptResult
```

使用本地 Whisper 模型转录完整音频缓冲区。内部将 int16 字节转换为 float32 numpy 数组，在线程池中执行推理。

```python
async def shutdown(self) -> None
```

卸载模型并释放 GPU 内存。

**依赖：** 需要安装 `faster-whisper`。

---

### `CloudAPITranscriber`

```python
class CloudAPITranscriber(BaseTranscriber)
```

使用云端语音转文本 API 的转录后端。默认使用 OpenAI 的 `whisper-1` 端点，可配置为任何兼容 OpenAI 的 API（如 Azure OpenAI、本地代理）。

**常量：**

```python
DEFAULT_API_URL = "https://api.openai.com/v1/audio/transcriptions"
DEFAULT_MODEL = "whisper-1"
```

**构造方法：**

```python
def __init__(
    self,
    model_name: str = DEFAULT_MODEL,
    api_key: str | None = None,
    api_url: str | None = None,
    language: str | None = None,
)
```

- `model_name` — API 模型名称（如 `"whisper-1"`）
- `api_key` — API 密钥；回退到 `OPENAI_API_KEY` 环境变量
- `api_url` — 转录端点的完整 URL
- `language` — 默认语言提示（`None` = 自动检测）

**方法：**

```python
async def initialize(self, config: EngineConfig) -> None
```

创建 `aiohttp` 客户端会话。

```python
async def transcribe(self, audio: bytes) -> TranscriptResult
```

将音频发送到云 API 进行转录。将原始 PCM 转换为 WAV 格式后，通过 multipart 表单请求发送。

```python
async def transcribe_stream(
    self, stream: AsyncIterator[bytes]
) -> AsyncIterator[TranscriptResult]
```

流式模式——默认实现缓冲所有块并发送单个请求。如果云提供商支持实时流式传输（如基于 WebSocket 的转录），请覆盖此方法。

```python
async def shutdown(self) -> None
```

关闭 HTTP 会话。

**依赖：** 需要安装 `aiohttp`。

---

## 6. 插件系统 (plugins)

**包路径：** `transcribe.plugins`

---

### `AbstractPlugin`

```python
class AbstractPlugin(TranscriptionPlugin)
```

转录流水线插件的便捷基类。添加了 **基于优先级的排序**（优先级值越低执行越早）。

**类属性：**

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `priority` | `int` | `100` | 执行顺序（0 = 最先，越大越晚） |
| `enabled` | `bool` | `True` | `False` 时跳过插件钩子 |

**抽象属性：**

```python
@property
@abc.abstractmethod
def name(self) -> str
```

唯一的、人类可读的插件名称（如 `"vad-filter"`）。

**生命周期方法：**

```python
async def setup(self, config: EngineConfig, plugin_config: dict | None = None) -> None
```

引擎初始化时调用一次。覆盖以执行一次性设置（加载模型、打开连接、读取配置）。

```python
async def teardown(self) -> None
```

引擎关闭时调用。覆盖以释放资源、关闭网络会话等。

**预处理/后处理钩子：**

```python
async def pre_process(self, audio: bytes) -> bytes
```

在 **转录前** 修改或检查原始音频。返回 `b""` 表示丢弃该音频（插件级 VAD）。

```python
async def post_process(self, result: TranscriptResult) -> TranscriptResult
```

在 **转录后** 修改、过滤或丰富 `TranscriptResult`。典型用途：语法纠正、实体提取、翻译、关键词高亮、置信度重评分。

**错误处理：**

```python
async def on_error(self, exception: Exception, context: dict[str, Any] | None = None) -> None
```

转录过程中发生错误时调用。插件可以记录日志、增加指标或执行回退逻辑。

---

### `PluginManager`

```python
class PluginManager
```

插件的发现、加载和生命周期管理器。使用 Python `importlib.metadata` 入口点（`transcribe.plugins` 组）。

**常量：**

```python
ENTRY_POINT_GROUP = "transcribe.plugins"
```

`importlib.metadata` 入口点组名。

**构造方法：**

```python
def __init__(self)
```

初始化空的插件列表。

**发现：**

```python
def discover(self, group: str = ENTRY_POINT_GROUP) -> list[AbstractPlugin]
```

扫描 `transcribe.plugins` 入口点组并实例化所有注册的插件。

- **返回：** 按优先级排序（升序）的插件列表
- **抛出：** `TypeError` — 入口点未解析为 `AbstractPlugin` 子类

```python
def register(self, plugin: AbstractPlugin) -> None
```

手动注册插件实例（绕过入口点发现）。按名称去重，同名时替换。

**生命周期：**

```python
async def setup_all(self, config: EngineConfig) -> None
```

按优先级顺序在每个已注册插件上调用 `setup()`。单个插件的失败不会中止其余插件。

```python
async def teardown_all(self) -> None
```

按逆序在每个活动插件上调用 `teardown()`。

**钩子执行：**

```python
async def run_pre_process(self, audio: bytes) -> bytes
```

按优先级顺序在每个已启用插件上运行 `pre_process`。一个插件的输出是下一个插件的输入。若某插件返回空字节，则流水线短路并立即返回 `b""`。

```python
async def run_post_process(
    self, result: TranscriptResult
) -> TranscriptResult
```

按优先级顺序在每个已启用插件上运行 `post_process`。

**访问器：**

```python
@property
def plugins(self) -> list[AbstractPlugin]
```

所有已注册的插件，按优先级排序（升序）。

```python
@property
def active_plugins(self) -> list[AbstractPlugin]
```

仅已启用的插件，按优先级排序（升序）。

```python
def get(self, name: str) -> AbstractPlugin | None
```

按 `name` 查找插件，未找到时返回 `None`。

```python
def count(self, *, only_enabled: bool = False) -> int
```

已注册（或已启用）的插件数量。

**特殊方法：**

```python
def __len__(self) -> int
```

```python
def __iter__(self)
```

```python
def __getitem__(self, index: int) -> AbstractPlugin
```

---

## 7. 流处理器 (stream)

**包路径：** `transcribe.stream`

---

### `VADStreamHandler`

```python
class VADStreamHandler(StreamHandler)
```

语音活动检测 (VAD) 流处理器。检查每个传入音频块是否包含语音。非语音（静音）块通过返回 `b""` 丢弃；语音块原样通过。

默认使用轻量级的 **基于能量**（RMS）检测器，适用于干净的近讲麦克风音频。对于嘈杂环境或更高精度，传入外部 `detector` 可调用对象（如 `webrtcvad` 或 `Silero VAD`）。

**构造方法：**

```python
def __init__(
    self,
    threshold: float = 0.02,
    sample_rate: int = 16000,
    sample_width: int = 2,
    frame_duration_ms: int = 30,
    detector: Callable[[bytes], bool] | None = None,
)
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `threshold` | `float` | `0.02` | RMS 能量阈值，仅无外部检测器时使用 |
| `sample_rate` | `int` | `16000` | 音频采样率（Hz） |
| `sample_width` | `int` | `2` | 每样本字节数（16-bit PCM） |
| `frame_duration_ms` | `int` | `30` | 子帧分析使用的帧时长（毫秒） |
| `detector` | `Callable[[bytes], bool] \| None` | `None` | 外部 VAD 可调用对象 |

**属性：**

```python
@property
def threshold(self) -> float
```

RMS 能量阈值。可设置（setter 会验证正值）。

```python
@property
def silence_frame_count(self) -> int
```

自上次语音以来的连续静默帧数。

**StreamHandler 接口：**

```python
async def process(self, chunk: bytes) -> bytes
```

若块包含语音则返回该块，否则返回 `b""`。

```python
async def reset(self) -> None
```

重置静音计数器。

---

### `BufferingStreamHandler`

```python
class BufferingStreamHandler(StreamHandler)
```

音频块累加器，具有可配置的清空条件。收集传入的音频块到内部缓冲区，当满足清空条件时发出累积的缓冲区。在清空前返回 `b""`，避免将不完整的话语转发给转录器。

**清空条件（任意一个触发）：**

1. **最大时长** — `max_chunk_seconds` 的音频累积（安全阀）
2. **静音间隔** — `silence_threshold_seconds` 的连续静音（在自然话语边界提前清空）
3. **最短时长** — 不短于 `min_chunk_seconds`（除非使用强制方法）

**构造方法：**

```python
def __init__(
    self,
    min_chunk_seconds: float = 2.0,
    max_chunk_seconds: float = 30.0,
    silence_threshold_seconds: float = 0.8,
    sample_rate: int = 16000,
    sample_width: int = 2,
)
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `min_chunk_seconds` | `float` | `2.0` | 允许清空的最小时长 |
| `max_chunk_seconds` | `float` | `30.0` | 触发强制清空的最大时长 |
| `silence_threshold_seconds` | `float` | `0.8` | 触发提前清空的连续静音时长（设为 0 禁用） |
| `sample_rate` | `int` | `16000` | 音频采样率（Hz） |
| `sample_width` | `int` | `2` | 每样本字节数 |

**属性：**

```python
@property
def buffer_duration(self) -> float
```

当前缓冲区中的音频时长（秒）。

```python
@property
def buffered_bytes(self) -> int
```

当前缓冲的字节数。

```python
@property
def is_flush_pending(self) -> bool
```

`True` 若累积的缓冲区达到清空阈值。

**StreamHandler 接口：**

```python
async def process(self, chunk: bytes) -> bytes
```

累加块，在满足清空条件前返回 `b""`，满足时返回累积的缓冲区。

```python
async def reset(self) -> None
```

丢弃当前缓冲区并重置静音跟踪。

**公共辅助方法：**

```python
async def flush(self) -> bytes
```

显式请求清空当前缓冲区。返回缓冲的音频（可能为空）并重置内部状态。适用于从外部信号表示话语结束。

---

## 8. 异步 API (api.async_api)

**包路径：** `transcribe.api.async_api`

---

### `JobStatus`

```python
class JobStatus(str, Enum)
```

转录作业的状态枚举。

| 成员 | 值 | 说明 |
|------|-----|------|
| `PENDING` | `"pending"` | 等待执行 |
| `RUNNING` | `"running"` | 正在执行 |
| `COMPLETED` | `"completed"` | 已完成 |
| `FAILED` | `"failed"` | 失败 |
| `CANCELLED` | `"cancelled"` | 已取消 |

---

### `TranscriptionJob`

```python
@dataclass
class TranscriptionJob
```

提交到 API 的单个转录作业描述符。

**属性：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `job_id` | `str` | — | 唯一作业标识符（自动生成） |
| `source` | `AudioSource` | — | 要转录的音频源 |
| `language` | `Optional[str]` | `None` | 可选的本次作业语言覆盖 |
| `status` | `JobStatus` | `PENDING` | 当前作业状态 |
| `result` | `Optional[TranscriptResult]` | `None` | 最终的转录结果（完成时填充） |
| `error` | `Optional[str]` | `None` | 作业失败时的异常消息 |
| `created_at` | `float` | `0.0` | 作业入队的时间戳（单调时钟） |
| `completed_at` | `Optional[float]` | `None` | 作业完成（或失败）的时间戳 |
| `metadata` | `dict[str, object]` | `{}` | 用户提供的任意元数据 |

---

### `AsyncTranscriptionAPI`

```python
class AsyncTranscriptionAPI
```

用于并发转录工作负载的高级异步包装器。

**功能：**

- **单次转录** — 转录音频源并立即获取结果
- **批处理排队** — 提交多个作业并等待结果
- **进度回调** — 每作业完成时接收回调
- **取消** — 按作业 ID 取消待定/运行中的作业
- **引擎生命周期** — 引擎惰性初始化，API 退出时自动关闭

**构造方法：**

```python
def __init__(
    self,
    transcriber: Transcriber,
    config: EngineConfig | None = None,
    *,
    max_concurrent: int = 4,
    progress_callback: ProgressCallback | None = None,
)
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `transcriber` | `Transcriber` | — | 核心 Transcriber 实现 |
| `config` | `EngineConfig \| None` | `None` | 引擎配置 |
| `max_concurrent` | `int` | `4` | 最大并发作业数 |
| `progress_callback` | `ProgressCallback \| None` | `None` | 每作业完成/失败/取消时调用的异步回调 |

**类型别名：**

```python
ProgressCallback = Callable[
    [TranscriptionJob],
    Coroutine[object, None, None],
]
```

**生命周期：**

```python
async def initialize(self) -> None
```

初始化底层引擎（幂等）。

```python
async def shutdown(self) -> None
```

关闭引擎并释放所有资源。取消所有待定/运行中的作业。

```python
async def __aenter__(self) -> AsyncTranscriptionAPI
```

```python
async def __aexit__(self, exc_type, exc_val, exc_tb) -> None
```

异步上下文管理器支持。

**单次转录：**

```python
async def transcribe(
    self,
    source: AudioSource,
    language: str | None = None,
    metadata: dict[str, object] | None = None,
) -> TranscriptResult
```

转录单个音频源并立即返回结果。是 `submit_job()` + `await_job()` 的便捷包装。

**批处理转录：**

```python
async def transcribe_batch(
    self,
    sources: list[AudioSource],
    language: str | None = None,
    metadata: dict[str, object] | None = None,
) -> list[TranscriptionJob]
```

提交多个源进行并发转录。返回按提交顺序排列的 `TranscriptionJob` 列表。

**作业管理：**

```python
async def submit_job(
    self,
    source: AudioSource,
    *,
    language: str | None = None,
    metadata: dict[str, object] | None = None,
) -> TranscriptionJob
```

提交单个作业并开始处理（受并发限制）。作业在有并发槽位可用时立即开始处理。

```python
async def await_job(
    self, job_id: str, *, timeout: float | None = None
) -> TranscriptResult
```

等待特定作业完成并返回其结果。

- **抛出：** `KeyError`（未知 ID）、`TimeoutError`（超时）、`RuntimeError`（失败/取消）

```python
async def await_all_jobs(
    self, *, timeout: float | None = None
) -> list[TranscriptionJob]
```

等待所有当前跟踪的作业完成。

- **抛出：** `TimeoutError`

```python
def cancel_job(self, job_id: str) -> None
```

按 ID 取消待定或运行中的作业。实际运行的转录调用 **不会** 被中断，但作业结果将被丢弃，状态设为 `CANCELLED`。

- **抛出：** `KeyError`

```python
def list_jobs(
    self, status: JobStatus | None = None
) -> list[TranscriptionJob]
```

返回所有跟踪的作业，可选的按状态过滤。

```python
def get_job(self, job_id: str) -> TranscriptionJob
```

按 ID 获取作业描述符。

- **抛出：** `KeyError`

**属性：**

```python
@property
def engine(self) -> TranscriptionEngine
```

访问底层引擎（高级用法）。

```python
@property
def max_concurrent(self) -> int
```

允许的最大并发作业数。

```python
@property
def pending_count(self) -> int
```

当前待定或运行中的作业数。

```python
@property
def is_initialized(self) -> bool
```

API 是否已初始化。

---

## 9. 顶层导出 (transcribe)

**包路径：** `transcribe`

顶层包导出了所有主要公开类和函数，方便用户直接从包根导入。

```python
from transcribe import (
    Transcriber,            # 来自 core.interfaces
    AudioSource,            # 来自 core.interfaces
    StreamHandler,          # 来自 core.interfaces
    TranscriptionPlugin,    # 来自 core.interfaces
    TranscriptResult,       # 来自 core.models
    TranscriptSegment,      # 来自 core.models
    EngineConfig,           # 来自 core.models
    TranscriptionEngine,    # 来自 core.engine
    AsyncTranscriptionAPI,  # 来自 api.async_api
)
```

**`__all__` 定义：**

```python
__all__ = [
    "Transcriber",
    "AudioSource",
    "StreamHandler",
    "TranscriptionPlugin",
    "TranscriptResult",
    "TranscriptSegment",
    "EngineConfig",
    "TranscriptionEngine",
    "AsyncTranscriptionAPI",
]
```

---

## 附录：快速参考

### 典型使用流程

```
1. 选择并实例化一个 Transcriber（LocalWhisperTranscriber / CloudAPITranscriber）
2. 可选：创建 EngineConfig 自定义设置
3. 创建 TranscriptionEngine(transcriber=..., config=...)
4. await engine.initialize()
5. 创建音频源（FileSource / MicrophoneSource / BytesSource）
6. await engine.transcribe(source)        # 批量模式
   — 或 —
   async for result in engine.transcribe_stream(source):  # 流式模式
7. await engine.shutdown()
```

### 插件链执行顺序

```
对每个已启用插件（按 priority 升序）：
    audio = await plugin.pre_process(audio)
→ 转录器转录
→ 对每个已启用插件（按 priority 升序）：
    result = await plugin.post_process(result)
```

### 流处理管道

```
AudioSource.stream()
    → StreamHandler[0].process()
    → StreamHandler[1].process()
    → ... → Transcriber.transcribe_stream()
    → plugin.post_process()
    → yield TranscriptResult
```
