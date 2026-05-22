# transcribe-engine

> 高扩展性、插件驱动的语音转文本引擎，支持本地 Whisper 和云端 API。

**transcribe-engine** 是一个全异步的语音转文本（STT）引擎，提供统一的编程接口，支持批处理文件转录、实时麦克风流式转录、云 API 调用，并通过强大的插件系统实现无侵入式扩展。基于 `async/await` 全程异步，从音频读取到转录再到回调，一气呵成。

---

## 功能特性

- **🎯 双后端支持** — 本地 `faster-whisper` 推理（CPU/CUDA/MPS）或云端 OpenAI Whisper API（兼容任何 OpenAI 端点）
- **⚡ 实时流式转录** — 通过 `AsyncIterator` 原生支持麦克风实时输入，逐段返回转录结果
- **📦 批处理转录** — 支持 WAV、FLAC、MP3、M4A、OGG 等多种音频格式，一次调用完成转录
- **🔌 插件系统** — 基于 `pre_process` / `post_process` 钩子，支持 entry points 自动发现和优先级排序
- **🔀 多后端组合** — `CompositeTranscriber` 支持主备切换（cloud → local）或多模型融合
- **📡 事件系统** — 轻量级内置事件总线，订阅 `segment`、`final`、`error` 等生命周期事件
- **🚀 并发批处理 API** — `AsyncTranscriptionAPI` 提供任务队列、并发控制、进度回调和取消功能
- **🎤 多种音频源** — 文件、麦克风、内存字节流，统一 `AudioSource` 抽象接口
- **🔊 流预处理链** — 内置 VAD 语音活动检测和缓冲处理器，可自由组合

---

## 快速安装

### 基础安装（仅核心，支持文件转录）

```bash
pip install transcribe-engine
```

### 按需安装扩展

```bash
# 本地 Whisper 支持
pip install transcribe-engine[local]

# 云端 API 支持（OpenAI 等）
pip install transcribe-engine[cloud]

# 实时麦克风流支持
pip install transcribe-engine[stream]

# 全部功能
pip install transcribe-engine[full]
```

### 从源码安装

```bash
git clone <repo-url>
cd transcription-engine
pip install -e ".[full]"
```

---

## 快速上手

### 批处理模式 — 转录音频文件

```python
import asyncio
from transcribe import TranscriptionEngine
from transcribe.source import FileSource
from transcribe.transcriber import LocalWhisperTranscriber

async def main():
    engine = TranscriptionEngine(
        transcriber=LocalWhisperTranscriber("base"),
    )
    await engine.initialize()

    result = await engine.transcribe(FileSource("meeting.mp3"))

    print(f"语言: {result.language}")
    print(f"全文: {result.text}")
    print(f"耗时: {result.processing_time:.2f}s")
    print(f"段落数: {len(result.segments)}")

    for seg in result.segments:
        print(f"  [{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")

    await engine.shutdown()

asyncio.run(main())
```

### 实时模式 — 麦克风流式转录

```python
import asyncio
from transcribe import TranscriptionEngine
from transcribe.source import MicrophoneSource
from transcribe.transcriber import LocalWhisperTranscriber
from transcribe.stream import VADStreamHandler, BufferingStreamHandler

async def main():
    engine = TranscriptionEngine(
        transcriber=LocalWhisperTranscriber("base"),
        stream_handlers=[
            VADStreamHandler(threshold=0.02),
            BufferingStreamHandler(
                min_chunk_seconds=2.0,
                max_chunk_seconds=30.0,
                silence_threshold_seconds=0.8,
            ),
        ],
    )
    await engine.initialize()

    print("正在聆听... 按 Ctrl+C 停止")
    async for result in engine.transcribe_stream(MicrophoneSource()):
        print(f"[实时] {result.text}")

    await engine.shutdown()

asyncio.run(main())
```

### 云端 API 模式 — OpenAI Whisper

```python
import asyncio
from transcribe import TranscriptionEngine
from transcribe.source import FileSource
from transcribe.transcriber import CloudAPITranscriber

async def main():
    engine = TranscriptionEngine(
        transcriber=CloudAPITranscriber(
            api_key="sk-xxx",          # 或设置 OPENAI_API_KEY 环境变量
            model_name="whisper-1",
        ),
    )
    await engine.initialize()

    result = await engine.transcribe(FileSource("speech.mp3"))
    print(result.text)

    await engine.shutdown()

asyncio.run(main())
```

### 并发批处理 API

```python
import asyncio
from transcribe import AsyncTranscriptionAPI
from transcribe.source import FileSource
from transcribe.transcriber import LocalWhisperTranscriber

async def on_progress(job):
    print(f"任务 {job.job_id}: {job.status.value}")

async def main():
    async with AsyncTranscriptionAPI(
        transcriber=LocalWhisperTranscriber("base"),
        max_concurrent=4,
        progress_callback=on_progress,
    ) as api:
        files = [
            FileSource("part1.mp3"),
            FileSource("part2.mp3"),
            FileSource("part3.mp3"),
        ]
        jobs = await api.transcribe_batch(files)

        for job in jobs:
            print(f"{job.job_id}: {job.result.text[:50]}...")

asyncio.run(main())
```

---

## 插件示例

插件通过继承 `AbstractPlugin` 实现，可挂接到转录管道的预处理和后处理阶段。

### 编写插件

```python
from transcribe.plugins.base import AbstractPlugin
from transcribe.core.models import TranscriptResult, EngineConfig

class TimestampFormatterPlugin(AbstractPlugin):
    """为每段转录文本添加时间戳前缀。"""

    @property
    def name(self) -> str:
        return "timestamp-formatter"

    priority = 50   # 在后处理阶段较早执行

    async def post_process(self, result: TranscriptResult) -> TranscriptResult:
        formatted = []
        for seg in result.segments:
            formatted.append(f"[{seg.start:.1f}s] {seg.text}")
        result.text = "\n".join(formatted)
        return result
```

### 使用插件

```python
from transcribe import TranscriptionEngine
from transcribe.source import FileSource
from transcribe.transcriber import LocalWhisperTranscriber

engine = TranscriptionEngine(
    transcriber=LocalWhisperTranscriber("base"),
    plugins=[TimestampFormatterPlugin()],
)
await engine.initialize()
result = await engine.transcribe(FileSource("speech.mp3"))
print(result.text)
# 输出：
# [0.0s] 大家好，欢迎参加今天的会议
# [3.5s] 首先我们来讨论上个季度的业绩
# [8.2s] ...
```

### 通过 Entry Points 自动发现

第三方包可以在 `pyproject.toml` 中注册插件：

```toml
[project.entry-points."transcribe.plugins"]
my_vad = "my_package.plugins:VADFilterPlugin"
```

引擎的 `PluginManager` 会自动扫描并加载它们。

---

## 项目结构

```
src/transcribe/
├── __init__.py                     # 包入口，导出主要类和函数
├── core/
│   ├── engine.py                   # TranscriptionEngine 编排器
│   ├── interfaces.py               # 抽象接口定义（所有 ABC）
│   └── models.py                   # 数据模型（dataclasses + enums）
├── source/
│   ├── file_source.py              # 本地文件音频源
│   ├── bytes_source.py             # 内存字节音频源
│   └── microphone_source.py        # 麦克风音频源
├── transcriber/
│   ├── base.py                     # BaseTranscriber 抽象基类
│   ├── local_whisper.py            # LocalWhisperTranscriber
│   └── cloud_api.py                # CloudAPITranscriber
├── stream/
│   └── stream_processor.py         # VADStreamHandler, BufferingStreamHandler
├── plugins/
│   ├── base.py                     # AbstractPlugin 基类
│   └── plugin_manager.py           # PluginManager 发现/生命周期管理
├── api/
│   └── async_api.py                # AsyncTranscriptionAPI 高层并发 API
└── utils/
    ├── audio.py                    # 音频格式转换工具
    └── logging.py                  # 日志配置工具

docs/
├── ARCHITECTURE.md                 # 架构文档
└── API.md                          # API 参考文档
```

---

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

Copyright (c) 2025 Hermes Agent / Nous Research

---

*由 Hermes Agent 自动生成*
