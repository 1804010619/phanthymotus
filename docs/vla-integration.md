# VLA 接入方案

状态：设计草案，未实现。撰写日期 2026-09-16。

本文给出把 VLA（Vision-Language-Action）策略接入 PhanthyMotus 的设计：卡片底座、本地/云双 provider、云端接口契约、以及本地模型选型。读之前请先读 `actucore/README.md`（卡片契约）和 `perception/README.md` 的「Plugin Concurrency」（并发规范）。

---

## 0. 结论摘要

| 问题 | 结论 |
|---|---|
| VLA 放哪 | `actucore/`，这一层就是为它建的，目前 `plugins/` 为空 |
| VLM 放哪 | 产出语义的放 `perception/`（同 `vop.py`）；产出动作的算 VLA |
| 一张卡片能否通吃各家模型 | 能，但必须拆成 **卡片（不变） + provider（每家一个 adapter）** |
| 端上能不能跑 π0 / UnifoLM | **不能**。实测 Orin NX 8GB 总内存 7 GB、跑着 perception 时仅余 ~3 GB；π0 类 3.3B 模型 bf16 光权重就 6.6 GB |
| 本地能跑什么 | SmolVLA-450M 级别。需实测，公开资料没有 Orin NX 的数字 |
| 云/边缘怎么放 | **局域网边缘服务器**是主路径；公网云只用于开发、评测、数据回流 |
| 最大的架构缺口 | ActuCore → Driver 的**动作通路目前是断的**，且没有指令仲裁和 watchdog |

---

## 1. 现状与缺口

### 1.1 已经具备的

- `actucore/` 骨架：MCP HTTP 15730、容器 `embodied-actucore`、`/opt/embodied/models` 挂载、注册与探活全通
- 卡片契约（duck typing，无基类）：`PREFIX` / `get_tools()` / `dispatch()`，`action.enum` 必须含 `info`
- ACP：`x-completion` 完成回调、`x-resource` 物理通道互斥（`mcp_client.py` 的 `parse_resources` / `resources_conflict`）
- 打断：`x-hooks` 的 `on_interrupt_*`（`hooks.py`，`event/llm.py` 里 `hooks.fire('on_interrupt_all')`）
- 授权面：画布连线即授权（`canvas_binding.py`）
- MCP 工具可返回 image content，`mcp_client.py` 会转成 `image_url` 交给主 LLM（`_trim` 只保留最近 5 张）

### 1.2 必须补的（按依赖顺序）

| # | 缺口 | 说明 |
|---|---|---|
| 1 | **ActuCore → Driver 的动作通路** | `control/velocity` / `control/joint` 只出现在 README 和 `mcp_manage.py` 的格式推断表里，**没有任何驱动订阅它们**。驱动里唯一以 topic 为输入的是 speaker（`topic_in: audio/pcm-16k`） |
| 2 | **`control/*` 的消息 schema** | 目前只有 format 字符串，没有字段、单位、关节顺序的约定 |
| 3 | **指令仲裁** | VLA 闭环在 ROS 上持续发布时，LLM 仍可直接调 `loco` / `arm`。`x-resource` 只在 agent-core 的 dispatch 路径生效，自跑的 ROS 节点不经过 barrier |
| 4 | **Watchdog / e-stop** | 远端推理断连、卡片崩溃、进程被 OOM kill 时谁停机器人 |
| 5 | **ACP 进度通道** | 只有 `/api/acp/complete`，没有中途进度。长任务期间 LLM 完全不知道状态 |
| 6 | **观测时间同步** | 现有 QoS 是 BEST_EFFORT / depth 2，没有多路相机 + 本体状态的对齐机制 |
| 7 | **`control/*` 渲染器** | `web/js/renderers/` 里没有，动作会落到默认 KV 面板 |
| 8 | **数据录制** | 观测-动作对无处存储，数据闭环无从谈起 |

**缺口 1–4 是安全相关的，必须在第一个 VLA 卡片上线前完成。** 5–8 可以后补。

---

## 2. 底座设计

### 2.1 两条正交的轴

换模型和换机器人是两件独立的事。混在一个配置里，每个「模型 × 机器人」组合都要改一次代码。

```
                  ┌──────────────────────────────┐
                  │   VLAPlugin（卡片，不变）      │
                  │  MCP 契约 / ROS 收发 / 生命周期 │
                  │  watchdog / 资源声明 / e-stop  │
                  └───────┬──────────────┬────────┘
                          │              │
              backend 轴  │              │  embodiment 轴
              （换模型）   │              │  （换机器人）
                          ▼              ▼
              ┌───────────────────┐  ┌──────────────────────┐
              │ VLAProvider       │  │ EmbodimentProfile    │
              │  local / remote   │  │  关节顺序、单位、limit │
              │  openpi / lerobot │  │  EEF vs joint、夹爪   │
              │  unifolm / openvla│  │  控制频率、相机对应    │
              └───────────────────┘  └──────────────────────┘
```

`EmbodimentProfile` 应当**从驱动的 `resource` 工具拉取**（URDF 已是现成锚点，`sensor/skeleton` 的 `model` 工具返回它），而不是写死在 actucore 镜像里。

### 2.2 卡片骨架

```python
# actucore/plugins/vla.py
#
# PREFIX 不能含下划线 —— dispatch 用 full_name.partition("_") 拆前缀。

class VLAPlugin:
    PREFIX = "vla"

    TOOLS = [{
        "name": "vla",
        "type": "processor",          # processor/actuator 都会过 ACP barrier
        "multiInstance": False,
        "description": "语言指令驱动的端到端操作策略",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string",
                           "enum": ["start", "stop", "info", "config"]},   # info 必须有，否则永远离线
                "task":   {"type": "string", "description": "自然语言指令"},
            },
            "required": ["action"],
            "x-completion": {"actions": ["start"], "timeout": 120},
            "x-resource":   ["arm_l", "arm_r"],      # 见 2.4
            "x-hooks":      {"on_interrupt_all": {"action": "stop"}},
        },
        "configSchema": {
            "type": "object",
            "properties": {
                # ── backend 轴 ────────────────────────────────────────
                "provider":    {"enum": ["local", "openpi_ws", "lerobot_async",
                                         "unifolm_http", "openvla_http"],
                                "scope": "shared"},
                "endpoint":    {"type": "string", "scope": "shared"},
                "model_dir":   {"type": "string", "default": "/models/vla",
                                "scope": "shared"},
                "unnorm_key":  {"type": "string", "scope": "shared"},   # 无默认值，见 2.5
                "fallback":    {"enum": ["none", "local"], "default": "none",
                                "scope": "shared"},
                # ── embodiment 轴 ─────────────────────────────────────
                "embodiment":      {"type": "string", "scope": "instance"},
                "control_rate_hz": {"type": "number", "default": 30, "scope": "instance"},
                "watchdog_ms":     {"type": "number", "default": 200, "scope": "instance"},
                "max_joint_delta": {"type": "number", "scope": "instance"},  # 单步限幅
            },
            "required": ["provider", "embodiment"],
        },
        "topic_in":  [{"format": "video/mjpeg", "desc": "主视角"},
                      {"format": "state/joint", "desc": "本体状态"}],
        "topic_out": [{"format": "control/joint", "desc": "关节指令"}],
    }]

    def get_tools(self) -> list[dict]: ...
    def dispatch(self, name, args) -> dict | None:
        ...   # 必须返回 plain dict，不要返回 [{"type": "text", ...}]
```

### 2.3 并发（actucore 和 perception 一样是 `ThreadingHTTPServer`）

`start` / `stop` / `config` 会并发到达同一个插件实例。规则和 `perception/README.md` 完全一致，这里只列最要命的四条：

1. per-instance 状态字典的读-改-写用 `threading.RLock` 保护
2. **不要**在持锁状态下调 `node.start()` / `node.stop()` / 加载模型 —— 否则 `stop` 排在 `start` 后面，**再也停不掉一个正在驱动电机的控制回路**
3. 节点**先注册再启动**，让并发的 `stop` 能找到它
4. `stop` 先非阻塞 `request_stop()` 置位，**再**去拿锁

对普通 perception 卡片这是"节点泄漏"，对 VLA 这是"停不下来的机器人"。

### 2.4 `x-resource` 必须声明

`parse_resources` 对未声明返回 `None`，语义是**与一切互斥**。VLA 卡片若不声明，一跑起来 TTS 讲解、导航全被 barrier 挡住，机器人连话都说不了。按实际占用的物理通道声明（`arm_l` / `arm_r` / `base` / `waist`），不要图省事写一个 `robot`。

### 2.5 会静默出错的配置必须设为必填

`unnorm_key`（OpenVLA / UnifoLM 系列都有）选错不会报错，只会让**动作尺度全错**。在一个真的驱动电机的系统里，这类配置：

- 不给默认值，`required` 里必须有
- `start` 时向 provider 查询可用值并校验，对不上直接拒绝启动
- `info` 要把当前生效值回显出来

---

## 3. Provider：本地与云统一

### 3.1 接口

```python
class VLAProvider(Protocol):
    def capabilities(self) -> dict:
        """握手。返回 {model, action_dim, chunk_size, control_hz,
        needs_state, n_cameras, image_size, supports_rtc, unnorm_keys}。
        卡片用它校验 EmbodimentProfile，不匹配就拒绝 start。"""

    def infer(self, obs: Observation, *, inference_delay: int = 0) -> np.ndarray:
        """返回 (T, D)，已按 EmbodimentProfile 反归一化到工程单位。
        单步模型（OpenVLA base）返回 T=1。"""

    def health(self) -> bool: ...
    def close(self) -> None: ...
```

```python
@dataclass
class Observation:
    images: dict[str, np.ndarray]   # {"main": HWC uint8, "wrist": ...}
    state:  np.ndarray | None       # 本体状态，OpenVLA base 为 None
    prompt: str
    t_capture_ms: int               # 观测采集时刻，用于 RTC 的 inference_delay
```

**`infer` 是唯一跨本地/云的抽象。** 上面的卡片逻辑（watchdog、限幅、发布、ACP）对两种 provider 完全一样。

### 3.2 LocalProvider

进程内加载模型，`infer` 直接推理。

- 只在 `provider: local` 时才 import torch / lerobot —— 依赖放在 Dockerfile 自己的 `RUN` 层，不要进基础层（actucore 镜像刻意做薄）
- **单独进程还是进程内**：如果本地模型用 ONNX Runtime，必须起独立子进程。同进程两份 ORT 共享一个 provider 桥，第二个 CUDA session 必挂（jp6.1 异常 / jp5.11 SIGSEGV），`perception/plugins/kokoro_worker.py` 有完整记录
- 显存/内存要在 `capabilities()` 里自检，装不下就明确报错，不要 OOM 到把 perception 一起拖死

### 3.3 RemoteProvider

按后端各写一个薄 adapter。它们只有传输和 payload 形状不同：

| adapter | 传输 | 说明 |
|---|---|---|
| `openpi_ws` | WebSocket | openpi `serve_policy.py`，用官方 `openpi-client` |
| `lerobot_async` | gRPC/队列 | LeRobot `PolicyServer` / `RobotClient`，自带 action queue |
| `unifolm_http` | HTTP | 宇树 `run_real_eval_server.sh`，README 给的是 ssh -L 隧道 |
| `openvla_http` | HTTP JSON/msgpack | `vla-scripts/deploy.py`，单步、无 state |

共同要求：客户端侧 resize（π 系预训练常用 224）、uint8、连接在回合外建立、断连即触发 watchdog。

### 3.4 fallback 策略

`fallback: local` 时，远端连续 N 次超时后切本地小模型。**但这不是"降级继续干活"**——本地模型的能力和远端不是一个量级，盲目接管更危险。建议语义是：

- `fallback: none`（默认）→ 断连即 watchdog 接管，机器人停住，向 LLM 报 error
- `fallback: local` → 本地模型只负责**把当前动作安全收尾**（回到 home / 松开夹爪 / 停住），不接着执行任务

这条要显式写进卡片文档，否则会被误当成高可用。

---

## 4. 云端设计与接口

### 4.1 部署拓扑

| 形态 | 用途 | 评价 |
|---|---|---|
| **局域网边缘服务器**（一台带 RTX 的机器跑 policy server） | 实时闭环 | **主路径**。网络 RTT 个位数 ms + 推理 20–50 ms，落在 RTC 的舒适区，链路自控 |
| 公网云 | 开发、评测、数据回流、离线微调 | 实时闭环仅限准静态任务。本环境网络状况不适合（办公 WiFi 有 MAC 认证门禁、机器人地址常变） |
| 端上 | 小模型 / 兜底 | 见第 5 节 |

### 4.2 线协议

即使用各家现成 server，**我们自己这一侧的契约要固定下来**，否则换后端就要改卡片。建议 RemoteProvider 对外统一成下面的形状，各 adapter 负责翻译：

```jsonc
// POST /infer  (或 ws 的一帧)
{
  "schema": "motus.vla/1",           // 版本号，必填，不匹配即拒绝
  "session_id": "uuid",              // 一次 start 一个，服务端可据此缓存/复位
  "seq": 128,                        // 单调递增，用于丢弃乱序回包
  "t_capture_ms": 1789234567123,     // 观测采集时刻（不是发送时刻）
  "prompt": "把红色方块放到黑色区域",
  "images": { "main": "<base64 jpeg>", "wrist": "<base64 jpeg>" },
  "state":  [0.12, -0.34, ...],      // 工程单位，服务端负责归一化
  "inference_delay": 3,              // RTC：以 timestep 计的预期延迟
  "unnorm_key": "g1_dex1"
}
```

```jsonc
// 响应
{
  "schema": "motus.vla/1",
  "seq": 128,
  "actions": [[...], [...]],         // (T, D)，工程单位
  "action_dim": 14,
  "chunk_size": 50,
  "t_server_recv_ms": ..., "t_server_done_ms": ...,   // 用于延迟归因
  "model": "pi05-ki@2026-02-01",     // 模型指纹，进日志
  "warnings": []
}
```

```jsonc
// GET /capabilities —— start 前握手一次，用于校验 EmbodimentProfile
{ "schema": "motus.vla/1", "model": "...", "action_dim": 14, "chunk_size": 50,
  "control_hz": 30, "needs_state": true, "n_cameras": 2, "image_size": 224,
  "supports_rtc": true, "unnorm_keys": ["g1_dex1", "libero_spatial"] }
```

设计要点：

1. **`schema` 版本号必填**。动作维度变了而客户端不知道，等于随机驱动电机。
2. **`t_capture_ms` 而非发送时刻**。RTC 的 `inference_delay` 是按观测的年龄算的，不是网络 RTT。
3. **`seq` + 丢弃乱序**。网络抖动时旧 chunk 后到，直接丢，不要覆盖新的。
4. **服务端不持有机器人状态**。每次请求自包含，服务端可随时重启、可水平扩容。`session_id` 只用于缓存，不用于正确性。
5. **超时由客户端定，且必须短于 watchdog**。`timeout_ms < watchdog_ms` 是硬约束。
6. **不做重试**。一次 infer 超时就让它过去，下一帧观测更新鲜；重试只会让动作更陈旧。
7. **单位在客户端侧统一**。服务端收工程单位、回工程单位，归一化是服务端的内部事务。归一化跨进程分摊是最容易出错的地方。

### 4.3 安全

云 policy server 是一个**能直接驱动电机的外部端点**，信任级别等同于 `operator` 角色的 peer。要求：

- 传输 mTLS，证书 pin 住；endpoint 白名单，不接受运行时任意改
- **watchdog 和 e-stop 的代码路径不得经过网络**，也不得在 actucore 里（见 4.4）
- 每次 start / 每次 fallback 切换都上活动流（参照 `peer_tool_call` / `peer_tool_result` 的做法）
- 画布绑定仍然是唯一的人工授权点：不连线，LLM 就调不到
- 相机帧持续外发，公网部署前要过隐私/合规

### 4.4 Watchdog 放在驱动侧，不在卡片里

```
VLA 卡片 ──publish──> /robot/arm/cmd ──subscribe──> Driver 命令卡片
                                                      │
                                            ┌─────────┴──────────┐
                                            │ watchdog: N ms 无新 │
                                            │ 指令 → hold/减速/停 │
                                            │ 限幅: max_joint_delta│
                                            └────────────────────┘
```

理由：卡片崩了、容器被 OOM kill 了、网络断了——这三种情况下卡片里的 watchdog 都不会执行。**只有驱动侧的超时才是无条件生效的。** 限幅同理：不信任上游给的任何一个数。

---

## 5. 本地模型选型

### 5.1 硬件事实

```
$ ssh nvidia@10.100.121.16 'cat /proc/device-tree/model; free -g'
NVIDIA Jetson Orin NX ... Super
Mem:  total 7   available 3      # 已在跑 perception
```

Orin NX 8GB，跑着 perception 时可用约 3 GB。这是选型的硬约束。

### 5.2 候选

| 模型 | 规模 | 本地可行性 | 备注 |
|---|---|---|---|
| **SmolVLA-450M** | 450M（含 ~100M action expert） | **唯一现实候选** | flow matching、action chunk、LeRobot 原生异步推理与 RTC。官方称可在 CPU 上跑 |
| π0 / π0.5 | ~3.3B | ✗ | bf16 权重 6.6 GB，装不下 |
| UnifoLM-VLA-0 | Qwen2.5-VL-7B + head | ✗ | 同上，且要 CUDA 12.4 + flash-attn |
| UnifoLM-WLA-1.0 | 6B | ✗ | 且 WLA-Base 权重尚未放出 |
| OpenVLA-7B | 7B | ✗ | 且 base 不吃本体状态、单步输出 |
| ACT / Diffusion Policy | ~10–100M | ✓ | 无语言条件，只能做单任务，作为兜底动作或对照组 |

### 5.3 SmolVLA 的证据与空白

- 架构上为边缘做了减法：视觉塔跳过一半层；action expert 取中间层（~L/2）特征而非最后一层；推理时不做 image tiling，只喂 global image + pixel shuffle
- 异步推理：响应快 ~30%，固定时间内完成任务数约 2×；关键参数 `actions_per_chunk`、`chunk_size_threshold`
- 支持 RTC（10 步 flow matching 时官方建议 guidance 10.0）

**公开资料里没有 Orin NX 上的 SmolVLA 延迟/内存数字。** 最接近的是 NanoVLA 在 Orin Nano Super 8GB 上对 SmolVLA 的对比（LIBERO-Goal，绝对 Hz 只在图里）。另一个量级参考：量化后的 LiteVLA-Edge 在 **AGX** Orin 上约 150.5 ms（~6.6 Hz）。Orin NX 带宽和算力都低于 AGX，要更保守。

**所以本地这条路必须先做一次实测**，不要按公开数字排期。实测项：单帧延迟 p50/p99、内存峰值、与 perception 共存时的相互影响、连续跑 30 min 的热降频。

### 5.4 本地模型的定位

按 3.4 节，本地模型的第一用途**不是**替代云端跑任务，而是：

1. 安全收尾（断连时把动作停在安全状态）
2. 低风险单任务（抓取固定物体、桌面整理）不必依赖网络
3. 作为云端结果的合理性对照（可选，后期）

---

## 6. 模型许可证（选型前必须过的一关）

| 模型 / 仓库 | 许可证 | 可商用 |
|---|---|---|
| openpi（π0 / π0.5） | Apache-2.0（仓库 LICENSE 已核对；权重条款以官方说明为准） | ✅ |
| LeRobot / SmolVLA | Apache-2.0 | ✅ |
| UnifoLM-WMA-0 | CC BY-NC-SA 4.0 | ❌ 非商用 + ShareAlike |
| UnifoLM-WLA-1.0 | CC BY-NC-SA 4.0 | ❌ |
| UnifoLM-VLA-0 | **仓库无 LICENSE 文件，README 未提** | ❌ 默认保留所有权利 |
| OpenVLA | 见其仓库 | 需确认 |

宇树系模型和我们机队的 embodiment 匹配度最好（训练数据就是 G1 + Dex1，12 个数据集开源，LeRobot v2.1 格式），但许可证是硬阻断。**研究和 demo 可用，产品化前必须先谈授权**。

---

## 7. 分阶段计划

### Phase 0 — 通路与安全（不含模型）

1. 定义 `control/joint` / `control/velocity` 的消息 schema，写进 `phanthymotus-driver/README_dev.md`
2. 在**一个**驱动上加命令订阅卡片（`topic_in: control/joint`）+ watchdog + 限幅
3. 加 `control/*` 的 Dashboard 渲染器
4. 用一个假策略（正弦轨迹）端到端验证：画布连线 → 发布 → 驱动执行 → 断连后 watchdog 停机

**这一阶段不涉及任何 VLA 模型，但它是全部风险所在。**

### Phase 1 — 卡片 + 远端 provider

5. 写 `actucore/plugins/vla.py`（第 2 节骨架）+ `openpi_ws` adapter
6. 局域网边缘服务器跑 π0.5，实测 RTT 分布（p50/p99，不要看均值）
7. 接 RTC，`inference_delay` 按实测配
8. 声明 `x-resource` / `x-completion` / `x-hooks`，验证"喊停能停"

### Phase 2 — 本地 provider

9. Orin NX 上实测 SmolVLA（5.3 的实测项）
10. 按结果决定 `LocalProvider` 是进程内还是独立子进程
11. 实现 fallback 的"安全收尾"语义

### Phase 3 — 数据闭环

12. 观测-动作对录制（LeRobot v2.1 格式，和上游生态对齐）
13. ACP 进度通道，让 LLM 在长任务期间不再是瞎的

---

## 8. 待决问题

1. **动作通路选 (a) 驱动订阅 topic 还是别的**？本文按 (a) 写，因为它和现有 speaker 模式同构。需要确认。
2. **仲裁归谁**：agent-core 的 ACP 资源锁，还是驱动侧 mux？两者都要有，但谁是权威需要定。
3. **边缘服务器是哪台机器**，谁维护，断电/重启策略是什么。
4. **宇树模型是否去谈授权**。若谈成，G1 上的 embodiment 适配工作量会显著小于 π0.5。
5. **π0.6 / UnifoLM-WLA-Base 未放出**，不进本期计划，只做跟踪。

---

## 附录：外部参考

- openpi 远端推理：<https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md>
- RTC（Real-Time Chunking）：<https://arxiv.org/abs/2506.07339>，LeRobot 实现 <https://huggingface.co/docs/lerobot/rtc>
- LeRobot 异步推理：<https://huggingface.co/docs/lerobot/async>
- SmolVLA：<https://huggingface.co/blog/smolvla>
- Jetson AI Lab，π0.5 on Thor（TRT FP8/NVFP4，132 ms → 54 ms → 49 ms）：<https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/>
- openpi 边缘部署 issue：<https://github.com/Physical-Intelligence/openpi/issues/657>
- 宇树 UnifoLM-VLA-0：<https://github.com/unitreerobotics/unifolm-vla> ｜ WMA-0：<https://github.com/unitreerobotics/unifolm-world-model-action> ｜ WLA-1.0：<https://github.com/unitreerobotics/unifolm-wla>
