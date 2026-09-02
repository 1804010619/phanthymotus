# llm_bench —— LLM 入口回测

回答「agent-core 该用哪个 LLM 入口 / 哪个模型」，用**本机真实请求历史**回放，输出带结论的报告。

测的是延迟、可用性、缓存命中。**不测回答质量。**

## 为什么在机器人本机跑

同一个入口在办公网和在 Orin 上的表现可以完全不同。这个工具随镜像进到 `/work/tools/`，所以哪台机器人上的疑问就在那台机器人上测。

## 快速开始

配置文件放机器人的 `/opt/phanthy-motus/data/`（容器内是 `/work/resource/`）。从仓库里的模板起手：

```bash
# 在机器人上
cp /work/tools/llm_bench/bench.yaml.example /opt/phanthy-motus/data/bench.yaml
chmod 600 /opt/phanthy-motus/data/bench.yaml
vi /opt/phanthy-motus/data/bench.yaml      # 填 url / key / models
```

`bench.yaml.example` 在仓库里，`bench.yaml`（含明文密钥）被 `.gitignore` 挡住。

然后：

```bash
# 1) 先探活：这些入口现在到底能不能用（秒级，几乎不花钱）
docker exec -w /work -e DB_PATH=/work/resource/data.db phanthy-motus-agent-core-1 \
    /work/.venv/bin/python /work/tools/llm_bench \
    --config /work/resource/bench.yaml --probe 10

# 2) 小样本验证流程通顺
docker exec -w /work -e DB_PATH=/work/resource/data.db phanthy-motus-agent-core-1 \
    /work/.venv/bin/python /work/tools/llm_bench \
    --config /work/resource/bench.yaml --count 3 --repeats 1 --warmup 0 --yes

# 3) 正式一轮，后台跑（SSH 断了前台会中断）
docker exec -d -w /work -e DB_PATH=/work/resource/data.db phanthy-motus-agent-core-1 \
    sh -c '/work/.venv/bin/python /work/tools/llm_bench \
    --config /work/resource/bench.yaml --count 50 --repeats 2 --yes > /tmp/bench.log 2>&1'
docker exec phanthy-motus-agent-core-1 tail -f /tmp/bench.log
```

`-e DB_PATH=` 只有用 `include_current` 时才需要。用 `/work/.venv/bin/python` 是为了走 httpx（连接复用，更贴近生产）；系统 `python3` 也能跑，自动退回 urllib。

**总请求数 = `count × 组数 × (warmup + repeats)`**，启动时会打印出来，看一眼再决定要不要跑。

输出在 `resource/llm_bench/<时间戳>/`，宿主机上是 `/opt/phanthy-motus/data/llm_bench/<时间戳>/`，可以直接 `cat`，不用进容器：

| 文件 | 内容 |
|---|---|
| `report.md` | 人读报告：结论 → 总表 → 配对比较 → 噪声基线 → 失败分析 → 缓存 → 方法与局限 |
| `run.json` | 配置快照（key 已打码）+ 全部统计量 |
| `results.jsonl` | 逐条原始结果，续跑和复算的依据 |

## 中断与续跑

结果逐条 flush 落盘。Ctrl-C / SSH 断线 / 容器重启后：

```bash
python3 tools/llm_bench --resume resource/llm_bench/20260902_143000
```

只补没跑完的 `(组, payload, 轮次, 阶段)`。**语料指纹不匹配会拒绝续跑**——把两批不同的 payload 混进一份报告，数字看着正常但毫无意义，这是比崩溃更危险的失败。

不重跑、只重新生成报告：

```bash
python3 tools/llm_bench --report-only resource/llm_bench/20260902_143000
```

## 评测组的三种来源

可混用，最终展开成一个扁平列表。

1. **YAML 显式列举** —— 一个 `url` + `key` 可带 `models: [a, b, c]`，自动展开成多组（组名 `{name}/{model}`），不用重复写 url 和 key。
2. **`include_current: true`** —— 从 ConfigDB 读当前生产配置加为一组（`current/{model}`），回答「换过去会不会比现在差」。用 sqlite 只读打开，**不 import `src/config.py`**：那个模块 import 时就会建库并跑端口迁移，评测工具不该有这种副作用。

   这一组测不测有命令行开关，优先于 YAML —— 「跟现在比一比」和「只比几个候选」是两种不同的问题，切换它不该去改配置文件：

   ```bash
   ... --include-current      # 强制带上 baseline
   ... --no-include-current   # 强制不测 baseline
   ...                        # 不给就沿用 YAML 里的 include_current
   ```

   显式要了 baseline 但 ConfigDB 里读不到 LLM 配置时会打一行 `[!]` 警告 —— 静默跳过会让人以为「跟现在比过了」，而报告里其实没有那一行。

   记得带 `-e DB_PATH=/work/resource/data.db`，否则读的是默认的 `resource/data.db`。
## 配置与密钥

**所有配置都在 `bench.yaml` 这一个文件里，密钥也一样。** 没有环境变量、没有外部密钥文件——多一种取密钥的方式就多一类「为什么没生效」的排查。

```yaml
groups:
  - name: router
    url: https://router.phanthy.com/v1
    key: sk-...
    models: [glm-5.3, zai-org/glm-5.3]   # 展开成 2 组，url/key 不用重复写
```

放在机器人的 `/opt/phanthy-motus/data/bench.yaml`（容器内 `/work/resource/bench.yaml`）——宿主机的运行时数据目录，不在仓库里。仓库里只有 `bench.yaml.example`，`bench.yaml` 被 `.gitignore` 挡住。建议 `chmod 600`。

`include_current` 那一组的密钥从 ConfigDB 读，不用写在这里。

**密钥永远不会出现在产物里**：报告、`run.json`、控制台输出里都只有 `sk-RjA…jpt5` 这种打码形式。保留头尾是为了让人认出用的是哪一把。测试里有断言明文 key 不出现在任何输出中；用了已废弃的 `key_env`/`key_file` 时报错也不回显值——因为那些字段里往往填的就是密钥本身。

## 为什么有这些"多余"步骤

**不要因为想跑快点就关掉它们。** 每一条都对应一次真实的误判——这个工具就是为了不再犯这些错才存在的。

### 预检（`--skip-preflight` 关闭）

先 `GET /models` 确认模型 id 在列表里，再发一次 `max_tokens=8` 的最小请求。

曾经把 `zai-org/glm-5.2` 配到一个根本没有 GLM 系模型的网关上，50×2 次请求全部 404 才发现。

两步都要：`/models` 只能证明"名字在列表里"，不能证明"能用"。router 的 `glm-5.3` 就长期在列表里却返回「无可用渠道」——**列在 `/models` 里 ≠ 能用**。

### 预热轮（`run.warmup`，默认 1）

跑完整批但结果不计入统计。

不预热就是在测「谁的缓存先被我喂热」。实测中一个入口第一轮缓存命中中位只有 48.8%，被判「慢 2.5 倍」；喂热后（命中 98.5%）实际只慢 1.8 倍。冷热不对称直接改变了结论的量级。

### 组顺序轮换（`run.order: rotate`）

第 i 条 payload 的组顺序按 `i % 组数` 旋转，两组时退化成 A/B 交替。

先把 50 条全发给 A 再全发给 B，两组差异里会混进时段差异。同一个入口在不同时段的波动可以超过组间差异。

### 计分轮 ≥ 2 与噪声基线（`run.repeats`）

用**同一组、同一条 payload 在不同轮次之间的差值**估计测量噪声。

这是最重要的一条约束。实测中：同一入口同一模型同一批 payload 跑两遍，中位差 **0.93s**；而被拿来做结论的两组之间中位差只有 **0.39s**。重复测量自身的波动比组间差异还大——那个"差异"是噪声。

**报告在组间差异 ≤ 噪声基线时直接判「测不出显著差异」，不排名。** `repeats=1` 时无法估噪声，报告会明确警告排名可能只是波动。

### 串行发送

并发会让各组互相争带宽和配额，污染的正是要测的单请求延迟。所以固定串行，不提供并发选项。

## 报告怎么读

**结论段是数据生成的，不是模板。** 三种形态：

- 有组成功率不满 → 首要结论是**可用性**而非延迟。一个入口再快，成功率不满就不是候选。
- 所有组间差异都在噪声内 → 明说「不可区分」，不给推荐。
- 否则 → 给推荐，同时列出哪些组对的差异仍在噪声内。

几个容易误读的地方，报告里都做了处理：

**幸存者偏差** —— 延迟分位只统计成功请求。某组成功率不满时，分位数旁强制标注 `(N/M)`，并单独给出失败样本的耗时分布。为什么重要：很多失败是秒拒（0.06s 的 503），混进平均值会让**失败更多的组看起来更快**。

**失败形态** —— 除失败率外还报最长连续失败段和首次失败位置。连续 14 次 503 是渠道断供，散布的 14 次是限流，处置完全不同，而聚合失败率区分不了。实测中正是「全部集中在 idx 36→49」这个形态才确认了是断供。

**中位与均值符号相反** —— 说明一方在多数请求上略快但吃到了更重的长尾。这种情况不应视为有差别。

**缓存两个口径** —— 总体命中（按 token 加权）反映实际省下的钱和 prefill 时间；逐条命中中位不被个别超长 prompt 带偏。两个都看。

## 探活模式

```bash
python3 tools/llm_bench --config ... --probe 30 --probe-interval 2
```

只发 `max_tokens=8` 的最小请求，报每组可用率和失败序列（`....XXXX....`）。便宜、快，用来回答「这个入口现在能不能用」。

实测中 router 的 `glm-5.3` 一次 `0/30` 全 503、对手 `30/30`——这个模式一分钟就给出了决定性证据，不必先烧掉 50 条完整 payload。**怀疑可用性时先跑这个，别直接上全量。**

## 语料

读 `resource/llm_data/llm_request_*.jsonl`（由 `src/llm_logger.py:145-163` 写出），取 `messages` + `tools` 原样重放，模型名由评测组决定而非沿用记录里的。

解析容错：非正常关机会让文件停在半条记录甚至 UTF-8 序列中间（`src/llm_logger.py:_scan_and_repair` 处理过的真实故障）。这里按字节读再手工解码，坏行跳过并计数报到报告里——**只读不修**，评测工具没有资格改生产数据。

抽样默认 `even`（跨越整个文件均匀取）。只取开头会让样本集中在某一段会话上，那段的 prompt 长度和缓存形态不具代表性。

## 语料从哪来

语料是 **agent-core 自己跑出来的**：`src/llm_logger.py` 每次 LLM 调用都把请求落盘到 `resource/llm_data/llm_request_*.jsonl`。

**只有真正跑过 agent-core 的机器上才有语料。** 本地 checkout 从没跑过服务，`resource/llm_data/` 不存在，完整回测会直接报 `语料路径不存在`（在预检之前就失败，不会发出任何请求，不花钱）。

`--probe` 不需要语料——它只发 `hi` + `max_tokens=8`，所以在任何地方都能跑。

想在本地跑完整回测，从机器人捞一份：

```bash
# 挑一个小的（另外几个可能 20MB+），别整目录拷
ssh nvidia@10.100.121.16 \
    'docker exec phanthy-motus-agent-core-1 ls -laS /work/resource/llm_data/'
scp nvidia@10.100.121.16:/opt/phanthy-motus/data/llm_data/llm_request_XXX.jsonl /tmp/corpus.jsonl

python3 tools/llm_bench --config my-bench.yaml --corpus /tmp/corpus.jsonl --count 20
```

`--corpus` 收目录（扫全部 `llm_request_*.jsonl`）也收单个文件。

> 这些是**真实对话日志**，含用户原话。往开发机上拷之前想清楚要不要，别久放。

各场景对数据的需求：

| 想干什么 | 在哪跑 | 需要语料 |
|---|---|---|
| 这个入口现在通不通 | 任何地方，`--probe` | 否 |
| **选型：该用哪个入口** | **目标机器人上** | 用它本机的 |
| 改工具本身 / 看报告格式 | 本地 + 捞来的 jsonl | 是 |
| 跑单元测试 | 本地 `pytest` | 否（合成数据） |

选型**必须**在目标机器人上做。本地测的是你办公网到入口的延迟，和机器人的网络路径不是一回事——这正是这套工具存在的理由。

## 依赖

stdlib + 可选 httpx + PyYAML（`pyproject.toml` 已含）。**无新增依赖。**

有 httpx 用 httpx（连接复用，贴近生产的 openai SDK 路径）；没有就退回 urllib，所以容器里那个没装 httpx 的系统 python3（3.10）也能跑。报告里会记下用的哪个后端。

## 不做

- 不做定时/常驻评测
- 不评估回答质量
- 不自动改 ConfigDB 切换入口——报告给结论，切换是人的决定
- 不做并发压测（会污染单请求延迟画像）
