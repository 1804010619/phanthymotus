# 飞书 Channel 配置与收发验收

本文说明如何从零创建飞书自建应用，并把它接入 Phanthy Motus Agent Core，实现用户与 Agent 收发消息，以及可选的群聊机器人互相 `@`。Agent Core 使用飞书 SDK 的 WebSocket 长连接接收事件，只需要设备能够访问飞书开放平台，不需要公网域名或 Webhook。

## 1. 创建并配置飞书应用

1. 打开[飞书开放平台开发者后台](https://open.feishu.cn/app/)，新建企业自建应用。
2. 进入 **应用能力 → 添加应用能力**，添加 **机器人**。
3. 进入 **权限管理 → 开通权限**，开通下表权限。

| 用途 | 权限名称 | Scope | 是否必需 |
|------|----------|-------|----------|
| 机器人发送文本消息 | 以应用的身份发消息 | `im:message:send_as_bot` | 是 |
| 接收用户发给机器人的单聊消息 | 读取用户发给机器人的单聊消息 | `im:message.p2p_msg:readonly` | 是 |
| 接收用户在群里的 @ | 接收群聊中提及机器人的消息 | `im:message.group_at_msg:readonly` | 使用用户群聊时必需 |
| 接收其他机器人在群里的 @ | 获取群组中其他机器人和用户 @ 当前机器人的消息 | `im:message.group_at_msg.include_bot:readonly` | 使用 Bot @ Bot 时必需 |
| 收发图片、音频、视频和文件 | 获取与上传图片或文件资源 | `im:resource` | 使用附件时必需 |

`im:message.group_at_msg.include_bot:readonly` 仅适用于企业自建应用。不要申请 `im:message.group_msg.include_bot:read`；本功能只需要收到明确 `@` 当前机器人的消息，不需要读取群内所有消息。

4. 进入 **事件与回调 → 事件配置**：

   - 订阅方式选择 **使用长连接接收事件** 并保存。
   - 添加应用身份事件 **接收消息 v2.0**，事件类型为 `im.message.receive_v1`。

5. 进入 **版本管理与发布**，创建并发布新版本。添加机器人能力、权限或事件后不发布，线上应用不会生效。
6. 在版本的 **可用范围** 中加入实际测试者和需要与机器人沟通的 HR 成员。除非确实需要跨租户沟通，否则保持“外部群”和“外部用户单聊”关闭。

## 2. 在 Agent Core 中添加飞书 Channel

1. 打开 Agent Core Web Dashboard。
2. 进入 **设置 → 渠道 → + Add Channel**。
3. 填写：

   - **Platform**：`Feishu (飞书)`
   - **ID**：稳定且可辨识的 Channel 名称，例如 `hr_feishu`
   - **App ID**：飞书开发者后台 **凭证与基础信息** 中的 App ID
   - **App Secret**：同页的 App Secret
   - **Allow group Bot @ Bot**：默认关闭；只有需要机器人群聊协作时才开启
   - **Enabled**：开启

4. 保存后确认 Channel 状态为 `connected`。已有 Channel 可以直接点击 `Bot @ Bot: Off` 开启；修改后 Adapter 会自动重启。
5. 如果先在 Core 中添加 Channel，后在飞书后台发布应用，等待 watchdog 自动恢复或在渠道面板点击 **Restart**。

App Secret 只应保存在设备的 Agent Core 配置中，不要写入仓库、文档、截图或聊天记录。Secret 泄露后应在飞书后台重置，并立即更新 Channel。

## 3. 把 Channel 接入 Decision Core

仅在“设置 → 渠道”中添加 Channel，只会建立飞书连接；消息不会自动进入 Agent。还需要在 Canvas 中完成两条连接：

1. 停止当前智能控制，进入 Canvas 编辑状态。
2. 添加 `Channel / channel_request` 卡片，在实例配置中选择刚创建的 Channel。
3. 将 `channel_request` 的 `data/json` 输出连接到 `AgentCore / decision_core` 输入。
4. 添加 `Channel / channel_reply` 卡片，在实例配置中选择同一个 Channel。
5. 从 `decision_core` 底部的 **执行器** 端口连接到 `channel_reply` 卡片。
6. 启动智能控制，确认两个 Channel 卡片均通过启动自检。

链路如下：

```text
飞书用户
  → 飞书长连接事件
  → channel_request
  → decision_core
  → channel_reply
  → 飞书用户
```

`channel_reply` 使用触发消息的 `source_message_id` 找回对应会话，不会被后到的群聊或私聊改写目标。因此必须先收到一条消息，并在回复时带上它的 `message_id`。

## 4. 机器人什么时候才会 @

Bot @ Bot 开启后，任意群中的任意机器人都可以通过明确 `@当前机器人` 触发 Agent。机器人单聊不会触发；bot 身份也不会加入人员 ACL。该 turn 只能调用 Canvas 已绑定的 sensor/resource，并通过 `channel_reply` 给当前消息发纯文本回复；不能读取本地文件/历史、访问网页、执行 actuator/processor/Shell、写文件或委派执行。

Agent 只有在以下情况才应使用 `channel_reply.mention_open_id`：

- 当前任务确实需要目标机器人提供信息或审查结果；
- 把对方请求的最终结果返回给对方。

以下消息不得继续 `@`：收到确认、感谢、复述、没有新增信息的状态，或已取得完成任务所需的信息。不能用“收到”“谢谢”“请确认收到我的确认”制造新一轮。

每条机器人 `@` 消息都带有明确语义：

| 标记 | `expect_reply` | 行为 |
|------|----------------|------|
| `【机器人协作请求·需要回复】` | `true` | 对方有一个具体的未完成任务，可以完成后回复，或确有新问题时继续询问 |
| `【机器人协作答复·无需回复】` | `false`（默认） | 最终结果；接收方在代码层不能再 `@` 机器人 |

`mention_open_id` 必须与当前触发事件的 `source_message_id` 一起使用。机器人不能选用历史消息 ID、跨会话发送或发送附件；回复只能 `@` 原发送机器人。其他机器人没有上述标记时，明确 `@` 当前机器人的消息按一次“需要回复”的请求处理。

本实现不设置轮次或时间熔断。对话何时继续由 Agent 根据是否还有具体未决任务决定；一旦使用默认的 `expect_reply=false` 返回最终结果，后续机器人 `@` 会被代码拒绝。

当前 Bot @ Bot 只支持单条纯文本，不支持附件或超长分片消息。

## 5. 五分钟收发验收

1. 在飞书中搜索应用机器人；找不到时，先检查用户是否位于应用可用范围。
2. 向机器人单聊发送：`请只回复：飞书链路测试成功`。
3. 在 Agent Core Activity 中确认收到来自 `channel:feishu` 的触发事件。
4. 确认机器人回复 `飞书链路测试成功`。
5. 若开通了 `im:resource`，再发送一张小图片，确认 Agent 能看到附件并可回复图片或文件。

验收通过需要同时满足：Channel 为 `connected`、Activity 有真实入站事件、Decision Core 实际运行、飞书客户端收到真实回复。只看到容器运行或 WebSocket 连接不算完整收发闭环。

### Bot @ Bot 真实验收

1. 为 A、B 两个自建应用都开通 `im:message.group_at_msg.include_bot:readonly`，发布新版本，并把两个机器人加入同一个测试群。
2. 两边 Core 的渠道面板都显示 `Bot @ Bot: On`，Canvas 的 `channel_request` 与 `channel_reply` 均已启动。
3. 人工发送：`@A 请向 @B 询问当前状态；@B 暂不处理这条人工消息，等待 A 的请求。`
4. 在 A 的 Activity 中确认触发事件带有 `message_id`，且 `mentions` 中 B 的 `open_id` 标记为 `is_self=false`。
5. A 使用该 B open_id、当前 `message_id` 和 `expect_reply=true` 发出真实可点击的 `@B`。
6. B 的 Activity 应显示 `sender_type=bot`、`chat_type=group`、`expect_reply=true`；B 用 `expect_reply=false` 在原群 `@A` 返回最终结果。
7. A 收到 `expect_reply=false` 后不再 `@B`。尝试再次带 `mention_open_id` 回复时，工具应明确拒绝且群里没有新消息。
8. 给任一机器人发送 bot 单聊，确认 Activity 没有对应触发事件。

验收截图至少包含渠道开关、群内 A→B→A 消息和双方 Activity 状态；不得截入 App Secret、token 或真实敏感消息正文。

## 6. 常见错误

| 表现或错误码 | 原因 | 处理 |
|--------------|------|------|
| `11205: app do not have bot` | 机器人能力未添加，或添加后未发布新版本 | 添加机器人能力并发布版本 |
| `230006: Bot ability is not activated` | 当前线上版本未启用机器人能力 | 发布包含机器人能力的新版本 |
| `99991672: Permission denied` | 缺少消息或资源权限，或权限变更尚未发布 | 对照权限表开通并发布；附件失败重点检查 `im:resource` |
| `Invalid topic name` | 旧版 Core 把含中文、空格或连字符的 Channel ID 直接拼入 ROS topic | 升级到已修复版本；升级前可把 Channel 重建为 `shanghai_g1` 这类 ASCII ID |
| Channel 已连接但消息无反应 | 未订阅 `im.message.receive_v1`，或 Canvas 没有配置 `channel_request` | 检查事件订阅、卡片配置和智能控制状态 |
| Agent 能收到但不能回复 | `decision_core` 未绑定 `channel_reply`，或尚无会话上下文 | 补执行器连接，并先由用户发送消息 |
| 飞书里找不到机器人 | 用户不在应用可用范围 | 在新版本中加入该用户并发布 |
| 群聊中没有事件 | 缺群聊 `@机器人` 权限，或机器人未加入群 | 开通 `im:message.group_at_msg:readonly`、发布版本并把机器人加入群 |
| 用户 @ 可以收到，机器人 @ 收不到 | 缺少 include-bot 权限，或 `Bot @ Bot` 仍为 Off | 开通 `im:message.group_at_msg.include_bot:readonly`、发布版本并在 **设置 → 渠道** 开启开关 |
| `unknown or expired source_message_id` | 该消息 ID 不存在，或已超出每个 Channel 保留的最近 100 条上下文 | 不发送；基于当前 Activity 事件重新处理 |
| `final answer ... do not @ another bot` | 收到的是 `expect_reply=false` 最终答复 | 正常结束，不再 @ |
| bot 群消息的 `user_role` 显示为 `viewer` | bot 不会写入人员 ACL | 正常；bot turn 会在 Decision Core 调用前强制 sensor/resource 与当前消息回复白名单 |

## 7. 最小安全边界

- 默认只开单聊所需权限；群聊和外部共享按需增加。
- 不把 App Secret、tenant token、WebSocket ticket 或用户消息写入 Git。
- 机器人可用范围只包含实际使用者；扩大到全员前先确认数据与行为边界。
- `channel_reply` 是 Agent 向飞书发送消息的唯一出口；bot 触发的 turn 在工具调用前强制只读，普通人工消息行为不变。
- `Bot @ Bot` 默认关闭。开启后接受机器人所在任意群中的任意机器人；不需要该信任范围时应保持关闭。
- 任意群机器人都能触发 Agent，但 bot turn 只能调用 Canvas 已绑定的 sensor/resource、`finish` 和精确绑定当前原消息的纯文本 `channel_reply`；其内容不写入共享对话历史，物理动作必须由人工来源单独触发。
- Bot @ Bot 消息只允许在来源群中回复，禁止机器人单聊、跨群回复、自 @ 和 bot 身份写入人员 ACL。

## 官方参考

- [发送消息](https://open.feishu.cn/document/server-docs/im-v1/message/create.md)
- [接收消息事件](https://open.feishu.cn/document/server-docs/im-v1/message/events/receive.md)
- [消息内容与结构化 @](https://open.feishu.cn/document/server-docs/im-v1/message-content-description/create_json.md)
- [获取机器人信息](https://open.feishu.cn/document/client-docs/bot-v3/obtain-bot-info.md)
- [上传图片](https://open.feishu.cn/document/server-docs/im-v1/image/create.md)
- [事件订阅概述](https://open.feishu.cn/document/server-docs/event-subscription-guide/overview.md)
