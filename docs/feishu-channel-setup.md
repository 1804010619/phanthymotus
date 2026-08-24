# 飞书 Channel 配置与收发验收

本文说明如何从零创建飞书自建应用，并把它接入 Phanthy Motus Agent Core，实现用户与 Agent 的飞书单聊收发。Agent Core 使用飞书 SDK 的 WebSocket 长连接接收事件，只需要设备能够访问飞书开放平台，不需要公网域名或 Webhook。

## 1. 创建并配置飞书应用

1. 打开[飞书开放平台开发者后台](https://open.feishu.cn/app/)，新建企业自建应用。
2. 进入 **应用能力 → 添加应用能力**，添加 **机器人**。
3. 进入 **权限管理 → 开通权限**，开通下表权限。

| 用途 | 权限名称 | Scope | 是否必需 |
|------|----------|-------|----------|
| 机器人发送文本消息 | 以应用的身份发消息 | `im:message:send_as_bot` | 是 |
| 接收用户发给机器人的单聊消息 | 读取用户发给机器人的单聊消息 | `im:message.p2p_msg:readonly` | 是 |
| 收发图片、音频、视频和文件 | 获取与上传图片或文件资源 | `im:resource` | 使用附件时必需 |

如果还要在群聊中响应用户的 `@机器人`，另开通 `im:message.group_at_msg:readonly`。不要为了单聊申请“读取群聊全部消息”等敏感权限。

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
   - **Enabled**：开启

4. 保存后确认 Channel 状态为 `connected`。如果先在 Core 中添加 Channel，后在飞书后台发布应用，等待 watchdog 自动恢复或在渠道面板点击 **Restart**。

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

`channel_reply` 默认回复最近一次从该 Channel 收到消息的会话，因此必须先由用户向机器人发送一条消息，Agent 才能获得目标会话上下文。

## 4. 五分钟收发验收

1. 在飞书中搜索应用机器人；找不到时，先检查用户是否位于应用可用范围。
2. 向机器人单聊发送：`请只回复：飞书链路测试成功`。
3. 在 Agent Core Activity 中确认收到来自 `channel:feishu` 的触发事件。
4. 确认机器人回复 `飞书链路测试成功`。
5. 若开通了 `im:resource`，再发送一张小图片，确认 Agent 能看到附件并可回复图片或文件。

验收通过需要同时满足：Channel 为 `connected`、Activity 有真实入站事件、Decision Core 实际运行、飞书客户端收到真实回复。只看到容器运行或 WebSocket 连接不算完整收发闭环。

## 5. 常见错误

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

## 6. 最小安全边界

- 默认只开单聊所需权限；群聊和外部共享按需增加。
- 不把 App Secret、tenant token、WebSocket ticket 或用户消息写入 Git。
- 机器人可用范围只包含实际使用者；扩大到全员前先确认数据与行为边界。
- `channel_reply` 是 Agent 向飞书发送消息的唯一出口；敏感工具仍受 Agent Core 的 ACL 与确认策略约束。

## 官方参考

- [发送消息](https://open.feishu.cn/document/server-docs/im-v1/message/create.md)
- [接收消息事件](https://open.feishu.cn/document/server-docs/im-v1/message/events/receive.md)
- [上传图片](https://open.feishu.cn/document/server-docs/im-v1/image/create.md)
- [事件订阅概述](https://open.feishu.cn/document/server-docs/event-subscription-guide/overview.md)
