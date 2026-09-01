# Agent Q&A user-story baseline

- Baseline revision: `agent-qa-v1`
- Status: `BASELINE_APPROVED`
- Approved by: user
- Approval evidence: user message `确认` on 2026-09-01 after the complete baseline was presented.
- GitHub issue source: https://github.com/johnymoo/ASR-App/issues/1
- Issue body SHA-256: `1d025b379f2a69a95e6ccf697dc07ec363040fdfe5fee7b533ba729ff3e1ac22`
- Design-input source: `planning/02-working/2026-09-01-agent-qa-design-inputs.md`
- Design-input SHA-256 at approval: `c0ac7791ea94dc96914c962412a19ffa2e6f96692ceef7439101b4816cd317ce`

## Required stories

### US-QA-01

作为播客用户，我希望选择“当前播客”或“全部播客”后提问，从而从 summary、官方大纲和转写原文中获得答案。

- Given 内容已发布
- When 用户在所选范围内提问
- Then 返回答案、播客名称、近似时间戳、原文片段和全文定位链接
- And 没有依据时明确说明未找到

### US-QA-02

作为站点唯一用户，我希望在配置页设置单一 Agent 的 OpenAI-compatible API 地址、API key、模型、system prompt、temperature 和超时，从而无需修改代码即可切换模型。

- Given 用户可以访问配置页
- When 用户保存有效配置
- Then 新配置立即用于后续问题
- And API key 仅显示掩码，不进入静态页面或日志
- And MVP 不要求登录鉴权

### US-QA-03

作为播客用户，我希望在同一会话中连续追问，并可在页面切换问答范围。

- Given 当前会话已有问答历史
- When 用户继续追问
- Then Agent 使用该会话历史理解追问
- When 用户切换“当前播客 / 全部播客”范围
- Then 后续问题使用新范围，并在界面明确显示当前范围
- And 不同会话不会混用上下文

## Constraints

- One active Agent profile.
- OpenAI-compatible API.
- Knowledge sources are limited to each published episode's structured summary, official outline, and transcript.
- Question execution is asynchronous and can be recovered through polling after refresh.
- Citation timestamps may use the existing SRT interpolation and are therefore approximate.
- This is a single-user LAN deployment; the configuration page does not require authentication.

## Non-goals

- Web search or external knowledge augmentation.
- Multiple Agent profiles or multi-Agent workflows.
- Multi-user permissions.
- Exact word-level timestamp alignment.

## Superseded issue assumption

Issue #1 originally excluded cross-episode global Q&A. The user's later instruction explicitly added an “all podcasts” scope, so that original non-goal is superseded by `US-QA-01`.
