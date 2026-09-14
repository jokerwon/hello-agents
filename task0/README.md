# Task00 环境配置与最小 Agent

## 1. 系统环境

| 项目   | 实测值                         | 获取命令           |
| ------ | ------------------------------ | ------------------ |
| macOS  | 26.2 (Build 25C56)             | `sw_vers`          |
| uv     | 0.12.13 (aarch64-apple-darwin) | `uv --version`     |
| Python | 3.14.0                         | `uv run python -V` |

## 2. 外部服务

| 服务                             | 用途                           | 是否需要 Key |
| -------------------------------- | ------------------------------ | ------------ |
| 内部 LLM 网关（OpenAI 兼容 API） | LLM 推理                       | 需要         |
| Tavily Search                    | 搜索推荐旅游景点               | 需要         |
| wttr.in                          | 查询实时天气（供天气工具调用） | **不需要**   |

天气工具走公开接口，缺 Key 只影响景点推荐（会返回错误字符串作为 Observation）。

## 3. 安装项目环境

```shell
uv init                                  # 1. 生成 pyproject.toml / .python-version
uv venv                                  # 2. 生成 .venv
uv add requests tavily-python python-dotenv openai   # 3. 装依赖并写入 pyproject + uv.lock
```

## 4. 环境变量

创建 `task0/.env`（真实 key 只放这里，`.gitignore` 已忽略 `.env`）：

```
# Tavily API 配置
TAVILY_API_KEY=your_tavily_api_key

# LLM 网关（OpenAI 兼容）
OPENAI_API_KEY=your_gateway_api_key
OPENAI_BASE_URL=https://aihubmix.com/v1
MODEL_NAME=qwen3.5-flash
```

## 5. 运行

```shell
uv run python task0/main.py
```

## 6. 代码组成

- `main.py`
  - `AGENT_SYSTEM_PROMPT`：工具说明书 + 输出格式约定。
  - `available_tools`：工具名字符串 → Python 函数的注册表。
  - `main()`：Agent 主循环——构造 prompt → 调 LLM → 截断多余的 Thought-Action 对 → 解析 Action → 执行工具 → 回灌 Observation。
  - `OpenAICompatibleClient`：OpenAI 兼容客户端，固定发 system + 1 条 user 两条 message。
- `tools/get_weather.py`：请求 wttr.in `?format=j1`，取当前天气描述与气温。
- `tools/get_attraction.py`：调用 Tavily 搜索，优先返回综合摘要 `answer`。

## 7. 运行结果解析

```
已初始化 LLM 客户端:model=qwen3.5-flash
用户输入: 你好，请帮我查询一下今天南京的天气，然后根据天气推荐一个合适的旅游景点。
========================================
--- 循环 1 ---
模型输出:
Thought: 用户首先要求查询南京的天气……需要先调用 `get_weather` 工具。
Action: get_weather(city="南京")

Observation: 南京当前天气:Smoky haze，气温25摄氏度
========================================
--- 循环 2 ---
模型输出:
Thought: 已经获取了南京的天气信息（Smoky haze，25摄氏度）……调用 `get_attraction`。
Action: get_attraction(city="南京", weather="Smoky haze")

Observation: Under smoky haze, visit Nanjing's Xuanwu Lake for a peaceful walk ... Also, explore the historic Nanjing City Wall ...
========================================
--- 循环 3 ---
模型输出:
Thought: 已经获取了天气与景点推荐……可以整合回答用户。
Action: Finish[今天南京的天气是雾霾（Smoky haze），气温25摄氏度……推荐玄武湖……或南京城墙……]

任务完成，最终答案: 今天南京的天气是雾霾（Smoky haze），气温25摄氏度……
```

全程 `real 1m9s`，几乎全花在 3 次 LLM 往返上。

### 7.1 流程

宏观：用户只发过一次请求，1 次交互。
微观：3 次模型调用——每轮循环恰好一次 `llm.generate`。

对应时序：

1. 用户提问。
2. 智能体调用大模型。
3. 大模型从系统提示词中得知可用工具，推理出需要调用 `get_weather`。
4. 智能体解析参数、调用工具，把结果拼回会话历史，再次调用大模型。
5. 大模型根据新历史推理出需要调用 `get_attraction`。
6. 智能体调用该工具，结果继续拼回历史，再次调用大模型。
7. 大模型判断信息已足够，输出最终回应。
8. 智能体识别出无需再调工具，把回应反馈给用户。

### 7.2 日志看不到的四个机制

1. **循环上限 5 是「模型调用次数」上限**，不是对话轮数。
2. **工具协议是手写文本协议，不是 OpenAI 原生 tool calling。** 工具签名写在 system prompt 里，模型靠模仿输出 `function_name(arg="value")`，智能体用两条正则解析。system prompt 里的「输出格式要求」就是接口契约，没有它解析必然失败。
3. **上下文是一条不断变长的单条 user message。** `prompt_history` 用 `"\n".join()` 拼成一个字符串发给模型，messages 始终只有 `[system, user]` 两条，没有 assistant/user 角色交替，工具结果（Observation）是纯文本回灌。
4. **`Finish` 是字符串匹配**后 `break`，不是语义层面的判停。

## 8. 知识点小结

这个极简 Agent 由四要素组成，缺一不可：

- **决策**：LLM 读 system prompt + 历史，产出 Thought/Action。
- **执行**：`available_tools` 注册表把工具名映射到真实函数。
- **控制**：`for` 循环 + `Finish` 判定 + 错误分支。
- **记忆**：`prompt_history` 累积文本。

两点体会：

- **ReAct 的闭环在于 Observation 由环境产生、再回灌上下文。** 模型第一轮的判断对不对不重要，重要的是它能看到工具的真实返回并据此修正——这是 Agent 与"一问一答 + 函数调用"的本质区别。
- **容错来自负反馈。** 调了不存在的工具会得到 `错误:未定义的工具 'xxx'`，格式不对会得到格式错误提示，它们都作为 Observation 回灌，模型下一轮有机会自我纠正。Agent 的鲁棒性不是"一次做对"，而是"错了能改"。
