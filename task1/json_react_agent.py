"""JSON 结构化输出的 ReAct 智能体（习题 6.2.3 的改造实现）。

与第四章原始 ReAct（react_agent.py，"Thought:/Action:" 文本 + 正则解析）相比，
本实现把模型的输出协议从"自由文本 + 正则"换成"单一 JSON 对象 + 结构化解析":

    {"thought": "...", "action": "工具名或Finish", "action_input": "..."}

解析侧不再依赖正则的形态假设，而是:
1. extract_json 做括号平衡扫描，容忍围栏/前后缀文字；
2. 校验必须字段与类型；
3. 解析失败时把格式错误作为 Observation 反馈给模型自行纠正（有界重试）。
"""

from typing import Any, Callable

from common import extract_json, make_llm

from hello_agents_llm import HelloAgentsLLM
from tools.tool_executor import ToolExecutor

JSON_REACT_PROMPT_TEMPLATE = """你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下JSON格式进行回应，只输出一个JSON对象，禁止输出JSON之外的任何内容:
{{"thought": "你的思考过程，用于分析问题、拆解任务和规划下一步行动", "action": "工具名称或Finish", "action_input": "工具的输入参数，或最终答案"}}

规则:
- action 必须是上方可用工具之一的名称，或者为 "Finish"。
- 当你收集到足够的信息、能够回答用户的最终问题时，action 必须为 "Finish"，并把最终答案完整写入 action_input。
- action_input 必须是字符串。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""


class JsonReActAgent:
    """输出协议为 JSON 的 ReAct 智能体。"""

    def __init__(
        self,
        llm_client: HelloAgentsLLM,
        tool_executor: ToolExecutor,
        max_steps: int = 8,
        max_format_failures: int = 3,
    ):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.max_format_failures = max_format_failures
        self.history: list[str] = []
        # 统计信息：便于实验对比
        self.stats = {"steps": 0, "format_failures": 0, "tool_calls": 0}

    def run(self, question: str) -> str | None:
        self.history = []
        self.stats = {"steps": 0, "format_failures": 0, "tool_calls": 0}
        format_failures = 0

        for step in range(1, self.max_steps + 1):
            self.stats["steps"] = step
            print(f"\n--- 第 {step} 步 ---")

            prompt = JSON_REACT_PROMPT_TEMPLATE.format(
                tools=self.tool_executor.getAvailableTools(),
                question=question,
                history="\n".join(self.history) or "无",
            )
            response_text = self.llm_client.think(
                messages=[{"role": "user", "content": prompt}]
            )
            if not response_text:
                print("错误：LLM 未能返回有效响应。")
                break

            # --- 结构化解析（替代正则） ---
            data = extract_json(response_text)
            if not isinstance(data, dict) or "action" not in data:
                format_failures += 1
                self.stats["format_failures"] = format_failures
                print("❌ 输出不是合法JSON或缺少 action 字段，已反馈给模型重试。")
                if format_failures >= self.max_format_failures:
                    print("连续格式错误次数过多，流程终止。")
                    break
                self.history.append(
                    "Observation: 格式错误——你的上一条回复不是合法的JSON对象或缺少 action 字段。"
                    '请重新输出形如 {"thought": "...", "action": "...", "action_input": "..."} 的JSON。'
                )
                continue

            thought = str(data.get("thought", "")).strip()
            action = str(data.get("action", "")).strip()
            action_input = data.get("action_input", "")
            if not isinstance(action_input, str):
                action_input = str(action_input)

            if thought:
                print(f"🤔 思考: {thought}")

            # --- 行动 ---
            if action == "Finish":
                print(f"🎉 最终答案: {action_input}")
                return action_input

            if not action_input.strip():
                observation = "错误：action_input 不能为空。请提供工具所需的输入参数。"
                print(f"👀 观察: {observation}")
                self.history.append(f"Action: {action}[{action_input}]")
                self.history.append(f"Observation: {observation}")
                continue

            observation = self._execute_action(action, action_input)

            print(f"👀 观察: {observation}")
            self.history.append(f"Action: {action}[{action_input}]")
            self.history.append(f"Observation: {observation}")

        print("已达到最大步数，流程终止。")
        return None

    def _execute_action(self, action: str, action_input: str) -> str:
        """执行一个工具 Action 并返回 Observation。

        子类（如习题 6.3.2 的失败处理机制）可覆写此方法，
        加入错误引导、失败计数与升级策略。
        """
        tool_function: Callable[..., Any] | None = self.tool_executor.getTool(action)
        if tool_function is None:
            available = "、".join(self.tool_executor.tools.keys())
            return (
                f"错误：工具 '{action}' 不存在。可用工具: {available}。"
                "请从可用工具中重新选择。"
            )
        print(f"🎬 行动: {action}[{action_input}]")
        self.stats["tool_calls"] += 1
        return self._invoke_tool(tool_function, action, action_input)

    def _invoke_tool(
        self, tool_function: Callable[..., Any], tool_name: str, action_input: str
    ) -> str:
        """执行工具；子类可覆写此方法加入更完善的失败处理（见习题 6.3.2）。"""
        try:
            return str(tool_function(action_input))
        except Exception as e:  # 工具自身的异常也转成 Observation
            return f"工具 '{tool_name}' 执行出错: {e}"


if __name__ == "__main__":
    from tools.search import search
    from tools.calculator import calculate

    llm = make_llm()
    executor = ToolExecutor()
    executor.registerTool(
        "Search", "一个网页搜索引擎。当你需要回答时事、事实性问题时使用。", search
    )
    executor.registerTool(
        "Calculate", "一个计算器。当你需要处理复杂的数学计算时使用。", calculate
    )
    agent = JsonReActAgent(llm_client=llm, tool_executor=executor)
    agent.run(
        "deepseek 最新发布的模型是什么？假如每百万tokens价格 0.02 元，花费 10 亿 tokens 将花费多少钱？（调用计算器）"
    )
