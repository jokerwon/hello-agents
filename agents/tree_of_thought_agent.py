import json
import re
from typing import Optional

from hello_agents import Config, HelloAgentsLLM, Message
from hello_agents.core.agent import Agent

THINK_PROMPT = """针对问题生成 {width} 条彼此不同的候选思路。
只返回 JSON 字符串数组，不要解释。
问题：{question}
已有最优路径：{path}
"""

SELECT_PROMPT = """选择最有希望解决问题的候选思路。
只返回 JSON：{{"index": 从0开始的整数, "reason": "理由"}}。
问题：{question}
候选：{candidates}
"""

ANSWER_PROMPT = """根据选出的思考路径回答问题。只输出最终答案。
问题：{question}
思考路径：{path}
"""


class TreeOfThoughtAgent(Agent):
    """每层生成多个候选，并由模型选择一条继续的轻量 ToT Agent。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        beam_width: int = 3,
        max_depth: int = 3,
    ):
        if beam_width < 2 or max_depth < 1:
            raise ValueError("beam_width must be >= 2 and max_depth must be >= 1")
        super().__init__(name, llm, system_prompt, config)
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.path: list[str] = []
        self.candidates_by_depth: list[list[str]] = []

    @staticmethod
    def _json_value(response: str, opening: str, closing: str):
        match = re.search(
            re.escape(opening) + r".*" + re.escape(closing), response, re.DOTALL
        )
        if not match:
            raise ValueError("LLM 响应不包含有效 JSON")
        return json.loads(match.group())
    def _invoke_json(self, prompt: str, opening: str, closing: str, **kwargs):
        response = ""
        for attempt in range(2):
            response = self.llm.invoke(
                [{"role": "user", "content": prompt}], **kwargs
            ) or ""
            try:
                return self._json_value(response, opening, closing)
            except (json.JSONDecodeError, ValueError):
                if attempt == 0:
                    prompt += "\n上次响应不是有效 JSON。请严格按指定 JSON 格式重新回答。"
        raise ValueError(f"LLM 连续两次未返回有效 JSON: {response}")


    def run(self, input_text: str, **kwargs) -> str:
        self.path = []
        self.candidates_by_depth = []

        for _ in range(self.max_depth):
            prompt = THINK_PROMPT.format(
                width=self.beam_width,
                question=input_text,
                path=" -> ".join(self.path) or "无",
            )
            candidates = self._invoke_json(prompt, "[", "]", **kwargs)
            if not isinstance(candidates, list) or len(candidates) != self.beam_width:
                raise ValueError(f"模型必须生成 {self.beam_width} 条候选思路")
            candidates = [str(candidate) for candidate in candidates]
            self.candidates_by_depth.append(candidates)

            prompt = SELECT_PROMPT.format(
                question=input_text,
                candidates=json.dumps(candidates, ensure_ascii=False),
            )
            index = self._invoke_json(prompt, "{", "}", **kwargs).get("index")
            if not isinstance(index, int) or not 0 <= index < len(candidates):
                raise ValueError("候选索引越界")
            self.path.append(candidates[index])

        answer = self.llm.invoke(
            [{"role": "user", "content": ANSWER_PROMPT.format(
                question=input_text,
                path=" -> ".join(self.path),
            )}],
            **kwargs,
        ) or ""
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(answer, "assistant"))
        return answer
