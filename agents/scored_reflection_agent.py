import json
import re
from typing import Optional

from hello_agents import Config, HelloAgentsLLM, Message
from hello_agents.agents.reflection_agent import ReflectionAgent

SCORE_PROMPT = """请评价下面回答对原任务的完成质量。
只返回 JSON：{{"score": 0到100的整数, "feedback": "具体改进建议"}}。

原任务：{task}
当前回答：{content}
"""


class ScoredReflectionAgent(ReflectionAgent):
    """达到质量阈值即停止优化的 ReflectionAgent。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        score_threshold: int = 85,
        custom_prompts: Optional[dict[str, str]] = None,
    ):
        if not 0 <= score_threshold <= 100:
            raise ValueError("score_threshold must be between 0 and 100")
        super().__init__(
            name, llm, system_prompt, config, max_iterations, custom_prompts
        )
        self.score_threshold = score_threshold
        self.scores: list[int] = []

    @staticmethod
    def _parse_score(response: str) -> tuple[int, str]:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            raise ValueError("评分响应不包含 JSON 对象")
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            score_match = re.search(r'"score"\s*:\s*(\d+)', match.group())
            if not score_match:
                raise ValueError("评分响应包含无效 JSON") from None
            data = {"score": int(score_match.group(1)), "feedback": response}
        score = data.get("score")
        if not isinstance(score, int) or not 0 <= score <= 100:
            raise ValueError("score 必须是 0 到 100 的整数")
        return score, str(data.get("feedback", ""))

    def run(self, input_text: str, **kwargs) -> str:
        self.memory = type(self.memory)()
        self.scores = []

        result = self._get_llm_response(
            self.prompts["initial"].format(task=input_text), **kwargs
        )
        self.memory.add_record("execution", result)

        for _ in range(self.max_iterations):
            score_response = self._get_llm_response(
                SCORE_PROMPT.format(task=input_text, content=result), **kwargs
            )
            score, feedback = self._parse_score(score_response)
            self.scores.append(score)
            self.memory.add_record("reflection", feedback)
            if score >= self.score_threshold:
                break
            result = self._get_llm_response(
                self.prompts["refine"].format(
                    task=input_text,
                    last_attempt=result,
                    feedback=feedback,
                ),
                **kwargs,
            )
            self.memory.add_record("execution", result)

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(result, "assistant"))
        return result
