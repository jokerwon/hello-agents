import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.scored_reflection_agent import ScoredReflectionAgent
from agents.tree_of_thought_agent import TreeOfThoughtAgent


class FakeLLM:
    provider = "fake"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def invoke(self, messages, **kwargs):
        self.calls += 1
        return next(self.responses)


def test_scored_reflection_stops_at_threshold():
    llm = FakeLLM([
        "初稿",
        '{"score": 92, "feedback": "完整准确"}',
    ])
    agent = ScoredReflectionAgent(
        "评分反思", llm, max_iterations=3, score_threshold=85
    )

    assert agent.run("解释二分查找") == "初稿"
    assert agent.scores == [92]
    assert llm.calls == 2


def test_scored_reflection_refines_low_score():
    llm = FakeLLM([
        "初稿",
        '{"score": 60, "feedback": "缺少复杂度"}',
        "补充复杂度后的答案",
        '{"score": 90, "feedback": "达标"}',
    ])
    agent = ScoredReflectionAgent(
        "评分反思", llm, max_iterations=3, score_threshold=85
    )

    assert agent.run("解释二分查找") == "补充复杂度后的答案"
    assert agent.scores == [60, 90]


def test_tree_of_thought_selects_one_candidate_per_depth():
    llm = FakeLLM([
        '["枚举", "代数", "画图"]',
        '{"index": 1, "reason": "更直接"}',
        '["设未知数", "列方程", "验算"]',
        '{"index": 1, "reason": "可求解"}',
        "答案是 42",
    ])
    agent = TreeOfThoughtAgent("思维树", llm, beam_width=3, max_depth=2)

    assert agent.run("求答案") == "答案是 42"
    assert agent.path == ["代数", "列方程"]
    assert all(len(candidates) == 3 for candidates in agent.candidates_by_depth)
    assert len(agent.get_history()) == 2

def test_tree_of_thought_retries_invalid_json():
    llm = FakeLLM([
        r'["错误转义 \q", "候选二"]',
        '["候选一", "候选二"]',
        r'{"index": 0, "reason": "错误转义 \q"}',
        '{"index": 1, "reason": "有效"}',
        "最终答案",
    ])
    agent = TreeOfThoughtAgent("思维树", llm, beam_width=2, max_depth=1)

    assert agent.run("求答案") == "最终答案"
    assert agent.path == ["候选二"]
    assert llm.calls == 5


def run_real():
    from dotenv import load_dotenv
    from core.my_llm import MyLLM

    load_dotenv(Path(__file__).with_name(".env"))
    llm = MyLLM(provider="wc", temperature=0)

    reflection = ScoredReflectionAgent(
        "评分反思", llm, max_iterations=1, score_threshold=80
    )
    reflection_answer = reflection.run("用一句话解释二分查找")
    assert reflection_answer and len(reflection.scores) == 1
    print("REFLECTION_SCORE", reflection.scores)

    tree = TreeOfThoughtAgent("思维树", llm, beam_width=2, max_depth=1)
    tree_answer = tree.run("鸡兔共两只，共有六条腿，各有几只？")
    assert tree_answer and len(tree.path) == 1
    assert len(tree.candidates_by_depth[0]) == 2
    print("TOT_PATH", tree.path)
    print("TOT_ANSWER", tree_answer)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="调用 task2/.env 中的真实模型")
    args = parser.parse_args()
    if args.real:
        run_real()
    else:
        test_scored_reflection_stops_at_threshold()
        test_scored_reflection_refines_low_score()
        test_tree_of_thought_selects_one_candidate_per_depth()
        test_tree_of_thought_retries_invalid_json()
        print("第七章实践题自检通过")
