import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# task1/ 目录（本文件位于 task1/exercises/ 下）
TASK1_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK1_DIR))

# 显式加载 task1/.env（不覆盖已存在的环境变量）
load_dotenv(TASK1_DIR / ".env")

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# 复用第四章基础模块
from hello_agents_llm import HelloAgentsLLM  # noqa: E402


class _Tee:
    """把写入 stdout 的内容同时写入日志文件。"""

    def __init__(self, log_path: Path):
        self._stdout = sys.__stdout__
        self._file = open(log_path, "w", encoding="utf-8")

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()


def start_logging(script_path: str) -> Path:
    """开始把 stdout 复制到 artifacts/<脚本名>.log，返回日志路径。"""
    name = Path(script_path).stem
    log_path = ARTIFACTS_DIR / f"{name}.log"
    sys.stdout = _Tee(log_path)
    print(f"===== {name} 运行日志 =====")
    return log_path


def extract_json(text: str):
    """从模型输出中提取第一个"括号平衡"的 JSON 对象并解析。

    相比直接 json.loads，这里容忍:
    - JSON 前后有说明文字 / markdown 代码围栏 (```json ... ```)
    - 字符串内部出现 { } 不会破坏平衡计数
    - 第一个候选解析失败时继续尝试下一个 '{' 起点
    解析失败返回 None。
    """
    if not text:
        return None
    text = text.replace("```json", "```")
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
        start = text.find("{", start + 1)
    return None


def make_llm() -> HelloAgentsLLM:
    """创建默认 LLM 客户端（配置来自 task1/.env）。"""
    return HelloAgentsLLM()


def save_text(filename: str, content: str) -> Path:
    """把实验产物保存到 artifacts/ 下，返回路径。"""
    path = ARTIFACTS_DIR / filename
    path.write_text(content, encoding="utf-8")
    print(f"\n💾 已保存产物: {path}")
    return path


if __name__ == "__main__":
    # 冒烟测试：一次最小 LLM 调用
    print(f"TASK1_DIR = {TASK1_DIR}")
    print(f"no_proxy = {os.environ.get('no_proxy')}")
    llm = make_llm()
    reply = llm.think([{"role": "user", "content": "请只回复两个字：你好"}])
    print(f"LLM 回复: {reply!r}")
    assert extract_json('前缀文字 ```json\n{"a": 1}\n``` 后缀') == {"a": 1}
    assert extract_json('{"a": "包含}花括号的字符串"}') == {"a": "包含}花括号的字符串"}
    assert extract_json("没有任何JSON") is None
    print("extract_json 自测通过。")
