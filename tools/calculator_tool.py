"""3.14 兼容计算器工具。

库的 CalculatorTool._eval_node 引用了 Python 3.12 已删除的 ast.Num，
在本项目 (requires-python>=3.14) 上任何运算表达式都会 AttributeError。
此处保留库工具的 name/description/参数声明（注册表与提示词依赖它们），
仅把求值委托给 task1 的白名单安全实现。
"""

from hello_agents.tools import CalculatorTool

from task1.tools.calculator import calculate


class SafeCalculatorTool(CalculatorTool):
    def run(self, parameters):
        expression = parameters.get("input", "") or parameters.get("expression", "")
        return calculate(expression)
