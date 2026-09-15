import ast
import operator
import re

# 白名单：二元/一元运算符
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# 计算器的参数说明，注册进 ToolExecutor 后用于失败引导（习题 6.3.2）
CALCULATOR_PARAM_HINT = (
    "action_input 必须是一个纯算术表达式字符串，只允许数字和 + - * / // % ** ( ) ，"
    '例如 "(123 + 456) * 789 / 12"。支持 × ÷ ^ （ ）等写法的自动转换，'
    "不要包含汉字、单位或其他文字。"
)


def _normalize(expr: str) -> str:
    """把模型常见的写法规范成合法的 Python 算术表达式。"""
    expr = expr.strip()
    # 去掉结尾的 "="、"=?"、"等于多少" 等
    expr = re.sub(r"(=|＝)\s*\??\s*$", "", expr)
    expr = re.sub(r"[?？]$", "", expr)
    # 全角/常见符号转换
    for src, dst in [
        ("×", "*"),
        ("x", "*"),
        ("X", "*"),
        ("÷", "/"),
        ("^", "**"),
        ("（", "("),
        ("）", ")"),
        ("＋", "+"),
        ("－", "-"),
        ("，", ""),
        (",", ""),
        (" ", ""),
        ("\t", ""),
        ("\n", ""),
    ]:
        expr = expr.replace(src, dst)
    return expr


def _safe_eval(node: ast.AST) -> float | int:
    """在语法树上递归求值，仅放行白名单节点。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"不支持的常量: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Pow):
            # 防御超大幂导致的资源耗尽
            if abs(right) > 10_000 or (abs(left) > 10_000 and right not in (0, 1, -1)):
                raise ValueError("指数过大，拒绝计算")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(
        f"不支持的表达式元素: {type(node).__name__}（仅允许数字与 + - * / // % ** 括号）"
    )


def calculate(expression: str) -> str:
    """计算器工具：接收算术表达式字符串，返回计算结果。

    供 ReAct 智能体在 Action 中调用，例如 Calculator[(123 + 456) × 789 / 12]。
    """
    print(f"🧮 正在执行 [Calculator] 计算: {expression}")
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("表达式为空")
    if len(expression) > 200:
        raise ValueError("表达式过长（>200字符）")

    expr = _normalize(expression)
    if not expr:
        raise ValueError("表达式为空")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(
            f"'{expression}' 不是合法的算术表达式（转换后为 '{expr}'）: {e.msg}。"
            f"参数要求: {CALCULATOR_PARAM_HINT}"
        ) from e

    try:
        result = _safe_eval(tree)
    except OverflowError as e:
        raise ValueError(f"计算结果溢出: {e}") from e

    # 整数值的浮点结果转成整数展示
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"计算结果: {expression.strip()} = {result}"


if __name__ == "__main__":
    # 冒烟测试
    assert (
        calculate("(123 + 456) × 789/ 12 = ?")
        == "计算结果: (123 + 456) × 789/ 12 = ? = 38069.25"
    )
    assert calculate("2^10 + 3^5") == "计算结果: 2^10 + 3^5 = 1267"
    assert calculate("（1+2）×3") == "计算结果: （1+2）×3 = 9"
    for bad in ["import os", "1+2; print('x')", "abc", "", "__import__('os')"]:
        try:
            calculate(bad)
            raise AssertionError(f"应当拒绝: {bad}")
        except ValueError:
            pass
    print("calculator 冒烟测试全部通过:")
    print(" ", calculate("(123 + 456) × 789/ 12 = ?"))
    print(" ", calculate("2^10 + 3^5"))
