import inspect
from typing import Callable
class _Function:

    def __init__(self, function: Callable, name: str, arity: int) -> None:
        self.function = function
        self.name = name
        self.arity = arity

    def __call__(self, *args):
        return self.function(*args)

def make_function(function: Callable, name: str, arity: int, wrap: bool =True) -> _Function:
    """
    构建函数节点

    Parameters
    ----------
    function:
    name
    arity
    wrap

    Returns
    -------

    """
    # 参数检查
    # arity必须为整数，代表函数的参数个数
    if not isinstance(arity, int):
        raise ValueError(f"arity must be an int, got {type(arity)}")
    # function必须是函数类型
    if not inspect.isfunction(function):
        raise ValueError(f"function must be a function, got {type(function)}")
    # 检查function的参数个数与arity是否一致
    elif function.__code__.co_argcount != arity:
        raise ValueError(f"arity {arity} does not match required number of "
                         f"function arguments of {function.__code__.co_argcount}")

    return _Function(function=function,
                     name=name,
                     arity=arity)