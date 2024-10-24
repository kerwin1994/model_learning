from numpy import *
import numpy as np

print(eye(4))
a = np.array([1, 2, 3])

print(a)


# *self._args 表示接受元组类参数("a")；
# **kwargs     表示接受字典类参数(a=1)；
def foo(*args, **kwargs):
    print("args = ", args)
    print("kwargs = ", kwargs)
    print("---------------------------------------")


if __name__ == "__main__":
    foo(1, 2, 3, 4)
    foo(a=1, b=2, c=3)
    foo(1, 2, 3, 4, a=1, b=2, c=3)
    foo("a", 1, None, a=1, b="2", c=3)
    # 同时使用*args和**kwargs时，必须*args参数列要在**kwargs前，像foo(a=1, b='2', c=3, a', 1, None, )这样调用的话，会提示
    # 语法错误“SyntaxError: non-keyword arg after keyword arg”
    # foo(a=1, b='2', c=3, a, 1, None)
    # 查询出实例foo的属性的集合
