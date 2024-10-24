# 类定义
# Python并没有真正的私有化支持，但可用下划线得到伪私有。   尽量避免定义以下划线开头的变量！
# （1）_xxx      " 单下划线 " 开始的成员变量叫做保护变量，意思是只有类实例和子类实例能访问到这些变量，
# 需通过类提供的接口进行访问； 不能用'from module import *'导入
# （2）__xxx    类中的私有变量/方法名 （Python的函数也是对象，所以成员方法称为成员变量也行得通。）,
# " 双下划线 " 开始的是私有成员，意思是 只有类对象自己能访问，连子类对象也不能访问到这个数据。
# （3）__xxx__ 系统定义名字， 前后均有一个“双下划线” 代表python里特殊方法专用的标识，如 __init__（）代表类的构造函数


class A:
    def __init__(self):
        self.__name = "python"  # 私有变量，翻译成 self._A__name='python'

    def __say(self):  # 私有方法,翻译成 def _A__say(self)
        print(self.__name)  # 翻译成 self._A__name


a = A()
# print a.__name #访问私有属性,报错!AttributeError: A instance has no attribute '__name'
print(a.__dict__)  # 查询出实例a的属性的集合
print(a._A__name)  # 这样，就可以访问私有变量了
# a.__say()#调用私有方法，报错。AttributeError: A instance has no attribute '__say'
print(dir(a))  # 获取实例的所有属性和方法
a._A__say()  # 这样，就可以调用私有方法了
