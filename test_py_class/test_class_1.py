# 类定义
# Python并没有真正的私有化支持，但可用下划线得到伪私有。   尽量避免定义以下划线开头的变量！
# （1）_xxx      " 单下划线 " 开始的成员变量叫做保护变量，意思是只有类实例和子类实例能访问到这些变量，
# 需通过类提供的接口进行访问； 不能用'from module import *'导入
# （2）__xxx    类中的私有变量/方法名 （Python的函数也是对象，所以成员方法称为成员变量也行得通。）,
# " 双下划线 " 开始的是私有成员，意思是 只有类对象自己能访问，连子类对象也不能访问到这个数据。
# （3）__xxx__ 系统定义名字， 前后均有一个“双下划线” 代表python里特殊方法专用的标识，如 __init__（）代表类的构造函数


class MyClass:
    __class_attribute = "这是一个私有类类属性"
    class_attribute = "这是一个普通类类属性"

    def __init__(self, instance_attribute):
        self.instance_attribute = instance_attribute


# 创建两个实例
obj1 = MyClass(1)
obj2 = MyClass(2)
# obj3 = MyClass1(2)
# 修改类属性
MyClass.class_attribute = "更新后的普通类属性"

# 访问类属性
print(obj1.class_attribute)  # 输出: 更新后的类属性
print(obj1.__dict__)
print(dir(obj1))
print(obj1._MyClass__class_attribute)  # 输出: 私有类属性

# 修改类属性
MyClass._MyClass__class_attribute = "更新后的私有类属性"
print(obj2.class_attribute)  # 输出: 更新后的类属性
print(obj2._MyClass__class_attribute)  # 输出: 私有类属性


class MyClass1(MyClass):
    class_attribute1 = "这是一个类1属性"

    def __init__(self, instance_attribute, class_attribute1):
        MyClass.__init__(self, instance_attribute)
        self.class_attribute1 = class_attribute1


obj3 = MyClass1(1, 2)
print(obj3.__dict__)
