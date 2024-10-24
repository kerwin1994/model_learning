# 类定义
# Python并没有真正的私有化支持，但可用下划线得到伪私有。   尽量避免定义以下划线开头的变量！
# （1）_xxx      " 单下划线 " 开始的成员变量叫做保护变量，意思是只有类实例和子类实例能访问到这些变量，
# 需通过类提供的接口进行访问； 不能用'from module import *'导入
# （2）__xxx    类中的私有变量/方法名 （Python的函数也是对象，所以成员方法称为成员变量也行得通。）,
# " 双下划线 " 开始的是私有成员，意思是 只有类对象自己能访问，连子类对象也不能访问到这个数据。
# （3）__xxx__ 系统定义名字， 前后均有一个“双下划线” 代表python里特殊方法专用的标识，如 __init__（）代表类的构造函数


class people:
    # 定义基本属性
    name = ""
    age = 0
    # 定义私有属性,私有属性在类外部无法直接进行访问
    __weight = 0

    # 定义构造方法
    def __init__(self, n, a, w):
        self.name = n
        self.age = a
        self.weight = w

    def speak(self):
        print("%s 说: 我 %d 岁。" % (self.name, self.age))


# 单继承示例
class student(people):  # student为子类，people为父类
    grade = ""

    def __init__(self, n, a, w, g):
        # 调用父类的构函
        super(student, self).__init__(n, a, w)  # 方法一
        # people.__init__(self, n, a, w) #方法二
        self.grade = g

    # 覆写父类的方法
    def speak(self):
        print(
            "%s 说: 我 %d 岁了，我在读 %d 年级，我 %d 斤了"
            % (self.name, self.age, self.grade, self.weight)
        )


s = student("ken", 10, 60, 3)
v = student("maggie", 8, 50, 2)

student.name_group = "kerwin"

# 查询出实例a的属性的集合
print(s.__dict__)
# 获取实例的所有属性和方法
print(dir(s))

s.speak()
print(s.name)  # 输出: 更新后的类普通变量
v.speak()
print(v.name)  # 输出: 更新后的类普通变量

print(s.name_group)  # 输出: 更新后的类属性
print(v.name_group)
