# 有类A，现在想创建类B继承类A，但是类B的self.c需要改为False，我们尝试实现


class A:
    def __init__(self):
        self.a = 100
        self.b = "xxx"
        self.c = True

    def func1(self):
        pass


class B(A):
    def __init__(self):
        A.__init__(self)
        super(B, self).__init__()
        self.c = False

    def func2(self):
        print(self.b)
        print(self.c)


b = B()
b.func2()


class A:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        print(self.b)
        self.prints()

    def prints(self):
        print("self.A:", self.a)


class Zilei(A):
    def __init__(self, a):
        # super目的：调用父类的__init__()函数进行初始化
        super(Zilei, self).__init__(a, a + 1)
        self.a = a + 100
        print("self.a:", self.a)

    def prints(self):
        print("self.b:", self.b)


ass = Zilei(1)

# 1/ 先执行super(Zilei, self).__init__(a,a+1)，即执行A.__init__
# 所以输出print(self.b)，即2
# 再执行self.prints() ,值得注意的是A.prints()，在子类zilei被重写，所以应该执行的是zilei.prints()，即输出：self.b:2
# 2/返回到zilei.__init__执行print('self.a:', self.a) 输出 self.a:101

# 最终输出的结果：
# 2
# self.b:2
# self.a:101
