from abc import ABC, abstractmethod
class Homework(ABC):
    # 检测程序是否正确
    @abstractmethod
    def check(self, *func) -> bool:
        pass
    # 打印答案
    @abstractmethod
    def show(self):
        pass
    # 打印提示
    @abstractmethod
    def hint(self) -> str:
        pass

    def assert_msg(self, str:str) -> str:
        return "Result error for input: " + str

class Homework_1_1_1(Homework):
    def check(self, func) -> bool:
        try:
            assert func([1,2,3,4,5]) == [1,2,3,4,5], self.assert_msg("[1,2,3,4,5]")
            assert func([1,2,2]) == [1,2], self.assert_msg("[1,2,2]")
            assert func(['12']) == ['12'], self.assert_msg("['12']")
            assert func([]) == [], self.assert_msg("[]")
        except AssertionError as e:
            print(f"Assertion Error: {e}")
            return False
        return True
    
    def show(self):
        code_string = 'def remove_duplicates(arr:list) -> str:\n\tres = []\n\tres=list(set(arr))\n\treturn res'
        print(code_string)
    
    def hint(self) -> str:
        return "Try using the set() function to eliminate duplicate elements and then convert it back to a list."

class Homework_1_3_1(Homework):
    def check(self, *func) -> bool:
        try:
            assert func[0](2).area() == 12.56, self.assert_msg("Circle(2)")
            assert func[1](12).area() == 144, self.assert_msg("Square(12)")
        except AssertionError as e:
            print(f"Assertion Error: {e}")
            return False
        return True
    def show(self):
        print('class Circle(Shape):\n\tdef __init__(self, radius):\n\t\tself.radius = radius\n\tdef area(self):\n\t\treturn 3.14 * self.radius ** 2\nclass Square(Shape):\n\tdef __init__(self, side):\n\t\tself.side = side\n\tdef area(self):\n\t\treturn self.side ** 2')
    def hint(self) -> str:
        return '圆的面积计算公式是3.14*r^2, 正方形的面积计算公式是side*side'
class HomeworkFactory:
    @staticmethod
    def get_homework(id: str) -> Homework:
        class_name = "Homework_"+id.replace('.','_')
        homework_class = globals().get(class_name)
        
        if homework_class and issubclass(homework_class, Homework):
            return homework_class()
        else:
            raise ValueError(f"Invalid homework id: {id}")