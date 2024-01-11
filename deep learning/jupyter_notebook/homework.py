from abc import ABC, abstractmethod
class Homework(ABC):
    # 检测程序是否正确
    @abstractmethod
    def check(self, func) -> bool:
        pass
    # 打印答案
    @abstractmethod
    def show(self) -> str:
        pass
    # 打印提示
    @abstractmethod
    def hint(self) -> str:
        pass

    def assert_msg(self, str:str) -> str:
        return "Result error for input" + str

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
    
    def show(self) -> str:
        code_string = 'def remove_duplicates(arr:list) -> str:\n\tres = []\n\tres=list(set(arr))\n\treturn res'
        print(code_string)
        return code_string
    
    def hint(self) -> str:
        return "Try using the set() function to eliminate duplicate elements and then convert it back to a list."

class HomeworkFactory:
    @staticmethod
    def get_homework(id: str) -> Homework:
        class_name = "Homework_"+id.replace('.','_')
        homework_class = globals().get(class_name)
        
        if homework_class and issubclass(homework_class, Homework):
            return homework_class()
        else:
            raise ValueError(f"Invalid homework id: {id}")