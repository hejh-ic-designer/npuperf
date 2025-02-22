from typing import Generator, Callable, List


class Stage:
    """Abstract superclass for Runnables"""
    def __init__(self, list_of_callables:List[Callable], **kwargs):
        """
        :param list_of_callables: a list of callables, that must have a signature compatible with this __init__ function
        (list_of_callables, *, required_kwarg1, required_kwarg2, kwarg_with_default=default, **kwargs)
        and return a Stage instance. This is used to flexibly build iterators upon other iterators.
        :param kwargs: any keyword arguments, irrelevant to the specific class in question but passed on down
        """
        self.kwargs = kwargs
        self.list_of_callables = list_of_callables

        if self.is_leaf() and list_of_callables not in ([], tuple(), set(), None):
            # 如果当前stage是叶节点，而且list_of_callables还不是空的话，说明stage还没走完，但是终止了
            raise ValueError("Leaf runnable received a non empty list_of_callables")

        if list_of_callables in ([], tuple(), set(), None) and not self.is_leaf():
            # 如果list_of_callables 已经空了，但是当前stage还不是叶stage，说明没用东西能迭代了，可能是没有声明叶stage
            raise ValueError("List of callables empty on a non leaf runnable, so nothing can be generated.\
                              Final callable in list_of_callables must return Stage instances that have is_leaf() == True")



    def run(self) -> Generator:
        """Runs the runnable.
        This requires no arguments and returns a generator yielding any amount of tuple, that each have
        a CostModelEvaluation as the first element and a second element that can be anything, meant only for manual
        inspection."""
        raise ImportError("Run function not implemented for runnable")
        # 这里的run 要raise error的原因在于，继承了Stage的其他Stage才是真正要run的，本stage只是作为父类提供一个规范或模板。
        # 如果子类stage没用写run的话，就会自动调用父类的stage，当然要给一个报错，这说明你子类的run没用运行

    def __iter__(self):
        return self.run()

    def is_leaf(self) -> bool:
        """
        :return: Returns true if the runnable is a leaf runnable, meaning that it does not use (or thus need) any substages
        to be able to yield a result. Final element in list_of_callables must always have is_leaf() == True, except
        for that final element that has an empty list_of_callables
        """
        # 这里要 return Flase 的意思是，只有最后一个stage会声明 is_leaf 这个函数而且返回True，其他的stage不声明，所以默认就调用父类，即此stage的 is_leaf函数，那当然是false了
        # 所以相当于，现在只用在最后一个stage声明 is_leaf 返回 True，其他stage不用写这个就好了
        return False

class MainStage:
    """
    Not actually a Stage, as running it does return (not yields!) a list of results instead of a generator
    Can be used as the main entry point
    """
    def __init__(self, list_of_callables, **kwargs):
        self.kwargs = kwargs
        self.list_of_callables = list_of_callables


    def run(self):
        """请不要使用mainstage.run() 的返回值, 因为它可能是一个不可控的复杂的数据结构 \\
        实际上, 每一个stage 都会对上一个stage 的返回值做一些处理, 将一些信息塞在extra_info 里一并传给上层, 
        有的stage在cme的位置可能是所有cme 的sum, 或者其他的cme 相关的对象, 这导致这里return 的值取决于
        stage pipeline之间的配合和各自的操作, 实际上是不可控的
        """
        answers = []
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        for cme, extra_info in substage.run():
            answers.append((cme, extra_info))
        return answers