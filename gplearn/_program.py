import numpy as np
from copy import copy
from typing import List, Tuple, Optional, Dict
from gplearn.functions import _Function

class _Program:
    """
    Attributes
    __________
    arities: Dict[int, _Function]
             A dictionary of the form '{arity: [functions]}'. airty是函数参数个数，functions必须在function_set中
    """
    def __init__(self,
                 function_set: List[_Function],
                 arities,
                 init_depth,
                 init_method,
                 n_features: int,
                 feature_names
                 ):
        self.function_set: List[_Function] = function_set
        self.arities: Dict[int, _Function] = arities
        self.init_depth = init_depth
        self.init_method = init_method
        self.n_features: int = n_features
        self.feature_names = feature_names

    def build_program(self, random_state):

        if self.init_method =='half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method

        # 最大深度
        max_depth = random_state.randint(*self.init_depth)

        # 初始化
        function = random_state.randint(len(self.function_set))  # 随机挑选一个函数结点
        function = self.function_set[function]                   # 随机挑选一个函数
        program = [function]                                     # 向program中添加函数
        terminal_stack = [function.arity]                        # 向terminal_stack中添加函数参数个数

        # 开始生成树
        while terminal_stack:
            # 树的深度
            depth: int = len(terminal_stack)
            # choice等于特征数+函数数，表示选择数
            choice: int = self.n_features + len(self.function_set)
            # 随机选择
            choice: int = random_state.randint(choice)

            # 如果树的深度小于最大深度且选择full的方法生成树且choice小于等于可选函数个数
            # 在full的情况下优先叠加函数
            if (depth < max_depth) and (method == 'full' or choice <= len(self.function_set)):
                #  随机挑选一个函数
                function = random_state.randint(len(self.function_set))
                function = self.function_set[function]
                # 向program中添加函数
                program.append(function)
                # 向terminal_stack中添加函数参数数目
                terminal_stack.append(function.arity)
            else:
                # 如果包括常数
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features+1)
                # 如果不包含常数
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1

        return None

    def validate_program(self):
        terminals = [0]
        for node in self.program:
            # 每一个结点都是一个_Function类型, terminals用来记录每个函数的参数个数
            if isinstance(node, _Function):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self):
        terminals = [0]
        output = ''
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                output += node.name + '('
            # 遇到node是数的情况
            else:
                # 遇到node是整数的情况
                if isinstance(node, int):
                    if self.feature_names is None:
                        output += f"X{node}"
                    else:
                        output += self.feature_names[node]
                # 遇到node是浮点数的情况
                else:
                    pass

    def _depth(self):
        pass

    def _length(self):
        """
        计算program中的函数和变量的总数

        Returns
        -------
        length: int
                program中函数和变量的总数
        """
        length: int = len(self.program)
        return length

    def execute(self, X):
        node = self.program[0]

    def fitness(self):
        pass

    def get_subtree(self, random_state, program=None):
        """
        从program中随机获取子树,即随机选取子数初始点

        Parameters
        ----------
        random_state: RandomState instance
                      随机数生成器

        program: List[Optional], default None
                 平摊(flatterned)的树，按照深度优先搜索的方式排列

        Returns
        -------
        start, end: tuple of two ints
                    子树的起点和终点索引
        """
        if program is None:
            program = self.program
        # 重组点的选择依照Koza's (1992) 使用的方法: 90%的概率选择函数，10%的概率选择变量
        probs = np.array([0.9 if isinstance(node, _Function) else 0.1 for node in program])
        # 累积概率，最后一个元素为1，长度与program一致
        probs = np.cumsum(probs / probs.sum())
        # 随机选出某个位置, 作为子树的起始点
        start = np.searchsorted(probs, random_state.uniform())

        # stack表示树的结点数目，初始为1（初始结点为1）
        stack = 1
        # 一开始end等于start
        end = start
        # stack > end-start表示没有到最终遍历所有的结点(深度优先搜索), 这可以通过例子证明
        # 案例一:
        # add(x, y)
        # 初始: start = 1, stack = 1, end = 1
        # 第一轮: start = 1, stack = 3, end = 2   add结点
        # 第二轮: start = 1, stack = 3, end = 3   x结点
        # 第三轮: start = 1, stack = 3, end = 4，遍历结束   y结点
        # 案例二:
        # add(x, add(y, z))
        # 初始: start = 1, stack = 1, end = 1
        # 第一轮: start = 1, stack = 3, end = 2  add结点
        # 第二轮: start = 1, stack = 3, end = 3  x结点
        # 第三轮: start = 1, stack = 5, end = 4  add结点
        # 第四轮: start = 1, stack = 5, end = 5  y结点
        # 第五轮: start = 1, stack = 5, end = 6  z结点
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1
        return start, end

    def crossover(self, donor, random_state) -> Tuple[Optional]:
        """
        重组

        从
        Parameters
        ----------
        donor
        random_state

        Returns
        -------

        """
        # 随机生成被替换的子树
        start, end = self.get_subtree(random_state)
        removed = range(start, end)
        # 随机生成被贡献的子树
        donor_start, donor_end = self.get_subtree(random_state, donor)
        donor_removed = list(set(range(len(donor))) - set(range(donor_start, donor_end)))

        return (self.program[:start] + donor[donor_start:donor_end] + self.program[end:], removed, donor_removed)

    def subtree_mutation(self, random_state):
        """

        Parameters
        ----------
        random_state

        Returns
        -------

        """
        pass

    def hoist_mutation(self, random_state):
        """
        hoist mutation

        先随机生成一个子树，再在子树中再生成子子树，删除子树中不在子子树的部分

        Parameters
        ----------
        random_state: RandomState instance
                      随机数生成器

        Returns
        -------

        """
        # 随机生成子树
        start, end = self.get_subtree(random_state)
        subtree = self.program[start:end]

        # 在子数中再随机生成子子树
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        hoist = subtree[sub_start:sub_end]

        # subtree中但不在sub_start与sub_end之间的结点被删除
        removed = list(set(range(start, end))-set(range(start+sub_start, start+sub_end)))

        return self.program[:start] + hoist + self.program[end:], removed

    def point_mutation(self, random_state):
        program: List[_Function] = copy(self.program)
        # 随机生成0-1的node结点数的随机数，获取其中小于p_point_replace的数的横坐标，代表这些点需要变异
        mutate = np.where(random_state.uniform(size=len(program)) < self.p_point_replace)[0]

        # 逐个遍历需要变异的node
        for node in mutate:
            # 如果node是函数，则将该node替换为其他具有相同参数的函数
            if isinstance(program[node], _Function):
                arity: int = program[node].arity
                # 用具有相同数量参数的函数来代替当前的node
                replacement: int = len(self.arities[arity])
                replacement = random_state.randint(replacement)
                replacement:_Function = self.arities[arity][replacement]
                program[node] = replacement
            else:
                pass
                













