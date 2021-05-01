import numpy as np
from copy import copy
from pandas import DataFrame
from numpy.random import RandomState
from gplearn.functions import _Function
from typing import List, Tuple, Optional, Dict, Union

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
                 const_range: Tuple[float],
                 random_state,
                 feature_names,
                 program: List[Optional] = None
                 ):
        self.function_set: List[_Function] = function_set
        self.arities: Dict[int, List[_Function]] = arities
        self.init_depth = init_depth
        self.init_method = init_method
        self.n_features: int = n_features
        self.const_range: Tuple[float] = const_range
        self.feature_names = feature_names
        self.program: List[Optional] = program

        # 检查program
        # 如果在实例化时输入了program, 则预先检查program
        if self.program is not None:
            # 如果program不有效，则报错
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        # 如果在实例化时没有输入program，则利用build_program先初始化program
        else:
            self.program = self.build_program(random_state)


    def build_program(self, random_state: RandomState) -> Union[List[Optional], None]:
        """
        在没有指定program的情况下，随机创建一个naive的program

        Parameters
        ----------
        random_state: numpy.random.RandomState
                      随机种子状态

        Returns
        -------
        program: List[Optional]
                 树的内容，元素为函数(_Function)或常数或变量
        """
        # 初始方法有full, grow, half and half三种方法
        if self.init_method =='half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method

        # 最大深度
        max_depth = random_state.randint(*self.init_depth)

        # 初始化，随机从function_set中选出一个函数
        # 将挑选出的函数添加到program中
        # 将挑选出的函数的参数个数添加到terminal_stack中
        function = random_state.randint(len(self.function_set))  # 随机挑选一个函数结点
        function = self.function_set[function]                   # 随机挑选一个函数
        program = [function]                                     # 向program中添加函数
        terminal_stack = [function.arity]                        # 向terminal_stack中添加函数参数个数

        # 开始生成树
        while terminal_stack:
            # 树的深度（函数个数=树的深度（指有多少层子结点），因为这里是按照深度优先搜索)
            depth: int = len(terminal_stack)
            # choice等于特征数+函数数，表示选择数
            choice: int = self.n_features + len(self.function_set)
            # 随机选择生成一个整数
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
                # 如果包括常数, cost_range代表常数数值范围
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features+1)
                # 如果不包含常数
                else:
                    terminal = random_state.randint(self.n_features)
                # 只有randint(self.n_features+1)的情况下才有可能满足条件，此时添加常数
                if terminal == self.n_features:
                    # 在cost_range所规定的范围内生成常数
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')

                # 如果是变量，则terminal为整数;如果是常数，则terminal为常数本身
                program.append(terminal)
                # 所在结点一个子结点完成处理
                terminal_stack[-1] -= 1
                # 如果一个结点的所有子结点都处理完毕，则删除该结点，同时上一层结点也-1
                # 如果所有子结点都处理完毕，则返回program，结束处理
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1
        # We should never get here
        return None

    def validate_program(self) -> bool:
        """
        检查self.program是否是有效

        检查方法原理: 如果是合法的，则函数的参数减1时必然会恰好将terminals中的元素pop，然后再减到-1，初始的0也是为了这个准备。

        检查方法举例:

        合法情况: 设表达式为add(x,y)

        1.初始: terminals = [0]

        2.第一轮: terminals = [0, 2]

        3.第二轮: terminals = [0, 1]

        4.第三轮: terminals = [0, 0], terminals = [0], terminals = [-1]

        非法情况: program = [add, x, y, z] (add仅接受2个参数)

        1.初始: terminals = [0]

        2.第一轮: terminals = [0, 2]

        3.第二轮: terminals = [0, 1]

        4.第三轮: terminals = [0, 0], terminals = [-1]

        5.第四轮: terminals = [-2]，最终返回False

        Returns
        -------
        terminals == [-1]: bool
                           program是否合法
        """
        # 最终结点的terminals
        terminals = [0]
        # 遍历program中的每个结点
        for node in self.program:
            # 如果结点为函数，则在terminals中添加函数参数个数
            if isinstance(node, _Function):
                terminals.append(node.arity)
            # 如果结点是变量(包括常量)，则不向terminals中添加, 反而对上一个函数结点进行减1操作
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self) -> str:
        """
        展示树代表的表达式

        Returns
        -------

        """
        terminals = [0]
        output = ''
        # 遍历（按深度优先搜索）
        for i, node in enumerate(self.program):
            # 如果结点是函数，则terminals中添加元素，代表深度增加了一层
            if isinstance(node, _Function):
                terminals.append(node.arity)
                output += node.name + '('
            # 遇到node是数的情况
            else:
                # 遇到node是整数的情况
                if isinstance(node, int):
                    # 如果没有feature_names属性，则直接为乘该整数
                    if self.feature_names is None:
                        output += f"X{node}"
                    # 如果有feature_names属性，则添加该属性
                    else:
                        output += self.feature_names[node]
                # 遇到node是浮点数的情况
                else:
                    output += '%.3f' % node
                # 处理完一个非函数结点后对terminals的最后一个元素减1
                terminals[-1] -= 1
                # 如果处理完一个函数结点
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                # 如果不是最后一个结点,则需要加逗号
                if i != len(self.program)-1:
                    output += ', '
        return output

    def _depth(self) -> int:
        """
        获取树的深度

        Returns
        -------
        depth-1: int
                 树的深度
        """
        # 初始化terminals
        terminals = [0]
        # 初始深度为1，但是实际深度为depth-1
        depth = 1
        # 遍历所有的结点
        for node in self.program:
            # 如果是函数结点
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] = -1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

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

    def execute(self, X: Dict[str, DataFrame]):
        # 最开始的结点
        node = self.program[0]
        # node为浮点数表示其为一个数字
        if isinstance(node, float):
            return np.repeat()


    # 从这里开始修改
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
                # 如果需要常数项
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features+1)
                else:
                    terminal = random_state.randint(self.n_features)
                # terminal如果能等于self.n_features, 说明需要常数项
                if terminal == self.n_features:
                    terminal = random_state.randint(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program[node] = terminal

        return program, list(mutate)

    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)

                













