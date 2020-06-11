import difflib
import textwrap

from deap import gp
from pycparser import c_ast

ASSIGN = 'set'
SEQ = 'seq'
IF = 'if'
RET = 'ret'


class Any(object):
    pass


class Unit(Any):
    """
    The type with only one value: the Unit object.
    """
    pass


class Val(Any):
    """
    The value type that is used in the shader.
    """
    pass


class Float(Val):
    pass


class Bool(Val):
    pass


class Id:
    """
    An identifier of a variable.
    """

    def __init__(self, name):
        self.name = name


class FunDef:
    """
    A function definition.
    """

    def __init__(self, name, params, ret):
        """
        Construct a function definition.

        :param name: The name of the function.
        :param params: The parameter types of the function.
        :param ret: The return type of the function.
        """
        self.name = name
        self.params = params
        self.ret = ret
        self.__name__ = f'{name}/{self.arity}'

    @property
    def arity(self):
        """
        Return the arity of the function.
        :return: The arity of the function.
        """
        return len(self.params)

    def format(self, args, ctx):
        """
        Format the function call with the specified arguments.

        :param args: The argument with which the function has been called.
        :param ctx: A context that is shared between every call of the same tree.
        :return: The formatted function call.
        """
        return f'{self.name}({", ".join(args)})'


class SequenceDef(FunDef):
    """
    The function definition for a seq/n function.
    """

    def __init__(self, arity):
        """
        Construct a sequence function definition.

        :param arity: The arity of the function.
        """
        super().__init__(SEQ, [Any] * arity, Unit)

    def format(self, args, ctx):
        return '\n'.join([f'{arg};' for arg in args])


class AssignDef(FunDef):
    """
    The function definition for a variable assignment.
    """

    def __init__(self):
        super().__init__(ASSIGN, [Id, Val], Unit)

    def format(self, args, ctx):
        name, value = args
        is_defined = name in ctx['defined']
        if is_defined:
            return f'{name} = {value}'
        else:
            ctx['defined'].add(name)
            return f'float {name} = {value}'


class IfDef(FunDef):
    """
    The function definition for a conditional statement.
    """

    def __init__(self):
        super().__init__(IF, [Bool, Any, Any], Unit)

    def format(self, args, ctx):
        return f'''if ({args[0]}) {{\n{textwrap.indent(args[1], '    ')}\n}} else {{\n{textwrap.indent(args[2], '    ')}\n}}'''


class ReturnDef(FunDef):
    """
    The function definition for a return statement.
    """

    def __init__(self):
        super().__init__(RET, [Val], Unit)

    def format(self, args, ctx):
        return f'return {args[0]};'


class UnaryOpDef(FunDef):
    """
    A function definition for a binary operator.
    """

    def __init__(self, op, input, output):
        """
        Construct a binary operator definition.
        :param op: The binary operator to represent.
        :param input: The input type of the binary operator.
        :param output: The output type of the binary operator.
        """
        super().__init__(op, [input] * 2, output)

    def format(self, args, ctx):
        return f'({args[0]} {self.name} {args[1]})'


class BinaryOpDef(FunDef):
    """
    A function definition for a binary operator.
    """

    def __init__(self, op, input, output):
        """
        Construct a binary operator definition.
        :param op: The binary operator to represent.
        :param input: The input type of the binary operator.
        :param output: The output type of the binary operator.
        """
        super().__init__(op, [input] * 2, output)

    def format(self, args, ctx):
        return f'({args[0]} {self.name} {args[1]})'


FUN = {
    # Sequencing functions (e.g. chaining statements via ';')
    f'{SEQ}/0': SequenceDef(0),
    f'{SEQ}/1': SequenceDef(1),
    f'{SEQ}/2': SequenceDef(2),
    f'{SEQ}/3': SequenceDef(3),
    f'{SEQ}/4': SequenceDef(4),
    f'{SEQ}/5': SequenceDef(5),
    f'{SEQ}/6': SequenceDef(6),
    f'{SEQ}/7': SequenceDef(7),
    f'{SEQ}/8': SequenceDef(8),
    f'{SEQ}/9': SequenceDef(9),

    # Special functions (e.g. assignment, if and return)
    f'{ASSIGN}/2': AssignDef(),
    f'{IF}/3': IfDef(),
    f'{RET}/1': ReturnDef(),

    # Unary and binary operators
    '-/1': UnaryOpDef('-', Float, Float),
    '+/2': BinaryOpDef('+', Float, Float),
    '-/2': BinaryOpDef('-', Float, Float),
    '//2': BinaryOpDef('/', Float, Float),
    '*/2': BinaryOpDef('*', Float, Float),
    '</2': BinaryOpDef('<', Float, Bool),
    '>/2': BinaryOpDef('>', Float, Bool),
    '>=/2': BinaryOpDef('>=', Float, Bool),
    '<=/2': BinaryOpDef('<=', Float, Bool),
    '!=/2': BinaryOpDef('!=', Float, Bool),
    '==/2': BinaryOpDef('==', Float, Bool),
    '&&/2': BinaryOpDef('&&', Bool, Bool),
    '||/2': BinaryOpDef('||', Bool, Bool),

    # Regular functions
    'sin/1': FunDef('sin', [Float], Float),
    'cos/1': FunDef('cos', [Float], Float),
    'sqrt/1': FunDef('sqrt', [Float], Float),
    'pow/2': FunDef('pow', [Float, Float], Float),
    'log/1': FunDef('log', [Float], Float),
    'min/2': FunDef('min', [Float, Float], Float),
    'max/2': FunDef('max', [Float, Float], Float)
}


def parse(node):
    """
    Convert the given C Abstract Syntax Tree (AST) to a DEAP genetic programming tree.

    :param node: The root node of the AST.
    :return: A primitive tree for DEAP.
    """
    parser = ShaderParser()
    return parser.parse(node)


class ShaderParser(c_ast.NodeVisitor):
    """
    A visitor which converts a shader definition into a DEAP genetic programming tree.
    """

    def __init__(self):
        self.results = []
        self.vars = {}

    def parse(self, node):
        """
        Convert the given C Abstract Syntax Tree (AST) to a DEAP genetic programming tree.

        :param node: The root node of the AST.
        :return: A primitive tree for DEAP.
        """
        self.results = []
        self.vars = {}
        self.visit(node)
        return self.results

    def visit_FuncDef(self, node):
        name = node.decl.name
        params = list([(decl.name, decl.type.type.names[0]) for decl in node.decl.type.args.params])

        for name, type in params:
            self.vars[name] = type

        self.tree = gp.PrimitiveTree([])
        self.visit(node.body)
        self.results.append((name, params, self.tree))
        self.tree = None

    def visit_Compound(self, node):
        arity = len(node.block_items)
        atom = self._sequence(arity)
        self.tree += [atom]

        if node.block_items:
            self.visit(node.block_items)

    def visit_Decl(self, node):
        atom = self._assign()
        name = self._id(node.name)
        type = node.type.type.names[0]

        if type == 'float' or type == 'double' or type == 'int':
            real_type = Float
        elif node.type == 'bool':
            real_type = Bool
        else:
            real_type = Val
        self.vars[node.name] = real_type

        if node.init:
            self.tree += [atom, name]
            self.visit(node.init)

    def visit_Constant(self, node):
        if node.type == 'float' or node.type == 'double' or node.type == 'int':
            type = Float
        elif node.type == 'bool':
            type = Bool
        else:
            type = Val
        atom = self._literal(node.value, type)
        self.tree += [atom]

    def visit_UnaryOp(self, node):
        name = node.op
        fun = self._resolve_fun(name, 1)
        atom = self._call(fun)
        self.tree += [atom]
        self.visit(node.expr)

    def visit_BinaryOp(self, node):
        name = node.op
        fun = self._resolve_fun(name, 2)
        atom = self._call(fun)
        self.tree += [atom]

        self.visit(node.left)
        self.visit(node.right)

    def visit_If(self, node):
        atom = self._if()
        nop = self._sequence(0)

        self.tree += [atom]

        self.visit(node.cond)
        if node.iftrue:
            self.visit(node.iftrue)
        else:
            self.tree += [nop]

        if node.iffalse:
            self.visit(node.iffalse)
        else:
            self.tree += [nop]

    def visit_ID(self, node):
        if node.name not in self.vars:
            raise ValueError('Variable referenced before declaration')

        atom = self._var(node.name, self.vars[node.name])
        self.tree += [atom]

    def visit_Assignment(self, node):
        atom = self._assign()
        self.tree += [atom]
        self.visit(node.lvalue)
        self.visit(node.rvalue)

    def visit_Return(self, node):
        atom = self._return()
        self.tree += [atom]
        self.visit(node.expr)

    def visit_FuncCall(self, node):
        name = node.name.name
        arity = len(node.args.exprs) if node.args else 0
        fun = self._resolve_fun(name, arity)
        atom = self._call(fun)
        self.tree += [atom]

        if node.args:
            self.visit(node.args)

    @staticmethod
    def _resolve_fun(name, arity):
        proper_name = f'{name}/{arity}'

        if proper_name not in FUN:
            raise ValueError(f'Unknown function {proper_name}')

        return FUN[proper_name]

    @staticmethod
    def _id(name):
        return gp.Terminal(name, True, Id)

    @staticmethod
    def _sequence(arity):
        name = f'{SEQ}/{arity}'
        fun = FUN[name]
        return gp.Primitive(name, fun.params, fun.ret)

    @staticmethod
    def _assign():
        name = f'{ASSIGN}/2'
        fun = FUN[name]
        return gp.Primitive(name, fun.params, fun.ret)

    @staticmethod
    def _if():
        name = f'{IF}/3'
        fun = FUN[name]
        return gp.Primitive(name, fun.params, fun.ret)

    @staticmethod
    def _return():
        name = f'{RET}/1'
        fun = FUN[name]
        return gp.Primitive(name, fun.params, fun.ret)

    @staticmethod
    def _var(name, type):
        return gp.Terminal(name, True, type)

    @staticmethod
    def _literal(value, type):
        return gp.Terminal(value, False, type)

    @staticmethod
    def _call(fun):
        return gp.Primitive(f'{fun.name}/{fun.arity}', fun.params, fun.ret)


def write(name, params, tree):
    """
    Convert the specified DEAP genetic programming tree into its shader source code.

    :param name: The name of the shader function to create.
    :param params: The parameters of the shader.
    :param tree: The tree to convert.
    :return: The shader source code as string.
    """
    writer = ShaderWriter(name, params)
    return writer.write(tree)


class ShaderWriter:
    """
    A class to convert a DEAP genetic programming tree into a GLSL shader.
    """

    def __init__(self, name, params):
        """
        Construct a {@link ShaderWriter} instance.

        :param name: The name of the function name to generate.
        :param params: A list of parameters of the shader.
        """
        self.name = name
        self.params = params

    def write(self, tree):
        """
        Convert the specified DEAP genetic programming tree into its shader source code.
        :param tree: The tree to convert.
        :return: The shader source code as a string.
        """
        string = ""
        stack = []
        ctx = {
            'defined': set([name for name, _ in self.params])
        }
        for node in tree:
            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()

                if isinstance(prim, gp.Terminal):
                    string = self.convert_terminal(prim)
                else:
                    string = self.convert_primitive(prim, args, ctx)

                if len(stack) == 0:
                    break  # If stack is empty, all nodes should have been seen
                stack[-1][1].append(string)

        param_list = ', '.join([f'{typ} {param}' for param, typ in self.params])
        header = f'float {self.name}({param_list}) {{\n'
        footer = '\n}'
        return header + textwrap.indent(string, '    ') + footer

    def convert_terminal(self, node):
        if node.name.startswith('ARG'):
            # DEAP names the function arguments as ARG0, ARG1, ..., ARGn. Make sure that we use the correct parameter
            # names.
            name, _ = self.params[int(node.name[3:])]
            return name
        elif node.ret == Unit:
            # In case the SEQ has not been given any arguments, do not do anything: NOP
            return ''
        elif node.ret == Bool:
            return '1' if node.name else '0'
        else:
            return str(node.name)

    def convert_primitive(self, node, args, ctx):
        if node.name not in FUN:
            raise ValueError(f'Unknown primitive {node.name}')

        fun = FUN[node.name]
        return fun.format(args, ctx)


def diff(name, params, lhs, rhs):
    """
    Diff the source code of the two shader representations.
    :param name: The name of the shader.
    :param params: The parameters of the shader.
    :param lhs: The initial shader source code.
    :param rhs: The final shader source code.
    :return: The difference between the shaders.
    """
    source_before = write(name, params, lhs)
    source_after = write(name, params, rhs)

    return difflib.Differ().compare(source_before.splitlines(True), source_after.splitlines(True))
