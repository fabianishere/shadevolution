import textwrap

from pycparser import c_ast, parse_file
from deap import base, creator, gp


class ShaderParser(c_ast.NodeVisitor):
    """
    A visitor which converts a shader definition into a DEAP genetic programming tree.
    """

    def convert(self, node):
        """
        Convert the given C Abstract Syntax Tree (AST) to a DEAP genetic programming tree.

        :param node: The root node of the AST.
        :return: A primitive tree for DEAP.
        """
        self.tree = gp.PrimitiveTree([])
        self.visit(node)
        return self.tree

    def visit_Compound(self, node):
        atom = gp.Primitive("seq", [object] * len(node.block_items), object)
        self.tree += [atom]

        if node.block_items:
            self.visit(node.block_items)

    def visit_Decl(self, node):
        atom = gp.Primitive("def", [object] * 2, object)
        name = gp.Terminal(node.name, True, object)

        if node.init:
            self.tree += [atom, name]
            self.visit(node.init)

    def visit_Constant(self, node):
        atom = gp.Terminal(node.value, False, object)
        self.tree += [atom]

    def visit_BinaryOp(self, node):
        atom = gp.Primitive(node.op, [object] * 2, object)
        self.tree += [atom]

        self.visit(node.left)
        self.visit(node.right)

    def visit_If(self, node):
        atom = gp.Primitive('if', [object] * 3, object)
        void = gp.Terminal('void', True, object)

        self.tree += [atom]

        self.visit(node.cond)
        if node.iftrue:
            self.visit(node.iftrue)
        else:
            self.tree += [void]

        if node.iffalse:
            self.visit(node.iffalse)
        else:
            self.tree += [void]

    def visit_ID(self, node):
        atom = gp.Terminal(node.name, True, object)
        self.tree += [atom]

    def visit_Assignment(self, node):
        atom = gp.Primitive('set', [object] * 2, object)
        self.tree += [atom]
        self.visit(node.lvalue)
        self.visit(node.rvalue)

    def visit_Return(self, node):
        atom = gp.Primitive('return', [object], object)
        self.tree += [atom]
        self.visit(node.expr)

    def visit_FuncCall(self, node):
        args_count = len(node.args.exprs) if node.args else 0
        atom = gp.Primitive(node.name.name, [object] * args_count, object)
        self.tree += [atom]

        if node.args:
            self.visit(node.args)


class ShaderWriter:
    """
    A class to convert a DEAP genetic programming tree into a GLSL shader.
    """
    def convert(self, tree):
        """
        Convert the specified DEAP genetic programming tree into its shader source code.
        :param tree: The tree to convert.
        :return: The shader source code as a string.
        """
        string = ""
        stack = []
        defined = set()
        for node in tree:
            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()

                if not prim.arity:
                    string = self.convert_terminal(prim)
                else:
                    string = self.convert_primitive(prim, args, defined)

                if len(stack) == 0:
                    break  # If stack is empty, all nodes should have been seen
                stack[-1][1].append(string)
        return string

    def convert_terminal(self, node):
        if node.name == 'void':
            return ''
        else:
            return node.name

    def convert_primitive(self, prim, args, defined):
        if prim.name == 'seq':
            return ';\n'.join(args) + ';'
        elif prim.name == 'def' or prim.name == 'set':
            if args[0] in defined:
                return f'{args[0]} = {args[1]}'
            else:
                defined.add(args[0])
                return f'float {args[0]} = {args[1]}'
        elif prim.name == 'if':
            return f'''if ({args[0]}) {{\n{textwrap.indent(args[1], '    ')}\n}} else {{\n{textwrap.indent(args[2], '    ')}\n}}'''
        elif prim.name == 'return':
            return f'return {args[0]}'
        elif prim.name in ('+', '-', '/', '*', '<', '>', '>=', '<=', '&&', '||'):
            return f'({args[0]} {prim.name} {args[1]})'
        else:
            return f'{prim.name}({", ".join(args)})'


if __name__ == "__main__":
    ast = parse_file("resources/programs/fresnel.glsl", use_cpp=False)
    v = ShaderParser()
    w = ShaderWriter()
    tree = v.convert(ast)
    print(tree)

    shader = w.convert(tree)
    print(shader)
