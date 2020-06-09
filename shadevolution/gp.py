import difflib
import random

from deap import base, gp, creator, tools

from shadevolution import shader


def generate_individual(container, pset, genesis):
    """
    Generate a single individual in the population which represents a shader variant.
    :param container: The container to create the individual with.
    :param genesis: The initial shader to base the individual on.
    :return: The generated individual.
    """
    individual = container(genesis)
    gp.mutShrink(individual)
    gp.mutNodeReplacement(individual, pset)
    return individual


def generate_pset(name, params, tree):
    """
    Generate the primitive set for the specified shader function.
    :param name: The name of the shader function.
    :param params: The params of the shader function.
    :param tree: The AST of the shader function.
    :return: The primitive set of the function.
    """
    pset = gp.PrimitiveSetTyped(name, [object] * len(params), object)
    for name, fun in shader.FUN.items():
        pset.addPrimitive(fun, fun.params, fun.ret, name=name)

    pset.addTerminal(0.0, shader.Float)
    pset.addTerminal(1.0, shader.Float)
    pset.addTerminal(False, shader.Bool)
    pset.addTerminal(True, shader.Bool)
    pset.addTerminal('void', shader.Unit)

    vars = ['cosi', 'cost', 'R', 'sint', 'r_ortho', 'r_par']
    for var in vars:
        pset.addTerminal(var, shader.Id)
        pset.addTerminal(var, float)

    for i, (name, _) in enumerate(params):
        pset.renameArguments(**{f'ARG{i}': name})
    return pset


def setup_creator(pset):
    """
    Setup the creator instance for DEAP.
    :return: The creator instance.
    """
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)
    return creator


def setup_toolbox(creator, genesis):
    """
    Setup a toolbox for DEAP.
    :param creator: The creator to use.
    :param genesis: The initial tree to start with.
    :return: The created toolbox.
    """
    toolbox = base.Toolbox()
    toolbox.register("expr", generate_individual, genesis=genesis)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.expr)
    return toolbox


def swap_children(tree, idx):
    """
    Swap two children of a node in the tree.
    :param tree: The tree to swap the children in.
    :param idx: The index of the node to swap the children.
    """
    slice = tree.searchSubtree(idx)
    subtree = tree[slice]
    arity = subtree[0].arity

    if arity <= 0:
        return

    i = random.randrange(arity)
    j = random.randrange(arity)
    subtree[i], subtree[j] = subtree[j], subtree[i]
    tree[slice] = subtree


def diff(name, params, lhs, rhs):
    """
    Diff the source code of the two shader representations.
    :param name: The name of the shader.
    :param params: The parameters of the shader.
    :param lhs: The initial shader source code.
    :param rhs: The final shader source code.
    :return: The difference between the shaders.
    """
    source_before = shader.write(name, params, lhs)
    source_after = shader.write(name, params, rhs)

    return difflib.Differ().compare(source_before.splitlines(1), source_after.splitlines(1))
