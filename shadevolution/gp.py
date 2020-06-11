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
    gp.mutInsert(individual, pset)
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

    for i, (name, _) in enumerate(params):
        pset.renameArguments(**{f'ARG{i}': name})
    return pset


def setup_creator():
    """
    Setup the creator instance for DEAP. Use negative weights as we want to minimize error and frame time
    :return: The creator instance.
    """
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    return creator


def setup_toolbox(creator, pset, genesis):
    """
    Setup a toolbox for DEAP.
    :param creator: The creator to use.
    :param pset: The primitive set to use for node creation.
    :param genesis: The initial tree to start with.
    :return: The created toolbox.
    """
    toolbox = base.Toolbox()
    toolbox.register("individual", generate_individual, creator.Individual, pset=pset, genesis=genesis)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox


def setup_operators(toolbox, pset):
    """
    Setup the genetic operators to be used during the evolution.
    :param toolbox: The toolbox to register the operators to.
    :param pset: The primitive set to pick new nodes from.
    """
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutInsert, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=16)


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
