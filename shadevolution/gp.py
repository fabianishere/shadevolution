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


def delete_statement(individual, pset):
    statement_names = ['set', 'if', 'ret', 'seq']
    # Find all statements amongst the primitives which we can somehow delete.
    statements = []
    for i, val in enumerate(individual):
        if isinstance(val, gp.Primitive):
            if val.name.split('/')[0] in statement_names:
                statements.append([i, val])

    # Choose random statement to delete
    st_idx = random.randrange(len(statements))
    index = statements[st_idx][0]
    node = statements[st_idx][1]
    slice_ = individual.searchSubtree(index)

    # In case the statement is an assignment (set), then set the right hand side to the default terminal of the same
    # type to prevent undefined variable errors
    if node.name.split('/')[0] == 'set':
        ret_type = individual[index + 2].ret
        slice_ = individual.searchSubtree(index + 2)
        del individual[slice_]
        if ret_type == shader.Float:
            node = gp.Terminal(0.0, False, shader.Float)
        elif ret_type == shader.Bool:
            node = gp.Terminal(False, False, shader.Bool)
        else:
            node = gp.Terminal('void', False, shader.Unit)
        individual.insert(index + 2, node)
    else:
        del individual[slice_]
    return individual,


# Mutates a single individual for the next generation by deleting, inserting or replacing subtrees
def mutate_individual(individual, pset):
    rand = random.random()
    # Mutate individual in one of three ways with equal probability
    if rand < 0.25:
        return delete_statement(individual, pset)
    if rand < 0.5:
        return gp.mutInsert(individual, pset)
    if rand < 0.75:
        return gp.mutShrink(individual)

    # Can throw errors if it cannot find a good replacement, in that case, return original to prevent further errors
    try:
        return gp.mutNodeReplacement(individual, pset)
    except:
        return individual,

def setup_operators(toolbox, pset):
    """
    Setup the genetic operators to be used during the evolution.
    :param toolbox: The toolbox to register the operators to.
    :param pset: The primitive set to pick new nodes from.
    """
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", mutate_individual, pset=pset)
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
