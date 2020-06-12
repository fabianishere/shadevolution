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
    mutate_individual(individual, pset)
    return individual


def mutate_individual(individual, pset):
    """
    Mutate a single individual for the next generation by deleting, inserting or replacing subtrees
    :param individual: The individual to mutate.
    :param pset:
    :return:
    """
    rand = random.random()

    # Perform pre-mutations
    if rand < 0.33:
        individual, = mutInlineChild(individual)

    rand = random.random()

    # Mutate individual in one of three ways with equal probability
    if rand < 0.33:
        return mutShrink(individual)
    elif rand < 0.66:
        return mutInsert(individual, pset)
    else:
        return mutNodeReplacement(individual, pset)


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
        # Ignore sequencing functions since they don't generate any mutations with side-effects
        if name.startswith('seq'):
            continue

        pset.addPrimitive(fun, fun.params, fun.ret, name=name)

    pset.addTerminal(0.0, shader.Float)
    pset.addTerminal(1.0, shader.Float)
    pset.addTerminal(2.0, shader.Float)
    pset.addTerminal(-1.0, shader.Float)
    pset.addTerminal(-2.0, shader.Float)
    pset.addTerminal(False, shader.Bool)
    pset.addTerminal(True, shader.Bool)
    pset.addTerminal('void', shader.Unit)

    # BUG: Ephemeral constants are not properly initialized by DEAP
    # pset.addEphemeralConstant('e', lambda: random.random(), shader.Float)

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
    toolbox.register("mutate", mutate_individual, pset=pset)
    toolbox.register("select", tools.selNSGA2)


def algorithm(population, toolbox, cxpb, mutpb, ngen, stats=None,
              halloffame=None, verbose=__debug__):
    """
    Run the genetic algorithm as described in the paper.
    :param population: The initial population to use.
    :param toolbox: The toolbox to use for the iterations.
    :param cxpb: The probability of a cross-over event.
    :param mutpb: The probability of a mutation event.
    :param ngen: The number of generations to perform.
    :param stats: The stats object to record the statistics with.
    :param halloffame: The HallOfFame object to record the fittest individuals.
    :param verbose: A flag to enable verbose printing.
    :return: A tuple consisting of the final population and the logbook.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    offspring = []
    mutated = []
    total = len(population)
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        population = toolbox.select(population, total)
        population = tools.selTournamentDCD(population, total)

        # Vary the pool of individuals
        offspring = [toolbox.clone(ind) for ind in population]
        # Apply crossover and mutation on the offspring
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                              offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        mutated = [toolbox.clone(ind) for ind in offspring]
        for i in range(len(offspring)):
            if random.random() < mutpb:
                mutated[i], = toolbox.mutate(offspring[i])
                del mutated[i].fitness.values

        # Evaluate the offspring individuals with an invalid fitness
        invalid_ind_off = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind_off)
        for ind, fit in zip(invalid_ind_off, fitnesses):
            ind.fitness.values = fit

        # Evaluate the mutated individuals with an invalid fitness
        invalid_ind_mut = [ind for ind in mutated if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind_mut)
        for ind, fit in zip(invalid_ind_mut, fitnesses):
            ind.fitness.values = fit

        # Add the offspring and mutated individuals to the population as per the paper
        population.extend(offspring)
        population.extend(mutated)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(population)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind_off) + len(invalid_ind_mut), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def mutShrink(individual):
    """
    Mutate the tree by randomly selecting and deleting a subtree of the tree, replacing them with the unit value of
    their return type.
    :param individual: The tree to mutate.
    """
    index = random.randrange(len(individual))
    prim = individual[index]
    slice_ = individual.searchSubtree(index)
    name = prim.name.split('/')[0]

    # In case the statement is an assignment (set), then set the right hand side to the default terminal of the same
    # type to prevent undefined variable errors
    if name == 'set':
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
    elif prim.ret == shader.Float:
        node = gp.Terminal(0.0, False, shader.Float)
        del individual[slice_]
        individual.insert(index, node)
    elif prim.ret == shader.Bool:
        node = gp.Terminal(False, False, shader.Bool)
        del individual[slice_]
        individual.insert(index, node)
    elif prim.ret == shader.Unit:
        node = gp.Terminal('void', False, shader.Unit)
        del individual[slice_]
        individual.insert(index, node)
    return individual,


def mutNodeReplacement(individual, pset):
    """
    Replace a randomly chosen primitive from *individual* by a randomly
    chosen primitive with the same number of arguments from the `pset`.
    attribute of the individual.
    :param individual: The normal or typed tree to be mutated.
    :returns: A tuple of one tree.
    """
    if len(individual) < 2:
        return individual,

    index = random.randrange(1, len(individual))
    node = individual[index]

    if node.arity == 0:  # Terminal
        terms = pset.terminals[node.ret]
        var = list([name for name, type in _find_in_scope(individual, index).items() if issubclass(type, node.ret)])
        choice = len(terms) + len(var)

        # If there is no replacement available: abort
        if choice <= 0:
            return individual,

        i = random.randrange(choice)

        if i < len(terms):
            term = random.choice(terms)
        else:
            term = gp.Terminal(random.choice(var), True, node.ret)

        # Fix for ephemeral constants
        if term.arity != 0:
            term.arity = 0
        individual[index] = term
    elif node.name not in {'set/2', 'return/1'}:
        # Prevent replacement of set and return calls due to their order sensitivity
        prims = [p for p in pset.primitives[node.ret] if _can_substitute_call(node, p)]

        # Make sure that a replacement is available
        if prims:
            individual[index] = random.choice(prims)

    return individual,


def mutInsert(individual, pset):
    """
    Insert a new branch at a random position in individual. The subtree
    at the chosen position is used as child node of the created subtree, in
    that way, it is really an insertion rather than a replacement.
    :param individual: The tree to be mutated.
    :returns: A tuple of one tree.
    """
    index = random.randrange(len(individual))
    node = individual[index]
    slice_ = individual.searchSubtree(index)
    choice = random.choice

    # As we want to keep the current node as children of the new one,
    # it must accept the return value of the current node
    primitives = [p for p in pset.primitives[node.ret] if any([issubclass(node.ret, pa) for pa in p.args])]

    if len(primitives) == 0:
        return individual,

    scope = _find_in_scope(individual, index)

    new_node = choice(primitives)
    new_subtree = []
    position = choice([i for i, a in enumerate(new_node.args) if issubclass(node.ret, a)])

    for i, arg_type in enumerate(new_node.args):
        if i != position:
            terms = _generate_node(arg_type, scope, pset)
            new_subtree.extend(terms)
        else:
            new_subtree.extend(individual[slice_])

    new_subtree.insert(0, new_node)
    individual[slice_] = new_subtree
    return individual,


def mutInlineChild(individual):
    """
    Mutate the specified individual by replacing a random expression by one of its children.
    :param individual: The individual to perform the mutation on.
    :return: A tuple that consists of the mutated individual.
    """
    index = random.randrange(1, len(individual))
    node = individual[index]

    children = _find_children(individual, index)

    # Bail out when the node has no children
    if not children:
        return individual,

    child_index = random.choice(children)
    child = individual[child_index]

    # Test whether the types are compatible
    if node.ret != child.ret:
        return individual,

    subtree = individual.searchSubtree(index)
    child_tree = individual.searchSubtree(child_index)
    new_individual = individual[:index] + individual[child_tree] + individual[subtree.stop:]
    individual.clear()
    individual.extend(new_individual)
    return individual,


def _generate_node(ret_type, scope, pset):
    """
    Generate a primitive value.
    :param ret_type:  The return type of the primitive to generate.
    :param scope: The scope in which to generate the primitive.
    :param pset: The primitive set to generate from.
    :return: The generated primitive and its children.
    """
    rand = random.random()

    # TODO: Parameterize this probability
    if rand < 0.2:
        return _generate_primitive(ret_type, scope, pset)
    else:
        return [_generate_terminal(ret_type, scope, pset)]


def _generate_terminal(arg_type, scope, pset):
    """
    Generate a terminal value.
    :param arg_type: The type of argument to generate.
    :param scope: The scope in which to generate the terminal.
    :param pset: The primitive set to generate from.
    :return: The generated terminal.
    """
    var = [gp.Terminal(name, True, type) for name, type in scope.items() if issubclass(type, arg_type)]
    return random.choice(pset.terminals[arg_type] + var)


def _generate_primitive(ret_type, scope, pset):
    """
    Generate a primitive value.
    :param ret_type:  The return type of the primitive to generate.
    :param scope: The scope in which to generate the primitive.
    :param pset: The primitive set to generate from.
    :return: The generated primitive and its children.
    """
    primitive = random.choice(pset.primitives[ret_type])
    nodes = [primitive]
    for i, arg_type in enumerate(primitive.args):
        nodes.extend(_generate_node(arg_type, scope, pset))
    return nodes


def _find_children(individual, index):
    """
    Find the children of a node at the given index.
    :param individual: The individual to find the nodes in.
    :param index: The index of the node to find the children of.
    :return: A list of indices of the children.
    """
    node = individual[index]
    children = []
    i = index + 1
    for _ in range(node.arity):
        children.append(i)
        subtree = individual.searchSubtree(i)
        i = subtree.stop
    return children


def _find_ancestors(individual, index):
    """
    Find the ancestors of the node at the given index.
    :param individual: The individual to find the nodes in.
    :param index: The index of the node to find the ancestors of.
    :return: A list of indices that represent the ancestors.
    """
    visited = 0
    ancestors = []
    for j in reversed(range(index)):
        arity = individual[j].arity
        if arity > visited:
            ancestors.append(j)
            visited = 0
        else:
            visited = (visited - arity) + 1
    return ancestors


def _find_in_scope(individual, index):
    """
    Determine the number of variables in scope for the node at the specified index.
    :param individual: The individual to search in.
    :param index: The index of the node to find its scope for.
    :return: The variables and their types in scope.
    """
    res = {}
    for ancestor in _find_ancestors(individual, index):
        i = ancestor + 1
        while i < index:
            subtree = individual.searchSubtree(i)
            node = individual[i]

            if node.name.startswith('set'):
                res[individual[i + 1].name] = individual[i + 2].ret
            i = subtree.stop

    return res


def _can_substitute_call(lhs, rhs):
    """
    Determine whether original call can be substituted by the other call.
    :param lhs: The original call.
    :param rhs: The call to check whether it can be substituted.
    :return: `True` if the new call can be substituted in place of the original call.
    """
    if lhs.arity != rhs.arity:
        # Functions do not accept the same amount of parameters
        return False

    # Arguments must be contravariant
    return all([issubclass(l, r) for l, r in zip(lhs.args, rhs.args)])
