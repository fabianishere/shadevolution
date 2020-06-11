#!/usr/bin/env python3
import sys
import argparse
import random

import numpy as np
import moderngl_window as mglw
from deap import tools, algorithms
from pycparser import parse_file

from shadevolution import shader, gp as sgp
from shadevolution.evaluator import Evaluator

def run_generations(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
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
        population.extend(offspring)
        population.extend(mutated)
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
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Evaluate the mutated individuals with an invalid fitness
        invalid_ind = [ind for ind in mutated if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

def run(pop_size, ngen, cxpb, mutpb):
    """
    Run the parameterized experiment.

    :param pop_size: The size of the population.
    :param ngen: The number of generations.
    :param cxpb: The probability of a cross-over event.
    :param mutpb: The probability of a mutation event.
    """
    window_cls = mglw.get_local_window_cls()
    window = window_cls(
        title="Genetic Programming for Shader Optimization",
        gl_version=(4, 1),
    )
    mglw.activate_context(ctx=window.ctx)

    if pop_size % 4 != 0:
        print("Population size must be multiple of 4")
        return

    # Parse shader
    ast = parse_file("resources/programs/fresnel.glsl", use_cpp=False)
    name, params, tree = shader.parse(ast.ext[0])[0]
    pset = sgp.generate_pset(name, params, tree)

    # Setup GP
    creator = sgp.setup_creator()
    toolbox = sgp.setup_toolbox(creator, pset, tree)
    sgp.setup_operators(toolbox, pset)

    # Setup evaluator
    evaluator = Evaluator(window)
    baseline = evaluator.determine_baseline()

    toolbox.register("evaluate", evaluator.eval, genesis=tree, baseline=baseline)

    pop = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()  # Child of tools.HallOfFame
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, log = run_generations(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                                   stats=stats, halloffame=hof, verbose=True)

    print("Hall of Fame:")
    for individual in hof:
        diff = shader.diff(name, params, tree, individual)
        sys.stdout.writelines(diff)
        print('')  # Ensure that we write a newline


def main():
    parser = argparse.ArgumentParser(description='Optimize shaders through genetic programming.')
    parser.add_argument('--population-size', type=int, default=24, help='size of the initial population',
                        dest='pop_size')
    parser.add_argument('--generations', type=int, default=5, help='number of generations to perform',
                        dest='ngen')
    parser.add_argument('--cross-over', type=float, default=0.5, help='probability of a cross-over event',
                        dest='cxpb')
    parser.add_argument('--mutation', type=float, default=0.2, help='probability of a mutation even', dest='mutpb')
    args = parser.parse_args()

    run(args.pop_size, args.ngen, args.cxpb, args.mutpb)


if __name__ == "__main__":
    main()
