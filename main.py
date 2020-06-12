#!/usr/bin/env python3
import argparse
import random
import sys

import moderngl_window as mglw
import numpy as np
from deap import tools
from pycparser import parse_file

from shadevolution import shader, gp as sgp
from shadevolution.evaluator import Evaluator


def run(pop_size, ngen, cxpb, mutpb, enable_plot=False):
    """
    Run the parameterized experiment.

    :param pop_size: The size of the population.
    :param ngen: The number of generations.
    :param cxpb: The probability of a cross-over event.
    :param mutpb: The probability of a mutation event.
    :param enable_plot: A flag to enable live plotting of results.
    """
    window_cls = mglw.get_local_window_cls()
    window = window_cls(
        title="Genetic Programming for Shader Optimization",
        gl_version=(4, 1),
        vsync=False
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
    evaluator = Evaluator(window, enable_plot=enable_plot)
    baseline = evaluator.determine_baseline()

    toolbox.register("evaluate", evaluator.eval, genesis=tree, baseline=baseline)

    pop = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()  # Child of tools.HallOfFame
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, log = sgp.algorithm(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                             stats=stats, halloffame=hof, verbose=True)

    print(f'Hall of Fame ({len(hof)} individuals):')
    for individual in hof:
        frame_time, error = individual.fitness.values
        print('Fitness:')
        print(f'Avg Frame Time [ns]: {frame_time}')
        print(f'Avg Frame Error: {error}')
        print('Variant:')
        diff = shader.diff(name, params, tree, individual)
        sys.stdout.writelines(diff)
        print('')  # Ensure that we write a newline


def main():
    parser = argparse.ArgumentParser(description='Optimize shaders through genetic programming.')
    parser.add_argument('--population-size', type=int, default=300, help='size of the initial population',
                        dest='pop_size')
    parser.add_argument('--generations', type=int, default=100, help='number of generations to perform',
                        dest='ngen')
    parser.add_argument('--cross-over', type=float, default=0.5, help='probability of a cross-over event',
                        dest='cxpb')
    parser.add_argument('--mutation', type=float, default=0.2, help='probability of a mutation event', dest='mutpb')
    parser.add_argument('--seed', type=int, default=False, help='seed for rng', dest='seed')
    parser.add_argument('--disable-plot', action='store_true', default=False, dest='disable_plot', help='disable live '
                                                                                                        'plot')
    args = parser.parse_args()

    if args.seed is not False:
        random.seed(args.seed)

    run(args.pop_size, args.ngen, args.cxpb, args.mutpb, enable_plot=not args.disable_plot)


if __name__ == "__main__":
    main()
