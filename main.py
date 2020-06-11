#!/usr/bin/env python3
import sys
import argparse

import numpy as np
import moderngl_window as mglw
from deap import tools, algorithms
from pycparser import parse_file

from shadevolution import shader, gp as sgp
from shadevolution.evaluator import Evaluator


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

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                                   stats=stats, halloffame=hof, verbose=True)

    print("Hall of Fame:")
    for individual in hof:
        diff = shader.diff(name, params, tree, individual)
        sys.stdout.writelines(diff)
        print('')  # Ensure that we write a newline


def main():
    parser = argparse.ArgumentParser(description='Optimize shaders through genetic programming.')
    parser.add_argument('--population-size', type=int, default=25, help='size of the initial population',
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
