import numpy as np
import moderngl_window as mglw
from deap import tools, algorithms
from pycparser import parse_file

from shadevolution import shader, gp as sgp
from shadevolution.evaluator import Evaluator

if __name__ == "__main__":
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

    pop = toolbox.population(n=100)
    hof = tools.ParetoFront()  # Child of tools.HallOfFame
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=5,
                                   stats=stats, halloffame=hof, verbose=True)

