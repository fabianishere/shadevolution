from pycparser import parse_file
from shadevolution import shader, gp as sgp
from deap import gp
import sys

if __name__ == "__main__":
    ast = parse_file("resources/programs/fresnel.glsl", use_cpp=False)
    name, params, tree = shader.parse(ast.ext[0])[0]
    pset = sgp.generate_pset(name, params, tree)

    cp = gp.PrimitiveTree(tree)
    for _ in range(2):
        gp.mutInsert(cp, pset)
        gp.mutNodeReplacement(cp, pset)
        gp.mutShrink(cp)

    res = sgp.diff(name, params, tree, cp)
    sys.stdout.writelines(res)
