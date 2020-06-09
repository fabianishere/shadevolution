from deap import base, creator, gp

pr = gp.Primitive("plus", (int, int), int)
x = gp.Terminal("x", True, int)
y = gp.Terminal("y", True, int)
tree = gp.PrimitiveTree([pr, x, pr, x, y])

if __name__ == "__main__":
    print(tree)

