from pycparser import parse_file

if __name__ == "__main__":
    ast = parse_file("../resources/programs/fresnel.glsl", use_cpp=False)
    ast.show()
