# Shadevolution
Genetic Programming for Shader Simplification

## Introduction
This project implements the work on "Genetic Programming for Shader Simplification" by Sitthi-Amorn et al: 
> Pitchaya Sitthi-Amorn, Nicholas Modly, Westley Weimer, and Jason Lawrence. 2011. Genetic programming for shader 
> simplification. ACM Trans. Graph. 30, 6 (December 2011), 1–12. DOI:https://doi.org/10.1145/2070781.2024186

# Quick Start
Ensure you have installed the correct dependencies:
```bash
pip3 install -r .
```
Then, you can run the program as follows:
```bash
python3 main.py
```
This will spawn a window where a model is shown twice side-by-side respectively using the original
shader and the evolved shader. In the console, you can observe the progress of the generations. Afterwards,
the individuals in the Pareto front will be displayed. 

To list the available options, you can run:
```bash
python3 main.py --help
```

## License
Shadevolution is available under the [MIT license](LICENSE.txt).
