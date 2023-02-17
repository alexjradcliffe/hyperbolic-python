import json
import os


if __name__ == "__main__":
    data = "/Users/alexradcliffe/kcl/hyperbolic-python/macData/"
    data = "/Users/alexradcliffe/kcl/hyperbolic-python/clusterData100/"
    outputJsons = f"{data}/outputJsons"
    files = os.listdir(outputJsons)
    bounds = {}
    coefficients = {}
    for file in files:
        with open(f"{outputJsons}/{file}") as f:
            j = json.load(f)
        lmax = j["lmax"]
        bounds[lmax] = j["bound"]
        coefficients[lmax] = {}
        for i, coeff in enumerate(j["coefficients"]):
            coefficients[lmax][i] = coeff
    lmaxs = sorted(bounds.keys())
    bounds = [bounds[i] for i in lmaxs]
    coefficients = [coefficients[i] for i in lmaxs]
    with open(f"{data}/lmaxs.json", "w") as f:
        json.dump(lmaxs, f, indent = 4)
    with open(f"{data}/bounds.json", "w") as f:
        json.dump(bounds, f, indent = 4)
    with open(f"{data}/coefficients.json", "w") as f:
        json.dump(coefficients, f, indent = 4)