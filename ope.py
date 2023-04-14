import json
import os
import numpy as np
from decimal import Decimal, getcontext
import operator
import sys
from hyperbolic import F, jsonInput

def read_output(lmax, directory):
    with open(f"{directory}/tmp/opepy{lmax}_out/out.txt") as f:
        return Decimal(f.readlines()[1][18:-2])

def max_index(normalization):
    return np.argmax(np.abs(list(map(Decimal, normalization))))

def outputFromTxt(lmax, normalization, directory):
    with open(f"{directory}/tmp/opepy{lmax}_out/y.txt") as f:
        y = list(map(Decimal, f.read().split()[2:]))
    n = list(map(Decimal, normalization))
    index = max_index(n)
    n0 = n.pop(index)
    y.insert(index, (1-sum(map(operator.mul, n, y))/n0))
    return y

def OPE_bound(lmax, pythonPrec, sdpbPrec, dualityGapThreshold, procsPerNode, mac):
    if mac:
        directory = "/Users/alexradcliffe/kcl/hyperbolic-python"
    else:
        directory = "/users/k21187236/scratch/hyperbolic-python"
    os.chdir(directory)
    nn = 6
    normalization = ["1"] + ["0"] * lmax
    objective = [str(F(nn, 2 * nn + l)(0)) for l in range(lmax + 1)]
    polynomials = [[list(map(str, F(nn, 2 * nn + l).coefficients)) for l in range(lmax + 1)]] + [
                [["0"] if i != j else ["-1"] for i in range(lmax+1)] for j in list(range(2, lmax, 2))]
    with open(f"{directory}/tmp/opepy{lmax}.json", "w") as f:
        json.dump(jsonInput(objective, normalization, polynomials), f, indent=4)
    if mac:
        sdp2input = f"/usr/local/bin/docker run -v {directory}/tmp/:/usr/local/share/sdpb wlandry/sdpb:2.5.1 mpirun --allow-run-as-root -n 4 sdp2input --precision={sdpbPrec} --input=/usr/local/share/sdpb/opepy{lmax}.json --output=/usr/local/share/sdpb/opepy{lmax}"
        sdpb = f"/usr/local/bin/docker run -v {directory}/tmp/:/usr/local/share/sdpb wlandry/sdpb:2.5.1 mpirun --allow-run-as-root -n 4 sdpb --precision={sdpbPrec} --procsPerNode={procsPerNode} --dualityGapThreshold={dualityGapThreshold} --primalErrorThreshold={dualityGapThreshold} --dualErrorThreshold={dualityGapThreshold} -s /usr/local/share/sdpb/opepy{lmax}"
        print(sdpb)
    else:
        sdp2input = f"sdp2input --precision={sdpbPrec} --input={directory}/tmp/opepy{lmax}.json --output={directory}/tmp/opepy{lmax}"
        sdpb = f"sdpb --precision={sdpbPrec} --procsPerNode={procsPerNode} --dualityGapThreshold={dualityGapThreshold} --primalErrorThreshold={dualityGapThreshold} --dualErrorThreshold={dualityGapThreshold} -s {directory}/tmp/opepy{lmax}"
    os.system(sdp2input)
    os.system(sdpb)
    bound = read_output(lmax, directory)
    coefficients = outputFromTxt(lmax, normalization, directory)
    jsonResults = {"lmax" : lmax,
                "pythonPrec" : pythonPrec,
                "sdpbPrec" : sdpbPrec,
                "dualityGapThreshold" : dualityGapThreshold,
                "procsPerNode" : procsPerNode,
                "bound" : str(bound),
                "coefficients" : list(map(str, coefficients))}
    with open(f"{directory}/outputJsons/opepy{lmax}_out.json", "w") as f:
        json.dump(jsonResults, f, indent=4)

# for i in range(2, 31):
#     print(f"lmax={i}")
#     OPE_bound(i, 200)


if __name__ == "__main__":
    config_file = sys.argv[1]
    with open(config_file) as f:
        config = json.load(f)
    pythonPrec = config["pythonPrec"]
    sdpbPrec = config["sdpbPrec"]
    lmaxLower = config["lmaxLower"]
    lmaxUpper = config["lmaxUpper"]
    procsPerNode = config["procsPerNode"]
    dualityGapThreshold = config["dualityGapThreshold"]
    oddOnly = config["oddOnly"]
    mac = config["mac"]
    getcontext().prec = pythonPrec
    step = 2 if oddOnly else 1
    for lmax in range(lmaxLower + 1 if lmaxLower % 2 == 0 else lmaxLower, lmaxUpper + 1, step):
        OPE_bound(lmax, pythonPrec, sdpbPrec, dualityGapThreshold, procsPerNode, mac)
