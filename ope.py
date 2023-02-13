import json
import os
import itertools
import numpy as np
from decimal import *
import operator
import sys


class Polynomial:

    def __init__(self, coefficients):
        """ input: coefficients are in the form a_0, a_1,... a_d 
        """
        self.coefficients = coefficients # list of coefficients
        

    def __repr__(self):
        """
        method to return the canonical string representation 
        of a polynomial.
   
        """
        return "Polynomial" + str(self.coefficients)

    
    def __call__(self, x):    
        res = 0
        for coeff in self.coefficients[::-1]:
            res = res * x + coeff
        return res 

    
    def degree(self):
        return len(self.coefficients)   

    
    def __add__(self, other):
        c1 = self.coefficients
        c2 = other.coefficients
        res = [sum(pair) for pair in itertools.zip_longest(c1, c2, fillvalue=0)]
        return Polynomial(res)


    def __neg__(self):
        return Polynomial([-c for c in self.coefficients])
    

    def __sub__(self, other):
        return self+(-other)

    
    def __mul__(self, other):
        c1 = self.coefficients
        c2 = other.coefficients
        res = [0] * (len(c1) + len(c2) - 1)
        for i, ci in enumerate(c1):
            for j, cj in enumerate(c2):
                res[i+j] += ci * cj
        return Polynomial(res)
 

    def derivative(self):
        coeffs = self.coefficients
        derived_coeffs = [i*ci for i, ci in enumerate(coeffs)[1:]]
        return Polynomial(derived_coeffs)

    
    def __str__(self):
    
        def x_expr(power):
            if power == 0:
                res = ""
            elif power == 1:
                res = "x"
            else:
                res = "x^"+str(power)
            return res

        degree = self.degree()
        res = ""
        for power, coeff in enumerate(self.coefficients):
            # nothing has to be done if coeff is 0:
            if coeff == 1:
                if power == 0:
                    res += "1"
                else:
                    res += "+" + x_expr(power)
            elif coeff == -1:
                if power == 0:
                    res += "-1"
                else:
                    res += "-" + x_expr(power)
            elif coeff > 0:
                res += "+" + str(coeff) + x_expr(power)
            elif coeff < 0:
                res += str(coeff) + x_expr(power)
        
        if res == "":
            res = "0"

        return res.lstrip('+')    # removing leading '+'
        

def product(l):
    res = l[0]
    for item in l[1:]:
        res *= item
    return res

def Pochhammer(a, n):
    product = 1
    for i in range(n):
        product *= a+i
    return product

def factorial(n):
    return Pochhammer(1, n)

def term(n, p, a, b, c):
    num = (-1) ** a * Pochhammer(2 * n + a, c) * Pochhammer(1 - p, b) ** 2
    den = factorial(c) * Pochhammer(2 - 2 * p, b) * factorial(b) * factorial(a) ** 2
    prodTerms = [Polynomial([k * (k + 1), 1]) for k in range(a)]
    return product([Polynomial([Decimal(num) / den])] + prodTerms)

def F(n, p):
    triples = [(a, b, p - 2 * n - a - b)
               for a in range(p - 2 * n + 1)
               for b in range(p - 2 * n - a + 1)]
    terms = [term(n, p, a, b, c) for a, b, c, in triples]
    return sum(terms, Polynomial([0]))

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
    DampedRational = {
                    "base": "0.367879",
                    "poles": [],
                    "constant": "1"
                    }
    jsonInput = {
        "objective": objective,
        "normalization": normalization,
        "PositiveMatrixWithPrefactorArray": [
            {
                "DampedRational": DampedRational,
                "polynomials": [
                    [
                        polynomial
                    ]
                ]
            } for polynomial in polynomials
        ]
    }
    with open(f"{directory}/tmp/opepy{lmax}.json", "w") as f:
        json.dump(jsonInput, f, indent=4)
    if mac:
        sdp2input = f"/usr/local/bin/docker run -v {directory}/tmp/:/usr/local/share/sdpb wlandry/sdpb:2.5.1 mpirun --allow-run-as-root -n 4 sdp2input --precision={sdpbPrec} --input=/usr/local/share/sdpb/opepy{lmax}.json --output=/usr/local/share/sdpb/opepy{lmax}"
        sdpb = f"/usr/local/bin/docker run -v {directory}/tmp/:/usr/local/share/sdpb wlandry/sdpb:2.5.1 mpirun --allow-run-as-root -n 4 sdpb --precision={sdpbPrec} --procsPerNode={procsPerNode} --dualityGapThreshold={dualityGapThreshold} -s /usr/local/share/sdpb/opepy{lmax}"
        print(sdpb)
    else:
        sdp2input = f"sdp2input --precision={sdpbPrec} --input={directory}/tmp/opepy{lmax}.json --output={directory}/tmp/opepy{lmax}"
        sdpb = f"sdpb --precision={sdpbPrec} --procsPerNode={procsPerNode} --dualityGapThreshold={dualityGapThreshold} -s {directory}/tmp/opepy{lmax}"
    os.system(sdp2input)
    os.system(sdpb)
    bound = read_output(lmax, directory)
    coefficients = outputFromTxt(lmax, normalization, directory)
    jsonResults = {"lmax" : lmax,
                "pythonPrec" : pythonPrec,
                "sdpbPrec" : sdpbPrec,
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
    for lmax in range(2 * int(lmaxLower / 2) + 1, lmaxUpper + 1, step): # 
        OPE_bound(lmax, pythonPrec, sdpbPrec, dualityGapThreshold, procsPerNode, mac)

