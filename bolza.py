import json
import os
import numpy as np
from decimal import Decimal, getcontext
import operator
import sys
from hyperbolic import F, Polynomial, jsonInput


def remove_exponent(d):
    string = str(d)
    if string[0] == "-":
        sign = "-"
        string = string[1:]
    else:
        sign = ""
    splitted = string.split("E")
    if len(splitted) == 1:
        return sign + string
    mantissa, exponent = splitted
    exponent = int(exponent)
    splitted = mantissa.split(".")
    if len(splitted) == 1:
        exponent += len(mantissa)
        before, after = "0", mantissa
    else:
        before, after = splitted
        if before == "0":
            pass
        else:
            exponent += len(before)
            after = before + after
            before = "0"
    while after[0] == "0":
        exponent -= 1
        after = after[1:]
    if exponent <= 0:
        return sign + "0." + ("0" * (-exponent)) + after
    elif exponent <= len(after):
        return sign + after[: exponent] + "." + after[exponent:]
    else:
        return sign + after + ((exponent - len(after)) * "0")

def read_output(lmax, workingDir):
    with open(f"{workingDir}/tmp/gapPy{lmax}_out/out.txt") as f:
        result = f.readlines()[0]
    if result == 'terminateReason = "found dual feasible solution";\n':
        return True
    elif result == 'terminateReason = "found primal feasible solution";\n':
        return False
    else:
        raise Exception("SDPB returned a weird response.")

def max_index(normalization):
    return np.argmax(np.abs(list(map(Decimal, normalization))))

def outputFromTxt(lmax, normalization, workingDir):
    with open(f"{workingDir}/tmp/gapPy{lmax}_out/y.txt") as f:
        y = list(map(Decimal, f.read().split()[2:]))
    norm = list(map(Decimal, normalization))
    index = max_index(norm)
    norm0 = norm.pop(index)
    y.insert(index, (1 - sum(map(operator.mul, norm, y)) / norm0))
    return y

def gap_test(lambdaGap, n, ln, lmax, sdpbPrec, procsPerNode, dualityGapThreshold, workingDir, mac):
    os.chdir(workingDir)
    print("Testing", lambdaGap)
    # os.system(f"rm -r {workingDir}/tmp/*")
    if ln == 1:
        normalization = [remove_exponent(F(n, 2 * n + l)(0)) for l in range(lmax + 1)]
        objective = ["0"] * (lmax + 1)
        polynomials = [[["0"] if i != j else ["-1"] for i in range(lmax + 1)] for j in list(range(0, lmax, 2))] + [
                    #    [[remove_exponent(F(n, 2 * n + l)(0))] for l in range(lmax + 1)]] + [
                       [list(map(remove_exponent, F(n, 2 * n + l).shift(-lambdaGap).coefficients)) for l in range(lmax + 1)]]
    else:
        normalization = [remove_exponent(Decimal(-(1 + ln) / ln) * F(n, 2 * n + l)(0)) for l in range(lmax + 1)] + [remove_exponent(-F(n, 2 * n + l)(0)) for l in range(lmax + 1)]
        objective = ["0"] * (2 * lmax + 2)
        polynomials = ([[["0"] if i != j else ["1"] for i in range(lmax + 1)] + ([["0"]] * (lmax + 1)) for j in list(range(0, lmax + 1, 2))] +
                       [([["0"]] * (lmax + 1)) + [["0"] if i != j else ["1"] for i in range(lmax + 1)] for j in list(range(1, lmax + 1, 2))] +
                       [[list(map(remove_exponent, (Polynomial([((Decimal(1) - Decimal(ln)) / Decimal(ln))]) * F(n, 2 * n + l).shift(-lambdaGap)).coefficients)) for l in range(lmax + 1)] +
                        [list(map(remove_exponent, (F(n, 2 * n + l).shift(-lambdaGap).coefficients))) for l in range(lmax + 1)]] +
                       [[list(map(remove_exponent, (Polynomial([(-(Decimal(1) + Decimal(ln)) / Decimal(ln))]) * F(n, 2 * n + l).shift(-lambdaGap)).coefficients)) for l in range(lmax + 1)] +
                        [list(map(remove_exponent, ((-F(n, 2 * n + l)).shift(-lambdaGap).coefficients))) for l in range(lmax + 1)]])
        
    with open(f"{workingDir}/tmp/gapPy{lmax}.json", "w") as f:
        json.dump(jsonInput(objective, normalization, polynomials), f, indent=4)
    if mac:
        sdp2input = f"/usr/local/bin/docker run -v {workingDir}/tmp/:/usr/local/share/sdpb wlandry/sdpb:2.5.1 mpirun --allow-run-as-root -n 4 sdp2input --precision={sdpbPrec} --input=/usr/local/share/sdpb/gapPy{lmax}.json --output=/usr/local/share/sdpb/gapPy{lmax}"
        sdpb = f"/usr/local/bin/docker run -v {workingDir}/tmp/:/usr/local/share/sdpb wlandry/sdpb:2.5.1 mpirun --allow-run-as-root -n 4 sdpb  --findPrimalFeasible --findDualFeasible --precision={sdpbPrec} --procsPerNode={procsPerNode} --dualityGapThreshold={dualityGapThreshold} --primalErrorThreshold={dualityGapThreshold} --dualErrorThreshold={dualityGapThreshold} -s /usr/local/share/sdpb/gapPy{lmax}"
        print(sdp2input)
        print(sdpb)
    else:
        sdp2input = f"sdp2input --precision={sdpbPrec} --input={workingDir}/tmp/gapPy{lmax}.json --output={workingDir}/tmp/gapPy{lmax}"
        sdpb = f"sdpb  --findPrimalFeasible --findDualFeasible --precision={sdpbPrec} --procsPerNode={procsPerNode} --dualityGapThreshold={dualityGapThreshold} --primalErrorThreshold={dualityGapThreshold} --dualErrorThreshold={dualityGapThreshold} -s {workingDir}/tmp/gapPy{lmax}"
    os.system(sdp2input)
    os.system(sdpb)
    result = read_output(lmax, workingDir)
    print("Tested:", lambdaGap, result)
    return result

# for i in range(2, 31):
#     print(f"lmax={i}")
#     OPE_bound(i, 200)

def binarySearch(lambdaMin, lambdaMax, n, ln, gapPrec, lmax, pythonPrec, sdpbPrec, procsPerNode, dualityGapThreshold, workingDir, mac):
    if lambdaMax - lambdaMin < gapPrec:
        bound = lambdaMax
        print("inputs", bound, n, ln, lmax, sdpbPrec, procsPerNode, dualityGapThreshold, workingDir, mac)
        gap_test(bound, n, ln, lmax, sdpbPrec, procsPerNode, dualityGapThreshold, workingDir, mac)
        if ln == 1:
            normalization = [remove_exponent(F(nn, 2 * nn + l)(0)) for l in range(lmax + 1)]
            coefficients = list(map(remove_exponent, outputFromTxt(lmax, normalization, workingDir)))
            jsonResults = {"n" : n,
                           "ln" : ln,
                           "lmax" : lmax,
                           "gapPrec" : float(gapPrec),
                           "pythonPrec" : pythonPrec,
                           "sdpbPrec" : sdpbPrec,
                           "procsPerNode" : procsPerNode,
                           "bound" : str(bound),
                           "coefficients" : coefficients}
        else:
            normalization = [remove_exponent(Decimal(-(1 + ln) / ln) * F(n, 2 * n + l)(0)) for l in range(lmax + 1)] + [remove_exponent(-F(n, 2 * n + l)(0)) for l in range(lmax + 1)]
            coefficients = list(map(remove_exponent, outputFromTxt(lmax, normalization, workingDir)))
            qs, rs = coefficients[: lmax + 1], coefficients[lmax + 1 :]
            assert len(qs) == len(rs)
            jsonResults = {"n" : n,
                           "ln" : ln,
                           "lmax" : lmax,
                           "gapPrec" : float(gapPrec),
                           "pythonPrec" : pythonPrec,
                           "sdpbPrec" : sdpbPrec,
                           "procsPerNode" : procsPerNode,
                           "bound" : str(bound),
                           "qs" : qs,
                           "rs" : rs}
        with open(f"{workingDir}/outputJsons/gapPy{lmax}_out.json", "w") as f:
            json.dump(jsonResults, f, indent=4)
        return lambdaMax
    else:
        lambdaMiddle = (lambdaMin + lambdaMax) / 2
        print("inputs", lambdaMiddle, n, ln, lmax, sdpbPrec, procsPerNode, dualityGapThreshold, workingDir, mac)
        if gap_test(lambdaMiddle, n, ln, lmax, sdpbPrec, procsPerNode, dualityGapThreshold, workingDir, mac):
            lambdaMax = lambdaMiddle
        else:
            lambdaMin = lambdaMiddle
        return binarySearch(lambdaMin, lambdaMax, n, ln, gapPrec, lmax, pythonPrec, sdpbPrec, procsPerNode, dualityGapThreshold, workingDir, mac)

if __name__ == "__main__":
    if sys.argv[1] == "test":
        getcontext().prec = 200
        # print(gap_test(Decimal(3.839141845703125), 1, 2, 23, 1024, 4, 1e-30, "/Users/alexradcliffe/kcl/hyperbolic-python/tmpDir", "mac"))
        
        # print(Decimal(3.839))
        # print(gap_test(Decimal(3.839), 1, 2, 23, 1024, 4, 1e-30, "/Users/alexradcliffe/kcl/hyperbolic-python/tmpDir", "mac"))
        n = 1
        ln = 2
        lmax = 5
        lambdaGap = Decimal(384262)/Decimal(100000)
        print(gap_test(lambdaGap, n, ln, lmax, 1024, 4, 1e-30, "/Users/alexradcliffe/kcl/hyperbolic-python/tmpDir", "mac"))
        os.system("cp /Users/alexradcliffe/kcl/hyperbolic-python/tmpDir/tmp/gapPy5.json /Users/alexradcliffe/kcl/hyperbolic-python/testJson.json")
        # numbers = ['5.5', '3.25', '4.375', '3.8125', '4.09375', '3.953125', '3.8828125', '3.84765625', '3.830078125', '3.8388671875', '3.84326171875', '3.841064453125', '3.8399658203125', '3.83941650390625', '3.839141845703125', '3.8392791748046875', '3.83921051025390625', '3.839244842529296875', '3.8392620086669921875', '3.83925342559814453125', '3.839257717132568359375', '3.8392555713653564453125', '3.83925449848175048828125', '3.839253962039947509765625', '3.83925449848175048828125']
        # numbers = ['5.5', '3.25', '4.375', '3.8125', '4.09375', 
        #            '3.953125', '3.8828125', '3.84765625',
        #            '3.830078125', '3.8388671875', '3.84326171875',
        #            '3.841064453125', '3.8399658203125', '3.83941650390625',
        #            '3.839141845703125']
        # numbers = map(Decimal, numbers)
        # results = []
        # for number in numbers:
        #     result = gap_test(number, 1, 2, 23, 2048, 4, 1e-60, "/Users/alexradcliffe/kcl/hyperbolic-python/tmpDir", True)
        #     results.append(result)
        # print(results)
        # print(numbers)
        assert False
    config_file = sys.argv[1]
    with open(config_file) as f:
        config = json.load(f)
    pythonPrec = config["pythonPrec"]
    getcontext().prec = pythonPrec
    lambdaMin = Decimal(config["lambdaMin"])
    lambdaMax = Decimal(config["lambdaMax"])
    n = config["n"]
    ln = config["ln"]
    gapPrec = Decimal(config["gapPrec"])
    lmaxLower = config["lmaxLower"]
    lmaxUpper = config["lmaxUpper"]
    sdpbPrec = config["sdpbPrec"]
    procsPerNode = config["procsPerNode"]
    dualityGapThreshold = config["dualityGapThreshold"]
    oddOnly = config["oddOnly"]
    workingDir = config["workingDir"]
    mac = config["mac"]
    step = 2 if oddOnly else 1
    if oddOnly and lmaxLower % 2 == 0:
        lmaxLower += 1
    for lmax in range(lmaxLower, lmaxUpper + 1, step):
        print(lambdaMin, lambdaMax, gapPrec)
        print(binarySearch(lambdaMin, lambdaMax, n, ln, gapPrec, lmax, pythonPrec, sdpbPrec, procsPerNode, dualityGapThreshold, workingDir, mac))