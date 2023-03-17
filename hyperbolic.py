from decimal import Decimal
import itertools

class Polynomial:

    def __init__(self, coefficients):
        """ input: coefficients are in the form a_0, a_1,..., a_d
        for a polynomial a_0 + a_1x + a_2 x^2 + a_3 x^3 + ..., a_d x^d
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
    
    def __pow__(self, k):
        """
        Returns self ** k
        """
        if k == 0:
            return Polynomial([1])
        else:
            return product([self for i in range(k)])

    def derivative(self):
        coeffs = self.coefficients
        derived_coeffs = [i*ci for i, ci in enumerate(coeffs)[1:]]
        return Polynomial(derived_coeffs)

    def shift(self, b):
        """
        Takes a polynomial p(x) and returns p(x-b)
        """
        shifted = Polynomial([0])
        for k, coeff in enumerate(self.coefficients):
            shifted += Polynomial([coeff])*Polynomial([-b, 1])**k
        return shifted

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