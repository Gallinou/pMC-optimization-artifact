import numpy as np
import sympy as sp
import sympy.abc as abc
import copy

rng = np.random.default_rng()

class bool_graph:
    def __init__(self, ni : int, n : int, no : int):
        self.ni = ni
        self.n = n
        self.no = no
    
        self.I = [[] for _ in range(ni)]
        self.M = [[] for _ in range(n)]
        self.F = [[] for _ in range(n)]

        self.n_transitions = 0

        self.acc = []
        self.not_acc = [i for i in range(n)]

        self.co_acc = []
        self.not_co_acc = [i for i in range(n)]

    def __len__(self):
        return len(self.not_acc) + len(self.not_co_acc)
    
    def is_acc(self, j : int):
        if len(self.acc) < len(self.not_acc):
            return j in self.acc
        else:
            return j not in self.not_acc
        
    def is_co_acc(self, i : int):
        if len(self.co_acc) < len(self.not_co_acc):
            return i in self.co_acc
        else:
            return i not in self.not_co_acc
        
    def is_edge_I(self, i : int, j : int):
        return j in self.I[i]
    
    def is_edge_M(self, i : int, j : int):
        return j in self.M[i]
    
    def is_edge_F(self, i : int, j : int):
        return j in self.F[i]

    def add_acc(self, j : int):
        new_acc = [j]
        while len(new_acc) != 0:
            k = new_acc.pop()
            self.not_acc.remove(k)
            self.acc.append(k)

            for l in self.not_acc:
                if self.is_edge_M(k, l):
                    new_acc.append(l)
    
    def add_co_acc(self, i : int):
        new_co_acc = [i]
        while len(new_co_acc) != 0:
            l = new_co_acc.pop()
            self.not_co_acc.remove(l)
            self.co_acc.append(l)

            for k in self.not_co_acc:
                if self.is_edge_M(k, l):
                    new_co_acc.append(k)
    
    def add_I(self, i : int, j : int):
        self.I[i].append(j)
        self.n_transitions += 1

        if not self.is_acc(j):
            self.add_acc(j)

    def add_M(self, i : int, j : int):
        self.M[i].append(j)
        self.n_transitions += 1

        if self.is_acc(i) and not self.is_acc(j):
            self.add_acc(j)

        if not self.is_co_acc(i) and self.is_co_acc(j):
            self.add_co_acc(i)

    def add_F(self, i : int, j : int):
        self.F[i].append(j)
        self.n_transitions += 1

        if not self.is_co_acc(i):
            self.add_co_acc(i)

def random_bool_graph(p : float, ni : int, n : int, no : int):
    graph = bool_graph(ni, n, no)

    for i in range(ni):
        j = int(rng.choice(n))
        graph.add_I(i, j)

    for j in range(no):
        i = int(rng.choice(n))
        graph.add_F(i, j)

    while len(graph) != 0:
        kj = int(rng.choice(len(graph)))

        if kj < len(graph.not_acc):
            ki = int(rng.choice(ni + len(graph.acc)))
            
            if ki < ni:
                i, j = ki, graph.not_acc[kj]
                graph.add_I(i, j)
            else:
                i, j = graph.acc[ki - ni], graph.not_acc[kj]
                graph.add_M(i, j)

        else:
            kj -= len(graph.not_acc)

            kk = int(rng.choice(no + len(graph.co_acc)))

            if kk < no:
                j, k = graph.not_co_acc[kj], kk
                graph.add_F(j, k)
            else:
                j, k = graph.not_co_acc[kj], graph.co_acc[kk - no]
                graph.add_M(j, k)

    nt = graph.n_transitions
    nt_max = n * (ni + n + no)

    current_p = nt / nt_max

    if current_p < p:
        new_p = (p * nt_max - nt) / (nt_max - nt)

        I = [[] for _ in range(ni)]
        M = [[] for _ in range(n)]
        F = [[] for _ in range(n)]

        for i in range(ni):
            for j in range(n):
                if rng.random() < new_p:
                    I[i].append(j)
                elif graph.is_edge_I(i, j):
                    I[i].append(j)

        for i in range(n):
            for j in range(n):
                if rng.random() < new_p:
                    M[i].append(j)
                elif graph.is_edge_M(i, j):
                    M[i].append(j)

        for i in range(n):
            for j in range(no):
                if rng.random() < new_p:
                    F[i].append(j)
                elif graph.is_edge_F(i, j):
                    F[i].append(j)
        
        return I, M, F
        
    return graph.I, graph.M, graph.F

def len_sparse_matrix(M):
    return sum([len(e) for e in M])

def density_sparse_graph(I, M, F, ni, n, no):
    nt = len_sparse_matrix(I) + len_sparse_matrix(M) + len_sparse_matrix(F)
    nt_max = n * (ni + n + no)
    return nt / nt_max

def random_float_sparse_matrix(M, n, xmin, xmax):
    for i in range(n):
        for id, j in enumerate(M[i]):
            M[i][id] = (j, rng.uniform(xmin, xmax))

def random_graph(p : float, ni : int, n : int, no : int, xmin : float, xmax : float, fast : bool = False):
    """Returns a non-null random transition system of size (ni, n, no) with p the probability of creating a transition and coef in (xmin, xmax)"""
    if fast == True:
        I, M, F = rec_copy_random_bool_graph(p, ni, n, no)
    else:
        I, M, F = random_bool_graph(p, ni, n, no)

    random_float_sparse_matrix(I, ni, xmin, xmax)
    random_float_sparse_matrix(M, n, xmin, xmax)
    random_float_sparse_matrix(F, n, xmin, xmax)
    
    return I, M, F

def sparse_matrix_normalization(M, n):
    for i in range(n):
        v = 0
        for _, vj in M[i]:
            v += vj

        for id, (j, vj) in enumerate(M[i]):
            M[i][id] = (j, vj / v)  

def random_probabilistic_graph(p : float, ni : int, n : int, no : int, eps : float = 1e-7, fast : bool = False):
    """Returns a non-null random probabilistic transition system of size (ni, n, no) with p the probability of creating a transition"""
    I, M, F = random_graph(p, ni, n, no, 0 + eps, 1 - eps, fast)

    MF = []
    for i in range(n):
        MFi = []
        for (j, vj) in M[i]:
            MFi.append((j, vj))
        for (j, vj) in F[i]:
            MFi.append((n+j, vj))
        MF.append(MFi)

    sparse_matrix_normalization(I, ni)
    sparse_matrix_normalization(MF, n)

    return I, MF

def random_multivariate_polynomial_aux(exps : tuple, j : int, d : int, imin : int, k : int, coeffs : dict, xmin : float, xmax : float, p : float):
    if j == d:
        if rng.random() < p:
            coeffs[exps] = rng.uniform(xmin, xmax)
    else:
        random_multivariate_polynomial_aux(exps, j + 1, d, k, k, coeffs, xmin, xmax, p)
        for i in range(imin, k):
            new_exps = exps[:i] + (exps[i] + 1,) + exps[i + 1:]
            random_multivariate_polynomial_aux(new_exps, j + 1, d, i, k, coeffs, xmin, xmax, p)

def random_multivariate_polynomial(d : int, xmin : float, xmax : float, p : float, symbols : list):
    """Returns a random multivariate polynomial of degree d and coef in (xmin, xmax)"""
    k = len(symbols)
    coeffs = {}
    random_multivariate_polynomial_aux((0,) * k, 0, d, 0, k, coeffs, xmin, xmax, p)
    return sp.Poly.from_dict(coeffs, *symbols, domain='RR')

def random_multivariate_interpolation(v : float, a : list, d : int, xmin : float, xmax : float, p : float, symbols : list):
    """Returns a random polynomial P such that P(a) = v and deg(P) = d with coef in (xmin, xmax)"""
    P = random_multivariate_polynomial(d, xmin, xmax, p, symbols)
    return P - sp.Poly(P.eval(a) - v, *symbols, domain='RR')
    
def random_multi_poly_sparse_matrix(M, n, a, d, xmin, xmax, p, symbols):
    for i in range(n):
        sum_poly = sp.Poly(0, *symbols, domain='RR')
        for id, (j, vj) in enumerate(M[i]):
            poly = random_multivariate_interpolation(vj, a, d, xmin, xmax, p, symbols)
            M[i][id] = (j, poly)
            sum_poly += poly

        id = rng.choice(len(M[i]))
        j, vj = M[i][id]
        M[i][id] = (j, 1 - sum_poly + vj)
    
def random_pMC(p : float, ni : int, n : int, no : int, d : int, q : float, xmin : float, xmax : float, symbols : list, fast : bool = False):
    """
    p is the probability of creating a transition in the graph
    (ni, n, no) is the size of the transition system
    d is the degree of the multivariate polynomials
    q is the probability of creating a monomial
    a is the point around whom the value are probabilistics
    (xmin, xmax) is the range of the polynomial coefs
    symbols is the list of symbols of the multivariate polynomials
    """
    I, MF = random_probabilistic_graph(p, ni, n, no, fast=fast)
    
    a = [.5 for _ in range(len(symbols))]

    random_multi_poly_sparse_matrix(I, ni, a, d, xmin, xmax, q, symbols)
    random_multi_poly_sparse_matrix(MF, n, a, d, xmin, xmax, q, symbols)

    return I, MF

def random_simple_poly(xmin : float, xmax : float, x : float, y : float):
    a = rng.uniform(xmin, xmax)

    a, x, y = sp.Rational(a), sp.Rational(x), sp.Rational(y)
    b = y - a * x

    return a, b
    
def domain_simple_poly(a : float, b : float, eps : float = 1e-6):
    if a == 0:
        return None, None

    v0 = (eps-b)/a
    v1 = (1-eps-b)/a

    assert 0 < a * v0 + b < 1
    assert 0 < a * v1 + b < 1
    
    if a < 0:
        return v1, v0
    else:
        return v0, v1

def precise_str(a : float):
    return f"{a:.30f}".rstrip('0').rstrip('.')
    
def simple_poly_str(a : float, b : float, symbol : sp.Symbol):
    a, b = float(a), float(b)

    if a == 0:
        return precise_str(b)
    elif b == 0:
        return precise_str(a) + "*" + str(symbol)
    elif b < 0:
        return precise_str(a) + "*" + str(symbol) + precise_str(b)
    else:
        return precise_str(a) + "*" + str(symbol) + "+" + precise_str(b)

def merge_domains(I1 : tuple, I2 : tuple):
    a1, b1 = I1
    a2, b2 = I2

    if a1 == None or b1 == None:
        return a2, b2
    
    if a2 == None or b2 == None:
        return a1, b1

    a, b = max(a1, a2), min(b1, b2)
    assert a < b
    return a, b

def random_simple_poly_matrix(M, xmin, xmax, symbols):
    dom = [(None, None) for _ in symbols]

    for V in M:
        idxsymbol = rng.choice(len(symbols))
        symbol = symbols[idxsymbol]
        idxj0 = rng.choice(len(V))
        aj0, bj0 = 0, 1
        for idxj, (j, vj) in enumerate(V):
            if idxj != idxj0:
                aj, bj = random_simple_poly(xmin, xmax, .5, vj)
                aj0, bj0 = aj0 - aj, bj0 - bj
                V[idxj] = j, simple_poly_str(aj, bj, symbol)
                dom[idxsymbol] = merge_domains(dom[idxsymbol], domain_simple_poly(aj, bj))

                vmin, vmax = dom[idxsymbol]
                assert 0 <= aj * vmin + bj <= 1 and 0 <= aj * vmax + bj <= 1, f"{aj * vmin + bj}, {aj * vmax + bj}"
                
        j0, _ = V[idxj0]
        V[idxj0] = j0, simple_poly_str(aj0, bj0, symbol)
        dom[idxsymbol] = merge_domains(dom[idxsymbol], domain_simple_poly(aj0, bj0))

    return dom

def sparse_nb_transitions(M):
    total = 0
    for V in M:
        total += len(V)

    return total

def random_simple_pMC(p, ni, n, no, xmin, xmax, symbols, fast = False):
    I, MF = random_probabilistic_graph(p, ni, n, no, fast=fast)

    ntrans = sparse_nb_transitions(I) + sparse_nb_transitions(MF)

    domI = random_simple_poly_matrix(I, xmin, xmax, symbols)
    domMF = random_simple_poly_matrix(MF, xmin, xmax, symbols)

    dom = [merge_domains(domI[i], domMF[i]) for i in range(len(symbols))]

    return I, MF, dom, ntrans

def minimal_rec_random_bool_graph(ni : int, n : int, no : int, nmin : int = 100):
    if n < nmin:
        return random_bool_graph(0, ni, n, no)

    n1 = n // 2
    n2 = n - n1

    I1, M1, F1 = minimal_rec_random_bool_graph(ni, n1, no, nmin)
    I2, M2, F2 = minimal_rec_random_bool_graph(ni, n2, no, nmin)

    for i in range(ni):
        for j in I2[i]:
            I1[i].append(j + n1)

    for V2 in M2:
        V = []
        for j in V2:
            V.append(j + n1)
        M1.append(V)

    for V2 in F2:
        V = []
        for j in V2:
            V.append(j)
        F1.append(V)

    return I1, M1, F1

def minimal_rec_copy_random_bool_graph(ni : int, n : int, no : int, nmin : int = 100):
    if n < nmin:
        return random_bool_graph(0, ni, n, no)

    n0 = n // 2

    I1, M1, F1 = minimal_rec_copy_random_bool_graph(ni, n0, no, nmin)
    I2, M2, F2 = copy.deepcopy(I1), copy.deepcopy(M1), copy.deepcopy(F1)

    for i in range(ni):
        for j in I2[i]:
            k = rng.choice(n0)
            M1[k].append(j + n0)

    for V2 in M2:
        V = []
        for j in V2:
            V.append(j + n0)
        M1.append(V)

    for V2 in F2:
        V = []
        for j in V2:
            V.append(j)
        F1.append(V)

    if n == 2 * n0 + 1:

        i = rng.choice(ni + n - 1)
        if i < ni:
            I1[i].append(n - 1)
        else:
            M1[i - ni].append(n - 1)

        i = rng.choice(no + n - 1)
        if i < no:
            M1.append([])
            F1.append([i])
        else:
            M1.append([i - no])
            F1.append([])

    return I1, M1, F1

def nb_edge(I, M, F):
    total = 0
    for V in I:
        total += len(V)
    for V in M:
        total += len(V)
    for V in F:
        total += len(V)
    return total

def rec_copy_random_bool_graph(p : float, ni : int, n : int, no : int, nmin : int = 100):
    I, M, F = minimal_rec_copy_random_bool_graph(ni, n, no, nmin)
    #nb = nb_edge(I, M, F)
    #nb_tot = n * (ni + n + no)
    #if nb < p * nb_tot:
    #    new_p = (nb_tot * p - nb) / (nb_tot - nb)
    #the goal of this part is given current_p, generate enough transition to get to p
    #the two main ways are adding enough transitions exaclty or sampling random edges enough time

    return I, M, F

def list_variables(k : int):
    if k == 1:
        return [abc.a]

    s = ""
    for i in range(k):
        s += f"a{i} "
    
    return list(sp.symbols(s))