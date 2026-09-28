"""
MMA (Method of Moving Asymptotes) optimization functions with IPOPT subproblem solver.

Place this file as: topology_opt_ipopt/mma_al.py
"""

from __future__ import division
from scipy.sparse import diags
import numpy as np
from typing import Tuple
import cyipopt

def mmasub(m: int, n: int, iter: int, xval: np.ndarray, xmin: np.ndarray, xmax: np.ndarray,
           xold1: np.ndarray, xold2: np.ndarray, f0val: float, df0dx: np.ndarray, fval: np.ndarray,
           dfdx: np.ndarray, low: np.ndarray, upp: np.ndarray, a0: float, a_mma: np.ndarray, c: np.ndarray,
           d: np.ndarray, move: float = 0.5, raa0: float = 0.00001, 
           albefa: float = 0.1) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, 
                                         np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the MMA (Method of Moving Asymptotes) subproblem for optimization.

    Returns:
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp
    """
    
    epsimin = 0.0000001
    eeen = np.ones((n, 1), dtype=float)
    eeem = np.ones((m, 1), dtype=float)
    zeron = np.zeros((n, 1), dtype=float)
    
    # Calculation of the bounds alfa and beta
    zzz1 = low + albefa * (xval - low)
    zzz2 = xval - move * (xmax - xmin)
    zzz = np.maximum(zzz1, zzz2)
    alfa = np.maximum(zzz, xmin)
    zzz1 = upp - albefa * (upp - xval)
    zzz2 = xval + move * (xmax - xmin)
    zzz = np.minimum(zzz1, zzz2)
    beta = np.minimum(zzz, xmax)

    # Calculations of p0, q0, P, Q and b
    xmami = xmax - xmin
    xmami_eps = 0.00001 * eeen
    xmami = np.maximum(xmami, xmami_eps)
    xmami_inv = eeen / xmami
    ux1 = upp - xval
    ux2 = ux1 * ux1
    xl1 = xval - low
    xl2 = xl1 * xl1
    ux_inv = eeen / ux1
    xl_inv = eeen / xl1
    p0 = zeron.copy()
    q0 = zeron.copy()
    p0 = np.maximum(df0dx, 0)
    q0 = np.maximum(-df0dx, 0)
    pq0 = 0.001 * (p0 + q0) + raa0 * xmami_inv
    p0 = p0 + pq0
    q0 = q0 + pq0
    p0 = p0 * ux2
    q0 = q0 * xl2
    P = np.zeros((m, n), dtype=float)
    Q = np.zeros((m, n), dtype=float)
    P = np.maximum(dfdx, 0)
    Q = np.maximum(-dfdx, 0)
    PQ = 0.001 * (P + Q) + raa0 * np.dot(eeem, xmami_inv.T)
    P = P + PQ
    Q = Q + PQ
    P = (diags(ux2.flatten(), 0).dot(P.T)).T
    Q = (diags(xl2.flatten(), 0).dot(Q.T)).T
    b = np.dot(P, ux_inv) + np.dot(Q, xl_inv) - fval

    # Solving the subproblem using IPOPT
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(
        m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a_mma, b, c, d)
    
    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp


def subsolv(m: int, n: int, epsimin: float, low: np.ndarray, upp: np.ndarray, alfa: np.ndarray, 
            beta: np.ndarray, p0: np.ndarray, q0: np.ndarray, P: np.ndarray, Q: np.ndarray, 
            a0: float, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, 
            np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Solve the MMA subproblem using IPOPT interior point optimizer.
    
    Problem:
        Minimize: sum[p0/(upp-x) + q0/(x-low)] + a0*z + sum[c*y + 0.5*d*y^2]
        Subject to: sum[P/(upp-x) + Q/(x-low)] - a*z - y <= b
                    alfa <= x <= beta
                    y >= 0, z >= 0
    
    Variables: [x (n), y (m), z (1)]
    """
    
    # Flatten arrays
    p0 = p0.flatten()
    q0 = q0.flatten()
    alfa = alfa.flatten()
    beta = beta.flatten()
    low = low.flatten()
    upp = upp.flatten()
    a = a.flatten()
    b = b.flatten()
    c = c.flatten()
    d = d.flatten()
    
    # Add small epsilon to avoid exact boundaries
    eps_bound = 1e-6
    alfa = alfa + eps_bound
    beta = beta - eps_bound
    
    # Define the optimization problem class
    class MMASubproblem:
        def __init__(self):
            self.n = n
            self.m = m
            self.nvar = n + m + 1
            
        def objective(self, x_all):
            """Objective function"""
            x = x_all[:n]
            y = x_all[n:n+m]
            z = x_all[n+m]
            
            # Ensure x stays away from asymptotes
            x = np.clip(x, low + 1e-8, upp - 1e-8)
            
            # MMA rational approximation
            obj = np.sum(p0 / (upp - x) + q0 / (x - low))
            # Penalty terms
            obj += a0 * z + np.sum(c * y + 0.5 * d * y**2)
            return obj
        
        def gradient(self, x_all):
            """Gradient of objective"""
            x = x_all[:n]
            y = x_all[n:n+m]
            
            # Ensure x stays away from asymptotes
            x = np.clip(x, low + 1e-8, upp - 1e-8)
            
            grad = np.zeros(self.nvar)
            grad[:n] = p0 / (upp - x)**2 - q0 / (x - low)**2
            grad[n:n+m] = c + d * y
            grad[n+m] = a0
            return grad
        
        def constraints(self, x_all):
            """Inequality constraints: g(x) <= 0"""
            x = x_all[:n]
            y = x_all[n:n+m]
            z = x_all[n+m]
            
            # Ensure x stays away from asymptotes
            x = np.clip(x, low + 1e-8, upp - 1e-8)
            
            g = np.zeros(m)
            for i in range(m):
                g[i] = np.sum(P[i,:] / (upp - x) + Q[i,:] / (x - low))
                g[i] -= a[i] * z + y[i] + b[i]
            return g
        
        def jacobian(self, x_all):
            """Jacobian of constraints (row-major flattened)"""
            x = x_all[:n]
            
            # Ensure x stays away from asymptotes
            x = np.clip(x, low + 1e-8, upp - 1e-8)
            
            jac_values = []
            for i in range(m):
                # w.r.t. x
                for j in range(n):
                    jac_values.append(P[i,j] / (upp[j] - x[j])**2 - 
                                     Q[i,j] / (x[j] - low[j])**2)
                # w.r.t. y (diagonal -1)
                for j in range(m):
                    jac_values.append(-1.0 if j == i else 0.0)
                # w.r.t. z
                jac_values.append(-a[i])
            
            return np.array(jac_values)
        
        def jacobianstructure(self):
            """Sparsity structure of Jacobian"""
            rows = []
            cols = []
            
            for i in range(m):
                # x columns
                for j in range(n):
                    rows.append(i)
                    cols.append(j)
                # y columns
                for j in range(m):
                    rows.append(i)
                    cols.append(n + j)
                # z column
                rows.append(i)
                cols.append(n + m)
            
            return (np.array(rows), np.array(cols))
    
    # Create problem instance
    problem = MMASubproblem()
    
    # Initial guess - start closer to feasibility
    x0 = np.zeros(problem.nvar)
    x0[:n] = 0.5 * (alfa + beta)  # Midpoint between bounds
    x0[n:n+m] = np.maximum(0.1, -b.flatten())  # Better initial y
    x0[n+m] = 1.0
    
    # Variable bounds with safer margins
    lb = np.zeros(problem.nvar)
    ub = np.zeros(problem.nvar)
    lb[:n] = alfa
    ub[:n] = beta
    lb[n:n+m] = 1e-8      # y >= 0
    ub[n:n+m] = 1e10
    lb[n+m] = 1e-8        # z >= 0
    ub[n+m] = 1e10
    
    # Constraint bounds: g(x) <= 0
    cl = -1e10 * np.ones(m)
    cu = np.zeros(m)
    
    # Create IPOPT problem
    nlp = cyipopt.Problem(
        n=problem.nvar,
        m=m,
        problem_obj=problem,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu
    )
    
    # Set IPOPT options - more relaxed for first solve
    nlp.add_option('print_level', 3)  # Enable output for debugging
    nlp.add_option('sb', 'yes')
    nlp.add_option('tol', 1e-6)
    nlp.add_option('acceptable_tol', 1e-5)
    nlp.add_option('max_iter', 1000)
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('linear_solver', 'mumps')  # Try mumps if available
    
    # Solve
    print(f"  [IPOPT] Solving subproblem with {n} design vars, {m} constraints...")
    try:
        x_sol, info = nlp.solve(x0)
        print(f"  [IPOPT] Status: {info['status']}, Objective: {info['obj_val']:.6f}")
    except Exception as e:
        print(f"  [IPOPT] FAILED: {e}")
        # Return a small perturbation instead of initial guess
        x_sol = x0.copy()
        x_sol[:n] += 0.01 * (np.random.rand(n) - 0.5)
        x_sol[:n] = np.clip(x_sol[:n], lb[:n], ub[:n])
        info = {
            'mult_g': 0.1 * np.ones(m),
            'mult_x_L': 0.1 * np.ones(problem.nvar),
            'mult_x_U': 0.1 * np.ones(problem.nvar),
            'status': -1
        }
    
    # Extract primal solution
    x_opt = x_sol[:n].reshape((n, 1))
    y_opt = x_sol[n:n+m].reshape((m, 1))
    z_opt = x_sol[n+m]
    
    # Extract dual variables (Lagrange multipliers)
    lam = np.abs(info['mult_g']).reshape((m, 1))
    
    mult_x_L = info['mult_x_L']
    mult_x_U = info['mult_x_U']
    
    xsi = np.abs(mult_x_L[:n]).reshape((n, 1))
    eta = np.abs(mult_x_U[:n]).reshape((n, 1))
    mu = np.abs(mult_x_L[n:n+m]).reshape((m, 1))
    zet = abs(mult_x_L[n+m])
    
    # Compute slack variables from complementarity
    g_val = problem.constraints(x_sol)
    s = np.maximum(-g_val, 1e-8).reshape((m, 1))
    
    # Ensure positivity
    lam = np.maximum(lam, 1e-6)
    xsi = np.maximum(xsi, 1e-6)
    eta = np.maximum(eta, 1e-6)
    mu = np.maximum(mu, 1e-6)
    zet = max(zet, 1e-6)
    
    return x_opt, y_opt, z_opt, lam, xsi, eta, mu, zet, s


def asymp(outeriter: int, n: int, xval: np.ndarray, xold1: np.ndarray, xold2: np.ndarray, xmin: np.ndarray,
          xmax: np.ndarray, low: np.ndarray, upp: np.ndarray, raa0: float, raa: np.ndarray, raa0eps: float,
          raaeps: float, df0dx: np.ndarray, dfdx: np.ndarray, asyinit: float = 0.5, asydecr: float = 0.7, 
          asyincr: float = 1.2, asymin: float = 0.01, asymax: float = 10) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Calculate the asymptotes and raa parameters at the beginning of each outer iteration.

    Returns:
        low, upp, raa0, raa
    """
    
    eeen = np.ones((n, 1))
    xmami = xmax - xmin
    xmamieps = 0.00001 * eeen
    xmami = np.maximum(xmami, xmamieps)
    
    # Update raa parameters
    raa0 = np.dot(np.abs(df0dx).T, xmami)
    raa0 = np.maximum(raa0eps, (0.1 / n) * raa0)
    raa = np.dot(np.abs(dfdx), xmami)
    raa = np.maximum(raaeps, (0.1 / n) * raa)
    
    # Update asymptotes
    if outeriter <= 2:
        low = xval - asyinit * xmami
        upp = xval + asyinit * xmami
    else:
        xxx = (xval - xold1) * (xold1 - xold2)
        factor = eeen.copy()
        factor[xxx > 0] = asyincr
        factor[xxx < 0] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        lowmin = xval - asymax * xmami
        lowmax = xval - asymin * xmami
        uppmin = xval + asymin * xmami
        uppmax = xval + asymax * xmami
        low = np.maximum(low, lowmin)
        low = np.minimum(low, lowmax)
        upp = np.minimum(upp, uppmax)
        upp = np.maximum(upp, uppmin)
    
    return low, upp, raa0, raa