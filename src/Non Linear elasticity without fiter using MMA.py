from dolfin import *
import os
import numpy as np
from mmaa_al import mmasub, asymp

# -------------------------------
# Problem parameters
# -------------------------------
volfrac = 0.5        # Target volume fraction
penal   = 3.0        # SIMP penalization factor
max_iter = 100       # Number of iterations

E0    = 1.0          # Young's modulus of solid material
Emin  = 1e-6         # Young's modulus of void 
nu    = 0.3          # Poisson's ratio

# Stabilization parameters
rho_min = 0.001      # Minimum density (avoid complete void)
rho_max = 1.0        # Maximum density

print("=" * 70)
print("NONLINEAR TOPOLOGY OPTIMIZATION WITH MMA (NO FILTER)")
print("=" * 70)
print(f"Target volume fraction: {volfrac}")
print(f"SIMP penalization: {penal}")
print(f"Emin (stabilized): {Emin}")
print("=" * 70)

# -------------------------------
# Create Mesh and Function Spaces
# -------------------------------
mesh = RectangleMesh(Point(0, 0), Point(8.0, 4.0), 180, 90)
V_d = FunctionSpace(mesh, "P", 1)
V_u = VectorFunctionSpace(mesh, "P", 1)

print(f"Mesh: {mesh.num_vertices()} nodes, {mesh.num_cells()} elements")

# -------------------------------
# Define design variable and initial guess
# -------------------------------
rho = Function(V_d)
rho.vector()[:] = volfrac

# Get number of design variables
n = len(rho.vector().get_local())
print(f"Number of design variables: {n}")

# -------------------------------
# Define nonlinear elasticity problem
# -------------------------------
u = Function(V_u)          # displacement
du = TrialFunction(V_u)
v = TestFunction(V_u)

def lame_parameters(rho_val):
    """SIMP with safeguards for numerical stability"""
    # Ensure rho is bounded
    rho_bounded = conditional(lt(rho_val, rho_min), rho_min, 
                             conditional(gt(rho_val, rho_max), rho_max, rho_val))
    E_val = Emin + rho_bounded**penal * (E0 - Emin)
    mu    = E_val/(2*(1+nu))
    lmbda = E_val*nu/((1+nu)*(1-2*nu))
    return mu, lmbda

I = Identity(2)

def neo_hookean_stress(rho_val, u_):
    """Neo-Hookean stress with numerical safeguards"""
    mu, lmbda = lame_parameters(rho_val)
    F = I + grad(u_)
    J = det(F)
    
    # Safeguard: ensure J > 0 (no element inversion)
    J_safe = conditional(lt(J, 0.1), 0.1, J)
    
    FinvT = inv(F).T
    return mu*(F - FinvT) + lmbda*ln(J_safe)*FinvT

# -------------------------------
# Boundary conditions
# -------------------------------
class LoadEdge(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 8.0, DOLFIN_EPS) and (1.8 <= x[1] <= 2.2)

class LeftEdge(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, DOLFIN_EPS)

facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
LoadEdge().mark(facets, 2)
LeftEdge().mark(facets, 1)

ds = Measure("ds", domain=mesh, subdomain_data=facets)
bc_left = DirichletBC(V_u, Constant((0.0, 0.0)), facets, 1)
bcs = [bc_left]

# Applied traction - REDUCED for stability
traction_magnitude = 0.05  # Reduced from 0.1
traction = Constant((0.0, 0.0))

print("\nBoundary conditions:")
print("  - Left edge: Fixed (u = 0)")
print("  - Right edge center: Applied traction")
print(f"  - Traction magnitude: {traction_magnitude} (reduced for stability)")

# -------------------------------
# Nonlinear elasticity solver function with robustness
# -------------------------------
def solve_nonlinear_elasticity(rho_values, load_steps=15, max_load=traction_magnitude,
                              max_newton_iters=25, verbose=False):
    """Solve with enhanced stability and error recovery"""
    # Ensure densities are bounded
    rho_values = np.clip(rho_values, rho_min, rho_max)
    
    rho.vector().set_local(rho_values)
    rho.vector().apply("insert")
    
    # Start from zero displacement
    u.vector()[:] = 0.0
    
    # More gradual load increments
    load_increments = np.linspace(0, max_load, load_steps + 1)[1:]
    
    for step_idx, lam in enumerate(load_increments):
        traction.assign(Constant((0.0, -lam)))
        
        R_total = inner(neo_hookean_stress(rho, u), grad(v))*dx - dot(traction, v)*ds(2)
        Jac_total = derivative(R_total, u, du)
        
        problem = NonlinearVariationalProblem(R_total, u, bcs, Jac_total)
        solver = NonlinearVariationalSolver(problem)
        
        prm = solver.parameters
        prm["nonlinear_solver"] = "newton"
        prm["newton_solver"]["relative_tolerance"] = 1e-6
        prm["newton_solver"]["absolute_tolerance"] = 1e-8
        prm["newton_solver"]["maximum_iterations"] = max_newton_iters
        prm["newton_solver"]["linear_solver"] = "mumps"
        prm["newton_solver"]["error_on_nonconvergence"] = False
        prm["newton_solver"]["relaxation_parameter"] = 1.0
        
        try:
            niter, converged = solver.solve()
            
            if not converged:
                if verbose:
                    print(f"    Warning: Load step {step_idx+1}/{load_steps} did not converge")
                # Try with smaller relaxation
                prm["newton_solver"]["relaxation_parameter"] = 0.8
                niter, converged = solver.solve()
                
                if not converged:
                    print(f"    ERROR: Could not converge at load step {step_idx+1}")
                    return u
                    
        except RuntimeError as e:
            print(f"    Runtime error in load step {step_idx+1}: {str(e)}")
            return u
    
    return u

# -------------------------------
# Compliance and sensitivity with error handling
# -------------------------------
def compute_compliance(rho_values):
    """Compute compliance with error handling"""
    try:
        u_sol = solve_nonlinear_elasticity(rho_values)
        
        # Check for NaN in solution
        if np.any(np.isnan(u_sol.vector().get_local())):
            print("  WARNING: NaN detected in displacement, returning large penalty")
            return 1e10
        
        W_ext = assemble(dot(traction, u_sol)*ds(2))
        
        if np.isnan(W_ext) or np.isinf(W_ext):
            print("  WARNING: Invalid compliance value, returning large penalty")
            return 1e10
            
        return W_ext
        
    except Exception as e:
        print(f"  ERROR in compliance computation: {str(e)}")
        return 1e10

def compute_sensitivity(rho_values):
    """Compute sensitivity with error handling"""
    try:
        u_sol = solve_nonlinear_elasticity(rho_values)
        
        # Check for NaN in solution
        if np.any(np.isnan(u_sol.vector().get_local())):
            print("  WARNING: NaN in displacement, returning zero sensitivity")
            return np.zeros(n)
        
        rho.vector().set_local(rho_values)
        rho.vector().apply("insert")

        F_expr = Identity(2) + grad(u_sol)
        C_expr = F_expr.T * F_expr
        J_expr = det(F_expr)
        
        # Safeguard J
        J_expr = conditional(lt(J_expr, 0.1), 0.1, J_expr)
        
        Ic_expr = tr(C_expr)
        
        # Bounded rho for derivative
        rho_bounded = conditional(lt(rho, rho_min), rho_min, 
                                 conditional(gt(rho, rho_max), rho_max, rho))
        
        dE_drho = penal*(E0 - Emin)*rho_bounded**(penal - 1)
        dmu_dE = 1.0/(2.0*(1.0 + nu))
        dlambda_dE = nu/((1.0 + nu)*(1.0 - 2.0*nu))
        
        # Strain energy density
        dpsi_dmu = 0.5*(Ic_expr - 2 - 2*ln(J_expr))
        dpsi_dlambda = 0.5*(ln(J_expr))**2
        dpsi_drho = dE_drho*(dpsi_dmu*dmu_dE + dpsi_dlambda*dlambda_dE)
        
        # Project to get nodal values
        sens_proj = project(dpsi_drho, V_d)
        sens_array = sens_proj.vector().get_local()
        
        # CRITICAL: Sensitivity is POSITIVE (more material = less compliance)
        # So we need NEGATIVE for minimization: dc/drho = -dpsi/drho
        sens_array = -sens_array
        
        # Check for invalid values
        if np.any(np.isnan(sens_array)) or np.any(np.isinf(sens_array)):
            print("  WARNING: Invalid sensitivity values, setting to zero")
            sens_array = np.nan_to_num(sens_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return sens_array
        
    except Exception as e:
        print(f"  ERROR in sensitivity computation: {str(e)}")
        return np.zeros(n)

# -------------------------------
# Volume constraint gradient (element volumes)
# -------------------------------
print("\nPrecomputing element volumes for volume constraint...")
# For unfiltered case, gradient is simply element volume / total volume
dx_elem = dx(domain=mesh)
one_func = Function(V_d)
one_func.vector()[:] = 1.0
total_volume = assemble(one_func * dx)

# Element-wise volume (for P1 elements, all nodes in element contribute equally)
element_volume = assemble(TestFunction(V_d) * dx)
volume_gradient = (element_volume.get_local() / total_volume).reshape((1, n))

# -------------------------------
# MMA Setup
# -------------------------------
print("\n" + "=" * 70)
print("SETTING UP MMA OPTIMIZATION")
print("=" * 70)

m = 1  # Number of constraints
epsimin = 0.0000001
eeen = np.ones((n, 1))
eeem = np.ones((m, 1))
zeron = np.zeros((n, 1))
zerom = np.zeros((m, 1))
tol_change = 1e-3

# Initialize design variables
xval_mma = volfrac * np.ones((n, 1))
xold1 = xval_mma.copy()
xold2 = xval_mma.copy()
xmin = rho_min * eeen  
xmax = rho_max * eeen

# Initialize MMA parameters
low = xval_mma - 0.5 * eeen
upp = xval_mma + 0.5 * eeen
c = 10000 * eeem
d = zerom
a0 = 1
a_mma = zerom  

# Initialize raa parameters
raa0 = 0.01
raa = 0.01 * eeem
raa0eps = 1e-6
raaeps = 1e-6

# History tracking
mma_compliance_history = []
mma_volume_history = []
mma_constraint_history = []
failed_iterations = []

# -------------------------------
# Create output directory and VTK files
# -------------------------------
output_dir = "nonlinear_topology_mma_nofilter"
os.makedirs(output_dir, exist_ok=True)
print(f"Created output directory: {output_dir}/")

vtkfile = File(os.path.join(output_dir, "density.pvd"))
displacement_file = File(os.path.join(output_dir, "displacement.pvd"))

# -------------------------------
# MMA Optimization Loop
# -------------------------------
print("\n" + "=" * 70)
print("STARTING NONLINEAR TOPOLOGY OPTIMIZATION WITH MMA (NO FILTER)")
print("=" * 70)

# Initial compliance
print("Computing initial compliance...")
initial_compliance = compute_compliance(xval_mma.flatten())
print(f"Initial compliance: {initial_compliance:.6f}")
print("-" * 70)

for iteration in range(max_iter):
    print(f"\n--- Iteration {iteration+1}/{max_iter} ---")
    
    # Use design variables directly (no filtering)
    rho_vals = np.clip(xval_mma.flatten(), rho_min, rho_max)
    
    # Compute objective (compliance)
    print("  Computing compliance...")
    f0val = compute_compliance(rho_vals)
    
    # Check if compliance is valid
    if f0val > 1e9:
        print("  ERROR: Invalid compliance detected, skipping iteration")
        failed_iterations.append(iteration + 1)
        xval_mma = xold1.copy()
        continue
    
    # Compute sensitivity
    print("  Computing sensitivity...")
    df0dx = compute_sensitivity(rho_vals).reshape((n, 1))
    
    # SANITY CHECK
    print(f"  Sensitivity stats: mean={np.mean(df0dx):.2e}, "
          f"min={np.min(df0dx):.2e}, max={np.max(df0dx):.2e}")
    
    # Check for invalid gradients
    if np.any(np.isnan(df0dx)) or np.any(np.isinf(df0dx)):
        print(f"  Warning: Invalid gradients detected, setting to zero")
        df0dx = np.nan_to_num(df0dx, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.all(np.abs(df0dx) < 1e-12):
        print("  ERROR: Zero sensitivity - optimization cannot proceed")
        break
    
    # Volume constraint: mean(x) - volfrac <= 0
    volume_fraction = np.mean(rho_vals)
    constraint_val = volume_fraction - volfrac
    fval = np.array([[constraint_val]])
    
    # Constraint gradient (precomputed)
    dfdx = volume_gradient
    
    # Update asymptotes (skip first iteration)
    if iteration > 0:
        low, upp, raa0, raa = asymp(
            outeriter=iteration + 1,
            n=n,
            xval=xval_mma,
            xold1=xold1,
            xold2=xold2,
            xmin=xmin,
            xmax=xmax,
            low=low,
            upp=upp,
            raa0=raa0,
            raa=raa,
            raa0eps=raa0eps,
            raaeps=raaeps,
            df0dx=df0dx,
            dfdx=dfdx,
            asyinit=0.5,
            asydecr=0.7,
            asyincr=1.2,
            asymin=0.01,
            asymax=10.0
        )
    
    # Solve MMA subproblem
    print("  Solving MMA subproblem...")
    try:
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
            m=m,
            n=n,
            iter=iteration + 1,
            xval=xval_mma,
            xmin=xmin,
            xmax=xmax,
            xold1=xold1,
            xold2=xold2,
            f0val=f0val,
            df0dx=df0dx,
            fval=fval,
            dfdx=dfdx,
            low=low,
            upp=upp,
            a0=a0,
            a_mma=a_mma,
            c=c,
            d=d
        )
    except Exception as e:
        print(f"  ERROR in MMA subproblem: {str(e)}")
        break
    
    # Apply bound constraints
    xmma = np.maximum(xmin, np.minimum(xmax, xmma))
    
    # Check for invalid design
    if np.any(np.isnan(xmma)) or np.any(np.isinf(xmma)):
        print(f"  Warning: Invalid design detected, reverting to previous")
        xmma = xval_mma.copy()
    
    # Compute change
    change = np.max(np.abs(xmma - xval_mma))
    
    # Update design variables
    xold2 = xold1.copy()
    xold1 = xval_mma.copy()
    xval_mma = xmma.copy()
    
    # Update density for visualization
    rho.vector().set_local(xval_mma.flatten())
    
    # Store history
    mma_compliance_history.append(f0val)
    mma_volume_history.append(volume_fraction)
    mma_constraint_history.append(constraint_val)
    
    # Print progress
    print(f"  Compliance: {f0val:.6f}")
    print(f"  Volume fraction: {volume_fraction:.4f} (target {volfrac:.4f})")
    print(f"  Constraint: {constraint_val:+.5f}")
    print(f"  Lambda: {lam[0,0]:.5f}")
    print(f"  Change: {change:.3e}")
    print(f"  Density range: [{xval_mma.min():.3f}, {xval_mma.max():.3f}]")
    
    # Save to VTK
    rho.rename("density", "Material Density")
    vtkfile << (rho, float(iteration))
    
    # Save displacement every 10 iterations
    if iteration % 10 == 0:
        print("  Saving displacement field...")
        u_final = solve_nonlinear_elasticity(xval_mma.flatten(), verbose=True)
        u_final.rename("displacement", "Displacement")
        displacement_file << (u_final, float(iteration))
    
    # Adjust penalty if constraint violated
    if abs(constraint_val) > 0.05:
        c = np.minimum(c * 2.0, 1000000 * eeem)
        print(f"  Adjusted penalty parameter c to {c[0,0]:.2f}")
    
    # Check convergence
    if iteration > 5:
        rel_change = abs(mma_compliance_history[-1] - mma_compliance_history[-2]) / abs(mma_compliance_history[-2])
        print(f"  Relative compliance change: {rel_change:.3e}")
        
        if change < tol_change and abs(constraint_val) < 0.05 and iteration > 15:
            print(f"\n{'='*70}")
            print(f"MMA CONVERGED at iteration {iteration+1}")
            print(f"{'='*70}")
            break

# -------------------------------
# Save final results
# -------------------------------
print("\n" + "=" * 70)
print("OPTIMIZATION COMPLETED")
print("=" * 70)

final_compliance = mma_compliance_history[-1]
final_volume = mma_volume_history[-1]

print(f"Final compliance: {final_compliance:.6f}")
print(f"Final volume fraction: {final_volume:.4f}")
print(f"Compliance reduction: {((initial_compliance - final_compliance)/initial_compliance)*100:.2f}%")

# Save final design
rho.rename("density_final", "Final Material Density")
vtkfile_final = File(os.path.join(output_dir, "density_final.pvd"))
vtkfile_final << (rho, 0.0)

print(f"\nResults saved to: {output_dir}/")