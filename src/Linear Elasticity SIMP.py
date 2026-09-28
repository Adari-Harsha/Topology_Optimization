from dolfin import *
import numpy as np
from mmaa_al import mmasub, asymp
import os
import matplotlib.pyplot as plt

# -------------------------------
# BEAM TYPE SELECTION
# -------------------------------
# Choose beam type: 'half_mbb', 'cantilever', or 'simply_supported'
BEAM_TYPE = 'half_mbb'  # Change this to switch between beam types

print("=" * 70)
print(f"SELECTED BEAM TYPE: {BEAM_TYPE.upper()}")
print(f"Running ALL optimization methods for comparison")
print("=" * 70)

# -------------------------------
# Create output directory
# -------------------------------
output_dir = f"results_{BEAM_TYPE}_comparison"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}/")

# -------------------------------
# Problem parameters
# -------------------------------
volfrac = 0.5        # Target volume fraction
penal   = 3.0        # SIMP penalization factor
max_iter = 100       # Number of iterations

E0    = 1.0         # Young's modulus of solid material
Emin  = 1e-9        # Young's modulus of void
nu    = 0.3         # Poisson's ratio
rmin  = 0.05        # Filter radius

# -------------------------------
# Create Mesh and Function Spaces
# -------------------------------
L = 8.0  # Length
H = 4.0  # Height
mesh = RectangleMesh(Point(0, 0), Point(L, H), 180, 90)
V_d = FunctionSpace(mesh, "P", 1)
V_u = VectorFunctionSpace(mesh, "P", 1)

print(f"Domain: {L} x {H}")
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
# Define elasticity problem
# -------------------------------
u = TrialFunction(V_u)
v = TestFunction(V_u)

def lame_parameters(rho_val):
    E_val = Emin + rho_val**penal * (E0 - Emin)
    mu    = E_val/(2*(1+nu))
    lmbda = E_val*nu/((1+nu)*(1-2*nu))
    return mu, lmbda

def eps(u):
    return sym(grad(u))

def sigma(u, rho_val):
    mu, lmbda = lame_parameters(rho_val)
    return 2*mu*eps(u) + lmbda*tr(eps(u))*Identity(2)

# -------------------------------
# Beam-specific Boundary Conditions
# -------------------------------
facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

if BEAM_TYPE == 'half_mbb':
    print("\n--- HALF MBB BEAM SETUP ---")
    
    # Left edge: Symmetry plane (models centerline of full beam)
    left_edge = CompiledSubDomain("near(x[0], 0.0, tol) && on_boundary", tol=DOLFIN_EPS)
    
    # Bottom left corner: Pin to prevent rigid body motion
    bottom_left = CompiledSubDomain("near(x[0], 0.0, tol) && near(x[1], 0.0, tol)", tol=DOLFIN_EPS)
    
    # Bottom right corner: Roller support (represents half-beam's right support)
    bottom_right = CompiledSubDomain("near(x[0], L, tol) && near(x[1], 0.0, tol)", L=L, tol=DOLFIN_EPS)
    
    # Load region at top-left
    class LoadRegion(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], H, DOLFIN_EPS) and (0.0 <= x[0] <= 2.0)
    
    # Mark boundaries
    load_marker = LoadRegion()
    load_marker.mark(facets, 1)
    left_edge.mark(facets, 2)
    
    # Boundary conditions
    bc_left_x = DirichletBC(V_u.sub(0), Constant(0.0), left_edge)  # u_x = 0 (symmetry)
    bc_bottom_left_y = DirichletBC(V_u.sub(1), Constant(0.0), bottom_left, method='pointwise')  # u_y = 0 (pin)
    bc_bottom_right_y = DirichletBC(V_u.sub(1), Constant(0.0), bottom_right, method='pointwise')  # u_y = 0 (roller)
    
    bcs = [bc_left_x, bc_bottom_left_y, bc_bottom_right_y]
    
    # Load
    load = Constant((0.0, -1.0))
    
    print(f"Boundary conditions:")
    print(f"  - Left edge (x=0): u_x = 0 (symmetry plane)")
    print(f"  - Bottom left corner (0,0): u_y = 0 (pin - prevent rigid body motion)")
    print(f"  - Bottom right corner ({L},0): u_y = 0 (roller support)")
    print(f"  - Load: Top-left region (0 <= x <= 2, was top-center of full beam)")

elif BEAM_TYPE == 'cantilever':
    print("\n--- CANTILEVER BEAM SETUP ---")
    
    # Left edge: Fixed (clamped)
    left_edge = CompiledSubDomain("near(x[0], 0.0, tol) && on_boundary", tol=DOLFIN_EPS)
    
    # Load region at right edge center
    class LoadRegion(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], L, DOLFIN_EPS) and (1.6 <= x[1] <= 2.4)
    
    # Mark boundaries
    load_marker = LoadRegion()
    load_marker.mark(facets, 1)
    left_edge.mark(facets, 2)
    
    # Boundary conditions
    bc_left = DirichletBC(V_u, Constant((0.0, 0.0)), left_edge)  # Fully clamped
    
    bcs = [bc_left]
    
    # Load
    load = Constant((0.0, -1.0))
    
    print(f"Boundary conditions:")
    print(f"  - Left edge (x=0): u_x = u_y = 0 (fully clamped)")
    print(f"  - Load: Right edge center (1.6 <= y <= 2.4)")

elif BEAM_TYPE == 'simply_supported':
    print("\n--- SIMPLY SUPPORTED BEAM SETUP ---")
    
    # Left support: Pin (both directions fixed)
    left_support = CompiledSubDomain("near(x[0], 0.0, tol) && near(x[1], 0.0, tol)", tol=DOLFIN_EPS)
    
    # Right support: Roller (vertical only)
    right_support = CompiledSubDomain("near(x[0], L, tol) && near(x[1], 0.0, tol)", L=L, tol=DOLFIN_EPS)
    
    # Load region at top center
    class LoadRegion(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], H, DOLFIN_EPS) and (3.5 <= x[0] <= 4.5)
    
    # Mark boundaries
    load_marker = LoadRegion()
    load_marker.mark(facets, 1)
    
    # Boundary conditions
    bc_left_pin = DirichletBC(V_u, Constant((0.0, 0.0)), left_support, method='pointwise')  # Pin
    bc_right_roller = DirichletBC(V_u.sub(1), Constant(0.0), right_support, method='pointwise')  # Roller
    
    bcs = [bc_left_pin, bc_right_roller]
    
    # Load
    load = Constant((0.0, -1.0))
    
    print(f"Boundary conditions:")
    print(f"  - Left support (0,0): u_x = u_y = 0 (pin)")
    print(f"  - Right support ({L},0): u_y = 0 (roller)")
    print(f"  - Load: Top center region (3.5 <= x <= 4.5)")

else:
    raise ValueError(f"Unknown beam type: {BEAM_TYPE}")

ds = Measure("ds", domain=mesh, subdomain_data=facets)

u_sol = Function(V_u)

def solve_elasticity(rho_values):
    """Solve elasticity problem with given density values"""
    rho.vector().set_local(rho_values)
    a_form = inner(sigma(u, rho), eps(v))*dx
    L_form = inner(load, v) * ds(1)
    A, b = assemble_system(a_form, L_form, bcs)
    solve(A, u_sol.vector(), b)
    return u_sol

def compliance(rho_values):
    """Compute compliance for given density values"""
    solve_elasticity(rho_values)
    rho.vector().set_local(rho_values)
    compliance_val = assemble(inner(load, u_sol) * ds(1))
    return compliance_val

def compute_sensitivity():
    """Compute sensitivity of compliance with respect to density using adjoint method"""
    rho_vals = rho.vector().get_local()
    solve_elasticity(rho_vals)
    
    # Use UNIT stiffness (E=1) for adjoint sensitivity
    rho_unit = Constant(1.0)
    unit_compliance = inner(sigma(u_sol, rho_unit), eps(u_sol))
    psi = project(unit_compliance, V_d)
    
    # Correct adjoint sensitivity
    sens = -penal * (E0 - Emin) * (rho_vals ** (penal - 1)) * psi.vector().get_local()
    return sens

def density_filter(rho_func):
    """Apply density filter to design variables"""
    V = rho_func.function_space()
    rho_filtered = Function(V)
    v_filter = TestFunction(V)
    u_filter = TrialFunction(V)
    
    a_filter = rmin**2 * inner(grad(u_filter), grad(v_filter)) * dx + u_filter * v_filter * dx
    solve(a_filter == rho_func * v_filter * dx, rho_filtered)
    return rho_filtered

def filter_sensitivity(sens_values, rho_func):
    """Apply filter to sensitivity values"""
    sens_func = Function(V_d)
    sens_func.vector().set_local(sens_values)
    filtered_sens_func = density_filter(sens_func)
    return filtered_sens_func.vector().get_local()

# Precompute filtered ones for volume constraint gradient
print("Precomputing filter gradient for volume constraint...")
ones_func = Function(V_d)
ones_func.vector().set_local(np.ones(n))
filtered_ones = density_filter(ones_func)
volume_gradient = (filtered_ones.vector().get_local() / n).reshape((1, n))

# -------------------------------
# Storage for all methods
# -------------------------------
all_results = {}

# ===============================
# OPTIMALITY CRITERIA (OC)
# ===============================
print("\n" + "=" * 70)
print("RUNNING OPTIMALITY CRITERIA (OC)")
print("=" * 70)

# OC parameters
move = 0.2
tol_change = 1e-2

def oc_update(xval, dc, dv, g):
    """Optimality Criteria update"""
    l1 = 0
    l2 = 1e9
    xnew = np.zeros(n)
    
    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        xnew = np.maximum(0.001, np.maximum(xval - move, 
               np.minimum(1.0, np.minimum(xval + move, 
               xval * np.sqrt(-dc / dv / lmid)))))
        
        if np.sum(dv * (xnew - xval)) > 0:
            l1 = lmid
        else:
            l2 = lmid
    
    return xnew

# History tracking
oc_compliance_history = []
oc_volume_history = []
oc_change_history = []

# VTK output
vtkfile_oc = File(os.path.join(output_dir, f"{BEAM_TYPE}_beam_oc.pvd"))

# Initialize design
xval_oc = volfrac * np.ones(n)

# Initial compliance
initial_compliance = compliance(xval_oc)
print(f"Initial compliance: {initial_compliance:.6f}")
print("-" * 70)

for iteration in range(max_iter):
    # Apply density filter
    rho.vector().set_local(xval_oc)
    rho_filtered = density_filter(rho)
    rho_filtered_vals = rho_filtered.vector().get_local()
    
    # Compute objective
    f0val = compliance(rho_filtered_vals)
    
    # Compute sensitivity
    raw_sens = compute_sensitivity()
    dc = filter_sensitivity(raw_sens, rho)
    
    # Volume gradient
    dv = volume_gradient.flatten()
    
    # Volume constraint
    volume_fraction = np.mean(rho_filtered_vals)
    g = volume_fraction - volfrac
    
    # OC update
    xold = xval_oc.copy()
    xval_oc = oc_update(xval_oc, dc, dv, g)
    
    # Compute change
    change = np.max(np.abs(xval_oc - xold))
    
    # Store history
    oc_compliance_history.append(f0val)
    oc_volume_history.append(volume_fraction)
    oc_change_history.append(change)
    
    # Print progress
    print(f"Iter {iteration+1:3d}: "
          f"Compliance = {f0val:10.6f}, "
          f"Volume = {volume_fraction:6.4f}, "
          f"Change = {change:8.5f}")
    
    # Save to VTK
    if iteration % 10 == 0:
        rho.vector().set_local(xval_oc)
        rho_filtered = density_filter(rho)
        rho_filtered.rename("rho", "Material Density")
        vtkfile_oc << (rho_filtered, float(iteration))

    if change < tol_change : 
        print (f"\n OC converged at iteration {iteration+1} (change = {change:.2e})")
        break
    
    
# Save final OC result
rho.vector().set_local(xval_oc)
rho_filtered_oc = density_filter(rho)
rho_filtered_oc.rename("rho_filtered", "Final Material Density")
vtkfile_oc_final = File(os.path.join(output_dir, f"{BEAM_TYPE}_beam_final_oc.pvd"))
vtkfile_oc_final << (rho_filtered_oc, 0.0)

all_results['OC'] = {
    'compliance': oc_compliance_history,
    'volume': oc_volume_history,
    'change': oc_change_history,
    'final_design': xval_oc.copy()
}

print(f"OC Final Compliance: {oc_compliance_history[-1]:.6f}")
print(f"OC Final Volume: {oc_volume_history[-1]:.6f}")

# ===============================
# IPOPT
# ===============================
print("\n" + "=" * 70)
print("RUNNING IPOPT")
print("=" * 70)

try:
    import cyipopt
    
    # History tracking - use lists that will be populated by intermediate callback
    ipopt_compliance_history = []
    ipopt_volume_history = []
    ipopt_designs = {}  # Store designs for VTK output
    
    # VTK output
    vtkfile_ipopt = File(os.path.join(output_dir, f"{BEAM_TYPE}_beam_ipopt.pvd"))
    
    class TopOptProblem:

        def __init__(self):
            self.current_x = None
            self.current_obj = None
            self.current_vol = None
            self.last_logged_obj = None  # Track last logged objective
            self.major_iter = 0  # Count major iterations
    
        def objective(self, x):
            self.current_x = x.copy()
            rho.vector().set_local(x)
            rho_filtered = density_filter(rho)
            rho_filtered_vals = rho_filtered.vector().get_local()
            obj = compliance(rho_filtered_vals)
        
            self.current_obj = obj
            self.current_vol = np.mean(rho_filtered_vals)
        
            return obj
    
        def gradient(self, x):
            rho.vector().set_local(x)
            rho_filtered = density_filter(rho)
            raw_sens = compute_sensitivity()
            grad = filter_sensitivity(raw_sens, rho)
            return grad
    
        def constraints(self, x):
            rho.vector().set_local(x)
            rho_filtered = density_filter(rho)
            rho_filtered_vals = rho_filtered.vector().get_local()
            volume_fraction = np.mean(rho_filtered_vals)
            return np.array([volume_fraction])
    
        def jacobian(self, x):
            return volume_gradient.flatten()
    
        def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du,
                    mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
                



                """Only log when objective value is different = new major iteration"""
        
        # Check if this is a new objective value (not a line search repeat)
                is_new_iteration = False
        
                if self.last_logged_obj is None:
            # First iteration
                    is_new_iteration = True
                elif abs(obj_value - self.last_logged_obj) > 1e-12:
            # Objective changed significantly = new accepted step
                    is_new_iteration = True
        
                if is_new_iteration:
                    self.major_iter += 1
                    self.last_logged_obj = obj_value
            
            # Store history
                    ipopt_compliance_history.append(obj_value)
                    ipopt_volume_history.append(self.current_vol)
                    ipopt_designs[self.major_iter] = self.current_x.copy()
            
                    print(f"Major Iter {self.major_iter:3d} (IPOPT iter {iter_count}): "
                        f"Compliance = {obj_value:10.6f}, "
                        f"Volume = {self.current_vol:6.4f}, "
                        f"Inf_pr = {inf_pr:.2e}")
            
            # Save to VTK every 10 major iterations
                    if self.major_iter % 10 == 0:
                        rho.vector().set_local(self.current_x)
                        rho_filtered_vtk = density_filter(rho)
                        rho_filtered_vtk.rename("rho", "Material Density")
                        vtkfile_ipopt << (rho_filtered_vtk, float(self.major_iter))
    
    # Initialize
    x0_ipopt = volfrac * np.ones(n)
    
    print(f"Initial compliance: {initial_compliance:.6f}")
    print("-" * 70)
    
    lb = 0.001 * np.ones(n)
    ub = 1.0 * np.ones(n)
    cl = np.array([volfrac])
    cu = np.array([volfrac])
    
    problem = TopOptProblem()
    
    nlp = cyipopt.Problem(
        n=n,
        m=1,
        problem_obj=problem,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu
    )
    
    nlp.add_option('max_iter', 300)
    nlp.add_option('tol', 1e-6)
    nlp.add_option('print_level', 0)  # Suppress IPOPT's own detailed output
    nlp.add_option('sb', 'yes')  # Suppress banne
    nlp.add_option('acceptable_tol', 1e-5)


    nlp.add_option('mu_init', 0.01)
    nlp.add_option('mu_strategy','adaptive')
    nlp.add_option('alpha_for_y', 'primal')


    nlp.add_option('hessian_approximation','limited-memory')
    nlp.add_option('limited_memory_max_history', 6 )
    
    nlp.add_option('max_soc', 4)
    nlp.add_option('constr_viol_tol', 1e-5)
    nlp.add_option('acceptable_constr_viol_tol', 1e-4)
    
    xval_ipopt, info = nlp.solve(x0_ipopt)
    
    print(f"\nIPOPT optimization completed with status: {info['status']}")
    print(f"Total iterations: {len(ipopt_compliance_history)}")
    
    # Save final IPOPT result
    rho.vector().set_local(xval_ipopt)
    rho_filtered_ipopt = density_filter(rho)
    rho_filtered_ipopt.rename("rho_filtered", "Final Material Density")
    vtkfile_ipopt_final = File(os.path.join(output_dir, f"{BEAM_TYPE}_beam_final_ipopt.pvd"))
    vtkfile_ipopt_final << (rho_filtered_ipopt, 0.0)
    
    # Also save displacement field
    solve_elasticity(xval_ipopt)
    u_sol.rename("displacement", "Displacement Field")
    vtkfile_ipopt_disp = File(os.path.join(output_dir, f"{BEAM_TYPE}_beam_displacement_ipopt.pvd"))
    vtkfile_ipopt_disp << u_sol
    
    all_results['IPOPT'] = {
        'compliance': ipopt_compliance_history,
        'volume': ipopt_volume_history,
        'final_design': xval_ipopt.copy()
    }
    
    print(f"IPOPT Final Compliance: {ipopt_compliance_history[-1]:.6f}")
    print(f"IPOPT Final Volume: {ipopt_volume_history[-1]:.6f}")
    print(f"IPOPT VTK files saved successfully")
    
except ImportError:
    print("WARNING: cyipopt not installed. Skipping IPOPT optimization.")
    print("Install with: pip install cyipopt")
    all_results['IPOPT'] = None
# ===============================
# MMA
# ===============================
print("\n" + "=" * 70)
print("RUNNING MMA (Method of Moving Asymptotes)")
print("=" * 70)

# MMA Optimization Setup
m = 1  # Number of constraints
epsimin = 0.0000001
eeen = np.ones((n, 1))
eeem = np.ones((m, 1))
zeron = np.zeros((n, 1))
zerom = np.zeros((m, 1))
tol_change = 1e-2

# Initialize design variables
xval_mma = volfrac * np.ones((n, 1))
xold1 = xval_mma.copy()
xold2 = xval_mma.copy()
xmin = 0.001 * eeen  
xmax = 1.0 * eeen

# Initialize MMA parameters
low = xval_mma - 0.5 * eeen
upp = xval_mma + 0.5 * eeen
c = 1000 * eeem
d = eeem
a0 = 1
a_mma = zerom  

# Initialize raa parameters
raa0 = 0.001
raa = 0.001 * eeem
raa0eps = 1e-6
raaeps = 1e-6

# History tracking
mma_compliance_history = []
mma_volume_history = []
mma_constraint_history = []

# VTK output
vtkfile_mma = File(os.path.join(output_dir, f"{BEAM_TYPE}_beam_mma.pvd"))

print(f"Initial compliance: {initial_compliance:.6f}")
print("-" * 70)

for iteration in range(max_iter):
    
    # Apply density filter to current design
    rho.vector().set_local(xval_mma.flatten())
    rho_filtered = density_filter(rho)
    rho_filtered_vals = rho_filtered.vector().get_local()
    
    # Compute objective (compliance) with filtered densities
    f0val = compliance(rho_filtered_vals)
    
    # Compute sensitivity with filtered densities
    raw_sens = compute_sensitivity()
    df0dx = filter_sensitivity(raw_sens, rho).reshape((n, 1))
    
    # Check for invalid gradients
    if np.any(np.isnan(df0dx)) or np.any(np.isinf(df0dx)):
        print(f"Warning: Invalid gradients at iteration {iteration+1}, setting to zero")
        df0dx = np.nan_to_num(df0dx, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Volume constraint: mean(x_filtered) - volfrac <= 0
    volume_fraction = np.mean(rho_filtered_vals)
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
    
    # Solve MMA subproblem with IPOPT
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
    
    # Apply bound constraints
    xmma = np.maximum(xmin, np.minimum(xmax, xmma))
    
    # Check for invalid design
    if np.any(np.isnan(xmma)) or np.any(np.isinf(xmma)):
        print(f"Warning: Invalid design at iteration {iteration+1}")
        xmma = xval_mma.copy()

    change = np.max(np.abs(xmma - xval_mma))
    
    # Update design variables
    xold2 = xold1.copy()
    xold1 = xval_mma.copy()
    xval_mma = xmma.copy()
    
    # Update filtered density for visualization
    rho.vector().set_local(xval_mma.flatten())
    rho_filtered = density_filter(rho)
    
    # Store history
    mma_compliance_history.append(f0val)
    mma_volume_history.append(volume_fraction)
    mma_constraint_history.append(constraint_val)
    
    # Print progress
    print(f"Iter {iteration+1:3d}: "
          f"Compliance = {f0val:10.6f}, "
          f"Volume = {volume_fraction:6.4f}, "
          f"Constraint = {constraint_val:8.5f}, "
          f"Lambda = {lam[0,0]:8.5f}")
    
    # Save to VTK every 10 iterations
    if iteration % 10 == 0:
        rho_filtered.rename("rho", "Material Density")
        vtkfile_mma << (rho_filtered, float(iteration))
    
    # Adjust penalty if constraint violated significantly
    if abs(constraint_val) > 0.01:
        c = np.minimum(c * 1.5, 100000 * eeem)
    elif abs(constraint_val) > 0.005:
        c = np.minimum(c*1.2 , 100000 * eeem)

    if change < tol_change and abs(constraint_val) < 7e-2:
        print(f"\n MMA converged at iteration {iteration+1} (change = {change:.2e})")
        break
        
    
    

# Save final MMA result
rho.vector().set_local(xval_mma.flatten())
rho_filtered_mma = density_filter(rho)
rho_filtered_mma.rename("rho_filtered", "Final Material Density")
vtkfile_mma_final = File(os.path.join(output_dir, f"{BEAM_TYPE}_beam_final_mma.pvd"))
vtkfile_mma_final << (rho_filtered_mma, 0.0)

all_results['MMA'] = {
    'compliance': mma_compliance_history,
    'volume': mma_volume_history,
    'constraint': mma_constraint_history,
    'final_design': xval_mma.flatten().copy()
}

print(f"MMA Final Compliance: {mma_compliance_history[-1]:.6f}")
print(f"MMA Final Volume: {mma_volume_history[-1]:.6f}")

# ===============================
# GENERATE COMPARISON PLOTS
# ===============================
print("\n" + "=" * 70)
print("GENERATING COMPARISON PLOTS")
print("=" * 70)

# Set plot style - use default matplotlib style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

colors = {'OC': '#1f77b4', 'IPOPT': '#ff7f0e', 'MMA': '#2ca02c'}
markers = {'OC': 'o', 'IPOPT': 's', 'MMA': '^'}

# -------------------------------------------------------
# Plot 1: Compliance vs Iteration (All Methods)
# -------------------------------------------------------
plt.figure(figsize=(12, 7))
plt.plot(range(1, len(oc_compliance_history)+1), oc_compliance_history, 
         color=colors['OC'], linewidth=2, marker=markers['OC'], 
         markevery=max(1, len(oc_compliance_history)//10), markersize=8,
         label='OC', alpha=0.8)

if all_results['IPOPT'] is not None:
    plt.plot(range(1, len(ipopt_compliance_history)+1), ipopt_compliance_history, 
             color=colors['IPOPT'], linewidth=2, marker=markers['IPOPT'],
             markevery=max(1, len(ipopt_compliance_history)//10), markersize=8,
             label='IPOPT', alpha=0.8)

plt.plot(range(1, len(mma_compliance_history)+1), mma_compliance_history, 
         color=colors['MMA'], linewidth=2, marker=markers['MMA'],
         markevery=max(1, len(mma_compliance_history)//10), markersize=8,
         label='MMA', alpha=0.8)

plt.xlabel('Iteration', fontsize=14, fontweight='bold')
plt.ylabel('Compliance', fontsize=14, fontweight='bold')
plt.title(f'{BEAM_TYPE.upper().replace("_", " ")} Beam: Compliance vs Iteration', 
          fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='best', framealpha=0.9)
plt.grid(True, alpha= 0.3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{BEAM_TYPE}_compliance_comparison.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, f'{BEAM_TYPE}_compliance_comparison.pdf'), bbox_inches='tight')
print(f"Saved: {BEAM_TYPE}_compliance_comparison.png")
plt.close()

# -------------------------------------------------------
# Plot 2: Volume Fraction vs Iteration (All Methods)
# -------------------------------------------------------
plt.figure(figsize=(12, 7))
plt.plot(range(1, len(oc_volume_history)+1), oc_volume_history, 
         color=colors['OC'], linewidth=2, marker=markers['OC'], 
         markevery=max(1, len(oc_volume_history)//10), markersize=8,
         label='OC', alpha=0.8)

if all_results['IPOPT'] is not None:
    plt.plot(range(1, len(ipopt_volume_history)+1), ipopt_volume_history, 
             color=colors['IPOPT'], linewidth=2, marker=markers['IPOPT'],
             markevery=max(1, len(ipopt_volume_history)//10), markersize=8,
             label='IPOPT', alpha=0.8)

plt.plot(range(1, len(mma_volume_history)+1), mma_volume_history, 
         color=colors['MMA'], linewidth=2, marker=markers['MMA'],
         markevery=max(1, len(mma_volume_history)//10), markersize=8,
         label='MMA', alpha=0.8)

plt.axhline(y=volfrac, color='red', linestyle='--', linewidth=2, label=f'Target = {volfrac}')
plt.xlabel('Iteration', fontsize=14, fontweight='bold')
plt.ylabel('Volume Fraction', fontsize=14, fontweight='bold')
plt.title(f'{BEAM_TYPE.upper().replace("_", " ")} Beam: Volume Fraction vs Iteration', 
          fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='best', framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{BEAM_TYPE}_volume_comparison.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, f'{BEAM_TYPE}_volume_comparison.pdf'), bbox_inches='tight')
print(f"Saved: {BEAM_TYPE}_volume_comparison.png")
plt.close()

# -------------------------------------------------------
# Plot 3: Compliance (Log Scale) vs Iteration
# -------------------------------------------------------
plt.figure(figsize=(12, 7))
plt.semilogy(range(1, len(oc_compliance_history)+1), oc_compliance_history, 
             color=colors['OC'], linewidth=2, marker=markers['OC'], 
             markevery=max(1, len(oc_compliance_history)//10), markersize=8,
             label='OC', alpha=0.8)

if all_results['IPOPT'] is not None:
    plt.semilogy(range(1, len(ipopt_compliance_history)+1), ipopt_compliance_history, 
                 color=colors['IPOPT'], linewidth=2, marker=markers['IPOPT'],
                 markevery=max(1, len(ipopt_compliance_history)//10), markersize=8,
                 label='IPOPT', alpha=0.8)

plt.semilogy(range(1, len(mma_compliance_history)+1), mma_compliance_history, 
             color=colors['MMA'], linewidth=2, marker=markers['MMA'],
             markevery=max(1, len(mma_compliance_history)//10), markersize=8,
             label='MMA', alpha=0.8)

plt.xlabel('Iteration', fontsize=14, fontweight='bold')
plt.ylabel('Compliance (Log Scale)', fontsize=14, fontweight='bold')
plt.title(f'{BEAM_TYPE.upper().replace("_", " ")} Beam: Compliance vs Iteration (Log Scale)', 
          fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='best', framealpha=0.9)
plt.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{BEAM_TYPE}_compliance_log_comparison.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, f'{BEAM_TYPE}_compliance_log_comparison.pdf'), bbox_inches='tight')
print(f"Saved: {BEAM_TYPE}_compliance_log_comparison.png")
plt.close()

# -------------------------------------------------------
# Plot 4: Convergence Rate (Relative Change in Compliance)
# -------------------------------------------------------
plt.figure(figsize=(12, 7))

# Calculate relative changes
oc_rel_change = [abs(oc_compliance_history[i] - oc_compliance_history[i-1])/oc_compliance_history[i-1] 
                 for i in range(1, len(oc_compliance_history))]
mma_rel_change = [abs(mma_compliance_history[i] - mma_compliance_history[i-1])/mma_compliance_history[i-1] 
                  for i in range(1, len(mma_compliance_history))]

plt.semilogy(range(2, len(oc_compliance_history)+1), oc_rel_change, 
             color=colors['OC'], linewidth=2, marker=markers['OC'], 
             markevery=max(1, len(oc_rel_change)//10), markersize=8,
             label='OC', alpha=0.8)

if all_results['IPOPT'] is not None and len(ipopt_compliance_history) > 1:
    ipopt_rel_change = [abs(ipopt_compliance_history[i] - ipopt_compliance_history[i-1])/ipopt_compliance_history[i-1] 
                        for i in range(1, len(ipopt_compliance_history))]
    plt.semilogy(range(2, len(ipopt_compliance_history)+1), ipopt_rel_change, 
                 color=colors['IPOPT'], linewidth=2, marker=markers['IPOPT'],
                 markevery=max(1, len(ipopt_rel_change)//10), markersize=8,
                 label='IPOPT', alpha=0.8)

plt.semilogy(range(2, len(mma_compliance_history)+1), mma_rel_change, 
             color=colors['MMA'], linewidth=2, marker=markers['MMA'],
             markevery=max(1, len(mma_rel_change)//10), markersize=8,
             label='MMA', alpha=0.8)

plt.xlabel('Iteration', fontsize=14, fontweight='bold')
plt.ylabel('Relative Change in Compliance', fontsize=14, fontweight='bold')
plt.title(f'{BEAM_TYPE.upper().replace("_", " ")} Beam: Convergence Rate', 
          fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='best', framealpha=0.9)
plt.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{BEAM_TYPE}_convergence_rate.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, f'{BEAM_TYPE}_convergence_rate.pdf'), bbox_inches='tight')
print(f"Saved: {BEAM_TYPE}_convergence_rate.png")
plt.close()

# -------------------------------------------------------
# Plot 5: Volume Constraint Violation
# -------------------------------------------------------
plt.figure(figsize=(12, 7))

oc_vol_error = [abs(v - volfrac) for v in oc_volume_history]
mma_vol_error = [abs(v - volfrac) for v in mma_volume_history]

plt.semilogy(range(1, len(oc_vol_error)+1), oc_vol_error, 
             color=colors['OC'], linewidth=2, marker=markers['OC'], 
             markevery=max(1, len(oc_vol_error)//10), markersize=8,
             label='OC', alpha=0.8)

if all_results['IPOPT'] is not None:
    ipopt_vol_error = [abs(v - volfrac) for v in ipopt_volume_history]
    plt.semilogy(range(1, len(ipopt_vol_error)+1), ipopt_vol_error, 
                 color=colors['IPOPT'], linewidth=2, marker=markers['IPOPT'],
                 markevery=max(1, len(ipopt_vol_error)//10), markersize=8,
                 label='IPOPT', alpha=0.8)

plt.semilogy(range(1, len(mma_vol_error)+1), mma_vol_error, 
             color=colors['MMA'], linewidth=2, marker=markers['MMA'],
             markevery=max(1, len(mma_vol_error)//10), markersize=8,
             label='MMA', alpha=0.8)

plt.xlabel('Iteration', fontsize=14, fontweight='bold')
plt.ylabel('Volume Constraint Violation', fontsize=14, fontweight='bold')
plt.title(f'{BEAM_TYPE.upper().replace("_", " ")} Beam: Volume Constraint Satisfaction', 
          fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='best', framealpha=0.9)
plt.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{BEAM_TYPE}_volume_constraint_violation.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, f'{BEAM_TYPE}_volume_constraint_violation.pdf'), bbox_inches='tight')
print(f"Saved: {BEAM_TYPE}_volume_constraint_violation.png")
plt.close()

# -------------------------------------------------------
# Plot 6: Side-by-side comparison of final designs
# -------------------------------------------------------
fig, axes = plt.subplots(1, 3 if all_results['IPOPT'] is not None else 2, figsize=(18, 5))

# OC
rho.vector().set_local(all_results['OC']['final_design'])
rho_filtered_oc = density_filter(rho)
rho_array_oc = rho_filtered_oc.vector().get_local().reshape((90+1, 180+1))

im0 = axes[0].imshow(rho_array_oc, cmap='gray_r', origin='lower', extent=[0, L, 0, H])
axes[0].set_title('OC Method', fontsize=14, fontweight='bold')
axes[0].set_xlabel('x', fontsize=12)
axes[0].set_ylabel('y', fontsize=12)
axes[0].set_aspect('equal')
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# IPOPT
if all_results['IPOPT'] is not None:
    rho.vector().set_local(all_results['IPOPT']['final_design'])
    rho_filtered_ipopt = density_filter(rho)
    rho_array_ipopt = rho_filtered_ipopt.vector().get_local().reshape((90+1, 180+1))
    
    im1 = axes[1].imshow(rho_array_ipopt, cmap='gray_r', origin='lower', extent=[0, L, 0, H])
    axes[1].set_title('IPOPT Method', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('y', fontsize=12)
    axes[1].set_aspect('equal')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    mma_idx = 2
else:
    mma_idx = 1

# MMA
rho.vector().set_local(all_results['MMA']['final_design'])
rho_filtered_mma = density_filter(rho)
rho_array_mma = rho_filtered_mma.vector().get_local().reshape((90+1, 180+1))

im2 = axes[mma_idx].imshow(rho_array_mma, cmap='gray_r', origin='lower', extent=[0, L, 0, H])
axes[mma_idx].set_title('MMA Method', fontsize=14, fontweight='bold')
axes[mma_idx].set_xlabel('x', fontsize=12)
axes[mma_idx].set_ylabel('y', fontsize=12)
axes[mma_idx].set_aspect('equal')
fig.colorbar(im2, ax=axes[mma_idx], fraction=0.046, pad=0.04)

plt.suptitle(f'{BEAM_TYPE.upper().replace("_", " ")} Beam: Final Designs Comparison', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{BEAM_TYPE}_final_designs_comparison.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, f'{BEAM_TYPE}_final_designs_comparison.pdf'), bbox_inches='tight')
print(f"Saved: {BEAM_TYPE}_final_designs_comparison.png")
plt.close()

# -------------------------------------------------------
# Plot 7: Compliance Reduction Percentage
# -------------------------------------------------------
plt.figure(figsize=(12, 7))

oc_reduction = [(initial_compliance - c)/initial_compliance * 100 for c in oc_compliance_history]
mma_reduction = [(initial_compliance - c)/initial_compliance * 100 for c in mma_compliance_history]

plt.plot(range(1, len(oc_reduction)+1), oc_reduction, 
         color=colors['OC'], linewidth=2, marker=markers['OC'], 
         markevery=max(1, len(oc_reduction)//10), markersize=8,
         label='OC', alpha=0.8)

if all_results['IPOPT'] is not None:
    ipopt_reduction = [(initial_compliance - c)/initial_compliance * 100 for c in ipopt_compliance_history]
    plt.plot(range(1, len(ipopt_reduction)+1), ipopt_reduction, 
             color=colors['IPOPT'], linewidth=2, marker=markers['IPOPT'],
             markevery=max(1, len(ipopt_reduction)//10), markersize=8,
             label='IPOPT', alpha=0.8)

plt.plot(range(1, len(mma_reduction)+1), mma_reduction, 
         color=colors['MMA'], linewidth=2, marker=markers['MMA'],
         markevery=max(1, len(mma_reduction)//10), markersize=8,
         label='MMA', alpha=0.8)

plt.xlabel('Iteration', fontsize=14, fontweight='bold')
plt.ylabel('Compliance Reduction (%)', fontsize=14, fontweight='bold')
plt.title(f'{BEAM_TYPE.upper().replace("_", " ")} Beam: Compliance Reduction Progress', 
          fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='best', framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{BEAM_TYPE}_compliance_reduction.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, f'{BEAM_TYPE}_compliance_reduction.pdf'), bbox_inches='tight')
print(f"Saved: {BEAM_TYPE}_compliance_reduction.png")
plt.close()

# ===============================
# GENERATE SUMMARY TABLE
# ===============================
print("\n" + "=" * 70)
print("OPTIMIZATION SUMMARY")
print("=" * 70)

summary_data = []
summary_data.append(["Method", "Final Compliance", "Final Volume", "Iterations", "Compliance Reduction (%)"])
summary_data.append(["-" * 10, "-" * 18, "-" * 14, "-" * 12, "-" * 24])

oc_iters = len(oc_compliance_history)
oc_final_comp = oc_compliance_history[-1]
oc_final_vol = oc_volume_history[-1]
oc_reduction_pct = ((initial_compliance - oc_final_comp)/initial_compliance)*100
summary_data.append(["OC", f"{oc_final_comp:.6f}", f"{oc_final_vol:.6f}", f"{oc_iters}", f"{oc_reduction_pct:.2f}%"])

if all_results['IPOPT'] is not None:
    ipopt_iters = len(ipopt_compliance_history)
    ipopt_final_comp = ipopt_compliance_history[-1]
    ipopt_final_vol = ipopt_volume_history[-1]
    ipopt_reduction_pct = ((initial_compliance - ipopt_final_comp)/initial_compliance)*100
    summary_data.append(["IPOPT", f"{ipopt_final_comp:.6f}", f"{ipopt_final_vol:.6f}", f"{ipopt_iters}", f"{ipopt_reduction_pct:.2f}%"])

mma_iters = len(mma_compliance_history)
mma_final_comp = mma_compliance_history[-1]
mma_final_vol = mma_volume_history[-1]
mma_reduction_pct = ((initial_compliance - mma_final_comp)/initial_compliance)*100
summary_data.append(["MMA", f"{mma_final_comp:.6f}", f"{mma_final_vol:.6f}", f"{mma_iters}", f"{mma_reduction_pct:.2f}%"])

# Print table
for row in summary_data:
    print(f"{row[0]:<10} {row[1]:<18} {row[2]:<14} {row[3]:<12} {row[4]:<24}")

# Save summary to file
summary_file = os.path.join(output_dir, f'{BEAM_TYPE}_optimization_summary.txt')
with open(summary_file, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write(f"{BEAM_TYPE.upper().replace('_', ' ')} BEAM TOPOLOGY OPTIMIZATION SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Domain: {L} x {H}\n")
    f.write(f"Mesh: {mesh.num_vertices()} nodes, {mesh.num_cells()} elements\n")
    f.write(f"Design variables: {n}\n")
    f.write(f"Target volume fraction: {volfrac}\n")
    f.write(f"Initial compliance: {initial_compliance:.6f}\n\n")
    
    for row in summary_data:
        f.write(f"{row[0]:<10} {row[1]:<18} {row[2]:<14} {row[3]:<12} {row[4]:<24}\n")
    
    f.write("\n" + "=" * 70 + "\n")

print(f"\nSummary saved to: {summary_file}")

# ===============================
# SAVE DATA TO CSV
# ===============================
import csv

# Save convergence data
csv_file = os.path.join(output_dir, f'{BEAM_TYPE}_convergence_data.csv')
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    
    # Header
    if all_results['IPOPT'] is not None:
        writer.writerow(['Iteration', 'OC_Compliance', 'OC_Volume', 'IPOPT_Compliance', 'IPOPT_Volume', 
                        'MMA_Compliance', 'MMA_Volume'])
    else:
        writer.writerow(['Iteration', 'OC_Compliance', 'OC_Volume', 'MMA_Compliance', 'MMA_Volume'])
    
    # Data
    max_len = max(len(oc_compliance_history), len(mma_compliance_history))
    if all_results['IPOPT'] is not None:
        max_len = max(max_len, len(ipopt_compliance_history))
    
    for i in range(max_len):
        row = [i+1]
        
        # OC data
        if i < len(oc_compliance_history):
            row.extend([oc_compliance_history[i], oc_volume_history[i]])
        else:
            row.extend(['', ''])
        
        # IPOPT data
        if all_results['IPOPT'] is not None:
            if i < len(ipopt_compliance_history):
                row.extend([ipopt_compliance_history[i], ipopt_volume_history[i]])
            else:
                row.extend(['', ''])
        
        # MMA data
        if i < len(mma_compliance_history):
            row.extend([mma_compliance_history[i], mma_volume_history[i]])
        else:
            row.extend(['', ''])
        
        writer.writerow(row)

print(f"Convergence data saved to: {csv_file}")

print("\n" + "=" * 70)
print("ALL OPTIMIZATIONS COMPLETED!")
print(f"All results saved to: {output_dir}/")
print("=" * 70)
print("\nGenerated files:")
print(f"  - Compliance comparison plot")
print(f"  - Volume fraction comparison plot")
print(f"  - Compliance log scale plot")
print(f"  - Convergence rate plot")
print(f"  - Volume constraint violation plot")
print(f"  - Final designs comparison plot")
print(f"  - Compliance reduction plot")
print(f"  - Optimization summary (TXT)")
print(f"  - Convergence data (CSV)")
print(f"  - VTK files for each method")
print("=" * 70)