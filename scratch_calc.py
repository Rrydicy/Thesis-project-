import sympy as sp

t = sp.Symbol('t')
theta = sp.Function('theta')(t)
phi = sp.Function('phi')(t)

M, Jx, Jz, m, Ra, Lc, g = sp.symbols('M Jx Jz m Ra Lc g')

# The new K(theta)
K = Jz + m*(Ra**2 + Lc**2 * sp.sin(theta)**2) + (Jx - Jz)*sp.sin(theta)**2

# The cross term coefficient C(theta)
C = m * Lc * Ra * sp.cos(theta)

T = 0.5 * M * theta.diff(t)**2 + 0.5 * K * phi.diff(t)**2 + C * theta.diff(t) * phi.diff(t)
V = m * g * Lc * sp.cos(theta) 

L = T - V

eq_theta = sp.diff(sp.diff(L, theta.diff(t)), t) - sp.diff(L, theta)
eq_phi = sp.diff(sp.diff(L, phi.diff(t)), t) - sp.diff(L, phi)

print("Eq theta (raw):")
print(sp.simplify(eq_theta))
print("\nEq phi (raw):")
print(sp.simplify(eq_phi))

# Energy approach for control law:
E = 0.5 * Jz * theta.diff(t)**2 + m * g * Lc * (sp.cos(theta) - 1)  # the paper uses this E
# Let's write the energy properly. E is just a candidate Lyapunov function.
# V_lyap = 0.5 * (E - E0)**2
# dV_lyap/dt = (E - E0) * dE/dt
# dE/dt = Jz * theta_dot * theta_ddot - m * g * Lc * sp.sin(theta) * theta_dot

# We need to find theta_ddot.
# Let's solve the eq_theta and eq_phi for theta_ddot and phi_ddot.
# eq_theta = 0, eq_phi = tau (or J_r Omega u, etc.)
tau = sp.Symbol('tau')
sols = sp.solve([eq_theta, eq_phi - tau], [theta.diff(t, 2), phi.diff(t, 2)])
theta_ddot = sp.simplify(sols[theta.diff(t, 2)])
phi_ddot = sp.simplify(sols[phi.diff(t, 2)])

print("\ntheta_ddot:")
print(theta_ddot)

print("\ndE/dt substitute theta_ddot:")
dE_dt = Jz * theta.diff(t) * theta_ddot - m * g * Lc * sp.sin(theta) * theta.diff(t)
print(sp.simplify(dE_dt))
