import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def solve_differential_equations(init_cond, t_span):
    def model(y, t):
        x, z, v_x, v_z, theta = y
        dxdt = v_x
        dzdt = v_z
        dv_xdt = 0.4 * abs(np.sin(t)) * np.cos(theta) + 0.2 * abs(np.cos(t)) * np.sin(theta)
        dv_zdt = 0.2 * abs(np.cos(t)) * np.cos(theta) - 0.4 * abs(np.sin(t)) * np.sin(theta)
        dthetadt = theta_dot_0
        return [dxdt, dzdt, dv_xdt, dv_zdt, dthetadt]

    sol = odeint(model, init_cond, t_span)
    return np.transpose(sol)

def perform_extended_kalman_filter(dt, H, R, gps, t_span, P, X_real, init_cond):
    X = np.zeros((5, len(t_span)))
    X[:, 0] = init_cond + dt * np.array([init_cond[1], init_cond[3], 0.4 * abs(np.sin(0)) * np.cos(init_cond[4]) + 0.2 * abs(np.cos(0)) * np.sin(init_cond[4]),
                                         0.2 * abs(np.cos(0)) * np.cos(init_cond[4]) - 0.4 * abs(np.sin(0)) * np.sin(init_cond[4]), theta_dot_0])
    std_k = np.sqrt(np.diag(P))[:, None]

    for i in range(len(t_span) - 1):
        X[:, i + 1] = X[:, i] + dt * np.array([X[2, i], X[3, i], 0.4 * abs(np.sin(t_span[i])) * np.cos(X[4, i]) + 0.2 * abs(np.cos(t_span[i])) * np.sin(X[4, i]),
                                              0.2 * abs(np.cos(t_span[i])) * np.cos(X[4, i]) - 0.4 * abs(np.sin(t_span[i])) * np.sin(X[4, i]), theta_dot_0])

        J = np.array([[1, 0, dt, 0, 0],
                      [0, 1, 0, dt, 0],
                      [0, 0, 1, 0, dt * (0.2 * abs(np.cos(t_span[i])) * np.cos(X[4, i]) - 0.4 * abs(np.sin(t_span[i])) * np.sin(X[4, i]))],
                      [0, 0, 0, 1, dt * (-0.4 * abs(np.sin(t_span[i])) * np.cos(X[4, i]) - 0.2 * abs(np.cos(t_span[i])) * np.sin(X[4, i]))],
                      [0, 0, 0, 0, 1]])

        P = J @ P @ J.T
        Residual_m = gps[:, i + 1] - H @ X[:, i + 1]
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        X[:, i + 1] = X[:, i + 1] + K @ Residual_m
        P = (np.eye(5) - K @ H) @ P
        std_k = np.column_stack((std_k, np.sqrt(np.diag(P))))

    return S, X, P, K, Residual_m, std_k

def define_gps_vector(X_real, R):
    gps = np.zeros((2, len(X_real[0, :])))
    gps[0, :] = X_real[0, :] + np.sqrt(R[0, 0]) * np.random.randn(len(X_real[0, :]))
    gps[1, :] = X_real[1, :] + np.sqrt(R[1, 1]) * np.random.randn(len(X_real[1, :]))
    return gps

def plot_results(gps, X, X_real):
    plt.plot(gps[0, :], gps[1, :], 'o',markerfacecolor='none', markeredgecolor='blue',  label='GPS Location')
    plt.plot(X[0, :], X[1, :], '*',markerfacecolor='none', markeredgecolor='orange', label='EKF Prediction')
    plt.plot(X_real[0, :], X_real[1, :], '-k', linewidth=0.8, label='Trajectory')
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    plt.legend(loc='best')
    plt.show()


# Initialize Parameters and Initial Conditions
x_0 = 100
z_0 = 200
v_x_0 = 2
v_z_0 = 1
theta_0 = 0.01
theta_dot_0 = 0.01

P = np.diag([10, 15, 2, 3, 0.08]) ** 2
R = np.diag([8, 4]) ** 2
H = np.array([[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0]])

dt = 1e-2
T = 40
t_span = np.arange(0, T + dt, dt)

init_cond = [x_0, v_x_0, z_0, v_z_0, theta_0]

# Solve Differential Equations
X_real = solve_differential_equations(init_cond, t_span)

# Define GPS Vector
gps = define_gps_vector(X_real, R)

# Perform Extended Kalman Filter
S, X, P, K, Residual_m, std_k = perform_extended_kalman_filter(dt, H, R, gps, t_span, P, X_real, init_cond)

# Plot The Result
plot_results(gps, X, X_real)