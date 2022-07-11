from time import time
import numpy as np
from utils import visualize
from casadi import *
import sys
import GPI

# Simulation params
np.random.seed(10)
time_step = 0.5 # time between steps in seconds
sim_time = 120    # simulation time

# Car params
x_init = 1.5
y_init = 0.0
theta_init = np.pi/2
v_max = 1
v_min = 0
w_max = 1
w_min = -1


# This function returns the reference point at time step k
def lissajous(k):
    xref_start = 0
    yref_start = 0
    A = 2
    B = 2
    a = 2*np.pi/50
    b = 3*a
    T = np.round(2*np.pi/(a*time_step))
    k = k % T
    delta = np.pi/2
    xref = xref_start + A*np.sin(a*k*time_step + delta)
    yref = yref_start + B*np.sin(b*k*time_step)
    v = [A*a*np.cos(a*k*time_step + delta), B*b*np.cos(b*k*time_step)]
    thetaref = np.arctan2(v[1], v[0])
    return [xref, yref, thetaref]

# This function implements a simple P controller
def simple_controller(cur_state, ref_state):
    k_v = 0.55
    k_w = 1.0
    v = k_v*np.sqrt((cur_state[0] - ref_state[0])**2 + (cur_state[1] - ref_state[1])**2)
    v = np.clip(v, v_min, v_max)
    angle_diff = ref_state[2] - cur_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi ) - np.pi
    w = k_w*angle_diff
    w = np.clip(w, w_min, w_max)
    return [v,w]


# This function implements a generalized policy iteration (GPI) controller
# This function implements a receding-horizon certainty equivalent control (CEC) controller
def CEC_controller(cur_state, ref_state, cur_iter):
    '''
    input:
        cur_state: np.array((x,y,theta))
        ref_state : np.array
        cur_iter : int
    '''

    gamma = 0.99 # Y => [0,1)
    T = 5
    Q = np.ones((2,2))*15
    R = np.eye(2)
    q = 1

    u = SX.sym('u',2,T)

    e_tilde = cur_state - ref_state #error
    p_tilde = e_tilde[:2].reshape((2,1)) #position deviation from ref
    theta_tilde = e_tilde[-1].reshape((-1,1)) #orientaion deviation from ref
    f = 0

    # stage cost
    for i in range(T-1):

        gamma_T = gamma ** cur_iter
        summation = p_tilde.T @ Q @ p_tilde + q*(1 - cos(theta_tilde))**2 + (u[:,i].T @ R @ u[:,i])
        f = f + gamma_T * summation

        r_t = lissajous(i+cur_iter+1) #next ref point
        # r_t = lissajous(i+1) #next ref point
        p_tilde_x = p_tilde[0,:] + time_step * cos(theta_tilde + ref_state[-1]) * u[0,i] + ref_state[0] - r_t[0]
        p_tilde_y = p_tilde[1,:] + time_step * sin(theta_tilde + ref_state[-1]) * u[0,i] + ref_state[1] - r_t[1]
        p_tilde = vertcat(p_tilde_x , p_tilde_y)
        theta_tilde = theta_tilde + time_step * u[1,i] + ref_state[2]-  r_t[-1]
        ref_state = r_t
        # print("HELLOO ")

    #terminal cost
    f = f + (p_tilde.T @ Q @ p_tilde) + q * (1 - cos(theta_tilde))**2

    # g and h constraints.
    next_x = cur_state[0] + time_step * cos(cur_state[-1]) * u[0,0]
    next_y = cur_state[1] + time_step * sin(cur_state[-1]) * u[0,0]

    # There are two circular obstacles C1 centered at (−2, −2) with radius 0.5 and C2 centered at (1, 2) with radius 0.5.
    g = (next_x + 2)**2 + (next_y + 2)**2 - (0.5)**2
    h = (next_x - 1)**2 + (next_y - 2)**2 - (0.5)**2

    nlp = {}                 # NLP declaration
    nlp['x']= u[:]           # decision vars
    nlp['f'] = f             # objective
    nlp['g'] = vertcat(g,h)  # constraints

    # Create solver instance
    F = nlpsol('F','ipopt',nlp);
    lbx = [v_min, w_min] * T # lower bound
    ubx = [v_max, w_max] * T # upper bound

    # Solve the problem using a guess
    ans = F(lbx = lbx,ubx = ubx, lbg= [0,0])
    # print(ans['x'])
    # print(np.array(ans['x']).reshape(2,T).T[0])
    result = np.array(ans['x']).reshape(T,2)
    # print("result: ", result)
    # sys.exit()
    return result[0]

# This function implement the car dynamics
def car_next_state(time_step, cur_state, control, noise = True):
    theta = cur_state[2]
    rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    f = rot_3d_z @ control
    mu, sigma = 0, 0.04 # mean and standard deviation for (x,y)
    w_xy = np.random.normal(mu, sigma, 2)
    mu, sigma = 0, 0.004  # mean and standard deviation for theta
    w_theta = np.random.normal(mu, sigma, 1)
    w = np.concatenate((w_xy, w_theta))
    if noise:
        return cur_state + time_step*f.flatten() + w
    else:
        return cur_state + time_step * f.flatten()

if __name__ == '__main__':
    # Obstacles in the environment
    obstacles = np.array([[-2,-2,0.5], [1,2,0.5]])
    # Params
    traj = lissajous
    ref_traj = []
    error = 0.0
    car_states = []
    times = []
    # Start main loop
    main_loop = time()  # return time in sec
    # Initialize state
    cur_state = np.array([x_init, y_init, theta_init])
    cur_iter = 0
    # Main loop
    while (cur_iter * time_step < sim_time):
        t1 = time()
        # Get reference state
        cur_time = cur_iter*time_step
        cur_ref = traj(cur_iter)
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        # Generate control input
        # TODO: Replace this simple controller by your own controller
        # control = simple_controller(cur_state, cur_ref)
        control = CEC_controller(cur_state, cur_ref, cur_iter)
        print("[v,w]", control)
        ################################################################

        # Apply control input
        next_state = car_next_state(time_step, cur_state, control, noise=True)
        # Update current state
        cur_state = next_state
        # Loop time
        t2 = time()
        print(cur_iter)
        print(t2-t1)
        times.append(t2-t1)
        error = error + np.linalg.norm(cur_state - cur_ref)
        cur_iter = cur_iter + 1

    main_loop_time = time()
    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('Average iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('Final error: ', error)

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    visualize(car_states, ref_traj, obstacles, times, time_step, save=True)
