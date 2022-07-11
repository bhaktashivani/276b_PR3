from time import time
import numpy as np
from utils import visualize
from casadi import *
import sys
from main import lissajous


def get_transition_prob():
    pass
    
def GPI_controller(cur_state, ref_state, cur_iter):
    r_t = lissajous(cur_iter + 1)


# ignore this CEC controller the final one is in the main file. 
def CEC_controller(cur_state, ref_state, cur_iter):
    '''
    input:
        cur_state: np.array((x,y,theta))
        ref_state : np.array
        cur_iter : int
    '''

    gamma = 0.99 # Y => [0,1)
    T = 10
    Q = np.ones((2,2))*15
    R = np.eye(2)
    q = 1

    u = SX.sym('u',2,T)
    e_tilde = cur_state - ref_state #error
    p_tilde = e_tilde[:2].reshape((2,1)) #position deviation from ref
    theta_tilde = e_tilde[-1].reshape((1,1)) #orientaion deviation from ref
    f = 0
    count = 0
    for i in range(cur_iter , cur_iter + T):
        gamma_T = gamma ** i
        # z = (p_tilde.T @ Q @ p_tilde)
        summation = (p_tilde.T @ Q @ p_tilde) + q*(1 - np.cos(theta_tilde))**2 + (u[:,count].T @ R @ u[:,count])
        f = f + (gamma_T * summation)

        r_t = traj(i+count+1) #ref(T,3)
        alpha_t = ref_state[2]

        p_tilde_x = p_tilde[0,:] + time_step * cos(theta_tilde + alpha_t) * u[0,count] + (ref_state[0] - r_t[0]) #noise is 0
        p_tilde_y = p_tilde[1,:] + time_step * cos(theta_tilde + alpha_t) * u[0,count] + (ref_state[1] - r_t[1]) #noise is 0
        p_tilde = vertcat(p_tilde_x,p_tilde_y)
        theta_tilde = theta_tilde + time_step * u[1,count] + ref_state[2] - r_t[2]
        ref_state = r_t
        count += 1

    f = f + (p_tilde.T @ Q @ p_tilde)  + q * (1-cos(theta_tilde))**2
    next_state_x = cur_state[0] + time_step * cos(cur_state[2]) * u[0,0]
    next_state_y = cur_state[1] + time_step * cos(cur_state[2]) * u[0,0]

    # g and h constraints.
    g = (next_state_x + 2)**2 + (next_state_y + 2)**2 - 0.3
    h = (next_state_x - 1)**2 + (next_state_y - 2)**2 - 0.3

    # f = np.linalg.norm(x_tilde,axia = 1) + np.sum(gamma_T * summation)

    nlp = {}                 # NLP declaration
    nlp['x']= u[:]           # decision vars
    nlp['f'] = f             # objective
    nlp['g'] = vertcat(g,h)  # constraints

    # Create solver instance
    F = nlpsol('F','ipopt',nlp);
    # bounds
    lbx = [v_min, w_min] * T # lower
    ubx = [v_max, w_max] * T # upper

    # Solve the problem using a guess
    ans = F(lbx = lbx,ubx = ubx,lbg=[0,0])
    #
    print("this:  ", ans)
    print("this:  ", ans['x'])
    x_optimal = ans['x']
    result = np.array(x_optimal).reshape(T,2)[0]
    # sys.exit()
    return result
