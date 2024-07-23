
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


def TT(x):
    return torch.tensor(x,dtype=torch.float32)




# Define the HummingBird environment. Similar to custom gym envs
class HB_simplified_open(gym.Env):
    def __init__(self, g=9.81, m=0.005, L=2, final_time = 2, S=7.7e-2, dt=0.05/20, R=5.6e-2,c_A = 0.05):

        self.final_time_estimate = 0.9 + 0.2
        self.mean_freq = 50
        self.g = g
        self.m = m
        self.L = L
        self.S = S
        self.R = R
        self.dt = dt
        self.c_A = c_A
        self.final_time = final_time
        self.t = torch.tensor(0, dtype=torch.float32)
        #self.states_ = [torch.tensor(0, dtype=torch.float32), torch.tensor(0, dtype=torch.float32), self.t]
        self.states_ = [torch.tensor(-self.L, dtype=torch.float32), torch.tensor(0, dtype=torch.float32), self.t]


    def reset(self):
        #print('reset')
        #self.states_ = [[TT(0), TT(0), TT(0)]]
        self.states_ = [[TT(-self.L), TT(0), TT(0)]]

        return self.states_[0]

    def drag(self, x_dot):
        if x_dot == 0:
            x_dot = 1e-5
        viscosity = 1.460e-5

        Re_body = np.abs(x_dot) * 0.1 / viscosity
        Re_wing = np.abs(x_dot) * (self.S / self.R) / viscosity
        c_d_wing =  7 / np.sqrt(Re_wing)
        c_d_body = 24/Re_body + 0.4 + 6/(1+np.sqrt(Re_body))

        c_d = np.sign(x_dot) * ( 2 * c_d_wing + c_d_body)
 
        return torch.tensor(0.5 * c_d * 1.225 * self.S * x_dot**2, dtype=torch.float32)


    def d_drag_dx_dot(self, x_dot):
        if x_dot == 0:
            x_dot = 1e-5
        viscosity = 1.460e-5

        Re_body = np.abs(x_dot) * 0.1 / viscosity
        Re_wing = np.abs(x_dot) * (self.S / self.R) / viscosity
        c_d_wing =  7 / np.sqrt(Re_wing)
        c_d_body = 24/Re_body + 0.4 + 6/(1 + np.sqrt(Re_body))

        dc_d_wing_dx_dot = -7 * (self.S / self.R) / (2 * viscosity * (Re_wing ** 1.5))
        dc_d_body_dx_dot = -24 * 0.1 / (viscosity * (Re_body ** 2)) - 6 * 0.1 / (2 * viscosity * (1 + np.sqrt(Re_body))**2 * Re_body ** 1.5)

        dc_d_dx_dot = np.sign(x_dot) * (2 * dc_d_wing_dx_dot + dc_d_body_dx_dot)
        c_d = np.sign(x_dot) * (2 * c_d_wing + c_d_body)

        # alpha_rad = (A_alphal_rad)*np.cos(2*np.pi*fl*t)
        # [1.7104774792288275]
        # c_list*np.sin(2*(alpha_rad))
        
 
        return torch.tensor(0.5 * 1.225 * self.S * (dc_d_dx_dot * x_dot**2 + 2 * c_d * x_dot), dtype=torch.float32)
    


    def d_Fp_du(self,u):
        return 1

    def linearize(self,state):
        x_s = state[0]
        x_dot_s = state[1]
        x=0
        x_dot = 0
        #u = action
        u = 1
        u_s = 1

        F_g = self.m * self.g
        F_p = 1
        #F_d = self.drag(x_dot_s.item())


        x_dot = state[1] + (F_p - F_g - self.drag(x_dot_s.item())) + self.d_drag_dx_dot(x_dot_s)*(-x_dot_s+x_dot) + self.d_Fp_du(u_s)*(u-u_s)
        x_dot = x_dot * self.dt / self.m 
        x = state[0] + x_dot * self.dt



        pass



    def hummingbird_dynamics(self, state, action):

        F_g = self.m * self.g
        F_p = self.c_A * action
        F_d = self.drag(state[1].item())
        x_dot = state[1] + (F_p - F_g - F_d) * self.dt / self.m
        x = state[0] + x_dot * self.dt
        #print(F_p,F_d,x_dot)


        return x, x_dot, state[2]+self.dt

    def step(self, action): #to let the system evolve

        state = self.states_[-1]
        x, x_dot, t = self.hummingbird_dynamics(state, action) 
        cost = 0 #not used right now
        #state_ = [x, x_dot] 
        
        self.states_.append([x, x_dot, t])

        done = t>=self.final_time #or x<0 #truncation conditions: either the bird is 'close' to L or the velocity is negative
        

        return [x, x_dot, t], -cost, done

    def evaluate_step(self, state, action): #to evaluate evolution of the system with different A(state) (used during optimization)
        x, x_dot, t = self.hummingbird_dynamics(state, action)

        return [x, x_dot, t]
    


# Define the HummingBird environment. Similar to custom gym envs
class HB_simplified_closed(gym.Env):
    def __init__(self, g=9.81, m=0.005, L=2, final_time = 2, S=7.7e-2, dt=0.05/40, R=5.6e-2,c_A = 0.05):

        self.final_time_estimate = 0.9 + 0.2
        self.mean_freq = 50
        self.g = g
        self.m = m
        self.L = L
        self.S = S
        self.R = R
        self.dt = dt
        self.c_A = c_A
        self.check = [0,0]
        self.final_time = final_time
        self.t = torch.tensor(0, dtype=torch.float32)
        self.dt = dt
        self.states_ = [torch.tensor(0, dtype=torch.float32), torch.tensor(0, dtype=torch.float32), self.t]
        self.drags = []

    def reset(self):
        self.check = [0,0]
        #print('reset')
        self.states_ = [[TT(0), TT(0), TT(0)]]
        self.drags = []
        return self.states_[0]

    def drag(self, x_dot):
        if x_dot == 0:
            x_dot = 1e-5

        viscosity = 1.460e-5

        Re_body = np.abs(x_dot) * 0.1 / viscosity
        Re_wing = np.abs(x_dot) * (self.S / self.R) / viscosity
        c_d_wing =  7 / np.sqrt(Re_wing)
        c_d_body = 24/Re_body + 0.4 + 6/(1+np.sqrt(Re_body))

        c_d = np.sign( x_dot ) * ( 2 * c_d_wing + c_d_body)
 
        return torch.tensor(0.5 * c_d * 1.225 * self.S * x_dot**2, dtype=torch.float32)

    def hummingbird_dynamics(self, state, action):

        F_g = self.m * self.g
        F_p = self.c_A * (action) #**2
        F_d = self.drag(state[1].item())

        x_dot = state[1] + (F_p - F_g - F_d) * self.dt / self.m
        x = state[0] + x_dot * self.dt

        self.drags.append(F_d.item())

        return x, x_dot, state[2]+self.dt




    def step(self, action): #to let the system evolve

        state = self.states_[-1]
        x, x_dot, t = self.hummingbird_dynamics(state, action) 

        cost = abs(action) #not used right now
        #state_ = [x, x_dot] 
        
        self.states_.append([x, x_dot, t])
        #done = t>=self.final_time #or x<0 or x_dot<0 #truncation conditions: either the bird is 'close' to L or the velocity is negative
        done = t>=self.final_time #truncation conditions

        return [x, x_dot, t], -cost, done

    def evaluate_step(self, state, action): #to evaluate evolution of the system with different A(state) (used during optimization)
        x, x_dot, t = self.hummingbird_dynamics(state, action)
        done = t>=self.final_time

        return [x, x_dot, t], done
    




# Define the HummingBird environment. Similar to custom gym envs
class HB_simp_delta_closed_(gym.Env):
    def __init__(self, g=9.81, m=0.005, L=2, final_time = 2, S=7.7e-2, dt=0.05/40, R=5.6e-2,c_A = 0.05):

        self.final_time_estimate = 0.9 + 0.2
        self.mean_freq = 50
        self.g = g
        self.m = m
        self.L = L
        self.S = S
        self.R = R
        self.dt = dt
        self.c_A = c_A
        self.check = [0,0]
        self.final_time = final_time
        self.t = torch.tensor(0, dtype=torch.float32)
        self.dt = dt
       # self.states_ = [torch.tensor(-self.L, dtype=torch.float32), torch.tensor(0, dtype=torch.float32), self.t]

        self.drags = []

    def reset(self):
        #self.states_ = [[TT(-self.L), TT(0.01), TT(0)]]
        self.states_ = [[TT(-1.5), TT(0.1), TT(0)]]

        self.drags = []
        return self.states_[0]
    



    def reset_validation(self,range):
        initial_point = (range[0] - range[1]) * torch.rand(1) + range[1]
        self.states_ = [[initial_point, TT(0.), TT(0)]]

        self.drags = []
        return self.states_[0]


    def drag(self, x_dot):
        if x_dot == 0:
            x_dot = 1e-5

        viscosity = 1.460e-5

        Re_body = np.abs(x_dot) * 0.1 / viscosity
        Re_wing = np.abs(x_dot) * (self.S / self.R) / viscosity
        c_d_wing =  7 / np.sqrt(Re_wing)
        c_d_body = 24/Re_body + 0.4 + 6/(1+np.sqrt(Re_body))

        c_d = np.sign( x_dot ) * ( 2 * c_d_wing + c_d_body)
 
        return torch.tensor(0.5 * c_d * 1.225 * self.S * x_dot**2, dtype=torch.float32)

    def hummingbird_dynamics(self, state, action):

        F_g = self.m * self.g
        F_p = self.c_A * (action) #**2
        F_d = self.drag(state[1].item())


        if torch.isinf(state[1]) or torch.isnan(1/state[1]):
            print('action nan')
        x_dot = state[1] + (F_p - F_g - F_d) * self.dt / self.m
        x = state[0] + x_dot * self.dt
        
        if torch.isnan(x_dot):
            input('xdot nan')

        elif torch.isinf(x_dot):
            input('xdot inf')

        
        if torch.isnan(x):
            input('xnan')

        elif torch.isinf(x):
            input('x inf')
        x_dd = (x_dot - state[1])/self.dt
        #print('state',x_dot, x_dd)
        self.drags.append((x_dot-state[1]).item()/self.dt)

        return x, x_dot, state[2]+self.dt 




    def step(self, action): #to let the system evolve

        state = self.states_[-1]
        x, x_dot, t = self.hummingbird_dynamics(state, action) 

        cost = abs(action) #not used right now
        #state_ = [x, x_dot] 
        
        self.states_.append([x, x_dot, t])
        #done = t>=self.final_time #or x<0 or x_dot<0 #truncation conditions: either the bird is 'close' to L or the velocity is negative
        done = t>=self.final_time  #truncation condition

        return [x, x_dot, t], -cost, done

    def evaluate_step(self, state, action): #to evaluate evolution of the system with different A(state) (used during optimization)
        x, x_dot, t = self.hummingbird_dynamics(state, action)
        done = t>=self.final_time

        return [x, x_dot, t], done
    
