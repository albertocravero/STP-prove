
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from RBF_Nets import RBFnet_closedv2, Sinusoidal_net, RBFnet_openv2
from Humming_envs import TT
from math import isnan





# Define the agent
class RBF_agent_open():
    def __init__(self, env, num_centers, final_point, final_time,action_limit, sigma, lr_sigma = 0.01, c = [1,0.1,1], lr=0.01, weight_limit=np.pi):

        self.name = "open"
        
        #self.A_val = RBFnet_open(num_centers=num_centers, lr=lr, c1 = c1, c2= c2, final_point=final_time, sigma=sigma) #!! #declaring the model
        self.A_val = RBFnet_openv2(num_centers=num_centers, lr=lr,action_limit =action_limit, final_time=final_time, sigma=sigma, lr_sigma=lr_sigma) #!!
        self.final_time = final_time
        self.final_point = final_point
        self.weight_limit = weight_limit 
        self.env = env #to get step evaluation
        self.c = c 
        self.int_loss = 0
        self.pos_loss = self.A_val.loss_fn(TT(0),TT(2.))
        self.vel_loss = self.A_val.loss_fn(TT(0),TT(0))
        self.action_loss = []




    def choose_action(self, obs): 

        #action = self.A_val.get_values(obs[2]) #simple action evaluation 
        action = self.A_val.get_values_integral(obs[2]) #integral action evaluation
        self.action_loss.append(action)
        return action

    # Learn function (called during every timestep)
    def learn(self, state):
        self.A_val.optimizer.zero_grad()

        self.action_loss = torch.stack(self.action_loss)-self.A_val.action_limit 
        weight_cost = torch.sum(self.A_val.loss_constraint_fun(abs(self.A_val.net)-self.weight_limit)) #Relu
        action_cost = torch.sum(self.A_val.loss_constraint_fun(self.action_loss)) #Relu
        const_loss = weight_cost + 0.1*action_cost

        print('pos:',self.pos_loss.item(),'vel:',self.vel_loss.item(),'weight',weight_cost.item(),'action',action_cost.item())
        #loss=0.01*(self.pos_loss+0.01*self.vel_loss) + const_loss
        loss=0.01*(self.pos_loss+0.001**self.vel_loss + 0.1* const_loss) #pos and vel: MSE


        loss.backward(retain_graph=True)

        self.A_val.optimizer.step()

        final_time = torch.tensor(self.final_time, dtype=torch.float32)
        t_values = torch.linspace(0,final_time,int(1000*final_time.item())) #to update net
        self.A_val.net = self.A_val.forward(t_values)

        self.action_loss = []
        return loss




class SinusoidalAgent():
    def __init__(self, num_centers, final_point, final_time, switch = 0, lr=0.01, weight_limit=np.pi, c1=1, c2=1):
        self.A_val = Sinusoidal_net(num_centers=num_centers,switch=switch,final_time=final_time, lr=lr, weight_limit=weight_limit)  # Dichiarazione del modello
        self.final_time = final_time
        self.final_point = final_point
        self.weight_limit = weight_limit
        self.c = [c1, c2]
        self.name = "sin"

    def choose_action(self, obs):
        return self.A_val.get_values(torch.tensor([obs[2]], dtype=torch.float32))

    def learn(self, state):
        pos = state[0]
        vel = state[1]
        if len(pos.size()) == 1:
            target = TT([self.final_point])
            targ_vel = TT([0])
        else:
            target = TT(self.final_point)
            targ_vel = TT(0)
        #time = TT(state[2])  # Posizione corrente
        t_values = torch.linspace(0, self.final_time, 1000)  # Per valutare la perdita integrale
        #print(pos)

        self.A_val.optimizer.zero_grad()

        self.dist_loss = self.A_val.loss_fn(pos,target)

        self.vel_loss = self.A_val.loss_fn(targ_vel,vel)
        self.integral_loss = torch.trapz(self.A_val.forward(t_values)**2, t_values)
        loss = self.c[0] * self.dist_loss + self.c[1] * self.integral_loss + self.c[0]* self.vel_loss
        #print(self.A_val.c)
        #print(loss)
        loss.backward(retain_graph=True)
        self.A_val.optimizer.step()

        self.A_val.net = self.A_val.forward(t_values)

        with torch.no_grad():
            if self.A_val.switch:
                #print(self.A_val.weight)
                self.A_val.weight.clamp(-self.weight_limit, self.weight_limit)
            else:
                self.A_val.weights.clamp(-self.weight_limit, self.weight_limit)
        #print(self.A_val.c)


        return loss





# Define the agent
class RBF_agent_closed_simple():
    def __init__(self, env, num_centers, action_limit, final_point, final_time, sigma, lr_sigma = 0.01, c = [1,0.1,1], lr=0.01, weight_limit=np.pi):

        self.name = 'closed'
        self.final_pos = final_point
        #self.A_val = RBFnet_open(num_centers=num_centers, lr=lr, c1 = c1, c2= c2, final_point=final_time, sigma=sigma) #!! #declaring the model
        self.A_val = RBFnet_closedv2(num_centers=num_centers,weight_limit=weight_limit, lr=lr, final_time=final_time, action_limit=action_limit, final_pos=self.final_pos, sigma=sigma, lr_sigma=lr_sigma) #!!
        self.final_time = final_time
        self.final_point = final_point
        self.weight_limit = weight_limit 
        self.env = env #to get step evaluation
        self.c = c 
        self.int_loss = 0
        #self.pos_loss = self.A_val.loss_fn(TT(0),TT(2))
        self.pos_loss = self.A_val.loss_fn(TT(0),TT(2))
        self.vel_loss = self.A_val.loss_fn(TT(0),TT(0))

        self.count = 0
        self.net_range = self.A_val.net_range



    def choose_action(self, obs): #not very useful in this case
        # #return self.A_val.forward(obs[2])
        # if obs[0]>2:
        #     return - self.A_val.get_values(obs[0])
        # else:
            #return self.A_val.get_values(obs[0])
            obs.append(obs[2])
            if obs[2].item() == 0:
                obs[2] = TT(0.) 
            else:
                x_dd = (self.env.states_[-1][1]-self.env.states_[-2][1])/self.env.dt
                obs[2] = x_dd
            #print(obs)
            return self.A_val.get_values(obs)

    # Learn function (called during every timestep)
    def learn(self, state):
        u1 = self.A_val.weights
        self.A_val.optimizer.zero_grad()

  
        #target = torch.tensor(self.final_point, dtype=torch.float32) #final point converted into tensor
        #pos, vel, time = self.env.evaluate_step(state, self.A_val(state[2])) #evaluating position and velocity starting from current state; the function will be evaluated during the opt step
        x_values = torch.linspace(-self.A_val.final_pos,self.A_val.net_range-self.A_val.final_pos,int(1000*self.A_val.net_range)) #to evaluate integral loss


        #self.vel_loss = self.A_val.loss_fn(targ_vel,vel)
        #self.dist_loss = self.A_val.loss_fn(pos, target)
        #self.dist_loss = torch.exp(10*self.A_val.loss_fn(pos,target))

        #self.dist_loss = torch.exp(self.A_val.loss_fn(pos,target)) + 2*torch.exp(10*self.A_val.loss_fn(state[2],time_target))
        #self.dist_loss = self.A_val.loss_fn(pos,target) + torch.exp(2*self.A_val.loss_fn(state[2],time_target))
        #self.dist_loss = self.A_val.loss_fn(pos,target) + self.A_val.loss_fn(state[2],time_target)
        #self.dist_loss =  torch.exp(10*self.A_val.loss_fn(state[2],time_target))


        #self.dist_loss = self.A_val.loss_fn()
        # loss = self.c[0]/5*self.vel_loss + self.c[0]*self.dist_loss + self.c[1]*self.int_loss
        #loss = self.c[0] * (self.dist_loss + self.final_time*self.env.dt*0.1 *self.pos_loss) + 0 * self.c[1] * self.vel_loss + self.c[2] * self.int_loss
        #print(self.pos_loss)
        #loss = torch.exp(5*self.A_val.loss_fn(pos,target)) + self.env.dt / self.final_time * self.pos_loss #+ self.c[1] * self.vel_loss 
        #loss = torch.exp(5*self.A_val.loss_fn(pos,target)) +  self.env.dt / self.final_time * self.pos_loss#/self.count #+ self.c[1] * self.vel_loss 
        #loss =  self.env.dt / self.final_time*self.pos_loss
        #loss = self.A_val.loss_fn(pos,target) +  self.env.dt / self.final_time * self.pos_loss + 0.1* self.vel_loss

        #loss =  torch.exp(self.A_val.loss_fn(target,pos))+ self.pos_loss/1000 
        #loss = self.pos_loss #* 0.001  + self.A_val.loss_fn(target,pos)
        #print('yo',self.vel_loss,self.pos_loss)
        #print(self.pos_loss,torch.exp(self.A_val.loss_fn(target,pos) ))
        #loss = 0.01*(self.vel_loss/10 + self.pos_loss)# +self.A_val.loss_fn(target,pos) #+ self.int_loss
        #loss = 0.001*self.pos_loss
       # print(self.vel_loss*0.1 - self.pos_loss)
        #loss.requires_grad = True
        #print(self.int_loss)
        #print(self.vel_loss, self.int_loss, self.dist_loss)
        #loss = self.c[0] * self.A_val.loss_fn(pos, target) + self.c[1] * (torch.trapz(self.A_val(t_values)**2,t_values)+self.A_val.loss_fn(TT(0),vel))


        action_loss = 0 #torch.matmul((self.A_val.ordinate_prime>self.A_val.action_limit)*1.,self.A_val.ordinate_prime - self.A_val.action_limit)
        const_loss = 0 # torch.matmul((abs(self.A_val.net)>self.weight_limit)*1.,abs(self.A_val.net)-self.weight_limit)

        #const_loss = torch.matmul((self.A_val.ordinate[:]>self.weight_limit)*1.,self.A_val.ordinate-self.weight_limit)
        #const_loss = torch.exp(1+const_loss) + action_loss
        const_loss = 10*const_loss + 0.1*action_loss
        torch.autograd.set_detect_anomaly(True)
        loss = 0.01*(self.vel_loss/10 + self.pos_loss) #+ const_loss
        #print(self.vel_loss,self.pos_loss)
        loss.backward(retain_graph=True)

        self.A_val.optimizer.step()
        #self.A_val.net = self.A_val.forward(t_values)

        #print(pos,loss)
        #clipping the values inside the limits
        if abs(self.A_val.weights.detach().numpy()[:]).any()>self.weight_limit:
            print('oltre il limite')
        # with torch.no_grad():
        #     self.A_val.weights.clamp_(-self.weight_limit, self.weight_limit)
        #     #self.A_val.sigma.clamp_(0,1)

        self.A_val.net = self.A_val.forward(self.A_val.points_for_net)
        #print(loss, pos,vel,u1,self.A_val.sigma)
        #print(loss,pos)
        return loss

        return loss
