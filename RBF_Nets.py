
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import optimize
from scipy.ndimage import  maximum_filter, minimum_filter
#from scipy.ndimage import laplace, sobel

def TT(x): # convert x into tensor
    return torch.tensor(x,dtype=torch.float32)



    
    
# Define the Radial Basis Function network
class RBFnet_openv2(nn.Module):
    def __init__(self, num_centers, action_limit, final_time, lr,  sigma=1.0, lr_sigma = 0.001):
        super(RBFnet_openv2, self).__init__()
        self.num_centers = num_centers+1

        self.sigma = nn.Parameter(torch.tensor(sigma),requires_grad=True)
        self.centers = nn.Parameter(torch.linspace(0, final_time, num_centers),requires_grad=True) 
        #self.weights = nn.Parameter(torch.normal(mean=0, std=0.5, size=(num_centers,)), requires_grad=True)
        self.weights = nn.Parameter(torch.rand(size=(num_centers,),requires_grad=True))
        self.signs = torch.tensor([(-1)**(i+1) for i in range(num_centers)], dtype=torch.float32)

        self.signs[0]= 0
        
        self.loss_fn = nn.MSELoss()
        self.loss_fn1 = nn.L1Loss()
        self.final_time = final_time

    
        self.w1 = nn.Parameter(torch.tensor(1.),requires_grad=True)

        if isinstance(self.sigma, float):
            self.optimizer = optim.Adam([self.weights], lr=lr)
            print('c')
        else:
            #self.optimizer = optim.Adam([{'params': self.weights}, {'params': self.sigma, 'lr': 0.0005}], lr=lr)
            #self.optimizer = optim.Adam([{'params': self.weights},  {'params': self.centers},{'params': self.sigma, 'lr': lr_sigma}], lr=lr)
            self.optimizer = optim.Adam([{'params': self.weights}, {'params': self.sigma, 'lr': lr_sigma}], lr=lr)

            
        self.action_limit = action_limit
        self.net = self.forward(torch.linspace(0,self.final_time,1000*self.final_time))



    def gaussian(self, x):
        if not isinstance(x,(np.ndarray,float)):
            x = x.unsqueeze(1) if x.dim() == 1 else x
        if isinstance(self.sigma, float):
            G = torch.exp(-((x - self.centers)**2) / (2 * self.sigma**2)) * self.signs
        else:
            G = torch.exp(-((x - self.centers)**2) / (2 * self.sigma.clone()**2)) * self.signs  
        return torch.matmul(G, self.weights.clone()) #+ self.w1


    def forward(self, x): 

        G = self.gaussian(x)
        
        points = torch.linspace(0, self.final_time, self.num_centers*6)
        G_=self.gaussian(points).detach().numpy()

        #G_smooth = gaussian_filter(G_, sigma=1)
        G_smooth = G_
        local_max = maximum_filter(G_smooth, size = 3) == G_smooth
        local_min = minimum_filter(G_smooth, size = 3) == G_smooth
        maxima = np.argwhere(local_max)[:,0]
        minima = np.argwhere(local_min)[:,0]
        self.m = np.sort(np.concatenate([maxima,minima]))
        self.maxmin = points[self.m]
        self.true_centers = self.maxmin
        self.ordinate = self.gaussian(TT(self.maxmin))
        
        self.ordinate_prime = self.g_(points)**2
        self.actions = self.calculate_integrals_vel(x,self.maxmin)


        return G
    


    # def forward(self, x): ##NO
    #     G = self.gaussian(x)

    #     pp = []
    #     points = np.linspace(0, self.final_time, self.num_centers*3) 
    #     b = optimize.fsolve(self.gaussian_zeros,points)
    #     b=np.insert(b,0,0)
    #     b=np.append(b,self.final_time)
    #     b = np.unique(np.unique(b.round(decimals=3)))
    #     b = b[(b >= 0) & (b <= self.final_time)]  
    #     #b = b[np.sign(self.first_derivative(b-1e-2))!=np.sign(self.first_derivative(b+1e-2))]

    #     self.true_centers = b.tolist() #altro
    #     self.true_centers.append(0)
    #     self.true_centers.append(self.final_time)
    #     self.maxmin = self.true_centers
    #     self.maxmin=np.unique(np.sort(np.round(self.maxmin,3)))

    #     for i in range(0,len(b)-1):
    #         val = (b[i]+b[i+1])/2
    #         pp.append((val))
    #         self.true_centers.append(val)
    #     pp=np.insert(pp,0,0)
    #     pp=np.append(pp,self.final_time)
    #     self.int_centers = np.unique(np.unique(pp.round(decimals=3)) )

    #     #self.true_centers = np.unique(np.sort(self.true_centers.round(decimals=3)))
    #     self.true_centers=np.unique(np.sort(np.round(self.true_centers,3)))
    #     self.ordinate = self.gaussian(TT(self.true_centers))

    #     #print(self.true_centers)
        
    #     #self.actions = self.calculate_integrals_v(x,self.int_centers)

    #     self.actions = self.calculate_integrals_vel(x,self.maxmin)
    #     #print(self.actions)
    #     self.test = x
    #     # print(self.test)
    #     # print(self.true_centers)
    #     return G
    



    # def get_values(self,x):  #get values for average motion 
    #     i=-1
    #     while ~((x>=self.maxmin[i]) and (x<=self.maxmin[i+1])):
    #         i+=1

    #     action = self.actions[i] 
    #     return action


        
    def get_values(self,x): #get values for inst motion
        action = (self.g_(x)**2)
        #with torch.no_grad():
        # action = action.clamp(0,self.action_limit)

        if torch.isnan(action):
            input('action nan')
        return action





    def calculate_integrals_vel(self, x, int_centers):

        output_vals = []

        for i in range(0,len(int_centers) -1):

            points = x[~((x >= int_centers[i]) ^ (x < int_centers[i+1]))]

            integral= torch.trapz(self.g_(points)**2,points)

            if len(points)==0:
                integral= TT(0)
            else:
                integral = integral/(int_centers[i+1]-int_centers[i]).item()

            output_vals.append(integral)
            

        return output_vals



    def g_(self,x): # first derivative

        if not isinstance(x,(np.ndarray,float)):
            x = x.unsqueeze(1) if x.dim() == 1 else x
        if isinstance(self.sigma, float):
            sigma = self.sigma
        else:
            sigma = self.sigma.clone()

        G = -((x - self.centers) / sigma**2) * torch.exp(-((x - self.centers)**2) / (2 * self.sigma**2)) 
        G*= self.signs
        return torch.matmul(G, self.weights) #+ self.w1


    def first_derivative(self, x): 
        weights = self.weights.detach().numpy()
        centers = self.centers.detach().numpy()
        if isinstance(self.sigma, float):
            sigma = self.sigma
        else:
            sigma = self.sigma.item()

        G = -((x[:, None] - centers) / sigma**2) * np.exp(-((x[:, None] - centers)**2) / (2 * sigma**2))
        G *= self.signs.detach().numpy()

        return np.dot(G, weights)





    def plot(self):

        tol = 1e-3
        t = torch.linspace(0, self.final_time, 1000)
        A = self.gaussian(t)
        pp = []
        points = np.linspace(0, self.final_time, self.num_centers*3) 
        b = optimize.fsolve(self.first_derivative, points)

        b=np.insert(b,0,0)
        b=np.append(b,self.final_time)
        b = np.unique(np.unique(b.round(decimals=4)))
        b = b[(b >= 0) & (b <= self.final_time)]  
        b = b[np.sign(self.first_derivative(b-1e-2))!=np.sign(self.first_derivative(b+1e-2))]

        
        self.forward(t)
        c = optimize.fsolve(self.second_derivative, points)
        c = np.sort(c)
        c = c[(c >= 0) & (c <= self.final_time)]         
        plt.plot(t.detach().numpy(), A.detach().numpy())
        # for i in self.centers.detach().numpy():
        #     plt.axvline(i, color='k', linestyle='-')

        for i in range(0,len(b)-1):
            pp.append((b[i]+b[i+1])/2)
        pp=np.insert(pp,0,0)
        pp=np.append(pp,self.final_time)
        pp = np.unique(np.unique(pp.round(decimals=4)))

        azioni = []
        for i in t:
            azioni.append(self.get_values(i).item())
        plt.plot(t,azioni)
        for i in pp: 
            plt.plot(i,self.gaussian(i).item(), marker = '^')
            plt.axvline(i, color='r', linestyle='--')

        for i in b:
            plt.plot(i,self.gaussian(i).item(), marker = 'o')

        
            #plt.axvline(i, color='r', linestyle='--')
        # for i in b:
        #     plt.plot(i,self.forward(i).item(), marker = 's')

            
        
        plt.xlabel('Time')
        plt.ylabel('A(t)')
        plt.show()



















class Sinusoidal_net(nn.Module):
    def __init__(self, num_centers, final_time, switch = 0, lr=0.01, weight_limit=np.pi):
        super(Sinusoidal_net, self).__init__()
        self.num_centers = num_centers
        self.final_time = final_time
        self.switch = switch
        self.c = final_time / num_centers

        #self.c = nn.Parameter(torch.normal(mean=0, std=1, size=(1,)))
#        self.num_centers = int(torch.round(abs(final_time/self.c)).item())
        if switch: #switch for 
            self.c = nn.Parameter(torch.normal(mean=self.c, std=1, size=(1,)))
            self.weight = nn.Parameter(torch.normal(mean=0, std=1, size=(1,)))
            self.weights = self.weight * torch.ones(num_centers)
            print(self.weights)

        else:
            self.weights = nn.Parameter(torch.normal(mean=0, std=1, size=(num_centers,)))
        #self.num_centers = num_centers
        self.c = nn.Parameter(torch.normal(mean=self.c, std=1, size=(1,)))

        #self.weights = nn.Parameter(torch.normal(mean=0, std=1, size=(num_centers,)))
        #self.c = final_time / num_centers
        #self.optimizer = optim.Adam([self.weights,self.c], lr=lr)
        if switch:
            self.optimizer = optim.Adam([self.weight,self.c], lr=lr)
        else:
            self.optimizer = optim.Adam([{'params': self.weights}, {'params': self.c, 'lr':0.01}],lr = lr)

        self.weight_limit = weight_limit
        self.discretize = 1000  # Number of points to discretize the interval
        self.loss_fn = nn.MSELoss()
        self.loss_fn_L1 = nn.L1Loss()

        self.amplitude = []
        self.times = torch.linspace(0,self.final_time,self.discretize)
        self.net=self.forward(self.times)
        self.w = np.pi / self.c
        self.last = -1
        #print(self.c,self.weights)


    def forward(self, t):
        w = np.pi / self.c
        sin_ = torch.sin(w * t)
        A = torch.zeros_like(sin_)

        if self.switch:
            self.weights = self.weight * torch.ones(self.num_centers)

       #print(t)
        period_length = self.c
        half_period_indices = [i for i in range(1, len(t)) if int(t[i] // period_length) != int(t[i-1] // period_length)]
        #print(self.c)
        current_index = 0
        for i in range(len(half_period_indices)):
            start_idx = half_period_indices[i-1] if i > 0 else 0
            end_idx = half_period_indices[i]
           # print(current_index, self.num_centers, self.)
            #print(len(self.weights))
            A[start_idx:end_idx] =  self.weights[current_index % self.num_centers] * sin_[start_idx:end_idx]
             # Change the weight every full period (two half periods)
            if i % 2 == 1:  # Cambia il peso ogni periodo completo (due semiperiodi)
                current_index += 1
            #current_index += 1
        
        # last segment:
        #print(current_index % self.num_centers,sin_[half_period_indices[-1]:])

        A[half_period_indices[-1]:] = self.weights[current_index % self.num_centers] * sin_[half_period_indices[-1]:]
        self.centers = [self.times[i] for i in half_period_indices]
        if self.centers[-1]!= self.times[-1]:
            self.centers.append(TT(self.times[-1]))
        
        #self.half_period_indices = half_period_indices
        return A
    


    def get_values(self,t):
        i=0
        #print(i,t, self.weights[i],self.centers)

        while t>self.centers[i] and not self.switch:
            i+=1
        
        #print(self.weight,self.c)
        #return self.weights[i].clone()*torch.sin(self.w * TT(t).clone())
        if self.switch:
            return abs(self.weight*np.pi/self.c)/(4/0.05)
            #return abs(self.weights[i]*2*self.c/np.pi)
        else:
            if i>=self.num_centers:
                i= self.num_centers-1
                print(t)
            return abs(self.weights[i].clone()*np.pi/self.c)/(4/0.05)
        




    def plot(self):
        t = torch.linspace(0, self.final_time, self.discretize)
        A = self.forward(t)
        
        plt.plot(t.detach().numpy(), A.detach().numpy())
        # print(self.get_values(1))
        plt.xlabel('Time')
        plt.ylabel('A(t)')
        plt.show()



# # Define the Radial Basis Function network
# class RBFnet_closedv2(nn.Module):
#     def __init__(self, num_centers, final_pos, final_time, lr, action_limit,weight_limit, sigma=1.0, lr_sigma = 0.001):
#         super(RBFnet_closedv2, self).__init__()
#         self.num_centers = num_centers+1

#         self.final_pos = final_pos
#         self.net_range = 1.5*final_pos
#         self.sigma = nn.Parameter(torch.tensor(sigma),requires_grad=True)

#         #self.centers = nn.Parameter(torch.linspace(-self.net_range, self.net_range, num_centers),requires_grad=True) 
#         #self.centers = nn.Parameter(torch.linspace(0, self.net_range, num_centers),requires_grad=True) 
#         self.centers = nn.Parameter(torch.linspace(-self.final_pos, self.net_range-self.final_pos, num_centers),requires_grad=True) 

#         #self.weights = nn.Parameter(torch.normal(mean=0, std=0.5, size=(num_centers,)), requires_grad=True)
#         self.weights = nn.Parameter(torch.rand(size=(num_centers,)), requires_grad=True)#*weight_limit

#         self.signs = torch.tensor([(-1)**(i+1) for i in range(num_centers)], dtype=torch.float32)
#         self.signs[0]= 0
#         self.loss_fn = nn.MSELoss()
#         self.loss_fn1 = nn.L1Loss()
#         self.final_time = final_time
#         self.w1 = nn.Parameter(torch.tensor(0.),requires_grad=True)
#         self.action_limit = action_limit
#         #print(self.sigma)
#         if isinstance(self.sigma, float):
#             self.optimizer = optim.Adam([self.weights], lr=lr)
#             print('c')
#         else:
#             #self.optimizer = optim.Adam([{'params': self.weights}, {'params': self.sigma, 'lr': 0.0005}], lr=lr)
#             self.optimizer = optim.Adam([{'params': self.weights}, {'params': self.w1}, {'params': self.centers},  {'params': self.sigma, 'lr': lr_sigma}], lr=lr)

#             #self.optimizer = optim.SGD([{'params': self.weights}, {'params': self.sigma, 'lr': 0.001}], lr=lr)

#             print('learning')

#         self.net = self.forward(torch.linspace(-self.final_pos,self.net_range-self.final_pos,round(1000*self.net_range)))

#         self.net_= self.forward(torch.linspace(-self.final_pos,self.net_range-self.final_pos,round(1000*self.net_range)))
        


#     def gaussian(self, x):
#         if not isinstance(x,(np.ndarray,float)):
#             x = x.unsqueeze(1) if x.dim() == 1 else x
#         if isinstance(self.sigma, float):
#             G = torch.exp(-((x - self.centers)**2) / (2 * self.sigma**2)) * self.signs
#         else:
#             G = torch.exp(-((x - self.centers)**2) / (2 * self.sigma.clone()**2)) * self.signs  
#         return torch.matmul(G, self.weights.clone()) + self.w1.clone()
     



    
#     def forward(self, x): 

#         G = self.gaussian(x)
        
#         points = torch.linspace(-self.final_pos, self.net_range-self.final_pos, self.num_centers*6)
#         G_=self.gaussian(points).detach().numpy()

#         #G_smooth = gaussian_filter(G_, sigma=1)
#         G_smooth = G_
#         local_max = maximum_filter(G_smooth, size = 3) == G_smooth
#         local_min = minimum_filter(G_smooth, size = 3) == G_smooth
#         maxima = np.argwhere(local_max)[:,0]
#         minima = np.argwhere(local_min)[:,0]
#         self.m = np.sort(np.concatenate([maxima,minima]))
#         self.maxmin = points[self.m]
#         self.true_centers = self.maxmin
#         self.ordinate = self.gaussian(TT(self.maxmin))
        
#         self.ordinate_prime = self.g_(points)**2
#         self.actions = self.calculate_integrals_vel(x,self.maxmin)
#         #self.action_x = [self.get_values(i).item() for i in x]

        

#         if torch.isnan(G).any():
#             input('G nan')
            

#         return G
    




#     def g_(self,x): # first derivative

#         if not isinstance(x,(np.ndarray,float)):
#             x = x.unsqueeze(1) if x.dim() == 1 else x
#         if isinstance(self.sigma, float):
#             sigma = self.sigma
#         else:
#             sigma = self.sigma.clone()

#         G = -((x - self.centers) / sigma**2) * torch.exp(-((x - self.centers)**2) / (2 * self.sigma**2)) 
#         G*= self.signs
#         return torch.matmul(G, self.weights)


#     # def forward(self, x): 
#     #     G = self.gaussian(x)

#     #     pp = []
#     #     points = np.linspace(0, self.net_range, self.num_centers*3) 
#     #     b = optimize.fsolve(self.first_derivative, points)

#     #     b=np.insert(b,0,0)
#     #     b=np.append(b,self.net_range)
#     #     b = np.unique(np.unique(b.round(decimals=3)))
#     #     b = b[(b >= 0) & (b <= self.net_range)]  

#     #     b = b[np.sign(self.first_derivative(b-1e-2))!=np.sign(self.first_derivative(b+1e-2))]

#     #     self.true_centers = b.tolist()
#     #     self.true_centers.append(0)
#     #     self.true_centers.append(self.net_range)
#     #     self.maxmin = self.true_centers
#     #     self.maxmin=np.unique(np.sort(np.round(self.maxmin,3)))

#     #     for i in range(0,len(b)-1):
#     #         val = (b[i]+b[i+1])/2
#     #         pp.append((val))
#     #         self.true_centers.append(val)
#     #     pp=np.insert(pp,0,0)
#     #     pp=np.append(pp,self.net_range)
#     #     self.int_centers = np.unique(np.unique(pp.round(decimals=3)) )

#     #     #self.true_centers = np.unique(np.sort(self.true_centers.round(decimals=3)))
#     #     self.true_centers=np.unique(np.sort(np.round(self.true_centers,3)))
#     #     self.ordinate = self.gaussian(TT(self.true_centers))

#     #     #print(self.true_centers)
        
        
#     #     #self.actions = self.calculate_integrals_v(x,self.int_centers)

#     #     self.actions = self.calculate_integrals_vel(x,self.maxmin)
#     #     self.action_x = [self.get_values(i).item() for i in x]
#     #     #print(self.actions)
#     #     self.test = x
#     #     # print(self.test)
#     #     # print(self.true_centers)
#     #     return G
    



#     # def get_values(self,x):
#     #     i=-1
#     #     vel = x[1]
#     #     x = x[0]
#     #     #while ~((x>=self.int_centers[i]) and (x<=self.int_centers[i+1])):
#     #     #while ~((x>=self.true_centers[i]) and (x<=self.true_centers[i+1])):
#     #     #print(x)
#     #     if (torch.sort(self.maxmin)[0]!= self.maxmin).any():
#     #         print('diversi')
#     #    # print(self.maxmin[-1])
#     #     while i<len(self.maxmin)-1 and ~((x>=self.maxmin[i]) and (x<=self.maxmin[i+1])):
#     #         #print(x)
#     #         i+=1
#     #     if i<len(self.maxmin)-1:
#     #         #action = self.actions[i]*0.5*1e-2/np.sqrt(10)
#     #         action = self.actions[i]
#     #     else:
#     #         if x<=-self.final_pos:
#     #             #action = self.actions[0]*0.5*1e-2/np.sqrt(10)
#     #             action = self.actions[0]
#     #         else:
#     #             #action = self.actions[-1]*0.5*1e-2/np.sqrt(10)
#     #             action = self.actions[-1]
#     #     action = (torch.sqrt(action)*vel)**2 #* 2*0.3/np.sqrt(10)
#     #     #print(action)
#     #     #action = torch.clamp(action,-self.action_limit,self.action_limit)
#     #     #print(action)
#     #     #return self.actions[i], self.maxmin[i+1]
#     #     return action


    
#     def get_values(self,x):
#         action = (self.g_(x[0]))*x[1]
#         #with torch.no_grad():
#         # action = action.clamp(0,self.action_limit)

#         if torch.isnan(action):
#             print('action nan')
#         return action**2
        


#     def calculate_integrals_vel(self, x, int_centers):

#         output_vals = []

#         for i in range(0,len(int_centers) -1):

#             points = x[~((x >= int_centers[i]) ^ (x < int_centers[i+1]))]

#             integral= torch.trapz(self.g_(points)**2,points)
            

#             if len(points)==0:
#                 integral= TT(0)
#             else:
#                 #integral = integral/len(points)
#                 integral = integral/(int_centers[i+1]-int_centers[i]).item()

#             output_vals.append(abs(integral))
            

#         return output_vals



    
#     def first_derivative(self, x):
#         weights = self.weights.detach().numpy()
#         centers = self.centers.detach().numpy()
#         if isinstance(self.sigma, float):
#             sigma = self.sigma
#         else:
#             sigma = self.sigma.item()

#         G = -((x[:, None] - centers) / sigma**2) * np.exp(-((x[:, None] - centers)**2) / (2 * sigma**2))
#         G *= self.signs.detach().numpy()

#         return np.dot(G, weights)





#     def plot(self):

#         tol = 1e-3
#         t = torch.linspace(0, self.final_time, 1000)
#         A = self.gaussian(t)
#         pp = []
#         points = np.linspace(0, self.final_time, self.num_centers*3) 
#         b = optimize.fsolve(self.first_derivative, points)

#         b=np.insert(b,0,0)
#         b=np.append(b,self.final_time)
#         b = np.unique(np.unique(b.round(decimals=4)))
#         b = b[(b >= 0) & (b <= self.final_time)]  
#         b = b[np.sign(self.first_derivative(b-1e-2))!=np.sign(self.first_derivative(b+1e-2))]

        
#         self.forward(t)
#         # if b[0]> tol:
#         #     np.insert(b,0,TT(0))
#         # if self.final_time-b[-1]>tol:
#         #     np.append(b,TT(self.final_time))
#         c = optimize.fsolve(self.second_derivative, points)
#         c = np.sort(c)
#         c = c[(c >= 0) & (c <= self.final_time)]          #print(self.second_derivative(t.detach().numpy()))
#         plt.plot(t.detach().numpy(), A.detach().numpy())
#         # for i in self.centers.detach().numpy():
#         #     plt.axvline(i, color='k', linestyle='-')

#         for i in range(0,len(b)-1):
#             pp.append((b[i]+b[i+1])/2)
#         pp=np.insert(pp,0,0)
#         pp=np.append(pp,self.final_time)
#         pp = np.unique(np.unique(pp.round(decimals=4)))

#         azioni = []
#         for i in t:
#             azioni.append(self.get_values(i).item())
#         plt.plot(t,azioni)
#         for i in pp: 
#             plt.plot(i,self.gaussian(i).item(), marker = '^')
#             plt.axvline(i, color='r', linestyle='--')

#         for i in b:
#             plt.plot(i,self.gaussian(i).item(), marker = 'o')

#         plt.xlabel('Time')
#         plt.ylabel('A(t)')
#         plt.show()

class RBFnet_closedv2(nn.Module):
    def __init__(self, num_centers, final_pos, final_time, lr, action_limit, weight_limit, sigma=1.0, lr_sigma=0.001):
        super(RBFnet_closedv2, self).__init__()
        self.num_centers = num_centers + 1

        self.final_pos = final_pos
        self.net_range = 1.5 * final_pos
        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=True)

        self.centers_x = nn.Parameter(torch.linspace(-self.final_pos, self.net_range - self.final_pos, num_centers), requires_grad=True)
        self.centers_x_dot = nn.Parameter(torch.linspace(-5, 5, num_centers), requires_grad=True)

        self.weights = nn.Parameter(torch.rand(size=(num_centers,)), requires_grad=True)

        self.signs = torch.tensor([(-1) ** (i + 1) for i in range(num_centers)], dtype=torch.float32)
        self.signs[0] = 0
        self.loss_fn = nn.MSELoss()
        self.final_time = final_time
        self.w1 = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.action_limit = action_limit

        self.optimizer = optim.Adam([
            {'params': self.weights},
            {'params': self.w1},
            {'params': self.centers_x},
            {'params': self.centers_x_dot},
            {'params': self.sigma, 'lr': lr_sigma}
        ], lr=lr)

        x_state = [torch.linspace(-self.final_pos, self.net_range - self.final_pos, round(1000 * self.net_range)),torch.linspace(-5, 5, round(1000 * self.net_range))]
        #self.net = self.forward(torch.linspace(-self.final_pos, self.net_range - self.final_pos, round(1000 * self.net_range)))
        self.net = self.forward(x_state)

    def gaussian(self, x):

        x_val = x[0].unsqueeze(1) if x[0].dim() == 1 else x[0]
        x_dot_val = x[1].unsqueeze(1) if x[1].dim() == 1 else x[1]
        r_2 = (x_val - self.centers_x) ** 2 + (x_dot_val - self.centers_x_dot) ** 2

        G = torch.exp(- r_2/ (2 * self.sigma ** 2)) * self.signs
        
        return torch.matmul(G, self.weights.clone()) + self.w1.clone()

    def forward(self, x):
        G = self.gaussian(x)
        #points = torch.linspace(-self.final_pos, self.net_range - self.final_pos, self.num_centers * 6)
        points = [torch.linspace(-self.final_pos, self.net_range - self.final_pos, round(1000 * self.net_range)),torch.linspace(-5, 5, round(1000 * self.net_range))]

        # G_ = self.gaussian(torch.stack([points, points], dim=1)).detach().numpy()

        # G_smooth = G_
        # local_max = maximum_filter(G_smooth, size=3) == G_smooth
        # local_min = minimum_filter(G_smooth, size=3) == G_smooth
        # maxima = np.argwhere(local_max)[:, 0]
        # minima = np.argwhere(local_min)[:, 0]
        # self.m = np.sort(np.concatenate([maxima, minima]))
        # self.maxmin = points[self.m]
        # self.true_centers = self.maxmin
        # self.ordinate = self.gaussian(torch.stack([self.maxmin, self.maxmin], dim=1))


        #self.ordinate_prime = self.g_(points)[0] + self.g_(points)
        #self.actions = self.calculate_integrals_vel(x, self.maxmin)

        return G

    def g_(self, x):

        pos = x[0].unsqueeze(1) if x[0].dim() == 1 else x[0]
        vel = x[1].unsqueeze(1) if x[1].dim() == 1 else x[1]
        r_x = pos - self.centers_x
        r_xd = vel - self.centers_x_dot
        r2 = r_x**2 + r_xd**2
        G = torch.exp(- r2 / (2 * self.sigma ** 2)) * self.signs
        dGdx = -(r_x / self.sigma ** 2) * G * self.signs
        dGdx_dot = -(r_xd / self.sigma ** 2) * G * self.signs
        #G = G_x * G_x_dot * self.signs
        
        return torch.matmul(dGdx, self.weights) , torch.matmul(dGdx_dot, self.weights)

    def get_values(self, x):
        x_dot, x_dd = x[1], x[2]
        phi, phi_dot = self.g_([x[0],x[1]])
        action = phi * x_dot + phi_dot * x_dd
        print(x_dot,  x_dd)
        return action ** 2



    def calculate_integrals_vel(self, x, int_centers):
        output_vals = []

        for i in range(0, len(int_centers) - 1):
            points = x[~((x[:, 0] >= int_centers[i]) ^ (x[:, 0] < int_centers[i + 1]))]

            integral = torch.trapz(self.g_(points) ** 2, points[:, 0])

            if len(points) == 0:
                integral = torch.tensor(0)
            else:
                integral = integral / (int_centers[i + 1] - int_centers[i]).item()

            output_vals.append(abs(integral))

        return output_vals

    def plot(self):
        tol = 1e-3
        t = torch.linspace(0, self.final_time, 1000)
        A = self.gaussian(t)
        pp = []
        points = np.linspace(0, self.final_time, self.num_centers * 3)
        b = optimize.fsolve(self.first_derivative, points)

        b = np.insert(b, 0, 0)
        b = np.append(b, self.final_time)
        b = np.unique(np.unique(b.round(decimals=4)))
        b = b[(b >= 0) & (b <= self.final_time)]
        b = b[np.sign(self.first_derivative(b - 1e-2)) != np.sign(self.first_derivative(b + 1e-2))]

        self.forward(t)
        c = optimize.fsolve(self.second_derivative, points)
        c = np.sort(c)
        c = c[(c >= 0) & (c <= self.final_time)]
        plt.plot(t.detach().numpy(), A.detach().numpy())

        for i in range(0, len(b) - 1):
            pp.append((b[i] + b[i + 1]) / 2)
        pp = np.insert(pp, 0, 0)
        pp = np.append(pp, self.final_time)
        pp = np.unique(np.unique(pp.round(decimals=4)))

        actions = []
        for i in t:
            actions.append(self.get_values(i).item())
        plt.plot(t, actions)
        for i in pp:
            plt.plot(i, self.gaussian(i).item(), marker='^')
            plt.axvline(i, color='r', linestyle='--')

        for i in b:
            plt.plot(i, self.gaussian(i).item(), marker='o')

        plt.xlabel('Time')
        plt.ylabel('A(t)')
        plt.show()