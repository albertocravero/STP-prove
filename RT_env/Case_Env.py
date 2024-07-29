# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:46:59 2023

@author: pedro
"""

#%% Import packages


import numpy as np
from gym import spaces

from simpleFlappingAero import simpleFlappingAeroEnv
from simpleFlappingMotion import simpleFlappingMotionEnv
from simpleFlappingProperties import simpleFlappingPropertiesEnv


from scipy.stats import multivariate_normal as mn
from scipy import interpolate


class Pigeon():
    
    def __init__(self, 
                 # --- Simulation paramters --- #
                 T: float = 5,                     # Simulation time [s] (not-needed, provided as input in Case_Run)
                 dt: float = 0.01,                  # Simulation time-step [s] (not-needed provided as input in Case_Run)
                 t0: float = 0.0,
                 # --- Number of env I/O --- #
                 Nstate:   int = 2,
                 Nactions: int = 1,
                 # --- Initial state --- #
                 s0: np.array = np.array([4, 0]),   # Initial state [position, velocity]
                 # --- Model (additional) parameters --- #
                 mass: float =0.003,
                 g: float =9.81,
                 rho: float =1.2,
                 CL_body: float =1,
                 S_body: float =0.0005,
                 # --- Action bounds --- #
                 bound_action: list = ((88,), (30,)), # ((Aphi_min,Aphi_max),(beta_min,beta_max)) #!td: not used
                 action_min = np.array([50]),
                 action_max = np.array([88]),
                 # --- State bounds --- #
                 bound_state: list =((-18,18),(-10,10)), #((-18,18),(-18,18),(-10,10),(-10,10)),
                 # --- Closure terms --- #
                 c_list_true: list = [1.7104774792288275, 0.043035892324197876, 1.594219883355539],       # True values of the real system
                 bounds_p: tuple = ((0,2),(0,2),(0,2),(0,2)), # Tuple with the bounds for each closure parameter
                 # --- Control parametrization --- #
                 control_mode: str = 'linear', # Not-needed, provided as input in Case_Run
                 s_ref: list = [0, 5],
                 # --- Visualisation --- #
                 plot: bool = False,
                 # --- !td To be seen if let --- #
                 reward_type='gaussian', windType='constant', windVelocity=0
    ):


        # Related env
        self.simpleFlappingPropertiesEnv = simpleFlappingPropertiesEnv()  # load global properties
        self.simpleFlappingAeroEnv = simpleFlappingAeroEnv(self.simpleFlappingPropertiesEnv)
        self.simpleFlappingAeroEnv.computeADNumber_lr()  # It should be done each time a new action is chosen, this is ignored here for simplicity
        self.simpleFlappingMotionEnv = simpleFlappingMotionEnv(self.simpleFlappingPropertiesEnv)
        self.simpleFlappingAeroEnv.computeADNumber_lr() # Compute dimensionless parameters

        # Drone info
        self.rho = rho
        self.m = mass
        self.g = g
        self.CL_body = CL_body
        self.S_body = S_body

        # Range for the actions and the states
        self.bound_action = bound_action #! td: not used
        self.action_min = action_min
        self.action_max = action_max
        self.bound_state  = bound_state
        self.error_normalisation = np.ones(Nstate)*5
        
        # Simulation time settings
        self.t0 = t0 # initial time [s]
        self.T  = T  # Simulation time [s]
        self.dt = dt # Time-step [s]
        self.t_next = 0 # store current time of the solution, starts at 0
        T_period = 1/self.simpleFlappingPropertiesEnv.properties['wing']['frequency']
        self.Nstrokes = int((T-t0)/T_period)
        self.t = np.linspace(t0,T,int((T-t0)/dt)+1)
        self.Nit = int(T_period/dt) # number of timestep per cycle

        self.i_cycle = 1  # count of the current flapping cycle

        self.numObsState    = Nstate
        self.numActionState = Nactions
        self.reward_type    = reward_type

        # Initial state
        self.s0 = np.zeros(Nstate)  # [velocity_y,pos_y]
        self.s = np.zeros(Nstate)  # [velocity_y,pos_y]
        # Desired reference state
        self.s_ref =  np.array(s_ref)
        self.error = self.s - self.s_ref

        self.rew_pos = 0
        self.rew_vel = 0

        #! td Disturbance
        self.windType = windType  # constant or gust
        self.windVelocity = windVelocity
        N = 100
        self.N = N
        self.cov = self.get_cov(size=N, length=10)
        self.gust = mn.rvs(cov=self.cov)
        self.fgust = interpolate.interp1d(np.linspace(t0, T, N), self.gust)
        self.averageCycleGustList = []
        self.averageCycleGust = 0
        self.cycleGust = []

        #!td: used in step but why ?
        self.actionEnv = (0,0) # Stored in the buffer

        # Closure parameters
        self.c_list_true = c_list_true # True value for the closure parameters in the real system (i.e., real environment)
        
        # Bounds for the closure parameters
        self.bounds_p = bounds_p # For no bounds ((-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf))
        
        # NOTE: not mentioned in template to include observation space and action space
        self.observation_space_bounds = np.array([[-5,5],[-5,5]])
        self.observation_space = spaces.Box(self.observation_space_bounds[:, 0],
                                            self.observation_space_bounds[:, 1],
                                            dtype=np.float64)

        self.control_mode = control_mode

        # Debug arrays
        self.Lift = []
        self.Drag = []

        self.rew_pos_list = []
        self.rew_vel_list = []

        # Cost function for assimilation
        def cost_assimilation(pred_state, s_tilde):
            if len(pred_state.shape) > 1:
                return 0.5 * ((pred_state[:,0] - s_tilde[:,0])**2 + (pred_state[:,1] - s_tilde[:,1])**2 + (pred_state[:,2] - s_tilde[:,2])**2 + (pred_state[:,3] - s_tilde[:,3])**2)
            else:
                return 0.5 * ((pred_state[0] - s_tilde[0])**2 + (pred_state[1] - s_tilde[1])**2 +(pred_state[2] - s_tilde[2])**2 + (pred_state[3] - s_tilde[3])**2)
        self.cost_assimilation = cost_assimilation

        # Gaussian used by the reward function
        def gaussian(x, mu, sig):
            return (
                    np.exp(-np.power((x - mu) / sig, 2.0) / 2)   # 1.0 / (np.sqrt(2.0 * np.pi) * sig)
            )
        self.gaussian = gaussian

        # Control cost function (mf + mb)
        def cost(m=4):
            s_current = self.s  # [xdd, ydd, ubx, uby ]
            s_ref = self.s_ref

            # self.reward_pos_x = (s_current[2] - s_ref[2]) ** 2 # 2D
            self.reward_pos_y = (s_current[1] - s_ref[1]) ** 2

            if self.reward_type == 'tanh':
                self.reward_vel_x = 0.1 * ((s_current[0] - s_ref[0]) ** 2) * (
                        np.tanh(m * (s_current[2] - 7 * s_ref[2] / 8)) - np.tanh(
                        m * (s_current[2] - 9 / 8 * s_ref[2]))) / 2 * s_ref[2] ** 2
                self.reward_vel_y = 0.1 * ((s_current[1] - s_ref[1]) ** 2) * (
                            np.tanh(m * (s_current[3] - 7 * s_ref[3] / 8)) - np.tanh(
                        m * (s_current[3] - 9 / 8 * s_ref[3]))) / 2 * s_ref[3] ** 2
            else:
                self.reward_vel_y = 0.1 * ((s_current[0] - s_ref[0]) ** 2) * gaussian(s_current[1], s_ref[1], 0.5)  * s_ref[1] ** 2
                # self.reward_vel_y = 0.1 * ((s_current[1] - s_ref[1]) ** 2) * gaussian(s_current[3], s_ref[3], 0.5) *  s_ref[3] ** 2 # 2D

            self.rew_pos    = self.reward_pos_y # + self.reward_pos_x
            self.reward_vel = self.reward_vel_y # + self.reward_vel_x

            self.rew_pos_list.append(self.rew_pos)
            self.rew_vel_list.append(self.reward_vel)

            reward = self.rew_pos + self.reward_vel  # + reward_vel
            return reward

        self.cost = cost
        
        # Derivative of the cost function wrt the state
        def dcost_ds(pred_state, s_tilde):
            return pred_state - s_tilde
        self.dcost_ds = dcost_ds
        
        # NOTE: not mentioned in template to include a log and its mandatory elements
        self.log = {'state': [], 'dstate_dt': [], 'd_k': [], 'actions_eff':[], 'closure':[]}
        # Enable/disable plotting
        self.plot = plot

    def get_cov(self,size=100, length=10):
        x = np.arange(size)
        cov = np.exp(-(1 / length) * (x - np.atleast_2d(x).T)**2)
        return cov

    #!td: Disturbance: to be seen if kept
    def fwindVelocity(self,amplitude, windType='constant',t=0):
        if windType == 'constant':
            return amplitude
        elif windType == 'gust':
            return amplitude + amplitude*np.clip(0.25*self.fgust(min(t,self.T)),-0.5*np.abs(amplitude),0.5*np.abs(amplitude))
        else:
            ValueError('windType not implemented')

    def model(self, u0, t, action,c_list,generateGust=False,gust=0):
        # State var.
        # ubx = u0[0] # 2D
        # uby = u0[1] # 2D
        uby = u0[0]   # 1D

        # Action coming from the controller
        self.simpleFlappingPropertiesEnv.properties['motion']['A_phil'] = action[0]
        # beta_w = np.deg2rad(action[1])                                                            # 2D
        # self.simpleFlappingPropertiesEnv.properties['motion']['A_strokel'] = np.deg2rad(
        #     action[1])  # affects the force computation                                           # 2D

        # Compute kinematics
        philFull_rad, phirFull_rad, alphalFull_rad, alpharFull_rad = self.simpleFlappingMotionEnv.computeMotion_lr(t)
        phi_rad = philFull_rad[0];
        phid_rad = philFull_rad[1];
        alpha_rad = alphalFull_rad[0];
        alphad_rad = alphalFull_rad[1]

        # Compute wind velocity
        if generateGust:
            uwind = self.fwindVelocity(self.windVelocity, self.windType, t)
            self.averageCycleGustList.append(uwind)

            #if (self.windType != 'constant') and (self.windVelocity != 0):
            self.cycleGust.append(uwind)
        else: # replay gust
            uwind = gust

        XYZ = 2 * self.simpleFlappingAeroEnv.FTot_lr_b(alpha_rad, phi_rad, phid_rad, alphad_rad, [0], uwind, 1,c_list)
        Lwing = -XYZ[2]  # oriented along z
        #Dwing = XYZ[0] # used only in the x-dir                                    # 2D

        self.Lift.append(Lwing)

        # Total velocity account for wind motion
        #ux = ubx - uwind                                                             # 2D
        uy = uby

        # Velocity angle w.r.t horizontal
        beta_b = np.pi/2 # np.arctan2(uy, ux)                                         # 2D

        # Drag force of the body
        #D = 0.5 * self.rho * self.CL_body * self.S_body * (ux ** 2 + uy ** 2)         # 2D
        D = 0.5 * self.rho * self.CL_body * self.S_body * (uy ** 2)

        ydd = -self.g + Lwing / self.m - D / self.m * np.sin(beta_b)
        #xdd = Dwing / self.m - D / self.m * np.cos(beta_b)

        #return xdd, ydd, ubx, uby                                                      # 2D
        return ydd, uby

    def oneCycleEuler(self,u0,action,t_start,t_end,c_list,generateGust=True,gust=None):
            t = t_start
            u = u0
            dt = self.dt
            #x = self.s[2] # 2D
            y = self.s[-1]

            i = 0
            self.cycleGust = []
            while t<=t_end and i<100:
                if generateGust is False and self.windType=='gust': # replay an already generated gust #(gust != None) and (
                    uwind = gust[i]
                else:
                    uwind = 0 # dummy, replaced in 'model'

                #xdd, ydd, ubx, uby = self.model(u,t,action,c_list,generateGust=generateGust,gust=uwind)        # 2D
                ydd, uby = self.model(u,t,action,c_list,generateGust=generateGust,gust=uwind)

                # x   += dt*ubx                     # 2D
                y   += dt*uby
                # ubx += dt*xdd                     # 2D
                uby += dt*ydd

                #u = np.array([ubx,uby])            # 2D
                u = np.array([uby])

                t += dt
                i += 1 # counter to take the gust speed at the different instant inside the cyle

            self.t_next = t

            #return ubx, uby, x, y                      # 2D
            return uby, y

    def forward_step(self, a, assimilate = False,generateGust=False):
        """
        This method is used to compute the forward dynamics of the system. It is called
        by the controller at each time step. It is used both for the "real" system dynamics
        computation and for the assimilated one. The switch between the two is defined by the
        `mode` parameter.
        =========
        Parameters
        =========
        :param mode: str, either 'true_dynamics' or 'assimilated'
        :param prev_state: array, previous (true) state of the system (i.e. s_{t-1}), shape (n,)
                            where n is the state dimension. Not to be confused with the observed
                            state, input of the controller
        :param a: float, control input at time t \in [-1, 1], of dimensionality M (number of control inputs).
                This input is provided by the controller, and it is scaled from the range [-1, 1] x M to the
                actual action space of the system (i.e. (0, 30) deg for pitch of wind turbine), by the
                `action_scaling` method (which you need to call).
        :param s_tilde: array, trajectory to be tracked during the control. Same dimensionality of the true state.
                        This is the set-point.
        :param coeffs: array, if mode == 'assimilated', this is the set of coefficients of the model to be used
                       in the digital twin.
        =========
        Returns
        =========
        :return: obs_state: array, observed state of the system (i.e. s_t), shape (n,) (input of the controller)
        :return reward: float, reward obtained at time t. Quadratic tracking of set-point r_t =0.5 * (s - s_tilde)^2
        :return done: bool, True if the episode is over, False otherwise. The episode is over if the system
                has evolved for T time steps.
        :return info: dict, additional information to be returned. Some of them are mandatory, namely:
            - 'd_k', is the disturbance ecountered during this episode. Needed to carry out the assimilation step
            - 'state', is the true state of the system at time t. Needed to carry out the assimilation step and to
                compute the optimal control
            - 'g', is the closure function of the model. Needed to carry out the assimilation step. E.g. in the case
            of the wind turbine, this is the `c_p` function.
        """

        # Shortcuts
        t = self.t
        s = self.s  # u0: yd, u1: y
        i_cycle = self.i_cycle
        Nit = self.Nit
        c_list = self.c_list_true

        # Action scaling
        if not assimilate:
            action_vec = (a + 1) / 2 * (self.action_max - self.action_min) + self.action_min
        else:
            action_vec = a

        # Step one flapping cycle
        t_start = self.t[Nit * (i_cycle - 1)]
        t_end   = self.t[Nit * i_cycle]
        #xd, yd, x, y = self.oneCycleEuler(s,action_vec,t_start,t_end,c_list,generateGust=generateGust,gust=None)   # 2D
        yd, y = self.oneCycleEuler(s,action_vec,t_start,t_end,c_list,generateGust=generateGust,gust=None)

        # Last time step dynamics
        #self.s = np.array([xd, yd, x, y])  # state used for state->action mapping  # 2D
        self.s = np.array([yd, y])  # state used for state->action mapping
        self.error = self.s - self.s_ref

        # Increment counter of flapping cycle
        self.i_cycle = self.i_cycle + 1

        # Compute the cost based on self.s and self.s_ref
        cost = self.cost()

        done = (self.i_cycle == (self.Nstrokes+1))

        self.actionEnv = (action_vec[0], )  # (action_vec[0], action_vec[1]) # 2D

        # Update the log
        self.log['state'].append(self.s)
        self.log['d_k'].append(None if self.cycleGust == [] else self.cycleGust) #self.log['d_k'].append(None if d_k == None else d_k) # NOTE: could be nice to state that all log entries should be arrays. Scalars raise an error when concatenating.
        self.log['actions_eff'].append(action_vec)
        #self.log['closure']
        
        return self.error, cost, done, self.log
        

    @staticmethod
    def check_array_lengths(forward_solution, control_history, s_tilde):
        """
        Utility function which checks that the arrays have the same length. If not, raises an error: the ajoint
        has exploded!
        ======
        Parameters
        ======
        :param forward_solution: array, solution of the forward model
        :param control_history: array, history of the control outputs
        :param s_tilde: array, trajectory to be tracked during the control
        :return:
        """
        if len(forward_solution) != len(control_history) or \
                len(forward_solution) != len(s_tilde):
            raise ValueError('Arrays must have the same length!')


    def reset(self, s0: np.array, verbose: bool) -> np.array:
        """
        Reset function, to be called at the beginning of each episode, to set-up parameters and initial conditions.
        ==========
        Parameters
        ==========
        :param s0: array, initial conditions of the system
        :param verbose: bool, flag to print or not some information about the previous episode
        ==========
        Returns
        ==========
        :return: s0: array, initial conditions of the system

        Notes:
        - Remember to reset the logging variables
        - Here you must load the disturbance you are going to apply during your simulation
        - Call also the actuator model, if any. I.e. in the case of the wind turbine is a first order model to
        simulate delay in the response. Idea is to define a common pool of controllers architectures' to be used
        in different testcases.
        """

        self.s0 = s0
        self.s = np.zeros(self.numObsState)

        self.i = 0

        self.rew_pos = 0
        self.rew_vel = 0
        self.i_cycle= 1
        self.Lift = []
        self.averageCycleGustList = []

        #Sample noise from Gaussian
        if self.windType == 'gust':
            self.gust = mn.rvs(cov=self.cov)
            print(self.gust[-5:-1])
            self.fgust = interpolate.interp1d(np.linspace(self.t0,self.T,self.N), self.gust)

        # if verbose:
        #     print('Resetting environment...')

        self.log = {'state': [], 'dstate_dt': [], 'd_k': [], 'actions_eff':[], 'damping':[]}
        self.t_next = 0

        return s0
    
    def get_reference(self, d_k):
        """ reference state remains the same, no matter the disturbance """
        return self.s_ref
    
    def action_scaling(self, action, target_min: float = -1, target_max: float = 1,
                       source_min: float = -1, source_max: float = 1,
                       ) -> np.array:
        """
        Here we scale the action to the desired range. The action outputted from the policy is in the range [-1, 1] x M
        (action dimensionality). We need to scale it to the desired range, which is generally different from the
        [-1, 1] x M.
        ==========
        Parameters
        ==========
        :param action: array, action to be scaled
        :param target_min: float, minimum value of the target range = min value of the control input to be applied
        in the system
        :param target_max: float, maximum value of the target range = max value of the control input to be applied
        in the system
        :param source_min: float, minimum value of the source range = min value of the action outputted by the policy = -1
        :param source_max: float, maximum value of the source range = max value of the action outputted by the policy = 1
        ==========
        Returns
        ==========
        :return: action: array, scaled action to be applied to the system
        """
        a_raw_out = ((action - source_min) / (source_max - source_min)) * (target_max - target_min) + target_min
        return a_raw_out
