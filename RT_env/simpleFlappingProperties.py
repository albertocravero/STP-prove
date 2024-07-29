# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 11:24:46 2021

@author: romai
"""

# !. virer ADD_parameters et garder uniquement ADD_parameters_lr
# A voir car ca doublertait les appels aux fonctions forces pour rien dans le 
# cas ou les duex ailes ont le mm mvt. 

import numpy as np;

class simpleFlappingPropertiesEnv():

    
    #def __init__(self): # see https://careerkarma.com/blog/python-class-variables/
    
    # USER ENTRY
    ####### from here ... ##############   
    # Fluid            
        rho = 1.2 #Dickinson: 0.92e3 #0.88e3#1.2                   # [kg/m^3]
        nu  = 1.5e-5              # [m^2/s]
    
    # Gravity
        g   = 9.81              # [m/s^2]
        
    # Body
        mass = 0.003 # [kg]
        Ix = 1e-5; Iy = 1e-5; Iz = 1e-5; # inertial matrix of th body
        #self.Izx = 2; # symmetrical body 
        
    # Wing
    # Careful: the wing root is the hinge, see function chordDistribution(shape)
        shape = 'semi-ellipsoid'         # Chose between'rectangle' or 'ellipsoid'   or 'Dickinson'     
        # Computed from AD parameters 
        chord  = 0.01538    #Dickinson: 0.0766 # 0.015 # mean chord
        span   = 0.05      #Dickinson: 0.25-0.06 #0.05 #AR*chord
        DeltaR = 0.0225 #Dickinson: 0.06 #0.0225
        frequency = 20  #Dickinson: 0.168#0.03 #20;       #Uref/lR2
        pitchingAxis = 0.5    # dimensionless pitching axis position: 0=LE and 1=TE
        surface = span*chord  # could be also filled in a force function with distribution function as : S=int_0^R c(r) dr
        aspectRatio = span/chord    # [-]
        
        # If same motion for both wings
        # Filled with computeADNumber in FlappingAero because needs the chordDistribution function
        R2   = 0     #np.sqrt((1/S)*np.trapz(self.chordDistribution()*(r**2),x=r))
        lR2  = 0     #4*np.deg2rad(A_phi)*R2
        Uref = 0     #frequency*lR2
        Re   = 0     #Uref*chord/nu
        Ro   = 0     #Ro = R2/wing['chord']
        k_hover = 0 # reduced frequency for hovering k = pi / (phi * AR) - AC
        # If independent wings 
        # Filled with computeADNumber_lr in FlappingAero because needs the chordDistribution function
        R2lr   = 0     #np.sqrt((1/S)*np.trapz(self.chordDistribution()*(r**2),x=r))
        lR2l  = 0     #4*np.deg2rad(A_phi)*R2
        Urefl = 0     #frequency*lR2
        Rel   = 0     #Uref*chord/nu
        Rol   = 0     #Ro = R2/wing['chord']
        lR2r  = 0     #4*np.deg2rad(A_phi)*R2
        Urefr = 0     #frequency*lR2
        Rer   = 0     #Uref*chord/nu
        Ror   = 0     #Ro = R2/wing['chord']
    
    # Numerics
        discretisationRate = 5000 # number of elements for BEM, default: 100
        r = np.linspace(DeltaR,span+DeltaR,discretisationRate); # vector form 0 to full span  

    # Motion
        kinematicMotionType = 'SHM_midStart_figure8' #'Dickinson' #'SHM' #' # Chose between 'SHM', 'SHM_midStart','SHM_midStart_figure8' 'SHM-TRAPZ','Dickinson', 'insectInspired', 'trianglePhitrapezoidalAoa_old', 'trianglePhitrapezoidalAoa'
        # If same motion for both wings
        A_phi = 60    # [°] 
        A_alpha = 45  # [°]
        lR2 = 4*np.deg2rad(A_phi)*R2
        # If independent wings
        A_phil = 75    # [°]  # Left
        A_phir = 75    # [°]  # Right
        A_alphal = 45  # [°]  # Left
        A_alphar = 45  # [°]  # Right
        A_strokel = 0  # [°]  # Left
        A_stroker = 0  # [°]  # Right
        A_devl = 0  # [°]    # w.r.t ground # Left
        A_devr = 0  # [°]    # w.r.t ground # Right
        Kphi   = 0.01 #0.5 #
        Kalpha = 0.01 #3 #8
        
        A_phil_LW = 62.5
        
        # NOT USED FOR SHM
        tstar_riseAlpha  = 0.1  # for the motion type "trianglePhitrapezoidalAoa": dimensionless time to reach constant alpha
        tstar_delayAlpha = 0    # imposed dimensionless delay on alpha (>0: delayed rotation, =0: symmetrical rotation and <0: advanced rotation)
        tstar_accPhi = 0.075     # (dimensionless time of acceleration phase)/2 for phi
        
    # Model output; filled out when running the model
        CL_tr = []; CD_tr = []  # [-]
        alphat = [] # deg
    ######## ... to here #############

        fluid    = {'rho': rho, 'nu': nu}
        body     = {'mass': mass, 'I': np.array([Ix,Iy,Iz])}
        wing     = {'shape': shape, 'chord': chord, 'span': span, 'DeltaR': DeltaR,'surface': surface, 'aspectRatio': aspectRatio, 'frequency': frequency, 'pitchingAxis': pitchingAxis}
        ADD_parameters = {'Re': Re, 'Ro': Ro, 'R2': R2, 'Uref': Uref, 'lR2': lR2, 'k_hover': k_hover} # !.
        ADD_parameters_lr = {'Rel': Rel, 'Rer': Rer,'Rol': Rol, 'Ror': Ror, 'R2lr': R2lr, 'Urefl': Urefl, 'Urefr': Urefr, 'lR2l': lR2l, 'lR2r': lR2r, 'k_hover': k_hover}
        numerics = {'discretisationRate': discretisationRate,'r': r}
        motion   = {'kinematicMotionType': kinematicMotionType, 'A_phi': A_phi, 'A_phil': A_phil, 'A_phir': A_phir, 'A_alpha': A_alpha, 'A_alphal': A_alphal, 
                    'A_alphar': A_alphar, 'A_strokel': A_strokel, 'A_stroker': A_stroker,  'A_devl': A_devl, 'A_devr': A_devr,
                    'tstar_riseAlpha': tstar_riseAlpha, 'tstar_delayAlpha': tstar_delayAlpha, 'tstar_accPhi': tstar_accPhi,
                    'Kphi': Kphi,'Kalpha': Kalpha, 'A_phil_LW': A_phil_LW}
        modelOutput = {'CL_tr': CL_tr, 'CD_tr': CD_tr, 'alphat': alphat}
        
        properties = {'g':g, 'fluid': fluid, 'body': body, 'wing': wing, 'ADD_parameters': ADD_parameters, 'ADD_parameters_lr': ADD_parameters_lr, 'numerics': numerics, 'motion': motion, 'modelOutput': modelOutput} #, 'debug': debug}

        def chordDistribution(self): 
            NR   = self.properties['numerics']['discretisationRate']
            wing = self.properties['wing']
            r    =  self.properties['numerics']['r']
    
            if wing['shape'] == 'rectangle':
                self.properties['wing']['cdistr'] = wing['chord']*np.ones(self.SimulationPropertiesEnv.properties['numerics']['discretisationRate'])
            elif wing['shape']=='ellipsoid':
                self.properties['wing']['cdistr'] = np.sqrt(1 - ((wing['span']/2 - np.linspace(0,wing['span'],NR))/(wing['span']/2))**2 )*wing['chord']
            elif wing['shape']=='semi-ellipsoid':
                self.properties['wing']['cdistr'] = np.sqrt(np.abs(1 - ((r - wing['DeltaR'])/(wing['span']))**2))*(wing['chord']*4/np.pi)
            else:
                ValueError("Shape is not implemented")
                
# Used by second term in the rotational force
        def invertChordDistribution(self):
            '''
            Computes the span distribution according to an input chord distribution
            :return:
            '''
            #NR = self.properties['numerics']['discretisationRate']
            wing = self.properties['wing']
            cvec =  wing['cvec']

            if wing['shape'] == 'semi-ellipsoid':
                self.properties['wing']['rdistr'] = wing['span'] * np.sqrt(1 - (np.pi*cvec/4/wing['chord'])**2) + wing['DeltaR']
            else:
                ValueError("Shape is not implemented")
                
        def intcr2(self): 
            wing = self.properties['wing'];
            r    =  self.properties['numerics']['r'];
            self.properties['wing']['intcr2'] = np.trapz(wing['cdistr']*r**2, x=r)

        # Used by Frot only
        def intrxx(self):
            NR = self.properties['numerics']['discretisationRate']
            wing = self.properties['wing'];
            x0 = wing['pitchingAxis']
            TEToLE = np.linspace(-x0 * wing['chord'], wing['chord'] - x0 * wing['chord'],NR)
            self.properties['wing']['rcvec'] =  np.trapz(['wing']['rdistr'] * TEToLE * np.abs(TEToLE), x=TEToLE)
            
        def intc(self): 
            wing = self.properties['wing'];
            r    =  self.properties['numerics']['r'];
            self.properties['wing']['intc'] = np.trapz(wing['cdistr'], x=r)
            
        def intcr(self): 
            wing = self.properties['wing'];
            r    =  self.properties['numerics']['r'];
            self.properties['wing']['intcr'] = np.trapz(wing['cdistr']*r, x=r)
      
        def intc2r(self): 
            wing = self.properties['wing'];
            r    =  self.properties['numerics']['r'];
            self.properties['wing']['intc2r'] = np.trapz((wing['cdistr']**2)*r, x=r)        
            
        def intc2(self): 
            wing = self.properties['wing'];
            r    =  self.properties['numerics']['r'];
            self.properties['wing']['intc2'] = np.trapz((wing['cdistr']**2), x=r)        
            
        def intc3(self): 
            wing = self.properties['wing'];
            r    =  self.properties['numerics']['r'];
            self.properties['wing']['intc3'] = np.trapz((wing['cdistr']**3), x=r)   