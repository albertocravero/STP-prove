#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 18:31:27 2023

@author: poletti
"""

import numpy as np;

from simpleFlappingProperties import simpleFlappingPropertiesEnv

class simpleFlappingAeroEnv():
    
    
    def __init__(self,simpleFlappingProperties):
        
        self.simpleFlappingPropertiesEnv = simpleFlappingProperties # load global properties 
        
        #self.simpleFlappingMotionEnv = simpleFlappingMotionEnv() # load global properties  
        
        self.Nt = 1000 # filled in the main
        tfull = np.linspace(0,1/self.simpleFlappingPropertiesEnv.properties['wing']['frequency'],self.Nt); t = tfull[:-1];
        self.t = t
        self.simpleFlappingPropertiesEnv.chordDistribution()
        # Constant term in the force computation
        self.simpleFlappingPropertiesEnv.intcr2()
        self.simpleFlappingPropertiesEnv.intcr()
        self.simpleFlappingPropertiesEnv.intc()
        self.simpleFlappingPropertiesEnv.intc2r()
        self.simpleFlappingPropertiesEnv.intc2()
        self.simpleFlappingPropertiesEnv.intc3()
        
        # Init of rotation matrix to identity matrix
        self.EtatoTheta = np.eye(3); self.ThetatoPhi = np.eye(3); self.PhitoSp = np.eye(3); self.SptoI = np.eye(3)
        self.ThetatoEta = np.eye(3); self.PhitoTheta = np.eye(3); self.SptoPhi = np.eye(3); self.ItoSp = np.eye(3)
        
        self.PsiMat = np.eye(3); self.ThetaMat = np.eye(3);  self.PhiMat = np.eye(3)
        
        self.alphaGeom = []
        self.alphaTrue = []
        
        self.testInt1 = []
        self.testInt2 = []
        self.testInt3 = []
        self.testInt4 = []
        
        self.testInt5 = []
        self.testInt6 = []
        self.testInt7 = []
        
        self.CL = 0
        self.CD = 0
  
    def LiftTr_lr(self, alpha_rad, phi_rad, phid_rad, lr): 
        # shortcut
        rho = self.simpleFlappingPropertiesEnv.properties['fluid']['rho']
        wing= self.simpleFlappingPropertiesEnv.properties['wing']
        r   =  self.simpleFlappingPropertiesEnv.properties['numerics']['r']
        # load lift coefficient
        if lr == 1: CL_tr = self.CL_func(alpha_rad,'Lee',self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['Rel'],self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['Rol'],wing['aspectRatio'])
        else: CL_tr = self.CL_func(alpha_rad,'Lee',self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['Rer'],self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['Ror'],wing['aspectRatio'])

        # compute force
        LiftTr = 0.5*rho* CL_tr * (phid_rad**2) * np.trapz(self.chordDistribution()*(r**2), x=r)
        self.simpleFlappingPropertiesEnv.properties['modelOutput']['CL_tr']=CL_tr  # save to output
        return -np.abs(LiftTr)*np.sign(alpha_rad)*np.sign(phid_rad) #*np.sign(np.pi/2-alpha_rad) # adapt the sign

    # Take into account incoming wind and body velocity
    def LiftTr_lr_v2(self, alpha_rad, phi_rad, phid_rad,alphad_rad, ub, uwind, lr,c_list):
        # shortcut
        rho = self.simpleFlappingPropertiesEnv.properties['fluid']['rho']
        wing= self.simpleFlappingPropertiesEnv.properties['wing']
        R2 = self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['R2lr']
        
        flappingVar   = np.array([phi_rad, alpha_rad, 0])
        flappingVard  = np.array([phid_rad, alphad_rad, 0]) # alpha d is not used to compute flapping vel  
                
        # Total angular velocity: (body + wing)_wf
        state = np.zeros(12)            # no body rotation
        #state[0] = ub[0] - uwind;       #2D !! to be changed if there is pitch rotation
        state[2] = -ub[0];  #2D:   -ub[1];         # z pointing down

        omega_tot = self.omegaWingTot_wf(flappingVar, flappingVard, state, lr)
        # (Linear velocity body)_wf AND WIND
        vb_wf = self.velocityBody_wf(state, flappingVar) # beta enter the dance here
        # (Mean wing velocity)_wf
        # training component due to wing flapping and body rotation: omega x R
        Umean_vec = np.array([-omega_tot[2]*R2,0,omega_tot[0]*R2])
        # relative component due to body linear motion: \circle{u_body}
        Umean_vec += vb_wf # add linear velocity of b.

        # Effective a.o.a: measured between the airfoil and the velocity vector
        # t-frame (= wind frame): x aligned with velocity, z +90° in the ccw
        # Body motion affects the aoa seen by the wing
        # positive if velocity vector lag ccw w.r.t.airfoil
        
        #alphat_rad = np.sign(Umean_vec[0]+1e-12) * np.arccos(-Umean_vec[-1]/(np.linalg.norm(Umean_vec)+1e-12)) 
        alphat_rad = -np.arctan2(-Umean_vec[0],-Umean_vec[2]+1e-12) 

        
        #omega_tot[1] = 0  to not consider radial velocity
        # Switch omega order since later cross producted with r: omega x r
        # (omega1 e1 + omega3 e3) x (r1 e1 + r2 e2 + r3 e3) = - omega3 r2 e1  - omega1 r3 e2 + omega3 r1 e2 + omega1 r2 e3
        # Minus in front to put ourself on the fluid side
        omega_tot = -np.array([-omega_tot[2],0,omega_tot[0]]) 
        vb_wf = -np.array([vb_wf[0],0,vb_wf[2]])

        if lr == 1: 
            CL_tr = self.CL_func(alphat_rad,'RMA',self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['Rel'],self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['Rol'],wing['aspectRatio'],c_list[0])
            CD_tr = self.CD_func(alphat_rad,'RMA',self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['Rel'],self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['Rol'],wing['aspectRatio'],c_list[1:])
        else: 
            CL_tr = self.CL_func(alphat_rad,'RMA',self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['Rer'],self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['Ror'],wing['aspectRatio'],c_list[0])
            CD_tr = self.CD_func(alphat_rad,'RMA',self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['Rer'],self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['Ror'],wing['aspectRatio'],c_list[1:])
        
        # For assimilation
        self.CL = CL_tr
        self.CD = CD_tr

        LiftTr = 0.5*rho * CL_tr * (np.sum(omega_tot**2) * self.simpleFlappingPropertiesEnv.properties['wing']['intcr2'] 
           + 2*np.sum(vb_wf*omega_tot)*self.simpleFlappingPropertiesEnv.properties['wing']['intcr']
           + np.sum(vb_wf**2) * self.simpleFlappingPropertiesEnv.properties['wing']['intc']) 
        
        DragTr = 0.5*rho * CD_tr * (np.sum(omega_tot**2) * self.simpleFlappingPropertiesEnv.properties['wing']['intcr2']
         + 2*np.sum(vb_wf*omega_tot)*self.simpleFlappingPropertiesEnv.properties['wing']['intcr'] 
         + np.sum(vb_wf**2) * self.simpleFlappingPropertiesEnv.properties['wing']['intc'])

        self.simpleFlappingPropertiesEnv.properties['modelOutput']['alphat'].append(np.rad2deg(alphat_rad))
        
        LiftTr = -np.abs(LiftTr) * np.sign(alphat_rad)
        DragTr = -np.abs(DragTr)

        self.simpleFlappingPropertiesEnv.properties['modelOutput']['CL_tr'].append(LiftTr)
        
        F_Xw =  +LiftTr*np.cos(alphat_rad) + DragTr*np.sin(alphat_rad)
        F_Zw =  +LiftTr*np.sin(alphat_rad) - DragTr*np.cos(alphat_rad)

        return F_Xw, F_Zw 
    
    
    # Total forces in the body frame: array([Fx, Fy, Fz])
    def FTot_lr_b(self, alpha_rad, phi_rad, phid_rad, alphad_rad,ub,uwind, lr,c_list):

        
        flappingVar   = np.array([phi_rad, alpha_rad, 0])
        state = np.zeros(12) # no body rotation
        #state[0] = ub[0] - uwind; #2D #!! to be chnaged if there is pitch rotation
        state[2] = -ub[0]; #2D -ub[1]; # z pointing down
        
        F_Xw,F_Zw   = self.LiftTr_lr_v2(alpha_rad, phi_rad, phid_rad,alphad_rad,ub,uwind, lr,c_list)

        aa = np.matmul(self.rotationMatrixWing_EtatoTheta(flappingVar[1]), np.array([F_Xw,0,F_Zw]))
        bb = np.matmul(self.rotationMatrixWing_ThetatoPhi(flappingVar[2]),aa)
        cc = np.matmul(self.rotationMatrixWing_PhitoSp(flappingVar[0]),bb)
        

        if lr ==1:
            beta = self.simpleFlappingPropertiesEnv.properties['motion']['A_strokel']
        else:
            beta = self.simpleFlappingPropertiesEnv.properties['motion']['A_stroker']
        
        F_I = np.matmul(self.rotationMatrixWing_SptoI(beta), cc)
        
        #!!
        # InertialToBody = 1
        # F_B = np.matmul(self.eulerRotationMatrix_Psi(state[6], InertialToBody),np.matmul(self.eulerRotationMatrix_Theta(state[7], InertialToBody),np.matmul(self.eulerRotationMatrix_Phi(state[8], InertialToBody),F_I)))
        # return F_B
        
        return F_I
    
# Sign: downstroke: D<0, upstroke: D>0 according to the wing frame (which has its x pointing fwd)
    def DragTr_lr(self,alpha_rad, phi_rad, phid_rad,lr): 
        # shortcut        
        rho = self.simpleFlappingPropertiesEnv.properties['fluid']['rho']
        wing= self.simpleFlappingPropertiesEnv.properties['wing']
        r   =  self.simpleFlappingPropertiesEnv.properties['numerics']['r']
        # load lift coefficient
        if lr == 1: CD_tr = self.CD_func(alpha_rad,'Lee',self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['Rel'],self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['Rol'],wing['aspectRatio'])
        else:  CD_tr = self.CD_func(alpha_rad,'Lee',self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['Rer'],self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['Ror'],wing['aspectRatio'])
        # compute force        
        DragTr = 0.5*rho * CD_tr * (phid_rad**2) * np.trapz(self.chordDistribution()*(r**2), x=r)
        self.simpleFlappingProperties.properties['modelOutput']['CD_tr']=CD_tr  # save to output
        return np.abs(DragTr) #* np.sign(phid_rad) # adapt the sign


# Translation lift coefficient        
    def CL_func(self,alpha_rad,model,Re=1,Ro=1,AR=1,c_list=0):
        if model == 'RMA':
            return c_list*np.sin(2*(alpha_rad))
        if model=='Lee': # Lee2016 model, to be used for flight dynamic simulation (aoa defined differently)
            AL  = 1.966 - 3.94 * Re**(-0.429)
            fRo = -0.205 * np.arctan(0.587*(Ro-3.105)) + 0.870 # Correction factors cause Lee CFD are done with rectangular wings
            fAR = 32.9 - 32.0 * AR **(-0.00361)                # Correction factors cause Lee CFD are done with rectangular wings
            return (AL*np.sin(2*(np.pi/2-alpha_rad))) *fRo*fAR
        if model=='Lee_orig':  # Lee2016 model, to be used for static body simulation
            AL  = 1.966 - 3.94 * Re**(-0.429)
            fRo = -0.205 * np.arctan(0.587*(Ro-3.105)) + 0.870 # Correction factors cause Lee CFD are done with rectangular wings
            fAR = 32.9 - 32.0 * AR **(-0.00361)                # Correction factors cause Lee CFD are done with rectangular wings
            #print(1.966*fRo*fAR,3.94*fRo*fAR*Re**(-0.429))
            return (AL*np.sin(2*(alpha_rad)))*fRo*fAR          # original implementation of alpha: from U vector to airfoil   
        else:
            ValueError("Model is not implemented: select 'Lee' or 'robotFly'")   
            
# Translation drag coefficient               
    def CD_func(self,alpha_rad,model,Re=1,Ro=1,AR=1,c_list=[0,0]):
        if model == 'RMA':
            return c_list[0]+c_list[1]*(1-np.cos(2*alpha_rad))
        if model=='Lee': # Lee2016 model, to be used for flight dynamic simulation (aoa defined differently)
            AD  = 1.873-3.14*Re**(-0.369)
            CD0 = 0.031 + 10.48*Re**(-0.764)
            fRo = -0.205 * np.arctan(0.587*(Ro-3.105)) + 0.870  # Correction factors cause Lee CFD are done with rectangular wings
            fAR = 32.9 - 32.0 * AR **(-0.00361)                 # Correction factors cause Lee CFD are done with rectangular wings
            return (CD0 + AD*(1-np.cos(2*(np.pi/2-alpha_rad))))*fRo*fAR
        if model=='Lee_orig': # Lee2016 model, to be used for static body simulation
            AD  = 1.873-3.14*Re**(-0.369)
            CD0 = 0.031 + 10.48*Re**(-0.764)
            fRo = -0.205 * np.arctan(0.587*(Ro-3.105)) + 0.870  # Correction factors cause Lee CFD are done with rectangular wings
            fAR = 32.9 - 32.0 * AR **(-0.00361)                 # Correction factors cause Lee CFD are done with rectangular wings
            #print(CD0*fRo*fAR,AD*fRo*fAR)
            return (CD0 + AD*(1-np.cos(2*alpha_rad)))*fRo*fAR        
        else:
            ValueError("Model is not implemented: select 'Lee'")
            
            
    def chordDistribution(self): 
            NR  = self.simpleFlappingPropertiesEnv.properties['numerics']['discretisationRate']
            wing= self.simpleFlappingPropertiesEnv.properties['wing']
            r   =  self.simpleFlappingPropertiesEnv.properties['numerics']['r']
    
            if wing['shape'] == 'rectangle':
                return wing['chord']*np.ones(self.simpleFlappingPropertiesEnv.properties['numerics']['discretisationRate'])
            elif wing['shape']=='ellipsoid':
                return np.sqrt(1 - ((wing['span']/2 - np.linspace(0,wing['span'],NR))/(wing['span']/2))**2 )*wing['chord']
            elif wing['shape']=='semi-ellipsoid':
                return np.sqrt(np.abs(1 - ((r - wing['DeltaR'])/(wing['span']))**2))*(wing['chord']*4/np.pi)
            elif wing['shape']=='Dickinson':
                pup = np.array([-8.02425753e+06,  9.49416529e+06, -4.77680589e+06,  1.33168822e+06, -2.24505839e+05,  2.33880675e+04, 
                                -1.46869336e+03,  5.10758929e+01,-7.38503508e-01])
                plow = np.array([ 2.22383376e+07, -2.65627922e+07,  1.35168022e+07, -3.81972312e+06, 6.54585062e+05, -6.95887311e+04,
                                 4.48244362e+03, -1.60224824e+02, 2.38522369e+00])
                yup = np.polyval(pup,r)
                ylow = np.polyval(plow,r)
                return  yup-ylow
            else:
                ValueError("Shape is not implemented")  
                
    def computeADNumber_lr(self):
        r   =  self.simpleFlappingPropertiesEnv.properties['numerics']['r']
        self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['R2lr']   = np.sqrt((1/self.simpleFlappingPropertiesEnv.properties['wing']['surface'])*np.trapz(self.chordDistribution()*(r**2),x=r))
        # Left
        self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['lR2l']  = 4*np.deg2rad(self.simpleFlappingPropertiesEnv.properties['motion']['A_phil'])*self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['R2lr']
        self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['Urefl'] = self.simpleFlappingPropertiesEnv.properties['wing']['frequency']*self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['lR2l']
        self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['Rel']   = self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['Urefl']*self.simpleFlappingPropertiesEnv.properties['wing']['chord']/self.simpleFlappingPropertiesEnv.properties['fluid']['nu']
        self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['Rol']   = self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['R2lr']/self.simpleFlappingPropertiesEnv.properties['wing']['chord']
         # Reduced frequency for hovering
        self.simpleFlappingPropertiesEnv.properties['ADD_parameters_lr']['k_hover_l'] = np.pi/ (np.deg2rad(2*self.simpleFlappingPropertiesEnv.properties['motion']['A_phil'])) / self.simpleFlappingPropertiesEnv.properties['wing']['aspectRatio'] # added reduced frequency for hover. 


# Way back           
    def  rotationMatrixWing_EtatoTheta(self, eta):         
        self.ThetatoEta[0,0] = np.cos(eta); self.ThetatoEta[0,2] = -np.sin(eta)
        self.ThetatoEta[2,0] = np.sin(eta); self.ThetatoEta[2,2] = np.cos(eta)   
        return self.ThetatoEta        

    def  rotationMatrixWing_ThetatoPhi(self, theta):         
         self.ThetatoPhi[1,1] = np.cos(theta); self.ThetatoPhi[1,2] = np.sin(theta)
         self.ThetatoPhi[2,1] = -np.sin(theta); self.ThetatoPhi[2,2] = np.cos(theta)   
         return self.ThetatoPhi
    
    def  rotationMatrixWing_PhitoSp(self, phi):         
         self.PhitoSp[0,0] = np.cos(phi); self.PhitoSp[0,1] = -np.sin(phi)
         self.PhitoSp[1,0] = np.sin(phi); self.PhitoSp[1,1] = np.cos(phi)   
         return self.PhitoSp
    
    def  rotationMatrixWing_SptoI(self, beta):         
         self.SptoI[0,0] = np.cos(beta); self.SptoI[0,2] =  np.sin(beta)
         self.SptoI[2,0] = -np.sin(beta) ; self.SptoI[2,2] = np.cos(beta)           
         return self.SptoI
    
# Way fwd
    def  rotationMatrixWing_ThetatoEta(self, eta):         
         self.ThetatoEta[0,0] = np.cos(eta); self.ThetatoEta[0,2] = np.sin(eta)
         self.ThetatoEta[2,0] =-np.sin(eta); self.ThetatoEta[2,2] = np.cos(eta)   
         return self.ThetatoEta
     
    def  rotationMatrixWing_PhitoTheta(self, theta):         
         self.PhitoTheta[1,1] = np.cos(theta); self.PhitoTheta[1,2] = -np.sin(theta)
         self.PhitoTheta[2,1] = np.sin(theta); self.PhitoTheta[2,2] = np.cos(theta)   
         return self.PhitoTheta
     
    def  rotationMatrixWing_SptoPhi(self, phi):         
         self.SptoPhi[0,0] = np.cos(phi); self.SptoPhi[0,1] = np.sin(phi)
         self.SptoPhi[1,0] =-np.sin(phi); self.SptoPhi[1,1] = np.cos(phi)   
         return self.SptoPhi
    
    def  rotationMatrixWing_ItoSp(self, beta):         
         self.ItoSp[0,0] = np.cos(beta); self.ItoSp[0,2] = -np.sin(beta)
         self.ItoSp[2,0] = np.sin(beta) ; self.ItoSp[2,2] =  np.cos(beta)           
         return self.ItoSp 
     
# Rotation from body frame to inertial frame (e.g to have p,q,r in the inertial frame)
    def  eulerRotationMatrix_Psi(self, psi_euler, BodyToInertial):  
       psi_euler = BodyToInertial*psi_euler 
       self.PsiMat[0,0]=np.cos(psi_euler); self.PsiMat[0,1]=np.sin(psi_euler);
       self.PsiMat[1,0]=-np.sin(psi_euler); self.PsiMat[1,1]=np.cos(psi_euler);           
       return self.PsiMat;  
               
    def  eulerRotationMatrix_Theta(self, theta_euler, BodyToInertial):  
       theta_euler = BodyToInertial*theta_euler 
       self.ThetaMat[0,0]=np.cos(theta_euler);  self.ThetaMat[0,2]=-np.sin(theta_euler);
       self.ThetaMat[2,0]=np.sin(theta_euler);  self.ThetaMat[2,2]=np.cos(theta_euler);           
       return self.ThetaMat; 
   
    def  eulerRotationMatrix_Phi(self, phi_euler, BodyToInertial):  
       phi_euler = BodyToInertial*phi_euler 
       self.ThetaMat[0,0]=np.cos(phi_euler);  self.ThetaMat[0,2]=-np.sin(phi_euler);
       self.ThetaMat[2,0]=np.sin(phi_euler);  self.ThetaMat[2,2]=np.cos(phi_euler);           
       return self.PhiMat; 

# Total angular rotation of the wing. Includes wing flapping, pitching, deviation + body roll, yaw, pitch.
# In the wing frame
    def  omegaWingTot_wf(self, flappingVar, flappingVar_dot, state, lr):
        return self.omegaWing_wf(flappingVar, flappingVar_dot, lr) + self.omegaBody_wf(state, flappingVar) 
    
    def  omegaBody_wf(self, state, flappingVar):
       
       # !! 
       # pqr_B = state[3:6]
       # InertialToBody = -1
       # phi    = flappingVar[0]
       # eta    = flappingVar[1];
       # theta  = flappingVar[2];
       # # Euler rotation twds inertial frame
       # # !! without body rot
       # pqr_I = np.dot(np.matmul(np.dot(self.eulerRotationMatrix_Psi(state[6], InertialToBody),self.eulerRotationMatrix_Theta(state[7], InertialToBody)),self.eulerRotationMatrix_Phi(state[8], InertialToBody)),pqr_B)
       # # Inertial frame twds stroke plane
       # pqr_sp = np.dot(self.rotationMatrixWing_ItoSp(self.simpleFlappingPropertiesEnv.properties['motion']['A_strokel']),pqr_I)
       # # Stroke plane twds wing plane
       # return np.dot(np.dot(self.rotationMatrixWing_ThetatoEta(eta),self.rotationMatrixWing_PhitoTheta(theta)), np.dot(self.rotationMatrixWing_SptoPhi(phi),pqr_sp))
       
       return np.zeros(3)
           
# Angular velocity of the wing in the wing frame (wf)
    def  omegaWing_wf(self, flappingVar, flappingVar_dot, lr):  
       phid    = flappingVar_dot[0]
       etad    = flappingVar_dot[1];
       thetad  = flappingVar_dot[2];
       A = np.dot(np.matmul(self.rotationMatrixWing_ThetatoEta(flappingVar[1]),self.rotationMatrixWing_PhitoTheta(flappingVar[2])), np.array([0,0,phid]))
       B = np.dot(self.rotationMatrixWing_ThetatoEta(flappingVar[1]),np.array([thetad,0,0]))
       return A + B + np.array([0,etad,0])
   
    
   
#Translational velocity of the body in the wing frame (wf)
    def  velocityBody_wf(self, state, flappingVar):
       uvw_B = state[0:3]
       phi    = flappingVar[0]
       eta    = flappingVar[1];
       theta  = flappingVar[2];
       # Euler rotation twds inertial frame
       InertialToBody = -1
       #!! in this simple test case Inertial frame is aligned with body frame
       uvw_I = uvw_B # np.matmul(np.matmul(np.matmul(self.eulerRotationMatrix_Psi(state[6],InertialToBody),self.eulerRotationMatrix_Theta(state[7],InertialToBody)),self.eulerRotationMatrix_Phi(state[8],InertialToBody)),uvw_B)
       # Inertial frame twds stroke plane
       uvw_sp = np.matmul(self.rotationMatrixWing_ItoSp(self.simpleFlappingPropertiesEnv.properties['motion']['A_strokel']),uvw_I)
       # Stroke plane twds wing plane
       return np.matmul(np.matmul(self.rotationMatrixWing_ThetatoEta(eta),self.rotationMatrixWing_PhitoTheta(theta)), np.matmul(self.rotationMatrixWing_SptoPhi(phi),uvw_sp))
