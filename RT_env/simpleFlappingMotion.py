  # -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 11:56:58 2020

@author: romai
"""

import numpy as np
from simpleFlappingProperties import simpleFlappingPropertiesEnv


class simpleFlappingMotionEnv():
    
    def __init__(self,simpleFlappingProperties):
        self.simpleFlappingPropertiesEnv = simpleFlappingProperties # load global properties  
    
# !. Make it work for scalar entry (e.g. t=0.5) or array entry (e.g. t = linspace(0,0.5,0.05))
# Left and right wing are independent     
    def computeMotion_lr(self,t):
        kinematicMotionType = self.simpleFlappingPropertiesEnv.properties['motion']['kinematicMotionType']
        f              = self.simpleFlappingPropertiesEnv.properties['wing']['frequency']
        A_phil          = self.simpleFlappingPropertiesEnv.properties['motion']['A_phil'];
        A_alphal        = self.simpleFlappingPropertiesEnv.properties['motion']['A_alphal'];


        tstar= t*f
    

        
        if kinematicMotionType == 'Dickinson':
            # phi
            factortanh_phi = self.simpleFlappingPropertiesEnv.properties['motion']['Kphi']; #0.99 #0.8
            #print(factortanh_phi)
            A_phil_rad = np.deg2rad(A_phil)
            phi_rad    =  A_phil_rad/np.arcsin(factortanh_phi) * np.arcsin(factortanh_phi*np.sin(2*np.pi*(tstar+0.25)))  
            phid_rad   =  A_phil_rad/np.arcsin(factortanh_phi) * 2*np.pi*f*factortanh_phi * np.cos(2*np.pi*(tstar+0.25)) / (np.sqrt(1-(factortanh_phi**2)*(np.sin(2*np.pi*(tstar+0.25)))**2)) 
            phidd_rad   = (4*np.pi**2*A_phil_rad*factortanh_phi**3*f**2*np.sin(2*np.pi*(tstar+0.25))* (np.cos(2*np.pi*(tstar+0.25)))**2)/(np.arcsin(factortanh_phi)*(1-factortanh_phi**2*(np.sin(2*np.pi*(tstar+0.25)))**2)**1.5) -  (4*np.pi**2*A_phil_rad*factortanh_phi*f**2*np.sin(2*np.pi*(tstar+0.25)))/(np.arcsin(factortanh_phi)*(1-factortanh_phi**2*(np.sin(2*np.pi*(tstar+0.25)))**2)**0.5)               
            
            philFull_rad = np.concatenate(([phi_rad],[phid_rad],[phidd_rad]))
            phirFull_rad = philFull_rad            
            # alpha
            factortanh = self.simpleFlappingPropertiesEnv.properties['motion']['Kalpha']; #8#3
            A_alphal_rad = np.deg2rad(A_alphal)
            alpha_rad    =  (A_alphal_rad/np.tanh(factortanh))*np.tanh(factortanh*np.sin(2*np.pi*tstar)) 
            alphad_rad   = 2*np.pi*f*(A_alphal_rad/np.tanh(factortanh))*factortanh*np.cos(2*np.pi*tstar)/np.cosh(factortanh*np.sin(2*np.pi*tstar))**2 #  np.concatenate((np.array([0]),(alpha_rad[1:]-alpha_rad[0:-1])/(t[1:] - t[0:-1]))) # #(1 - (np.tanh(factortanh*np.sin(2*np.pi*tstar)))**2)*A_alphal_rad*(factortanh*2*np.pi*f)*np.cos(2*np.pi*tstar)
            alphadd_rad  = -4 * np.pi**2 * f**2 *(A_alphal_rad/np.tanh(factortanh))*factortanh* 1/np.cosh(factortanh*np.sin(2*np.pi*tstar))**2 * (2*factortanh*np.cos(2*np.pi*tstar)**2 * np.tanh(factortanh*np.sin(2*np.pi*tstar)) + np.sin(2*np.pi*tstar))  #np.concatenate((np.array([0]),(alphad_rad[1:]-alphad_rad[0:-1])/(t[1:] - t[0:-1]))) #-(((factortanh*2*np.pi*f)*np.cos(2*np.pi*tstar))**2) * A_alphal_rad* (2*np.sinh(factortanh*np.sin(2*np.pi*tstar))/np.cosh(factortanh*np.sin(2*np.pi*tstar))**3)
     
            alphalFull_rad = np.concatenate(([alpha_rad],[alphad_rad],[alphadd_rad]))
            alpharFull_rad = alphalFull_rad
            return philFull_rad, phirFull_rad, alphalFull_rad, alpharFull_rad
        
        
        elif kinematicMotionType == 'SHM_midStart_figure8': # Simple Harmonic Motion, 3 dof. t=0: middle of the downstroke
            # phi
            # Left wing
            fl = f
            A_phil_rad = np.deg2rad(A_phil)
            phi_rad   = -(A_phil_rad)*np.sin(2*np.pi*fl*t)
            phid_rad  = -(A_phil_rad)*2*np.pi*fl*np.cos(2*np.pi*fl*t)
            phidd_rad = (A_phil_rad)*((2*np.pi*fl)**2)*np.sin(2*np.pi*fl*t)
            philFull_rad = np.concatenate(([phi_rad],[phid_rad],[phidd_rad]))
            # Right wing
            # phi_rad   = -(A_phir)*np.sin(2*np.pi*fr*t)
            # phid_rad  = -(A_phir)*2*np.pi*fr*np.cos(2*np.pi*fr*t)
            # phidd_rad =  (A_phir)*((2*np.pi*fr)**2)*np.sin(2*np.pi*fr*t)
            # phirtFull_rad = np.concatenate(([phi_rad],[phid_rad],[phidd_rad]))
            # alpha
            #Left wing
            A_alphal_rad = np.deg2rad(A_alphal)
            alpha_rad    =   (A_alphal_rad)*np.cos(2*np.pi*fl*t)
            alphad_rad   =   -(A_alphal_rad)*(2*np.pi*fl)*np.sin(2*np.pi*fl*t)
            alphadd_rad  =   -(A_alphal_rad)*((2*np.pi*fl)**2)*np.cos(2*np.pi*fl*t)
            alphalFull_rad = np.concatenate(([alpha_rad],[alphad_rad],[alphadd_rad]))   
                        
            #Right wing
            # alpha_rad    =   (A_alphar)*np.cos(2*np.pi*fr*t)
            # alphad_rad   =   -(A_alphar)*(2*np.pi*fr)*np.sin(2*np.pi*fr*t)
            # alphadd_rad  =   -(A_alphar)*((2*np.pi*fr)**2)*np.cos(2*np.pi*fr*t)
            # alpharFull_rad = np.concatenate(([alpha_rad],[alphad_rad],[alphadd_rad]))
            
            # Deviation angle
            #Left wing
            # fl_theta = 2*fl
            # theta_rad    =    -(A_devl)*np.sin(2*np.pi*fl_theta*t)  #+  A_strokel
            # thetad_rad    =   -(A_devl)*(2*np.pi*fl_theta)*np.cos(2*np.pi*fl_theta*t)
            # thetadd_rad    =    (A_devl)*((2*np.pi*fl_theta)**2)*np.sin(2*np.pi*fl_theta*t)
            # thetalFull_rad =  np.concatenate(([theta_rad],[thetad_rad],[thetadd_rad]))  # np.zeros(3)
            # #Right wing
            # fr_theta = 2*fr
            # theta_rad    =    -(A_devr)*np.sin(2*np.pi*fr_theta*t)  #+  A_stroker
            # thetad_rad    =   -(A_devr)*(2*np.pi*fr_theta)*np.cos(2*np.pi*fr_theta*t)
            # thetadd_rad    =    (A_devr)*((2*np.pi*fr_theta)**2)*np.sin(2*np.pi*fr_theta*t)
            # thetarFull_rad = np.concatenate(([theta_rad],[thetad_rad],[thetadd_rad]))  #np.zeros(3)  
            phirFull_rad = philFull_rad            
            alpharFull_rad = alphalFull_rad
              
            return philFull_rad, phirFull_rad, alphalFull_rad, alpharFull_rad #, thetalFull_rad, thetarFull_rad

        
            