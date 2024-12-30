#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 13:01:34 2023

@author: moenning
"""
import numpy as np
import functools
from usefull_functions import resettable_cache



class wbPBTK():
    def __init__(self,cont_interval,physio_intervals=1):
        self.cont_interval=cont_interval
        self.physio_intervals=physio_intervals
    
    #Decorator, which makes the given function constant on intervals. 
    #Values are always the values the function would return at the center ofthe interval
    class Deco(object):
        def simplfy_timer(func):
            @functools.wraps(func)
            def wrapper(self,*args):
                this_t=args[0]
                
                simplfied_t=int(this_t/self.physio_intervals)*self.physio_intervals
                
                list_args=list(args)
                list_args[0]=simplfied_t+self.physio_intervals/2
                
                args=tuple(list_args)
                return func(self,*args)
            return wrapper
    
    #setting the parameters and reseting the caches of function using these parameters in some way
    def set_parameters(self,parameters):
        self.parameters=parameters
        self.TransitionMatrix.reset_cache()
        self.q_blood.reset_cache()
        self.q_heart.reset_cache()
        self.q_fat.reset_cache()
        self.q_ovary.reset_cache()
        self.q_muscle.reset_cache()
        self.q_kidney.reset_cache()
        self.q_spleen.reset_cache()
        self.q_remain.reset_cache()
        self.q_brain.reset_cache()
    
    #setting the daily amount given each day during the supplementation period
    def set_cont_amount(self,cont_amount):
        self.cont_amount=cont_amount
    
    
    @Deco.simplfy_timer
    @resettable_cache()
    #Transition matrix describing the cows kinec behaviour
    def TransitionMatrix(self,t):
        matrix=np.array([[-1/self.v_blood(t)*self.q_blood(t),
                              self.q_liver(t)/(self.v_liver(t)*self.parameters["Liver"]),
                              self.q_udder(t)/(self.v_udder(t)*self.parameters["Udder"]),
                              self.q_fat(t)/(self.v_adipose(t)*self.parameters["Adipose"]),
                              self.q_muscle(t)/(self.v_muscle(t)*self.parameters["Muscle"]),
                              self.q_spleen(t)/(self.v_spleen(t)*self.parameters["Spleen"]),
                              self.q_kidney(t)/(self.v_kidney(t)*self.parameters["Kidney"]),
                              self.q_brain(t)/(self.v_brain(t)*self.parameters["Brain"]),
                              self.q_heart(t)/(self.v_heart(t)*self.parameters["Heart"]),
                              self.q_ovary(t)/(self.v_ovary(t)*self.parameters["Ovary"]),
                              self.q_remain(t)/(self.v_remain(t)*self.parameters["Rest"])],
                         [self.q_liver(t)/self.v_blood(t),
                              -self.q_liver(t)/(self.v_liver(t)*self.parameters["Liver"])-self.parameters["Met"],
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,    
                              0],
                         [self.q_udder(t)/self.v_blood(t),
                              0,
                              -self.q_udder(t)/(self.v_udder(t)*self.parameters["Udder"])-self.milk_fat(t)*self.parameters["Milk"]/self.v_udder(t),
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0],
                         [self.q_fat(t)/self.v_blood(t),
                              0,
                              0,
                              -self.q_fat(t)/(self.v_adipose(t)*self.parameters["Adipose"]),
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0],
                         [self.q_muscle(t)/self.v_blood(t),
                              0,
                              0,
                              0,
                              -self.q_muscle(t)/(self.v_muscle(t)*self.parameters["Muscle"]),
                              0,
                              0,
                              0,
                              0,
                              0,
                              0],
                         [self.q_spleen(t)/self.v_blood(t),
                              0,
                              0,
                              0,
                              0,
                              -self.q_spleen(t)/(self.v_spleen(t)*self.parameters["Spleen"]),
                              0,
                              0,
                              0,
                              0,
                              0],
                         [self.q_kidney(t)/self.v_blood(t),
                              0,
                              0,
                              0,
                              0,
                              0,
                              -self.q_kidney(t)/(self.v_kidney(t)*self.parameters["Kidney"]),
                              0,
                              0,
                              0,
                              0],
                         [self.q_brain(t)/self.v_blood(t),
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              -self.q_brain(t)/(self.v_brain(t)*self.parameters["Brain"]),
                              0,
                              0,
                              0],
                         [self.q_heart(t)/self.v_blood(t),
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              -self.q_heart(t)/(self.v_heart(t)*self.parameters["Heart"]),
                              0,
                              0],
                         [self.q_ovary(t)/self.v_blood(t),
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              -self.q_ovary(t)/(self.v_ovary(t)*self.parameters["Ovary"]),
                              0],
                         [self.q_remain(t)/self.v_blood(t),
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              -self.q_remain(t)/(self.v_remain(t)*self.parameters["Rest"])]])
        return matrix
    
    #----------------------------------------------------------------------------------------------------------
    #Volumes [kg]
    #----------------------------------------------------------------------------------------------------------
    
    #TODO: Implement a weight function
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def v_total(self,t):
        a=668.6795147117443
        b=-0.0924604624296137
        c=0.0009262829110809889
        return a+b*t+c*t**2
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def v_blood(self,t):
        return 4.52/100*self.v_total(t)
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def v_liver_tot(self,t):
        return 1.31/100*self.v_total(t)
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def v_liver(self,t):
        return self.v_liver_tot(t)*13.4/100
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def v_udder_tot(self,t):
        return 3.20/100*self.v_total(t)
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def v_udder(self,t):
        return self.v_udder_tot(t)*31.6/100
    
    #TODO: Implement a fat weight function
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def v_adipose_tot(self,t):
        a=51.02042326788586
        b=-0.25344219938703816
        c=0.0007620500873457559
        return a+b*t+c*t**2
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def v_adipose(self,t):
        return self.v_adipose_tot(t)*92.9/100
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def v_muscle_tot(self,t):
        return 28.5/100*self.v_total(t)
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def v_muscle(self,t):
        return self.v_muscle_tot(t)*12.8/100
    
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def v_spleen_tot(self,t):
        return 0.17/100*self.v_total(t)
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def v_spleen(self,t):
        return self.v_spleen_tot(t)*3.9/100
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def v_kidney_tot(self,t):
        return 0.25/100*self.v_total(t)
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def v_kidney(self,t):
        return self.v_kidney_tot(t)*12.6/100
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def v_brain_tot(self,t):
        return 0.07/100*self.v_total(t)
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def v_brain(self,t):
        return self.v_brain_tot(t)*49/100
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def v_heart_tot(self,t):
        return 0.37/100*self.v_total(t)
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def v_heart(self,t):
        return self.v_heart_tot(t)*10.5/100
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def v_ovar_tot(self,t):
        return 0.003/100*self.v_total(t)
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def v_ovary(self,t):
        return self.v_ovar_tot(t)*7.1/100
    
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def v_remain_tot(self,t):
        return self.v_total(t)*(100-(4.52+1.31+3.20+28.5+0.17+0.25+0.07+0.37+0.003))/100-self.v_adipose_tot(t)
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def v_remain(self,t):
        return self.v_remain_tot(t)*0.25/100
    
    #------------------------------------------------------------------------------
    #Bloodflow rates [L/d]
    #----------------------------------------------------------------------------------------------------------
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def cardiac_output(self,t):
        return 5.45*24*self.v_total(t)
    @Deco.simplfy_timer
    @resettable_cache()    
    def q_blood(self,t):
        return self.q_liver(t)+self.q_fat(t)+self.q_muscle(t)+self.q_spleen(t)+self.q_kidney(t)+self.q_brain(t)+self.q_heart(t)+self.q_ovary(t)+self.q_udder(t)+self.q_remain(t)

    @functools.lru_cache(maxsize=1000)    
    def q_liver(self,t):
        return 3.17*24*self.v_total(t)
    
    @functools.lru_cache(maxsize=1000)
    def q_udder(self,t):
        return 2397+23500*(1-np.exp(-0.02091*self.milk_yield(t)))
    
    @functools.lru_cache(maxsize=1000)
    def pre_q_fat(self,t):
        return np.sqrt(self.v_adipose_tot(t)/self.v_total(t)*0.64*self.cardiac_output(t)*self.v_adipose_tot(t))
    
    @Deco.simplfy_timer
    @resettable_cache()
    def q_fat(self,t):
        return self.pre_q_fat(t)*self.parameters["Adipose_blood"]
    
    @functools.lru_cache(maxsize=1000)
    def pre_q_muscle(self,t):
        return np.sqrt(7*self.v_total(t)**1.1625*self.v_muscle_tot(t))
    
    @Deco.simplfy_timer
    @resettable_cache()    
    def q_muscle(self,t):
        return self.pre_q_muscle(t)*self.parameters["All_other_blood"]
 
    
    @functools.lru_cache(maxsize=1000)
    def pre_q_spleen(self,t):
        return np.sqrt(self.v_total(t)*0.17/0.28*4.54/100*5.45*24*self.v_spleen_tot(t))

    @Deco.simplfy_timer
    @resettable_cache()    
    def q_spleen(self,t):
        return self.pre_q_spleen(t)*self.parameters["All_other_blood"]
    
    @functools.lru_cache(maxsize=1000)
    def pre_q_kidney(self,t):
        return np.sqrt(0.36*self.v_total(t)*24*self.v_kidney_tot(t))
    
    @Deco.simplfy_timer
    @resettable_cache()    
    def q_kidney(self,t):
        return self.pre_q_kidney(t)*self.parameters["All_other_blood"]

    @functools.lru_cache(maxsize=1000)
    def pre_q_brain(self,t):
        return np.sqrt(self.v_total(t)*0.08/0.21*2.95/100*5.45*24*self.v_brain_tot(t))
    
    @Deco.simplfy_timer
    @resettable_cache()
    def q_brain(self,t):
        return self.pre_q_brain(t)*self.parameters["All_other_blood"]
    
    @functools.lru_cache(maxsize=1000)
    def pre_q_heart(self,t):
        return np.sqrt(0.50571428571*self.v_total(t)*24*self.v_heart_tot(t))

    @Deco.simplfy_timer
    @resettable_cache()
    def q_heart(self,t):
        return self.pre_q_heart(t)*self.parameters["All_other_blood"]

    @functools.lru_cache(maxsize=1000)
    def pre_q_ovary(self,t):
        return np.sqrt(0.0003*self.v_total(t)*24*self.v_ovar_tot(t))

    @Deco.simplfy_timer
    @resettable_cache()    
    def q_ovary(self,t):
        return self.pre_q_ovary(t)*self.parameters["All_other_blood"]
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def pre_q_remain(self,t):
        return np.sqrt(5.45*24)*self.v_remain_tot(t)
    
    @Deco.simplfy_timer
    @resettable_cache()
    def q_remain(self,t):
        return self.pre_q_remain(t)*self.parameters["All_other_blood"]
    
        
    #----------------------------------------------------------------------------------------
    #Fluxes in and out of the cow
    #----------------------------------------------------------------------------------------
    
    #TODO: implement milk yield curve
    #Returns the amount of milk fat excreted each day
    #@Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def milk_fat(self,t):
        return self.milk_yield(t)*4.5/100
    
    #Returns the amount of excreted each day
    #based on https://www.sciencedirect.com/science/article/pii/S0022030205727843#tbl3 
    #(Woodmodel)
    @Deco.simplfy_timer
    @functools.lru_cache(maxsize=1000)
    def milk_yield(self,t):
        if t<0:
            return 0
        else:
            a=25.69573438797686
            b=0.19502964828828284
            c=-0.0032733937164936
            return (a*t**b)*np.exp(c*t)
    
    #Returns the amount of conataminant absorpt each day from the GIT of the animals
    def Input(self,t):
        if self.cont_interval[0]<=t and self.cont_interval[1]>t:
            blood_in=self.cont_amount*self.parameters["Abs"]
        else:
            blood_in=0
        this_input=np.zeros(self.model_size())
        this_input[0]=blood_in
        return this_input
        
    
    #----------------------------------------------------------------------------------------
    #Meta stuff of the model
    #----------------------------------------------------------------------------------------
    
    #Converts the array containting the amount of contaminant in each tissue in the concentration of the conaminant in milk fat
    def conc_milk(self,Array,t):
        return Array[2]/self.v_udder(t)*self.parameters["Milk"]
    
   #Converts the array containting the amount of contaminant in each tissue in the amount of contaminant excreted via milk at day t
    def amount_milk(self,Array,t):
        return Array[2]/self.v_udder(t)*self.parameters["Milk"]*self.milk_fat(t)
    
   #Converts the array containting the amount of contaminant in each tissue in the concentration of the conaminant in blood (whole blood)
    def conc_blood(self,Array,t):
        return Array[0]/self.v_blood(t)
    
    #Converts the array containting the amount of contaminant in each tissue in the concentration of the conaminant in a specific organ (fat basis)
    def conc_organ(self,Array,t,organ):
        loc=self.compartment_loc()[organ]
        return Array[loc]/eval("self.v_"+organ.lower())(t)
    
    #Converts the array containting the amount of contaminant in each tissue in the concentration of the conaminant in a specific organ (fat basis)
    def amount_organ(self,Array,t,organ):
        loc=self.compartment_loc()[organ]
        return Array[loc]
    
    #size of the model
    @staticmethod
    def model_size():
        return 11
    
    #return a dictionary containing the index of each compartment in the array induced by the model 
    @staticmethod
    def compartment_loc():
        return {"Blood":0,
                "Liver":1,
                "Udder":2,
                "Adipose":3,
                "Muscle":4,
                "Spleen":5,
                "Kidney":6,
                "Brain":7,
                "Heart":8,
                "Ovary":9,
                "Remain":10}

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    
    this_model=wbPBTK(cont_interval=[0,100],physio_intervals=20)
    
    t=np.linspace(0,300,num=301)
    values=[this_model.milk_fat(this_t) for this_t in t]
    
    plt.plot(t,values)
    
    
    
    
    
    
    
    
    
    
    
    
    