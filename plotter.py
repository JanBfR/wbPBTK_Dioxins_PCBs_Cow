#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:16:08 2024

@author: moenning
"""

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Model import wbPBTK
from Solve import Solution

def plot_ct_profile(contamination_dic,cont_interval,physio_intervals=30,adding_type="TEQ",sample_size=1000,comps="Milk"):
    prediction_dic=None
    sub_keys=list(contamination_dic.keys())
    for p in sub_keys:
        if contamination_dic[p]==0:
            try:
                del contamination_dic[p]
            except KeyError:
                pass
    model=wbPBTK(cont_interval=cont_interval,physio_intervals=physio_intervals)
    all_samples_df=pd.read_csv('nested_bayes_all_samples.csv')
    for sub in contamination_dic:
        model.set_cont_amount(contamination_dic[sub])
        if adding_type=="TEQ":
            factor=get_TEF(sub)
        elif adding_type=="SUM":
            factor=1
        
        df=all_samples_df[all_samples_df['name']==sub]
        samples=df.to_dict(orient='records')
        samples=random.sample(samples,sample_size)
        t=np.linspace(0,300,num=301)
        if isinstance(prediction_dic, type(None)):
            prediction_dic=calc_ct_profiles(samples,t,comps,model)
            prediction_dic={key: prediction_dic[key]*factor for key in prediction_dic}
        elif factor!=0:
            new_prediction_dic=calc_ct_profiles(samples,t,comps,model)
            prediction_dic={key: new_prediction_dic*factor+prediction_dic for key in prediction_dic}
    for comp in comps:
        prediction_list=prediction_dic[comp]
        min_list=[]
        max_list=[]
        median_list=[]
        for i in range(len(prediction_list[0])):
            today_predicitons=[this_predict[i] for this_predict in prediction_list]
            
            min_list.append(np.percentile(today_predicitons, 2.5))
            max_list.append(np.percentile(today_predicitons, 97.5))
            median_list.append(np.percentile(today_predicitons, 50))
        
        #Creating the plot
        plt.plot(t,median_list,"r-",label="Median")
        plt.plot(t,min_list,"b-")
        plt.plot(t,max_list,"b-")
        plt.fill_between(t, min_list, max_list,  alpha=0.4,color="b",label="Credible interval")
        plt.grid()
        plt.legend()
        #Creating title of the plot
        if len(contamination_dic)==1:
            title=list(contamination_dic.keys())[0]+"-"
        else:
            if adding_type=="TEQ":
                title="WHO-TEQ-"
            elif adding_type=="SUM":
                title="Sum dioxins and PCBs-"
        title=title+comp
        if comp!="Blood":
            title=title+" (fat basis)"
        plt.title(title)
        #Creating the axis description
        plt.xlabel("Time in days")
        if adding_type=="TEQ":
            plt.ylabel("Concentration in ng/kg TEQ")
        elif adding_type=="Sum":
            plt.ylabel("Concentration in ng/kg")
        plt.show()
    
    
            
def calc_ct_profiles(samples,t,comps,model):
    start=np.zeros(model.model_size())
    comp_prediction_dic={key: [] for key in comps}
    for sample in samples:
        model.set_parameters(sample)
        solver=Solution(model)
        values=solver.AnaSolv(start,t)
        for comp in comps:
            if comp=="Milk":
                predict=np.array([model.conc_milk(values[i],t[i]) for i in range(len(t))])
            else:
                predict=np.array([model.conc_organ(values[i],t[i],comp) for i in range(len(t))])
            comp_prediction_dic[comp].append(predict)
    return comp_prediction_dic


def get_TEF(sub):
    dic={"2378-TCDD": 1,
        "12378-PeCDD": 1,
        "123478-HxCDD": 0.1,
        "123678-HxCDD ": 0.1,
        "123789-HxCDD": 0.1,
        "1234678-HpCDD": 0.01,
        "OCDD": 0.0003,
        "2378-TCDF": 0.1,
        "12378-PeCDF": 0.03,
        "23478-PeCDF": 0.3,
        "123478-HxCDF": 0.01,
        "123678-HxCDF": 0.01,
        "123789-HxCDF": 0.01,
        "234678-HxCDF": 0.01,
        "1234678-HpCDF": 0.01,
        "1234789-HpCDF": 0.01,
        "OCDF": 0.0003,
        "PCB-138": 0,
        "PCB-153": 0,
        "PCB-180": 0,
        "PCB-77": 0.0001,
        "PCB-81": 0.0003,
        "PCB-105": 3e-05,
        "PCB-114": 3e-05,
        "PCB-118": 3e-05,
        "PCB-123": 3e-05,
        "PCB-126": 0.1,
        "PCB-156": 3e-05,
        "PCB-157": 3e-05,
        "PCB-167": 3e-05,
        "PCB-169": 0.03,
        "PCB-189": 3e-05}
    return dic[sub]

if __name__ == '__main__':
    #Amount of each contaminant consumed each day during con_interval    
    contamination_dic={'2378-TCDD': 1, 
            '12378-PeCDD':0, 
            '123478-HxCDD':0, 
            '123678-HxCDD ':0, 
            '123789-HxCDD':0, 
            '1234678-HpCDD':0, 
            '2378-TCDF':0, 
            '12378-PeCDF':0, 
            '23478-PeCDF':0, 
            '123478-HxCDF':0, 
            '123678-HxCDF':0, 
            '234678-HxCDF':0, 
            '1234678-HpCDF':0, 
            '1234789-HpCDF':0, 
            'PCB-138':0,
            'PCB-153':0,
            'PCB-180':0,
            'PCB-77':0,
            'PCB-81':0,
            'PCB-105':0,
            'PCB-114':0,
            'PCB-118':0,
            'PCB-123':0,
            'PCB-126':0,
            'PCB-156':0,
            'PCB-157':0,
            'PCB-167':0,
            'PCB-169':0,
            'PCB-189':0}
    
    plot_ct_profile(contamination_dic,
                    cont_interval=[0,100],#Timeframe in which the animal was exposed to the contamination mix
                    physio_intervals=3,#Update frequency of the physiology of the animal, i.e. every physio_intervals the physiology of the animal is updated
                    adding_type="TEQ",#Type of adding all contaminants togheter either TEQ or SUM
                    sample_size=100,#How many samples should be used to derive the distribution of each congener ct-profile. Maximum is 3000
                    comps=["Milk","Blood","Adipose"])#Type of matrix which should be plotted
    
    #If only one congenere is of interest set adding_type="SUM" and let contamination dic
    #contain only the congenere of interest with non-zero value