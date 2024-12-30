'''
Created on 04.08.2021

@author: jan-l
'''

import numpy as np
from usefull_functions import expm_wrapper
from scipy.linalg import inv


class Solution():
    def __init__(self,Model):
        self.Model=Model
        self.Size=Model.model_size()
    
    #Analytically solves the differential equation induced by the model and return an array containing the amount a array at each time step of t
    #start is the amount in each compartment at the start
    #t is a vector containg the times the function should be evaluated. t[0] is the time at start
    def AnaSolv(self,start,t):
        currentT=t[0]
        Solution=np.zeros(shape=(len(t),len(start)))
        Solution[0]=start
        for i in range(1,len(t)):
            while currentT<t[i]-1:
                nextT=int(currentT)+1
                start=self.MatrixDifSolv(currentT,start,nextT-currentT)
                currentT=nextT
                
            tSolution=self.MatrixDifSolv(currentT,start,t[i]-currentT)
            start=tSolution
            currentT=t[i]
            Solution[i]=tSolution

        return Solution
    
    '''
    Solves Linear Matrix equation with Input if the Neither Input or Matrix changes. 
    Changes on the matrix are only registerd at a daily basis 
    '''
    def MatrixDifSolv(self,startTime,start,t):
        Matrix=self.Model.TransitionMatrix(startTime)
        In=self.Model.Input(startTime)
        start=start.reshape((-1, 1))
        if np.all((In == 0)):
            #Solution during depuration phase
            SolutionDif=expm_wrapper(Matrix*t)@start
        else:   
            #Solution during assimilation phase
            In=np.array(In.reshape((-1, 1)))
            x=-inv(Matrix)@In
            SolutionDif=x+expm_wrapper(Matrix*t)@(start-x)
        SolutionDif=np.transpose(SolutionDif)[0]
        return SolutionDif

    
        
        
    
if __name__=="__main__":   
    from Model import wbPBTK
    import pandas as pd
    import matplotlib.pyplot as plt
    import time as timer
    
    #Testing the solver--------------------------------------------------------
    #gaining the parameters for samples
    all_samples_df=pd.read_csv('nested_bayes_all_samples.csv')
    sub='2378-TCDD'
    df=all_samples_df[all_samples_df['name']==sub]
    this_samples=df.to_dict(orient='records')
    
    #Creating the model
    this_model=wbPBTK(cont_interval=[0,1],physio_intervals=3)
    this_model.set_cont_amount(1)
    
    #Timepoints of interest
    time=np.linspace(0,300,num=301)
    #No start contamination
    start=np.zeros(this_model.model_size())
    
    #Plotting and Simulating the model
    sample_index=0
    for i in range(0,1):
        #setting the parameters of the model
        test_sample=this_samples[0]
        sample_index=sample_index+1
        this_model.set_parameters(test_sample)
        #Solving the model
        solver=Solution(this_model)
        values=solver.AnaSolv(start,time)
        milk_conc=[this_model.conc_milk(values[i], time[i]) for i in range(len(time))]
        plt.plot(time,milk_conc)
    plt.show()
        
        
        
        