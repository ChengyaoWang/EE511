#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 4 15:59:41 2019
Project5 for EE511
Author:Chengyao Wang
USCID:6961599816
Contact Email:chengyao@usc.edu
"""
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import scipy.stats as pdfset
import seaborn as sns

def target_func(x):
    if int(x>0.0)&int(x<1.0):
        return 0.6*pdfset.beta.pdf(x,1,8)+0.4*pdfset.beta.pdf(x,9,1)
    else:
        return 0
def ideal_dis(total):
    ideal_distribution=[0.0]*10
    for i in range(0,10):
        ideal_distribution[i]+=0.6*(pdfset.beta.cdf(0.1*i+0.1,1,8)-pdfset.beta.cdf(0.1*i,1,8))
        ideal_distribution[i]+=0.4*(pdfset.beta.cdf(0.1*i+0.1,9,1)-pdfset.beta.cdf(0.1*i,9,1))
        ideal_distribution[i]*=total
    return ideal_distribution
#Kernel 1: Standard Normal(0,1)
#Kernel 2: Exponential(0.25)
#Kernel 3: Standard Cauchy
#Kernel 4: Normal(0,10)
#Kernel 5: Normal(0,0.1)
def kernel(xt,k_type):
    if k_type==1:
        while 1:
            x=np.random.normal(xt,1,1)
            if int(x<1)&int(x>0):
                return x
    elif k_type==2:
        while 1:
            dx=np.random.exponential(4)
            if rd.uniform(0,1)<0.5:
                if (xt+dx)<1:
                    return xt+dx
            else:
                if (xt-dx)>0:
                    return xt-dx
    elif k_type==3:
        while 1:
            x=xt+np.random.standard_cauchy(1)
            if int(x<1)&int(x>0):
                return x
    elif k_type==4:
        while 1:
            x=np.random.normal(xt,10,1)
            if int(x<1)&int(x>0):
                return x
    elif k_type==5:
        while 1:
            x=np.random.normal(xt,0.1,1)
            if int(x<1)&int(x>0):
                return x
def scwefel_2(x,y):
    return 418.9829*2-x*np.sin(np.sqrt(np.abs(x)))-y*np.sin(np.sqrt(np.abs(y)))
#Cooling Schedule1: Exponential
#Cooling Schedule2: Polynomial(-0.1)
#Cooling Schedule3: Polynomial(-0.5)
#Cooling Schedule4: Polynomial(-0.9)
#Cooling Schedule5: Logarithmic
def cooling(i,choose):
    if choose==1:
        return np.exp(-i)
    elif choose==2:
        return i**(-0.1)
    elif choose==3:
        return i**(-0.5)
    elif choose==4:
        return i**(-0.9)
    elif choose==5:
        return 1/np.log(1+i)
def distance(city,city_number):
    d=0
    for i in range(1,city_number):
        d+=np.sqrt((city[0][i-1]-city[0][i])**2+(city[1][i-1]-city[1][i])**2)
    return d
def city_generate(city_number):
    city=np.zeros((2,city_number))
    for i in range(0,city_number):
        city[:,i]=rd.uniform(0,1000),rd.uniform(0,1000)
    return city
def city_swap(city,switch1,switch2,city_number):
    new_city=np.zeros((2,city_number))
    for i in range(0,city_number):
        if (i!=switch1)&(i!=switch2):
            new_city[0][i]=city[0][i]
            new_city[1][i]=city[1][i]
        elif i==switch1:
            new_city[0][i]=city[0][switch2]
            new_city[1][i]=city[1][switch2]
        else:
            new_city[0][i]=city[0][switch1]
            new_city[1][i]=city[1][switch1]
    return new_city
def Gibbs_energy(e1,e0,t):
    return np.exp((-e1+e0)/t)
#Generate a Sample MCMC path via MH
#Kernel:Standard Normal
#Total Time:50000
#Inital Points:0.5
def func_1a():
    sample_path=[0.0]*50000
    sample_path[0]=0.5
    for i in range(1,50000):
        current_state=sample_path[i-1]
        potential_next=kernel(current_state,1)
        A=min(1,target_func(potential_next)/target_func(current_state))
        if rd.uniform(0,1)<A:
            sample_path[i] = potential_next
        else:
            sample_path[i] = current_state
    print "The initial Point of this Sample Path is", sample_path[0]
    plt.hist(sample_path,20)
    plt.title("MH Sampling With Kernal: Standard Normal")
    plt.show()
    plt.scatter(np.arange(0,50000),sample_path,s=1)
    plt.title("MCMC Sample Path")
    plt.show()
#Generate Sample MCMC paths via MH with different Inital Points
#Repetition:5 times 
#Kernel:Standard Normal
#Total Time:20000
#Samples Tested with GOF: 15000~20000
#Inital Points: Random(0,1)
def func_1b():
    ideal_distribution=ideal_dis(5000)
    for repeat in range(0,5):
        sample_path=[0.0]*20000
        sample_path[0]=rd.uniform(0,1)
        observed_distribution=[0.0]*10
        chi2_result=0
        for i in range(1,20000):
            current_state=sample_path[i-1]
            potential_next=kernel(current_state,1)
            A=min(1,target_func(potential_next)/target_func(current_state))
            if rd.uniform(0,1)<A:
                sample_path[i] = potential_next
            else:
                sample_path[i] = current_state
        #Count the Observed Distribution in 5 bins
        for i in range(0,10):
            up_bound=0.1*i+0.1
            low_bound=0.1*i
            for j in range(15000,20000):
                if int(low_bound<sample_path[j]) & int(sample_path[j]<up_bound):
                    observed_distribution[i]+=1
        #Print out Results
        print "The initial Point of the",repeat+1,"th Sample Path is", sample_path[0]
        plt.hist(sample_path,10)
        plt.title("MH Sampling with different Starting Points\nKernel: Standard Normal\nDistribution of the Entire Sample Path")
        plt.show()
        plt.scatter(np.arange(0,20000),sample_path,s=1)
        plt.title("Entire Sample Path")
        plt.show()
        arr1,arr2=np.split(sample_path,[15000])
        plt.hist(arr2,10)
        plt.title("MH Sampling with different Starting Points\nKernel: Standard Normal\nDistribution of the Last 5000 Sample")
        plt.show()
        plt.scatter(np.arange(0,5000),arr2,s=1)
        plt.title("Last 5000 Points of the Sample Path")
        plt.show()
        #Chi_Square Goodness of fit
        #Degree of Freedom 9, 16.92
        for i in range(0,10):
            chi2_result+=(ideal_distribution[i]-observed_distribution[i])**2/ideal_distribution[i]
        print "The Chi_Square Result of the",repeat+1,"th Sample is",chi2_result
        if chi2_result<16.92:
            print "The",repeat+1,"th Path CAN be Considered as convergent\n\n"
        else:
            print "The",repeat+1,"th Path CANNOT be Considered as convergent\n\n"
    plt.scatter(np.linspace(0,1,10),ideal_distribution,s=100)
    plt.title("Ideal Distribution of Target PDF")
    plt.show()
#Generate Sample MCMC paths via MH with different Kernels
#Kernel 1: Standard Normal(0,1)
#Kernel 2: Exponential(0.25)
#Kernel 3: Standard Cauchy
#Kernel 4: Normal(0,25)
#Kernel 5: Normal(0,0.04)
#Total Time:10000
#Samples Tested with GOF: 9000~10000
#Inital Points: 0.5
def func_1c():
    ideal_distribution=ideal_dis(5000)
    for repeat in range(1,6):
        sample_path=[0.0]*20000
        sample_path[0]=0.5#rd.uniform(0,1)
        observed_distribution=[0.0]*10
        chi2_result=0
        for i in range(1,20000):
            current_state=sample_path[i-1]
            potential_next=kernel(current_state,repeat)
            A=min(1,target_func(potential_next)/target_func(current_state))
            if rd.uniform(0,1)<A:
                sample_path[i] = potential_next
            else:
                sample_path[i] = current_state
        #Count the Observed Distribution in 5 bins
        for i in range(0,10):
            up_bound=0.1*i+0.1
            low_bound=0.1*i
            for j in range(15000,20000):
                if int(low_bound<sample_path[j]) & int(sample_path[j]<up_bound):
                    observed_distribution[i]+=1
        #Print out Results
        plt.hist(sample_path,10)
        if repeat==1:
            plt.title("MH Sampling with Kernel: Standard Normal")
        elif repeat==2:
            plt.title("MH Sampling with Kernel: Exp(0.25)")
        elif repeat==3:
            plt.title("MH Sampling with Kernel: Standard Cauchy")
        elif repeat==4:
            plt.title("MH Sampling with Kernel: Normal(0,10)")
        elif repeat==5:
            plt.title("MH Sampling with Kernel: Normal(0,0.1)")
        plt.show()
        plt.scatter(np.arange(0,20000),sample_path,s=1)
        plt.title("MCMC Sample Path")
        plt.show()
        arr1,arr2=np.split(sample_path,[15000])
        plt.hist(arr2,10)
        plt.title("MH Sampling with different Starting Points\nDistribution of the Last 5000 Sample")
        plt.show()
        plt.scatter(np.arange(0,5000),arr2,s=1)
        plt.title("Last 5000 Points of the Sample Path")
        plt.show()
        #Chi_Square Goodness of fit
        #Degree of Freedom 9, 16.92
        for i in range(0,10):
            chi2_result+=(ideal_distribution[i]-observed_distribution[i])**2/ideal_distribution[i]
        print "The Chi_Square Result of this Sample is",chi2_result
        if chi2_result<16.92:
            print "The Path CAN be Considered as convergent\n\n"
        else:
            print "The Path CANNOT be Considered as convergent\n\n"
#Plot a Contour Plot for 2-D Scwefel Function
#Save as a file for a higher resolution
def func_2a():
    x=np.linspace(-500,500,1000)
    y=np.linspace(-500,500,1000)
    X,Y=np.meshgrid(x,y)
    height=np.zeros((1000,1000))
    for i in range(0,1000):
        x0=-500+i
        for j in range(0,1000):
            y0=-500+j
            height[j][i]=scwefel_2(x0,y0)
    plt.contourf(X,Y,height,15)
    C=plt.contour(X,Y,height,15,colors='black')
    plt.clabel(C,s=5)
    plt.title("Contour Plot of Scwefel Function")
    plt.grid(True)
    plt.savefig('contour_plot.png', dpi=300)
#Simulated Annealing, with Candidate proposal Routine Bivariate Normal N(0,300)
#Initial Tempreture: 1000
#Cooling Schedule: Polynomial a=-0.5
#Total Time: 10000
#Starting Point: (0,0)
def func_2b():
    sample_path=np.zeros((2,10000))
    tempreture=[0.0]*10000
    energy=[0.0]*10000
    for i in range(1,10000):
        tempreture[i]=1000*cooling(i,3)
        current_x=sample_path[0][i-1]
        current_y=sample_path[1][i-1]
        while 1:
            candidate_x=np.random.normal(current_x,300)
            candidate_y=np.random.normal(current_y,300)
            if int(candidate_x>-500)&int(candidate_x<500):
                if int(candidate_y>-500)&int(candidate_y<500):
                    break
        e0=scwefel_2(current_x,current_y)
        e1=scwefel_2(candidate_x,candidate_y)
        A=min(1,Gibbs_energy(e1,e0,tempreture[i]))
        if rd.uniform(0,1)<A:
            sample_path[0][i]=candidate_x
            sample_path[1][i]=candidate_y
            energy[i]=e1
        else:
            sample_path[0][i]=current_x
            sample_path[1][i]=current_y
            energy[i]=e0
    print "The Sample Path Ends in:(",sample_path[0,9999],",",sample_path[1,9999],")"
    print "With Function Value:",energy[9999]
    plt.plot(sample_path[0,:],sample_path[1,:])
    plt.title("Sample Path")
    plt.show()
    plt.scatter(np.arange(0,10000),tempreture,s=1)
    plt.title("Cooling Schedule: Polynomial(-0.5)")
    plt.show()
    plt.scatter(np.arange(0,10000),energy,s=5)
    plt.title("Energy Trace")
    plt.show()
    return (sample_path[0,9999],sample_path[1,9999])
#Simulated Annealing, with Candidate proposal Routine Bivariate Normal N(0,300)
#Initial Tempreture: 1000
#Cooling Schedule1: Exponential
#Cooling Schedule2: Polynomial(-0.1)
#Cooling Schedule3: Polynomial(-0.5)
#Cooling Schedule4: Polynomial(-0.9)
#Cooling Schedule5: Logarithmic
#Total Time: 20/50/100/1000
#Starting Point: (0,0)
def func_2c(total_steps,schedule_num):
    data=np.zeros((2,100))
    energy=[0.0]*100
    min_energy=10000
    for repetition in range(0,100):
        path=np.zeros((2,total_steps))
        current_x=0
        current_y=0
        for i in range(1,total_steps):
            t=1000*cooling(i,schedule_num)
            #Next Candidate, have boarders [-500,500]
            while 1:
                candidate_x=np.random.normal(current_x,300)
                candidate_y=np.random.normal(current_y,300)
                if int(candidate_x>-500)&int(candidate_x<500):
                    if int(candidate_y>-500)&int(candidate_y<500):
                        break
            #Calculate Energy
            e0=scwefel_2(current_x,current_y)
            e1=scwefel_2(candidate_x,candidate_y)
            A=min(1,Gibbs_energy(e1,e0,t))
            energy[repetition]=e0
            if rd.uniform(0,1)<A:
                (current_x,current_y)=(candidate_x,candidate_y)
                path[:,i]=(candidate_x,candidate_y)
            else:
                path[:,i]=(current_x,current_y)
        energy[repetition]=scwefel_2(current_x,current_y)
        if min_energy>energy[repetition]:
            optimal_path=path
            min_energy=energy[repetition]
        data[0][repetition]=current_x
        data[1][repetition]=current_y
    print "-----------------------------------------------"
    print "Total_Steps=",total_steps,"Cooling Schedule=",schedule_num
    plt.scatter(data[0,:],data[1,:])
    plt.title("The Final Point of 100 runs with the above condition")
    plt.xlim(-500,500)
    plt.ylim(-500,500)
    plt.show()    
    plt.hist(energy,bins=20)
    plt.title("Distribution of the minimum of 100 samples")
    plt.show()
    with sns.axes_style("dark"):
        sns.jointplot(data[0,:],data[1,:],marginal_kws=dict(bins=20))
    plt.show()
    return optimal_path
#Go through all Cooling_Schedules and Total_steps
def func_2d():
    for i in range(1,6):
        func_2c(20,i)
    for i in range(1,6):
        func_2c(50,i)
    for i in range(1,6):
        func_2c(100,i)
    for i in range(1,6):
        func_2c(1000,i)
#Trajectory of the best run on the Contour Plot
#According to Results from Func_2d, Choose:
#Total Steps=1000, Cooling Schedule: Polynomial(-0.9)
def func_2e():
    optimal_path=func_2c(1000,4)
    print optimal_path
    x=np.linspace(-500,500,1000)
    y=np.linspace(-500,500,1000)
    X,Y=np.meshgrid(x,y)
    height=np.zeros((1000,1000))
    for i in range(0,1000):
        x0=-500+i
        for j in range(0,1000):
            y0=-500+j
            height[j][i]=scwefel_2(x0,y0)
    plt.contourf(X,Y,height,15)
    C=plt.contour(X,Y,height,15,colors='black')
    plt.clabel(C,s=5)
    plt.title("Contour Plot of Scwefel Function")
    plt.grid(True)
    plt.plot(optimal_path[0,:],optimal_path[1,:],color='red')
    plt.savefig('contour_plot with Trajectory.png', dpi=300)
#Traveling Salesman Problem (TSP)
def func_3a(city_number):
    city=city_generate(city_number)
    distance_store=[0.0]*10000
    distance_store[0]=distance(city,city_number)
    count=0
    i=0
    total_step=0
    while 1:
        t=1000*cooling(i+1,3)
        #ensure the two cities are not the same
        while True:
            switch1=rd.randint(0,city_number-1)
            switch2=rd.randint(0,city_number-1)
            if switch1!=switch2:
                break
        new_city=city_swap(city,switch1,switch2,city_number)
        new_distance=distance(new_city,city_number)
        A=min(1,Gibbs_energy(new_distance,distance_store[total_step],t))
        if rd.uniform(0,1)<A:
            city=new_city
            distance_store[total_step+1]=new_distance
            total_step+=1
            count=0
        else:
            count+=1
        if count==500:
            break
        i+=1
    print "---------------------------------------------------------"
    print "Minimun distance Found",distance_store[total_step],"at Step:",total_step
    arr1,arr2=np.split(distance_store,[total_step])
    plt.scatter(np.arange(0,total_step),arr1,s=10)
    plt.title("Values of objective function from each step")
    plt.show()
    plt.plot(city[0,:],city[1,:])
    plt.scatter(city[0,:],city[1,:],s=50)
    plt.title("Optimal City Tour Map")
    plt.show()
#func_1a()
#func_1b()
#func_1c()
#func_2a()
#func_2b()
#func_2c(1000,4)
#func_2d()
#func_2e()
#func_3a(10)
#func_3a(40)
#func_3a(400)
#func_3a(1000)
