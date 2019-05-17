#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 12:11:18 2019
Project3 for EE511
Author:Chengyao Wang
USCID:6961599816
Contact Email:chengyao@usc.edu
"""
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math
from scipy.stats import expon
from scipy.stats import norm
from scipy.stats import beta

def func_1a(repeat_times,total_sample_number,plot):
    x=[0.0]*total_sample_number
    y=[0.0]*total_sample_number
    area=[0.0]*repeat_times
    sample_mean=0.0
    sample_variance=0.0
    for repeat in range(0,repeat_times):
        for i in range(0,total_sample_number):
            x[i]=rd.uniform(0,1)
            y[i]=rd.uniform(0,1)
            area[repeat]+=((x[i]**2+y[i]**2)<=1)&(((x[i]-1)**2+(y[i]-1)**2)<=1)
        area[repeat]/=total_sample_number
        sample_mean+=area[repeat]/repeat_times
    for i in range(0,repeat_times):
        sample_variance+=(area[i]-sample_mean)**2/repeat_times
    if plot==1:
        plt.scatter(np.arange(0,repeat_times),area)
        plt.title("Area estimation of the samples")
        plt.xlabel("Samples")
        plt.ylabel("Area Estimation")
        plt.grid(True)
        plt.show()
        print "The average of this ",repeat_times," samples is: ",sample_mean
        print "The sample variance of this ",repeat_times," samples is :",sample_variance
    return sample_variance
    print "\n\n\n\n"

def func_1b():
    variance=[0]*100
    for i in range(1,101):
        variance[i-1]=func_1a(50,500*i,0)
    plt.scatter(np.arange(500,50500,500),variance,s=1)
    plt.title("Variance-Number of Samples")
    plt.xlabel("Number of Samples")
    plt.ylabel("Variance")
    plt.ylim(ymin=-0.00075)
    plt.ylim(ymax=0.00075)
    plt.grid(True)
    plt.show()
    
def function1(x):
    return 1/(1+np.sinh(x)*np.log(x))
def function2(x,y):
    return np.exp(-x*x*x*x-y*y*y*y)
def function3(x,y):
    return 20+x**2+y**2-10*(np.cos(2*math.pi*x)+np.cos(2*math.pi*y))

def func_2a1(repeat_times,sample_number):
    result=[0]*repeat_times
    result_mean=0
    result_variance=0
    for i in range(0,repeat_times):
        for j in range(0,sample_number):
            result[i]+=2.2*function1(rd.uniform(0.8,3))/sample_number
        result_mean+=result[i]/repeat_times
    for i in range(0,repeat_times):
        result_variance+=(result[i]-result_mean)**2
    result_variance/=repeat_times
    print "The Variance of the 50 samples is ",result_variance
    print "The average of this",repeat_times,"samples is: ",result_mean
    plt.scatter(np.arange(0,repeat_times),result)
    plt.title("Function 1 with standard Monte Carlo Estimation")
    plt.xlabel("Trial")
    plt.ylabel("Estimation Result")
    plt.grid(True)
    plt.show()
    print "\n\n\n\n"
def func_2a2(repeat_times,sample_number):
    result=[0]*repeat_times
    result_mean=0
    result_variance=0
    for i in range(0,repeat_times):
        for j in range(0,sample_number):
            result[i]+=39.478*function2(rd.uniform(-math.pi,math.pi),rd.uniform(-math.pi,math.pi))/sample_number
        result_mean+=result[i]/repeat_times
    for i in range(0,repeat_times):
        result_variance+=(result[i]-result_mean)**2
    result_variance/=repeat_times
    print "The Variance of the 50 samples is ",result_variance
    print "The average of this",repeat_times,"samples is: ",result_mean
    plt.scatter(np.arange(0,repeat_times),result)
    plt.title("Function 2 with standard Monte Carlo Estimation")
    plt.xlabel("Trial")
    plt.ylabel("Estimation Result")
    plt.grid(True)
    plt.show()
    print "\n\n\n\n"
def func_2b1(repeat_times,sample_number):
    #use pdf of exponential distribution to determine the importance of each interval
    #determine the parameter for exponential pdf 
    lamda=1.0/(np.log(function1(0.8)/function1(1.8)))
    sample_allocation=[0]*10
    for i in range(0,10):
        x1=expon.cdf(0.22*i+0.8,loc=0,scale=lamda)
        x2=expon.cdf(0.22*i+1.02,loc=0,scale=lamda)
        sample_allocation[i]=sample_number*(x2-x1)
    condition=expon.cdf(3,loc=0,scale=lamda)-expon.cdf(0.8,loc=0,scale=lamda)
    sum_allocation=0
    for i in range(0,9):
        sample_allocation[9-i]=int(sample_allocation[9-i]/condition)
        sum_allocation+=sample_allocation[9-i]
    sample_allocation[0]=sample_number-sum_allocation
    print "The allocation of Sample Numbers Derived from Exponential pdf with Lamda=",lamda,"is:\n"
    print sample_allocation
    result=[0]*repeat_times
    result_mean=0
    result_variance=0
    for i in range(0,repeat_times):
        for j in range(0,10):
            for k in range(0,sample_allocation[j]):
                result[i]+=0.22*function1(rd.uniform(0.22*j+0.8,0.22*j+1.02))/sample_allocation[j]
        result_mean+=result[i]/repeat_times
    for i in range(0,repeat_times):
        result_variance+=(result[i]-result_mean)**2
    result_variance/=repeat_times
    print "The Variance of the 50 samples is ",result_variance
    print "The average of this",repeat_times,"samples is: ",result_mean
    plt.scatter(np.arange(0,repeat_times),result)
    plt.title("Function 1 with Monte Carlo Estimation Imported with stratification")
    plt.xlabel("Trial")
    plt.ylabel("Estimation Result")
    plt.grid(True)
    plt.show()
    print "\n\n\n\n"
#Monte Carlo Method Using Importance Sampling 
#Use Exponential pdf with lamda=1.0/(np.log(function1(0.8)/function1(1.8)))
def func_2b11(repeat_times,sample_number):
    result=[0]*repeat_times
    result_mean=0
    result_variance=0
    lamda=1.0/(np.log(function1(0.8)/function1(1.8)))
    envelope_size=expon.cdf(3,loc=0,scale=lamda)-expon.cdf(0.8,loc=0,scale=lamda)
    for i in range(0,repeat_times):
        for j in range(0,sample_number):
            x=rd.uniform(0.8,3)
            result[i]+=(function1(x)*envelope_size)/(expon.pdf(x,loc=0,scale=lamda)*sample_number)
        result_mean+=result[i]/repeat_times
    for i in range(0,repeat_times):
        result_variance+=(result[i]-result_mean)**2
    result_variance/=repeat_times
    print "The Variance of the 50 samples is ",result_variance
    print "The average of this",repeat_times,"samples is: ",result_mean
    plt.scatter(np.arange(0,repeat_times),result)
    plt.title("Function 1 with Monte Carlo Estimation Imported with Importance Sampling")
    plt.xlabel("Trial")
    plt.ylabel("Estimation Result")
    plt.grid(True)
    plt.show()
    print "\n\n\n\n"
def func_2b2(repeat_times,sample_number):
    #use bivariate normal distribution of uncorrelated X & Y to determine the importance of each interval
    #use normal distribution with mean=0, variance=0.8
    sample_allocation=np.empty((5,5),dtype=float)
    allocation=[0.0]*5
    sum_allocation=0.0
    for i in range(0,5):
        x1=norm.cdf((2.0*i/5-1)*math.pi,loc=0,scale=0.8)
        x2=norm.cdf((2.0*i/5-0.6)*math.pi,loc=0,scale=0.8)
        allocation[i]=(x2-x1)
    condition=(norm.cdf(-math.pi,loc=0,scale=1)-norm.cdf(math.pi,loc=0,scale=1))**2
    for i in range(0,5):
        for j in range(0,5):
            sample_allocation[i][j]=int(allocation[i]*allocation[j]*sample_number/condition)
            sum_allocation+=sample_allocation[i][j]
    sample_allocation[2][2]+=(sample_number-sum_allocation)
    print "The allocation of Sample Numbers Derived from Exponential pdf is:\n"
    print sample_allocation
    result=[0]*repeat_times
    result_mean=0
    result_variance=0
    for i in range(0,repeat_times):
        for j1 in range(0,5):
            for j2 in range(0,5):
                if sample_allocation[j1][j2]!=0:
                    for k in range(0,int(sample_allocation[j1][j2])):
                        x=rd.uniform((2.0*j1/5-1)*math.pi,(2.0*j1/5-0.6)*math.pi)
                        y=rd.uniform((2.0*j2/5-1)*math.pi,(2.0*j2/5-0.6)*math.pi)
                        result[i]+=1.579*function2(x,y)/sample_allocation[j1][j2]
        result_mean+=result[i]/repeat_times
    for i in range(0,repeat_times):
        result_variance+=(result[i]-result_mean)**2
    result_variance/=repeat_times
    print "The Variance of the 50 samples is ",result_variance
    print "The average of this",repeat_times,"samples is: ",result_mean
    plt.scatter(np.arange(0,repeat_times),result)
    plt.title("Function 2 with Monte Carlo Estimation Imported with stratification")
    plt.xlabel("Trial")
    plt.ylabel("Estimation Result")
    plt.grid(True)
    plt.show()
    print "\n\n\n\n"
#Monte Carlo Method Using Importance Sampling 
def func_2b21(repeat_times,sample_number):
    result=[0.0]*repeat_times
    result_mean=0.0
    result_variance=0.0
    envelope_size=np.sqrt(2)*math.pi*(norm.cdf(math.pi)-norm.cdf(-math.pi))**2
    for i in range(0,repeat_times):
        for j in range(0,sample_number):
            x=rd.uniform(-math.pi,math.pi)
            y=rd.uniform(-math.pi,math.pi)
            result[i]+=(function2(x,y)*envelope_size)/(norm.pdf(x)*norm.pdf(y)*sample_number)
            #((function2(x,y)*envelope_size)/(norm.pdf(x)*norm.pdf(y)*sample_number))
        result_mean+=result[i]/repeat_times
    for i in range(0,repeat_times):
        result_variance+=(result[i]-result_mean)**2
    result_variance/=repeat_times
    print "The Variance of the 50 samples is ",result_variance
    print "The average of this",repeat_times,"samples is: ",result_mean
    plt.scatter(np.arange(0,repeat_times),result)
    plt.title("Function 2 with Monte Carlo Estimation Imported with Importance Sampling")
    plt.xlabel("Trial")
    plt.ylabel("Estimation Result")
    plt.grid(True)
    plt.show()
    print "\n\n\n\n"

def func_3a1(repeat_times):
#Normal Monte Carlo Integration Estimate
    result=[0]*repeat_times
    result_mean=0
    result_variance=0
    for i in range(0,repeat_times):
        for j in range(0,10000):
            result[i]+=100*function3(rd.uniform(-5,5),rd.uniform(-5,5))/10000
        result_mean+=result[i]/repeat_times
    for i in range(0,repeat_times):
        result_variance+=(result[i]-result_mean)**2
    result_variance/=repeat_times
    print "The Variance of the 50 samples is ",result_variance
    print "The average of this",repeat_times,"samples is: ",result_mean
    plt.scatter(np.arange(0,repeat_times),result)
    plt.title("Function 3 with Standard Monte Carlo Estimation")
    plt.xlabel("Trial")
    plt.ylabel("Estimation Result")
    plt.grid(True)
    plt.show()
    print "\n\n\n\n"
def func_3a2(repeat_times,sample_number):
    #use bivariate beta distribution which are uncorrelated to determine the allocation of samples
    #and make them even symmetric
    #use beta distribution with a=3 b=1
    sample_allocation=np.empty((5,5),dtype=float)
    allocation=[0.0]*5
    sum_allocation=0.0
    condition=0
    alpha_1=0.8
    beta_1=0.7
    allocation[4]=beta.cdf(1,alpha_1,beta_1)-beta.cdf(0.6,alpha_1,beta_1)
    allocation[3]=beta.cdf(0.6,alpha_1,beta_1)-beta.cdf(0.2,alpha_1,beta_1)
    allocation[2]=2*(beta.cdf(0.2,alpha_1,beta_1)-beta.cdf(0,alpha_1,beta_1))
    allocation[1]=allocation[3]
    allocation[0]=allocation[4]
    for i in range(0,5):
        condition+=allocation[i]
    condition=condition**2
    for i in range(0,5):
        for j in range(0,5):
            sample_allocation[i][j]=int(allocation[i]*allocation[j]*sample_number/condition)
            sum_allocation+=sample_allocation[i][j]
    sample_allocation[2][2]+=(sample_number-sum_allocation)
    print "The allocation of Sample Numbers Derived from Exponential pdf is:\n"
    print sample_allocation
    result=[0.0]*repeat_times
    result_mean=0.0
    result_variance=0.0
    for i in range(0,repeat_times):
        for j1 in range(0,5):
            for j2 in range(0,5):
                if sample_allocation[j1][j2]!=0:
                    for k in range(0,int(sample_allocation[j1][j2])):
                        x=rd.uniform(2.0*j1-5,2.0*j1-3)
                        y=rd.uniform(2.0*j2-5,2.0*j2-3)
                        result[i]+=4*function3(x,y)/sample_allocation[j1][j2]
        result_mean+=result[i]/repeat_times
    for i in range(0,repeat_times):
        result_variance+=(result[i]-result_mean)**2
    result_variance/=repeat_times
    print "The Variance of the 50 samples is ",result_variance
    print "The average of this",repeat_times,"samples is: ",result_mean
    plt.scatter(np.arange(0,repeat_times),result)
    plt.title("Function 3 with Monte Carlo Estimation Imported with stratification")
    plt.xlabel("Trial")
    plt.ylabel("Estimation Result")
    plt.grid(True)
    plt.show()
    print "\n\n\n\n"
    
def func_3a3(repeat_times):
#devide the interval into 5*5 bins
#Initial the number of samples in each bin equally
    sample_allocation=np.full((5,5),400,dtype=float)
    mean=[0.0]*1000
    iteration_count=0
    func_val_store=[0]*10000
    #make sure to iterate at least once
    while iteration_count<999:
        iteration_count+=1
        count=0
        func_val_avg=np.empty((5,5))
        for i in range(0,5):
            for j in range(0,5):
                step=int(math.sqrt(sample_allocation[i][j]))
                for k1 in range(1,step+1):
                    for k2 in range(1,step+1):
                        x=2*i-5+float(2*k1)/(step+1)
                        y=2*j-5+float(2*k2)/(step+1)
                        func_val_store[count]=function3(x,y)
                        func_val_avg[i][j]+=func_val_store[count]
                        count+=1
                if step==0:
                    func_val_avg[i][j]=0
                else:
                    func_val_avg[i][j]/=(step**2)
        #estimated integration in this iteration
        for i in range(0,5):
            for j in range(0,5):
                mean[iteration_count]+=func_val_avg[i][j]*4
        #Compare with the estimated mean in the last iteration
        if abs(mean[iteration_count]-mean[iteration_count-1])<0.00001:
            break
        #re-distribute the number in each bins, with totol sample_number of samples
        #The sample numbers in each bin is linearly positively propotional to abs of funv_val
        func_val_total=0
        number_in_bin_count=0
        for i in range(0,5):
            for j in range(0,5):
                func_val_total+=abs(func_val_avg[i][j])
        for i in range(0,5):
            for j in range(0,5):
                sample_allocation[i][j]=int(10000*abs(func_val_avg[i][j])/func_val_total)
                number_in_bin_count+=sample_allocation[i][j]
        #add the rest few sample to the interval that has the largest func_val
        sample_allocation[2][2]+=(10000-number_in_bin_count)
    #use the optimized sample_allocation to estimate the Integration
    print "The allocation of Sample Numbers Derived from Exponential pdf is:\n"
    print sample_allocation
    result=[0]*repeat_times
    result_mean=0
    result_variance=0
    for i in range(0,repeat_times):
        for j1 in range(0,5):
            for j2 in range(0,5):
                if sample_allocation[j1][j2]!=0:
                    for k in range(0,int(sample_allocation[j1][j2])):
                        x=rd.uniform(2*j1-5,2*j1-3)
                        y=rd.uniform(2*j2-5,2*j2-3)
                        result[i]+=4*function3(x,y)/sample_allocation[j1][j2]
        result_mean+=result[i]/repeat_times
    for i in range(0,repeat_times):
        result_variance+=(result[i]-result_mean)**2
    result_variance/=repeat_times
    print "The Variance of the 50 samples is ",result_variance
    print "The average of this",repeat_times,"samples is: ",result_mean
    plt.scatter(np.arange(0,repeat_times),result)
    plt.title("Function 3 with Monte Carlo Estimation Imported with own sample allocation strategy")
    plt.xlabel("Trial")
    plt.ylabel("Estimation Result")
    plt.grid(True)
    plt.show()
    print "\n\n\n\n"

func_1a(50,500,1)
func_1a(50,5000,1)
func_1a(50,50000,1)
func_1b()
func_2a1(50,1000)
func_2a2(50,1000)
func_2b1(50,1000)
func_2b11(50,1000)
func_2b2(50,1000)
func_2b21(50,1000)
func_3a1(50)
func_3a2(50,10000)
func_3a3(50)