#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 10:38:04 2019
Project1 for EE511
Author:Chengyao Wang
USCID:6961599816
Contact Email:chengyao@usc.edu
"""
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math
def one_trial_bernoulli_test(fail_prob):
    i=rd.uniform(0,1)
    #print(i)
    if ((i>0.)&(i<fail_prob)):
        return 0
    elif ((i>fail_prob)&(i<1.)):
        return 1

def func_1a(fail_prob):
    result=[0]*100
    Sum=([0,0])
    for trials in range(0,100):
        one_trial_result=one_trial_bernoulli_test(fail_prob)
        #print(one_trial_result)
        result[trials]=one_trial_result
        if one_trial_result==0:
            Sum[0]+=1
        elif one_trial_result==1:
            Sum[1]+=1
        trials+=1
    print("Result for 1(a):\nSuccess %d Times\nFailure %d Times\n"%(Sum[0],Sum[1]))
    plt.hist(result,10)
    plt.text(0.125,Sum[0]/2,'Fail\n%dTimes'%Sum[0])
    plt.text(0.75,Sum[1]/2,'Succeed\n%dTimes'%Sum[1])
    plt.title('Results for 100 Bernoulli Trials')
    plt.xlabel('$Result$')
    plt.ylabel('$Number of Times$')
    plt.show()

def func_1b(fail_prob):
    result=[0]*100
    Sum=([0,0,0,0,0,0,0,0])
    x=([0,1,2,3,4,5,6,7])
    for samples in range(0,100):
        for trials in range(0,7):
            one_trial_result=one_trial_bernoulli_test(fail_prob)
            result[samples]+=one_trial_result
        Sum[result[samples]]+=1
    print(result)
    plt.bar(x,Sum)
    for i in range(0,8):
        plt.text(i-0.1,Sum[i]+0.3,'%d'%Sum[i])
    plt.title('Results of 100 Samples')
    plt.xlabel('Times of Success in 7 Trials')
    plt.ylabel('Occurence of Certain Number of Successes')
    plt.show()

def func_1c(fail_prob):
    result=[0]*100
    for samples in range(0,100):
        one_trial_result=one_trial_bernoulli_test(fail_prob)
        while one_trial_result==1:
            result[samples]+=one_trial_result
            one_trial_result=one_trial_bernoulli_test(fail_prob)
    print(result)
    max_result=max(result)
    Sum=[0]*(max_result+1)
    for i in range(0,100):
        Sum[result[i]]+=1
    x=np.arange(0,max_result+1)
    print(x)
    print(Sum)
    plt.bar(x,Sum)
    for i in range(0,max_result+1):
        plt.text(i-0.1,Sum[i]+0.3,'%d'%Sum[i])
    plt.title('Results of 100 Samples')
    plt.xlabel('Longest Run of Heads')
    plt.ylabel('Occurence of Certain Number of Head Runs')
    plt.show()

def func_2a(fail_prob):
    number_of_each_sample=([5,10,30,50])
    numberofsuccess=0
    for i in range(0,4):
        Sum=[0]*(number_of_each_sample[i]+1)
        for repeat_times in range(0,300):
            for trials in range(0,number_of_each_sample[i]):
                numberofsuccess+=one_trial_bernoulli_test(fail_prob)
            Sum[numberofsuccess]+=1
            numberofsuccess=0
        x=np.arange(0,number_of_each_sample[i]+1)
        plt.bar(x,Sum)
        for j in range(0,number_of_each_sample[i]+1):
            plt.text(j-0.1,Sum[j]+0.3,'%d'%Sum[j])
        plt.title('Results of %d Samples'%number_of_each_sample[i])
        plt.xlabel('SuccessTimes')
        plt.ylabel('Occurence of Certain Number of SuccessTimes')
        plt.show()
        
def inverse_exp(y,lamda):
    x=math.log(1.0/(1.0-y))/lamda
    return x
def inverse_cauchy(y,x0,gama):
    x=x0+gama*math.tan(math.pi*(y-0.5))
    return x
def exp_dis(x,lamda):
    y=1-math.exp((-1)*lamda*x)
    return y
def cauchy_dis(x,x0,gama):
    y=math.atan((x-x0)/gama)/math.pi
    return y

def func_3a(sample_number):
    y_sample=np.random.rand(sample_number)
    x_exp=[0]*sample_number
    x_cauchy=[0]*sample_number
    x_test=[0]*sample_number
    for i in range(0,sample_number):
        x_test[i]=2*y_sample[i]+1
        x_exp[i]=inverse_exp(y_sample[i],5)
        x_cauchy[i]=inverse_cauchy(y_sample[i],0,2)
    #print(x_test,y_sample)
    plt.scatter(x_cauchy,y_sample)
    plt.xlim(-50,50)
    plt.grid(True)
    plt.show()
    plt.scatter(x_exp,y_sample)
    plt.xlim(-0.1,2)
    plt.grid(True)
    plt.show()
    #Chi-Squared Test for Goodness-of-Fit
    #Divide as: <-40,-40~-35,-35~-30,.....,30~35,35~40,>40 for exp_distribution
    #Divide as: 0~0.1,0.1~0.2,0.2~0.3,.....,0.15~0.16,0.16~0.17,>0.17 for cauchy_distribution
    E_exp=[0]*18
    E_cauchy=[0]*18
    O_exp=[0]*18
    O_cauchy=[0]*18
    goodness_exp=0
    goodness_cauchy=0
    new_goodness_exp=0
    new_goodness_cauchy=0
    for i in range(0,18):
        uplimit_cauchy=5*i-40+100000*(i==17)
        downlimit_cauchy=5*i-45-100000*(i==0)
        uplimit_exp=0.1*i+0.1+100000*(i==17)
        downlimit_exp=0.1*i-0.1
        E_exp[i]=sample_number*(exp_dis(uplimit_exp,5)-exp_dis(downlimit_exp,5))
        E_cauchy[i]=sample_number*(cauchy_dis(uplimit_cauchy,0,2)-cauchy_dis(downlimit_cauchy,0,2))
    for j in range(0,18):
        uplimit_cauchy=5*j-40+100000*(j==17)
        downlimit_cauchy=5*j-45-100000*(j==0)
        uplimit_exp=0.1*j+0.1+100000*(j==17)
        downlimit_exp=0.1*j-0.1
        for i in range(0,1000):
            O_exp[j]+=((x_exp[i]>downlimit_exp)&(x_exp[i]<uplimit_exp))
            O_cauchy[j]+=((x_cauchy[i]>downlimit_cauchy)&(x_cauchy[i]<uplimit_cauchy))
    #Calculate the Chi-Squared Goodness of Fit 
    for i in range(0,18):
        goodness_exp+=(O_exp[i]-E_exp[i])**2/E_exp[i]
        goodness_cauchy+=(O_cauchy[i]-E_cauchy[i])**2/E_cauchy[i]
    #Calculate a second Goodness of Fit
    for i in range(0,18):
        new_goodness_exp+=np.abs(O_exp[i]-E_exp[i])/E_exp[i]
        new_goodness_cauchy+=np.abs(O_cauchy[i]-E_cauchy[i])/E_cauchy[i]
    print("Number of dots observed in a certain interval")
    print(O_exp)
    print(O_cauchy)
    print("Number of dots ideal in a certain interval")
    print(E_exp)
    print(E_exp)
    print("Chi-Squared Test Result for exp is %d"%goodness_exp)
    print("Chi-Squared Test Result for cauchy is %d"%goodness_cauchy)
    print("New Goodness of Fit Result for exp is %d"%new_goodness_exp)
    print("New Goodness of Fit Result for cauchy is %d"%new_goodness_cauchy)

#Fuction starts here
func_1a(0.5)
func_1b(0.5)
func_1c(0.5)
func_2a(0.5)
func_3a(1000)
