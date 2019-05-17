#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 4 15:59:41 2019
Project4 for EE511
Author:Chengyao Wang
USCID:6961599816
Contact Email:chengyao@usc.edu
"""
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import random as rd

def matrix_gen(a,b):
    return np.array([[1-a,b],[a,1-b]])
def func_1():
    choice=np.array([[1.0/10,1.0/15],[0.5,0.5],[1.0,1.0],[0.0,0.0]])
    for i in range(0,4):
        matrix=matrix_gen(choice[i,0],choice[i,1])
        w,v=la.eig(matrix)
        count=0
        print "The Stationary Distribution for\n",matrix,"is:"
        for j in range(0,2):
            if w[j]==1:
                v[:,j]/=(v[0,j]+v[1,j])
                count+=1
                print v[:,j]
        if count>1:
            print "All Distributions Corresponding to Normalized Vectors that belongs to Span of Previous Vectors are stationary Distributions\n\n"
        print "\n\n" 
mc1=np.array([[0,1],[1,0]])
mc2=np.array([[0.75,0.25],[0.1,0.9]])
mc3=np.array([[0.48,0.48,0.04],[0.22,0.7,0.08],[0,0,1]])
def next_state_mc1(a):
    x=rd.uniform(0,1)
    if x<mc1[int(a),0]:
        return 0
    else:
        return 1
def next_state_mc2(a):
    x=rd.uniform(0,1)
    if x<mc2[int(a),0]:
        return 0
    else:
        return 1
def next_state_mc3(a):
    x=rd.uniform(0,1)
    if x<mc3[int(a),0]:
        return 0
    elif int(x>mc3[int(a),0]) & int(x<(mc3[int(a),1]+mc3[int(a),0])):
        return 1
    else:
        return 2
def func_2():
    sample=np.zeros((75,500))
    tail_count=np.zeros((75,2))
    goodness_of_fit=np.zeros((75,2))
    count=0.0
    ensemble_avg=[0.0]*2
    time_avg=[0.0]*2    
    for repeat_times in range(0,75):
        for time in range(0,499):
            sample[repeat_times][time+1]=next_state_mc1(sample[repeat_times][time])
        for time in range(425,500):
            tail_count[repeat_times][int(sample[repeat_times][time])]+=1
    arr1,arr2=np.split(sample[rd.randint(0,75),:],[425])
    plt.plot(np.arange(425,500),arr2)
    plt.title("The Last 75 Samples of a Randomly Chosen Sample Path")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.grid(True)
    plt.show()
    #Goodness of Fit threshold: degree of freedom=1, 3.84
    #stationary distribution is [0.5,0.5]
    for i in range(0,75):
        goodness_of_fit[i][0]=(37.5-tail_count[i][0])**2/37.5+(37.5-tail_count[i][1])**2/37.5
        if goodness_of_fit[i][0]<3.84:
            count+=1
            goodness_of_fit[i][1]=1
    #Count Ensemble_avg and Time_avg
    for i in range(0,75):
        ensemble_avg[int(sample[i][499])]+=1.0/75
        time_avg[int(arr2[i])]+=1.0/75
    print "Ensemble Avg of this MC is",ensemble_avg
    print "Time Avg of this MC is",time_avg
    plt.plot(np.arange(0,75),goodness_of_fit[:,1])
    plt.title("Path that can be considered convergent(convergent=1)")
    plt.xlabel("Index of Path")
    plt.ylim(-0.25,1.25)
    plt.grid(True)
    plt.show()
    #Plot State Distribution of MC1
    plt.hist(arr2)
    plt.title("State Distribution of The last 75 samples of one sample path")
    plt.xlabel("State")
    plt.ylabel("Time spent in Each State")
    plt.grid(True)
    plt.show()
    print "Ratio of path that can be considered convergent by Chi_square:",count," among 75 paths"
    print "\n\n\n\n"
def func_3():
    sample=np.zeros((75,500))
    tail_count=np.zeros((75,2))
    goodness_of_fit=np.zeros((75,2))
    count=0.0
    ensemble_avg=[0.0]*2
    time_avg=[0.0]*2    
    for repeat_times in range(0,75):
        for time in range(0,499):
            sample[repeat_times][time+1]=next_state_mc2(sample[repeat_times][time])
        for time in range(425,500):
            tail_count[repeat_times][int(sample[repeat_times][time])]+=1
    arr1,arr2=np.split(sample[rd.randint(0,74),:],[425])
    plt.plot(np.arange(425,500),arr2)
    plt.title("The Last 75 Samples of a Randomly Chosen Sample Path")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.grid(True)
    plt.show()
    #print sample[35,:]
    #print tail_count[35,:]
    #Goodness of Fit threshold: degree of freedom=1, 3.84
    #stationary distribution is [0.2857,0.7143]
    for i in range(0,75):
        goodness_of_fit[i][0]=(21.43-tail_count[i][0])**2/21.43+(53.57-tail_count[i][1])**2/53.57
        if goodness_of_fit[i][0]<3.84:
            count+=1
            goodness_of_fit[i][1]=1
    #Count Ensemble_avg and Time_avg
    for i in range(0,75):
        ensemble_avg[int(sample[i][499])]+=1.0/75
        time_avg[int(arr2[i])]+=1.0/75
    print "Ensemble Avg of this MC is",ensemble_avg
    print "Time Avg of this MC is",time_avg
    plt.scatter(np.arange(0,75),goodness_of_fit[:,1],s=10)
    plt.title("Path that can be considered convergent(convergent=1)")
    plt.xlabel("Index of Path")
    plt.ylim(-0.25,1.25)
    plt.grid(True)
    plt.show()
    plt.hist(arr2)
    plt.title("State Distribution of the Randomly Chosen Sample Path")
    plt.xlabel("State")
    plt.ylabel("Time spent in Each State")
    plt.grid(True)
    plt.show()
    print "Ratio of path that can be considered convergent:",count/75
    print "\n\n\n\n"    
def func_4():
    sample=np.zeros((75,500))
    tail_count=np.zeros((75,3))
    goodness_of_fit=np.zeros((75,2))
    state3_store=[0.0]*75
    count=0.0
    ensemble_avg=[0.0]*3
    time_avg=[0.0]*3
    for repeat_times in range(0,75):
        flag=1
        for time in range(0,499):
            sample[repeat_times][time+1]=next_state_mc3(sample[repeat_times][time])
            if (sample[repeat_times][time+1]==2)&flag:
                state3_store[repeat_times]=time
                flag=0
        for time in range(425,500):
            tail_count[repeat_times][int(sample[repeat_times][time])]+=1
    arr1,arr2=np.split(sample[rd.randint(0,74),:],[425])
    plt.plot(np.arange(425,500),arr2)
    plt.title("The Last 75 Samples of a Randomly Chosen Sample Path")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.ylim(-0.25,2.25)
    plt.grid(True)
    plt.show()
    #print sample[35,:]
    #print tail_count[35,:]
    #Goodness of Fit threshold: degree of freedom=2, 5.99
    #stationary distribution is [0,0,1]
    for i in range(0,75):
        goodness_of_fit[i][0]=(tail_count[i][0]-0.001)**2/0.001+(tail_count[i][1]-0.001)**2/0.001+(tail_count[i][2]-74.998)**2/74.998
        if goodness_of_fit[i][0]<5.99:
            count+=1
            goodness_of_fit[i][1]=1
    #Count Ensemble_avg and Time_avg
    for i in range(0,75):
        ensemble_avg[int(sample[i][499])]+=1.0/75
        time_avg[int(arr2[i])]+=1.0/75
    print "Ensemble Avg of this MC is",ensemble_avg
    print "Time Avg of this MC is",time_avg
    plt.scatter(np.arange(0,75),goodness_of_fit[:,1],s=10)
    plt.title("Path that can be considered convergent(convergent=1)")
    plt.xlabel("Index of Path")
    plt.ylim(-0.25,1.25)
    plt.grid(True)
    plt.show()
    plt.scatter(np.arange(0,75),state3_store,s=10)
    plt.title("Time Entering State 2 of every path")
    plt.xlabel("Index of Path")
    plt.ylabel("Time Entering State 2")
    plt.grid(True)
    plt.show()
    plt.hist(arr2)
    plt.title("State Distribution of the Randomly Chosen Sample Path")
    plt.xlabel("State")
    plt.xlim((-0.25,2.25))
    plt.ylabel("Time spent in Each State")
    plt.grid(True)
    plt.show()
    print "Ratio of path that can be considered convergent:",count/75
    print "\n\n\n\n"    
#func_1()
func_2()
#func_3()
#func_4()           