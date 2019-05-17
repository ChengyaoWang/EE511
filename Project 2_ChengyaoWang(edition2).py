#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 12:22:58 2019
Project2 for EE511
Author:Chengyao Wang
USCID:6961599816
Contact Email:chengyao@usc.edu
"""
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import networkx as nx
from scipy.stats import beta
from scipy.stats import poisson

sample_from_1a=[0]*1000
#The domain of target_pdf =[0,6]
def target_pdf(x):
    if (int(x>0.0)&int(x<=1.0)):
            return beta.pdf(x,8,5)/2
    elif(int(x>4.0)&int(x<=5.0)):
            return (x-4)/2
    elif int(x>5.0):
        if x<=6.0:
            return (6-x)/2
    else:
        return 0
#determine the expansion coefficient for envelope(height)
#use uniform distribution as the envelope
def envelope_height():
    x=np.arange(0,6,0.001)
    max_betapdf=0;
    for i in range(0,6000):
        y=target_pdf(x[i])
        if(max_betapdf<=y):
            max_betapdf=y
        #print(y)
    print "Maxium of target_pdf() is",max_betapdf
    return max_betapdf
#sample until index=1000
def func_1a():
    height=envelope_height()
    rejection_store=[0.0]*10000
    scatter_size=[0.01]*10000
    count=0     #total number of samples
    index=0     #number of accepted samples
    while index<1000:
        x=rd.uniform(0,6)
        y=rd.uniform(0,height)
        y_beta=target_pdf(x)
        count+=1
        if y<=y_beta:
            sample_from_1a[index]=x
            index+=1
        rejection_store[count]=float(index)/count
    rejection_rate=1-float(index)/count
#extend the last rejection sample to the rest of rejection_store
    for i in range(index+1,10000):
        rejection_store[i]=rejection_store[index]
    plt.scatter(range(0,10000),rejection_store,s=scatter_size)
    plt.title("Dynamic Trace of Acception Rate")
    plt.xlabel("Number of Total Samples")
    plt.ylabel("Acception Rate")
    plt.grid(True)
    plt.show()
    print "The rejection rate is",rejection_rate
    print "The 1000 sample generated in stored in sample_from_1a"
    
#use 1000 samples generated from func_1a(): sample_from_1a
def func_2a():
    estimated_mean=0
    estimated_mean_2=0
    covariance=0
    sample_from_1a_x1=[0]*1000
    sample_from_1a_x2=[0]*1000
#Derive the distribution of X and X+5
    for i in range(0,1000):
        sample_from_1a_x1[i]=sample_from_1a[i]
        if i>=5:
            sample_from_1a_x2[i-5]=sample_from_1a[i];
#mean of target PDF calculation
    for i in range(0,1000):
        estimated_mean+=sample_from_1a_x1[i]
        estimated_mean_2+=sample_from_1a_x2[i]
    estimated_mean/=1000
    estimated_mean_2/=995
#centralized sample
    for i in range(0,1000):
        sample_from_1a_x1[i]-=estimated_mean
        sample_from_1a_x2[i]-=estimated_mean_2
#covariance calculation
    for i in range(0,995):
            covariance+=sample_from_1a_x1[i]*sample_from_1a_x2[i]
    covariance/=994
    print "Calculated covariance is ",covariance
    if int(covariance<0.01)&int(covariance>-0.01):
        print "The two RV are nearly independent\n"
    elif covariance>0.01:
        print "The two RV are positively related\n"
    else :
        print "The two RV are negatively related\n"

def func_2b():
    sample_x=np.random.beta(8,5,1000)
    sample_y=np.random.beta(4,7,1000)
    observed_result=np.zeros((5,5))
    estimated_result=np.zeros((5,5))
    difference_result=np.zeros((5,5))
    estimated_x=[0]*5
    estimated_y=[0]*5
    chi_squared_result=0
#take (sample_x[i],sample_y[i]) as a sample of (X,Y)
    #print sample_x,"\n",sample_y
#divide the 2-way table into 5*5 areas
    for i in range(0,5):     #row
        for j in range(0,5): #column
            row_down=0.2*i
            row_up=0.2*i+0.2
            column_down=0.2*j
            column_up=0.2*j+0.2
            for sample_index in range(0,1000):
                if (int(sample_y[sample_index]>=row_down)&int(sample_y[sample_index]<row_up)):
                    if (int(sample_x[sample_index]>=column_down)&int(sample_x[sample_index]<column_up)):
                        observed_result[i][j]+=1
            #calculation_result[i][j]/=1000
    for i in range(0,5):
        estimated_x[i]=beta.cdf(0.2*i+0.2,8,5)-beta.cdf(0.2*i,8,5)
        estimated_y[i]=beta.cdf(0.2*i+0.2,4,7)-beta.cdf(0.2*i,4,7)
    for i in range(0,5):
        for j in range(0,5):
            estimated_result[i][j]=(estimated_y[i]*estimated_x[j]*1000)
    difference_result=estimated_result-observed_result
    for i in range(0,5):
        for j in range(0,5):
            chi_squared_result+=(difference_result[i][j]**2/estimated_result[i][j])
    #print estimated_result
    print "The observed result of RV X&Y is:\n",observed_result
    #print difference_result
    print "The Result of Chi-squared_test=\n",chi_squared_result
#Chi-squared distribution threshold can be acquired from charts, pick 95.0%
#the degree of freedom is 4*4=16, threshold=26.3
    if chi_squared_result<=26.3:
        print "X & Y can be considered as independent"
    else:
        print "X & Y cannot be considered as independent"

def func_3a(number_of_nodes,probability,times):
    nodes_list=np.arange(0,number_of_nodes)
    G=[0]*3 #a total of 3 graphs
    degree_node=[0]*(number_of_nodes)
    for number_of_graphes in range(0,times):
        G[number_of_graphes]=nx.Graph()
        G[number_of_graphes].add_nodes_from(nodes_list)
        edge_count=0
        potential_edge=0
        for i in range(0,number_of_nodes):
            for j in range(i+1,number_of_nodes):
                potential_edge+=1
                if rd.uniform(0,1)<probability:
                    G[number_of_graphes].add_edge(i,j)
                    edge_count+=1
                    if times==1:
                        degree_node[i]+=1
                        degree_node[j]+=1
        nx.draw(G[number_of_graphes])
        plt.show()
        print "The graph with",number_of_nodes,"nodes","and probability",probability,"\n"
        print "The total number of edges are",edge_count,"\n"
    if(times==1):
        #degree_node[number_of_nodes]=edge_count
        return degree_node

def func_3b():
    degree_sum=[0]*100
    max_degree=0
    degree_of_node=func_3a(100,0.06,1)
    for i in range(0,100):
        degree_sum[degree_of_node[i]]+=1
        if max_degree<=degree_of_node[i]:
            max_degree=degree_of_node[i]
    #print max_degree
    x=np.arange(0,max_degree+1)
    arr1,arr2=np.split(degree_sum,[max_degree+1])
    print "Data source of the histogram:\n",arr1
    #print degree_sum
#index 0~99 is the number of connections for each node
#100 is the totoal number of nodes
    plt.bar(x,arr1)
    plt.title('Results of 100 Samples')
    plt.xlabel('Node-degree')
    plt.ylabel('Occurence of Certain node-degree')
    plt.show()

#Chi-squared test
#Degree of freedom when judged by Chi-Squared: 100-2=98
#Threshold(0.05)=122.08
#Binomial Distribution
    goodness_binomial=0.0
    threshold_binomial=122.08
    for i in range(0,100):
        expected_binomial=combination_calculation(i,100)*pow(0.06,i)*pow(0.94,100-i)*100
        goodness_binomial+=(degree_sum[i]-expected_binomial)**2/expected_binomial
    print "Result for Goodness test compared with Binomial-Dis=",goodness_binomial
    if goodness_binomial<threshold_binomial:
        print "The Sample generated follows Binomial Distribution"
    else:
        print "The Sample generated doesn't follow Binomial Distribution"
#Poisson Distribution
#Degree of freedom when judged by Chi-Squared: 100-1=99
    goodness_poisson=0.0
    mu=100*0.06
    threshold_poisson=123.225
    for i in range(0,100):
        expected_poisson=poisson.pmf(i,mu)*100
        goodness_poisson+=(degree_sum[i]-expected_poisson)**2/expected_poisson
    print "Result for Goodness test compared with Poisson-Dis=",goodness_poisson
    if goodness_binomial<threshold_poisson:
        print "The Sample generated follows Poisson Distribution"
    else:
        print "The Sample generated doesn't follow Poisson Distribution"
def combination_calculation(k,n):
    result=1
    for i in range(n-k+1,n+1):
        result*=i
    for i in range(1,k+1):
        result/=i
    return result
#x=np.arange(0,6,0.1)
#for i in range(0,60):
#print(target_pdf(x[i]))  
func_1a()
#print index_for_2a
#print sample_for_2a[210]
#print sample_for_2a[40291]
#print sample_for_2a[21094]
#print sample_for_2a[81904]
func_2a()
#print np.random.beta(8,5,15)
func_2b()
func_3a(100,0.03,3)
func_3a(100,0.12,3)
#print combination_calculation(3,15)
func_3b()
