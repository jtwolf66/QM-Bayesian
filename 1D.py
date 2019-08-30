# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:33:25 2016

@author: Joseph
"""


import numpy as np
import scipy as sp
from numpy.linalg import inv
from numpy import pi, dot, transpose, radians, exp
import math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import time
# from sympy import *

t1 = time.time()
m = 5.0
N = 5000
dphi = 500
Eo = 1.0/2.0 * m
eps = 20 * np.sqrt(2.0/m)/dphi
space = np.arange(-10.0 * np.sqrt(2.0/m),10.0 * np.sqrt(2.0/m)+eps,eps)
P = np.zeros(N) 

'''
phi = Symbol('phi')
phi2 = Dummy('phi2')
En = Dummy('En')


Psi = exp(-m/2.0 * phi**2.0)**(1.0/2.0)

Lagr = -(1.0/2.0) * (((phi - phi2)/eps)**2.0 + m**2.0*((phi + phi2)/2.0)**2.0)
Fn = (exp(eps*En)/(2.0*pi*eps)**(1.0/2.0))*exp(eps*Lagr)

Pn = Fn*Psi/Psi.subs(phi,phi2)
'''

def Psi(x):
    return (exp(-m/2.0 * x**2.0))
 
def Lagr(x,space):
    return -1.0/2.0 * ( np.add(((space - x)/eps)**2.0, m**2.0*((space + x)/2.0)**2.0)) - Eo
    
def Fn(x,En,limits=space):
    return exp(eps*En)/np.sqrt(2.0*pi*eps) * exp(eps*Lagr(x,limits))
    
def Fn2(x,y,En):
    return exp(eps*En)/np.sqrt(2.0*pi*eps) * exp(eps*Lagr(x,y))
 
def Pn(x,En,limits=space):
    #return Fn(x,En,limits)*Psi(limits)/Psi(x)
    return exp(eps*En)/np.sqrt(2.0*pi*eps) * exp(eps*-1.0/2.0 * ( np.add(((space - x)/eps)**2.0, m**2.0*((space + x)/2.0)**2.0)))*(exp(-m/2.0 * space**2.0))/(exp(-m/2.0 * x**2.0))

def Randvar(Pn,limits=space):
    Pnorm = (Pn/np.sum(Pn))
    return np.random.choice(limits,p=Pnorm)
 
R = np.zeros(N+1)
P = np.zeros(N+1)

R[0] = Randvar(np.power(Psi(space),2.0),space)

s2 = np.zeros((len(space),len(space)))
for i in xrange(0,len(space)):
    for j in xrange(0,len(space)):
        s2[i,j] = Fn2(space[i],space[j],Eo)

def Pn2(x,En,limits=space):
    return np.dot(s2,v[:,0])/(w[0]*v[500,0])

for i in xrange(1,N+1):
    Prob = Pn(R[i-1],Eo,space)
    R[i] = Randvar(Prob,space)
t2 = time.time()
print "complete"
print t2-t1

w,v = np.linalg.eigh(s2)
sm = np.sum(np.dot(s2,v[:,-1])/(w[-1]*v[int(len(space)/2),-1]))

print sm
print sm*eps

Cor = np.zeros((N-2,N))
Corr = np.zeros(N-2)

for j in xrange(1,len(space)-1):
    for i in xrange(0,len(space)):
        if (i + j) <= N:
            Cor[j-1][i] = R[i]*R[i+j]
    Corr[j-1] = np.sum(Cor[j-1]/len(Cor[j-1]))
print Corr

plt.figure()
plt.hist(R, 50, normed=True)
plt.figure()
plt.plot(space,np.power(Psi(space),2.0))
