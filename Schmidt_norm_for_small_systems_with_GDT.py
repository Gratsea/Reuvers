#algorithm for finding the correct subspace with gradient descent techniques

import numpy as np
from numpy import linalg 
import math
from scipy.stats import ortho_group
from scipy import optimize
import random

def tensor(vectorA,vectorB) : #finds the tensor product between states A and B
    m = np.size(vectorA,0)
    n = np.size(vectorB,0)
    tens=np.zeros((m,n))
    for i in range(m) :
        for j in range(n) :
            tens[i][j] = vectorA[i]*vectorB[j]
    return (tens);

def func(z) : #function that is minimized by the bashin-hopping algorithm
    f = open("Reuvers_n2m4_initial_n10_xmax20.txt","a+")
    #random 2 spins initial state
    #initial = np.array([[ 0.17123127 , 0.79461077], [ 0.21939553 , 0.53957314]])
    #initial = np.array([[ 0.1, 0.7], [ 0.2 , 0.5]])
    
    #random 3 spins initial state
    initial=np.array([[0.52619533, 0.27592482, 0.26093188, 0.2510644 ], [0.2290105 , 0.48391584, 0.3214155 , 0.35487592]])
    
    #random 10 m sites intiial staet
    #initial = np.random.rand(2,3)
    
    initial = initial/np.linalg.norm(initial)
    Initial = initial
   
    #write initial state at the .txt
    f.write("initial state")
    f.close()
    with open('Reuvers_n2m4_initial_n10_xmax20.txt', 'a+') as f:
        print (Initial,file=f)
    f.close()
    
    n = np.size(initial,0) 
    m = np.size(initial,1) 
    k=min(m,n)
   

    #define the subspace from the initial guess z ( z is a list)
    l=0
    A = np.zeros((m*n,k))
    for i in range (0,m*n,1):
        for j in range (0,k,1):
            A[i][j] = z[l]
            l+=1
            
    #write the subspace A at the txt file
    with open('Reuvers_n2m4_initial_n10_xmax20.txt', 'a+') as f:
        print ("ok3,A",A,file=f)
    f.close()
    
    #define the projector from the subpace A
    A=A/np.linalg.norm(A)   
    AtA=np.dot( A.transpose(),A)
    AtAinv=np.linalg.inv(AtA)
    Q=np.dot(AtAinv,A.transpose())
    Q=np.dot(A,Q)
    Proj=Q

    #project the state at the given subspace
    initial=initial.reshape(m*n,1) #reshape of the initial state so as to act with the projecotr operator
    initial= Proj.dot(initial)/np.linalg.norm(Proj.dot(initial))
    initial=initial.reshape(m,n)

    #repeated Schmidt decompotion
    NORM=0.
    x=1 
    xmax=20
    while (x<=xmax) :
        NORM=0.

        '''step 1 of the algorithm : psiA,psiB Schmidt vector A,B and l Schmidt coefficients '''
        psiA, l, psiB = np.linalg.svd(initial,full_matrices=1) #decomposition of initial matrix
        
        '''step 2 of the algorithm : definition of the normalized vector '''   
           
        p=1.0 # p has to be larger or equal than 1 for the algorithm to work
        sum=np.zeros((m,n))
        for i in range(k) :
            tens= tensor(psiA[i],psiB[i])
            sum += math.pow(l[i],p-1)*tens
            NORM = NORM + math.pow(l[i],p)      
         
        NORM = math.pow(NORM,1./p)
            
        sum=sum.reshape(m*n,1) 
        psi= np.dot(Proj,sum) #projection on the subspace
        psi=psi/np.linalg.norm(psi)
           
           #write norm at the .txt
        f=open("Reuvers_n2m4_initial_n10_xmax20.txt","a+")
        f.write("t,NORM ")
        f.close()
        with open('Reuvers_n2m4_initial_n10_xmax20.txt', 'a+') as f:
            print (x,NORM,file=f)
            f.close()  
    
            
        '''step 3 of the algorithm '''
        initial=psi
        initial=initial.reshape(m,n)
    
        x+=1 
        print (NORM)
    return -NORM+math.sqrt(2)




n=2 #total states fo the subsystem A
m=4 #total states fo the subsystem B
k=min(n,m)

#initial guess for 2 spins
#x0= [1,0.5,1,0.5,0.2,0.3,0.5,0.5]

#initial guess for 3 spins
x0=  [1.,   0.5 , 1. ,  0.5 , 0.2  ,0.3,  0.5  ,0.5,  0.2,  0.3 , 0.5 , 0.5 , 1. ,  0.5, 1.5 , 0.45]



my_randoms=[]
for i in range (m*n*k):
    my_randoms.append(random.randrange(1,10,1))

print (my_randoms)

x0 = my_randoms

f = open("Reuvers_n2m4_initial_n10_xmax20.txt","a+")
f.write("initial guess of A")
f.close()
with open('Reuvers_n2m4_initial_n10_xmax20.txt', 'a+') as f:
    print (x0,file=f)
f.close()

minimizer_kwargs = {"method": "BFGS"}
ret = optimize.basinhopping(func,x0, minimizer_kwargs=minimizer_kwargs,niter=10,T=1.0, disp = True )
   

#define the correct subspace form the result ret.x of the bashin-hopping algorithm
l=0
B=np.zeros((m*n,k))
for i in range (0,m*n,1):
    for j in range (0,k,1):
        B[i][j] = ret.x[l]
        l+=1





