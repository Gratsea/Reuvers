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
    #random 2 spins initial state
    initial = np.array([[ 0.17123127 , 0.79461077], [ 0.21939553 , 0.53957314]])
    
    #random 3 spins initial state
    #initial=np.array([[0.52619533, 0.27592482, 0.26093188, 0.2510644 ], [0.2290105 , 0.48391584, 0.3214155 , 0.35487592]])
    
    #initial 2*8 random
    initial = np.array([[ 0.16445999,  0.0128692 ,  0.42034664,  0.19269148,  0.18763073,
         0.14401979,  0.31242185,  0.40329757],
       [ 0.3475274 ,  0.1107391 ,  0.29153246,  0.40336107,  0.01893523,
         0.02017586,  0.06197022,  0.23963927]])    
    
    
    #initial 2*25 random
    initial = np.array([[ 0.08395086,  0.35905596,  0.74519182,  0.58482998,  0.5709971 ,
         0.00188753,  0.10295313,  0.5355014 ,  0.01022064,  0.34866683,
         0.94218152,  0.65833488,  0.40761131,  0.79912806,  0.24291228,
         0.29893263,  0.19921267,  0.65889753,  0.20343142,  0.66810297,
         0.39740005,  0.22344591,  0.19931446,  0.46663114,  0.93425912],
       [ 0.42244557,  0.55658125,  0.95720225,  0.85128365,  0.1742795 ,
         0.55630994,  0.42689132,  0.85863061,  0.78078997,  0.47637764,
         0.77069141,  0.97726147,  0.12773686,  0.75773197,  0.97954399,
         0.77119188,  0.66874919,  0.19866114,  0.06484508,  0.20189229,
         0.70340717,  0.71977434,  0.64616132,  0.29447508,  0.95302879]])
    
    #initial 2*50 random
    initial = np.array([[ 0.43936808,  0.19595436,  0.66521295,  0.48874568,  0.25541025,
         0.82626083,  0.77867474,  0.37298978,  0.41238743,  0.07425773,
         0.60079406,  0.94353635,  0.74490519,  0.84987852,  0.90070506,
         0.0019357 ,  0.06947893,  0.86867889,  0.57526162,  0.92996132,
         0.96311033,  0.67940904,  0.41445507,  0.88890145,  0.98541201,
         0.49572883,  0.90678064,  0.51829683,  0.51349091,  0.46925381,
         0.31393399,  0.15330637,  0.30562494,  0.80608406,  0.29690816,
         0.03177704,  0.1022391 ,  0.33485398,  0.89392416,  0.08365688,
         0.80334503,  0.38246654,  0.25832347,  0.25156204,  0.17301436,
         0.57658989,  0.26927897,  0.30913527,  0.07104117,  0.14993112],
       [ 0.58284227,  0.16956595,  0.0375655 ,  0.76787974,  0.48638735,
         0.17357312,  0.21403614,  0.05356584,  0.15869978,  0.6365467 ,
         0.72290134,  0.11257087,  0.2916442 ,  0.76796205,  0.00254486,
         0.33413924,  0.94799251,  0.12702397,  0.70986751,  0.25058729,
         0.01025468,  0.50957219,  0.78715512,  0.06223669,  0.26127694,
         0.4001143 ,  0.09957726,  0.37562226,  0.15814467,  0.21770245,
         0.15140124,  0.13933993,  0.80024204,  0.26593252,  0.69656718,
         0.42388463,  0.6538354 ,  0.9875039 ,  0.97180309,  0.88513469,
         0.26205028,  0.33484531,  0.40428954,  0.55883557,  0.99674292,
         0.13217504,  0.43063181,  0.43067501,  0.17489478,  0.31970233]])
    
    initial = initial/np.linalg.norm(initial)
    Initial = initial
    
    #initial 2*100 random
    initial = np.array([[ 0.97665844,  0.95750406,  0.03271502,  0.00581718,  0.75326103,
         0.42090098,  0.05777608,  0.36385444,  0.77925131,  0.48679857,
         0.75298814,  0.21078591,  0.80945967,  0.40945785,  0.65163067,
         0.97084134,  0.9847005 ,  0.98048906,  0.85180357,  0.17726631,
         0.2142386 ,  0.77630585,  0.39611428,  0.0113059 ,  0.48309224,
         0.08266567,  0.25872249,  0.02812381,  0.05573787,  0.45735282,
         0.02606287,  0.15692817,  0.76003303,  0.88962458,  0.20016448,
         0.28102736,  0.34623372,  0.41796777,  0.05089445,  0.24520444,
         0.75408916,  0.99938743,  0.33841862,  0.78133493,  0.71211481,
         0.99696014,  0.99613546,  0.53147885,  0.89938489,  0.19039678,
         0.17815047,  0.71516141,  0.23132947,  0.93848227,  0.81350492,
         0.1968643 ,  0.45311155,  0.92871007,  0.00425793,  0.56769485,
         0.06160494,  0.8953797 ,  0.86951071,  0.82065772,  0.32179128,
         0.37144852,  0.54846107,  0.42796179,  0.55969656,  0.44669268,
         0.61123395,  0.16936111,  0.2328948 ,  0.77969569,  0.13924452,
         0.91607878,  0.35735768,  0.49425793,  0.93077064,  0.35861028,
         0.27830127,  0.41066325,  0.40058592,  0.63938132,  0.71501647,
         0.63328468,  0.94666696,  0.00186661,  0.21478808,  0.82069927,
         0.66078568,  0.89309345,  0.08740547,  0.5898963 ,  0.33406322,
         0.95293804,  0.27126105,  0.80908096,  0.64361073,  0.91884982],
       [ 0.1410154 ,  0.23378287,  0.86277615,  0.46195547,  0.85591931,
         0.64633902,  0.24505287,  0.61047091,  0.43316428,  0.74921257,
         0.13719463,  0.41421003,  0.65311778,  0.56415643,  0.98927146,
         0.07921332,  0.47367136,  0.82073036,  0.79671794,  0.05769988,
         0.73586324,  0.54851267,  0.54374085,  0.74149884,  0.93549049,
         0.04258554,  0.34599344,  0.55134441,  0.2749176 ,  0.93491757,
         0.36611495,  0.43260244,  0.83826849,  0.9911186 ,  0.12426802,
         0.36558696,  0.61977703,  0.32504721,  0.36111586,  0.46606772,
         0.24870376,  0.52107697,  0.12735579,  0.17526848,  0.17233384,
         0.37906054,  0.70848099,  0.47396076,  0.35338456,  0.28716993,
         0.85685955,  0.49379057,  0.83172229,  0.83839338,  0.64035139,
         0.94613018,  0.20814217,  0.22025412,  0.96669821,  0.42949032,
         0.70682227,  0.7643414 ,  0.71941786,  0.80184449,  0.16006022,
         0.2735027 ,  0.41076794,  0.04037073,  0.4818489 ,  0.71783373,
         0.11031807,  0.00634837,  0.07737157,  0.30327055,  0.35530241,
         0.29339109,  0.11278454,  0.54622749,  0.3403016 ,  0.73388964,
         0.10194951,  0.0657555 ,  0.46220374,  0.15886147,  0.65854526,
         0.64224593,  0.00663153,  0.28882211,  0.49851745,  0.28379566,
         0.51959189,  0.75907969,  0.52977569,  0.01555053,  0.19439624,
         0.18353673,  0.3520964 ,  0.53183152,  0.67932248,  0.26928893]])
   
    #write initial state at the .txt

    
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
    initial=initial.reshape(n,m)

    #repeated Schmidt decompotion
    NORM=0.
    x=1 
    xmax=50
    while (x<=xmax) :
        NORM=0.

        '''step 1 of the algorithm : psiA,psiB Schmidt vector A,B and l Schmidt coefficients '''
        psiA, l, psiB = np.linalg.svd(initial,full_matrices=1) #decomposition of initial matrix
        
        '''step 2 of the algorithm : definition of the normalized vector '''   
           
        p=1.0 # p has to be larger or equal than 1 for the algorithm to work
        sum=np.zeros((n,m))
        for i in range(k) :
            tens= tensor(psiA[i],psiB[i])
            sum += math.pow(l[i],p-1)*tens
            NORM = NORM + math.pow(l[i],p)      
         
        NORM = math.pow(NORM,1./p)
            
        sum=sum.reshape(m*n,1) 
        psi= np.dot(Proj,sum) #projection on the subspace
        psi=psi/np.linalg.norm(psi)
           
        with open('Reuvers_n2m100_initial_niter1_T1_xmax50_onlyNORM.txt', 'a+') as f:
            print (x,',',NORM,file=f)
        f.close()  
    
    
            
        '''step 3 of the algorithm '''
        initial=psi
        initial=initial.reshape(n,m)
        
        if (-NORM+math.sqrt(2)<0.00001) :
            f = open("Reuvers_n2m100_initial_niter1_T1_xmax50.txt","a+")
            f.write("initial state")
            f.close()
            with open('Reuvers_n2m100_initial_niter1_T1_xmax50.txt', 'a+') as f:
                print (Initial,file=f)
            f.close()
            
            f = open("Reuvers_n2m100_initial_niter1_T1_xmax50.txt","a+")
            f.write("final")
            f.close()
            with open('Reuvers_n2m100_initial_niter1_T1_xmax50.txt', 'a+') as f:
                print (initial,file=f)
            f.close()
            
                #write the subspace A at the txt file
            with open('Reuvers_n2m100_initial_niter1_T1_xmax50.txt', 'a+') as f:
                print ("ok3,A",A,file=f)
            f.close()
            
                       #write norm at the .txt
            f=open("Reuvers_n2m100_initial_niter1_T1_xmax50.txt","a+")
            f.write("t,NORM ")
            f.close()
            with open('Reuvers_n2m100_initial_niter1_T1_xmax50.txt', 'a+') as f:
                print (x,' ',NORM,file=f)
            f.close()  
    

        x+=1 
    return (-NORM+math.sqrt(2))




n=2 #total states fo the subsystem A
m=100 #total states fo the subsystem B
k=min(n,m)

my_randoms=[]
for i in range (m*n*k):
    my_randoms.append(random.randrange(1,10,1))

print (my_randoms)

x0 = my_randoms


f = open("Reuvers_n2m100_initial_niter1_T1_xmax50.txt","a+")
f.write("initial guess of A")
f.close()
with open('Reuvers_n2m100_initial_niter1_T1_xmax50.txt', 'a+') as f:
    print (x0,file=f)
f.close()

minimizer_kwargs = {"method": "BFGS"}
ret = optimize.basinhopping(func,x0, minimizer_kwargs=minimizer_kwargs,niter=1,T=1.0, disp = True )
   

#define the correct subspace form the result ret.x of the bashin-hopping algorithm
l=0
B=np.zeros((m*n,k))
for i in range (0,m*n,1):
    for j in range (0,k,1):
        B[i][j] = ret.x[l]
        l+=1





