import random
import numpy as np
## k_{b} = J = 1

##Generates N^2 lattice with random states from \in [-1,1]
def random_lattice(n):
 return np.random.choice([-1,1],size =(n,n))

## E_{i} = -J * \sum_{ij} S_{i}*S_{j} i-->state j-->nearest neighbour (w PBC)
def Energy_of_state_I(intial_state,i,j,n):
 return (-1 * intial_state[i,j]*( intial_state[(i-1)%n,j] + intial_state[(i+1)%n,j] + intial_state[i,(j+1)%n] + intial_state[i,(j-1)%n] ))

##Metropolis condition prob of accepting \in min{1,\exp{\cfrac{deltaE}{-k_{b}T}}}
def Metropolis(E_diff,T): 
  return min(1,np.exp(-1* E_diff/T))

#Intial_state = random_lattice(3)
#print(Intial_state)
#print(np.roll(Intial_state,(1,1),0))
#y = np.array([[1 , 2, 3],[4,5,6],[7,8,9]])
#print(y)
#print(np.roll(y,-1,axis=1))

def mcmc_updater(intial_state, T,n,i,j):

 #i = random.choice(range(n-1))
 #j = random.choice(range(n-1))
 S_state = intial_state[i][j]
 E_int =  Energy_of_state_I(intial_state,i,j,n)
 intial_state[i,j] *= -1 
 E_diff = 2.0 * E_int
 probability = Metropolis(E_diff,T)
 if random.random() < probability:
   return intial_state
 else:
    return S_state





 