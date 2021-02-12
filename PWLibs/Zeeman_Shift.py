import numpy as np
from scipy.interpolate import interp1d
from sys import platform

# Location of the field source file, depending on operating platform
if platform == "win32":
	slash = "\\"
elif platform == "linux" or "linux2":
	slash = "/"

"""Importing Data files from 2P3/2 breit rabi diagram for calculating 2P3/2 zeeman shift"""
Pdata1= np.genfromtxt(f'Input{slash}A.txt', delimiter=',')  
Pdata2=  np.genfromtxt(f'Input{slash}B.txt', delimiter=',')  
Pdata3=  np.genfromtxt(f'Input{slash}C.txt', delimiter=',')  
Pdata4=  np.genfromtxt(f'Input{slash}D.txt', delimiter=',')  
Pdata5=  np.genfromtxt(f'Input{slash}E.txt', delimiter=',')  
Pdata6=  np.genfromtxt(f'Input{slash}F.txt', delimiter=',') 
Pdata12=  np.genfromtxt(f'Input{slash}L.txt', delimiter=',')  
Pdata13=  np.genfromtxt(f'Input{slash}M.txt', delimiter=',')  
Pdata14=  np.genfromtxt(f'Input{slash}N.txt', delimiter=',')  
Pdata15=  np.genfromtxt(f'Input{slash}O.txt', delimiter=',')  
Pdata16=  np.genfromtxt(f'Input{slash}P.txt', delimiter=',')  
Pdata17=  np.genfromtxt(f'Input{slash}Q.txt', delimiter=',') 

P25data1= np.genfromtxt(f'Input{slash}A25.txt', delimiter=',')  
P25data2=  np.genfromtxt(f'Input{slash}B25.txt', delimiter=',')  
P25data3=  np.genfromtxt(f'Input{slash}C25.txt', delimiter=',')  
P25data4=  np.genfromtxt(f'Input{slash}D25.txt', delimiter=',')  
P25data5=  np.genfromtxt(f'Input{slash}E25.txt', delimiter=',')  
P25data9=  np.genfromtxt(f'Input{slash}I25.txt', delimiter=',')  
P25data10=  np.genfromtxt(f'Input{slash}J25.txt', delimiter=',')   
P25data14=  np.genfromtxt(f'Input{slash}N25.txt', delimiter=',')  

New_Pdata1 = np.concatenate((Pdata1,P25data1))
New_Pdata11 = np.concatenate((Pdata2,P25data2))
New_Pdata12 = np.concatenate((Pdata3,P25data3))
New_Pdata13 = np.concatenate((Pdata4,P25data4))
New_Pdata14 = np.concatenate((Pdata5,P25data5))
New_Pdata15 = np.concatenate((Pdata14,P25data9))
New_Pdata16 = np.concatenate((Pdata13,P25data10))
New_Pdata17 = np.concatenate((Pdata12,P25data14))

Pmodel3_3 = interp1d(New_Pdata1, New_Pdata11, kind='cubic',bounds_error=False) #F=3 mF=-3
Pmodel3_2 = interp1d(New_Pdata1, New_Pdata12, kind='cubic',bounds_error=False) #F=3 mF=-2
Pmodel3_1 = interp1d(New_Pdata1, New_Pdata13, kind='cubic',bounds_error=False) #F=3 mF=-1
Pmodel30 = interp1d(New_Pdata1, New_Pdata14, kind='cubic',bounds_error=False) #F=3 mF=0
Pmodel31 = interp1d(New_Pdata1, New_Pdata15, kind='cubic',bounds_error=False) #F=3 mF=1
Pmodel32 = interp1d(New_Pdata1, New_Pdata16, kind='cubic',bounds_error=False) #F=3 mF=2
Pmodel33 = interp1d(New_Pdata1, New_Pdata17, kind='cubic',bounds_error=False) #F=3 mF=3

"""Modeling the S-state"""

BreitRabiE = np.genfromtxt(f'Input{slash}2S_BreitRabi.csv', delimiter=',') #get breit rabi data for 2S1/2 state
#Smodel11 = interp1d(BreitRabiE[:,0], BreitRabiE[:,1], kind='cubic',bounds_error=False) #F=1 mF=1
#Smodel10 = interp1d(BreitRabiE[:,0], BreitRabiE[:,2], kind='cubic',bounds_error=False) #F=1 mF=0
#Smodel1_1 = interp1d(BreitRabiE[:,0], BreitRabiE[:,3], kind='cubic',bounds_error=False) #F=1 mF=-1
Smodel2_2 = interp1d(BreitRabiE[:,0], BreitRabiE[:,4], kind='cubic',bounds_error=False) #F=2 mF=-2
Smodel2_1 = interp1d(BreitRabiE[:,0], BreitRabiE[:,5], kind='cubic',bounds_error=False) #F=2 mF=-1
Smodel20 = interp1d(BreitRabiE[:,0], BreitRabiE[:,6], kind='cubic',bounds_error=False) #F=2 mF=0
Smodel21 = interp1d(BreitRabiE[:,0], BreitRabiE[:,7], kind='cubic',bounds_error=False) #F=2 mF=1
Smodel22 = interp1d(BreitRabiE[:,0], BreitRabiE[:,8], kind='cubic',bounds_error=False) #F=2 mF=2


class zeeman_shift:
    
    def __init__(self):
        
        self._S = [Smodel22, Smodel21, Smodel20, Smodel2_1, Smodel2_2] #array for models for S levels

        self._P = [Pmodel33, Pmodel32, Pmodel31, Pmodel30, Pmodel3_1, Pmodel3_2, Pmodel3_3] #holds models for P levels

    def S_energy(self, mF, Bfield):

        return self._S[2-mF](Bfield) #return the zeeman shift (2pi*Hz) for 2S1/2 state for the given mF value

    def P_adjacent_energies(self, mF, Bfield):
        #doesn't work for mF=+-3 but that should never be an input, max input should be mF=+-2
        return (self._P[1 + 3-mF](Bfield), self._P[3-mF](Bfield), self._P[-1 + 3-mF](Bfield)) #return the zeeman shifts (2pi*Hz) for the 2P3/2 state for adjacent mF levels as to the one specified