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
Pdata7=  np.genfromtxt(f'Input{slash}G.txt', delimiter=',')  
Pdata8=  np.genfromtxt(f'Input{slash}H.txt', delimiter=',')  
Pdata9=  np.genfromtxt(f'Input{slash}I.txt', delimiter=',')  
Pdata10=  np.genfromtxt(f'Input{slash}J.txt', delimiter=',')  
Pdata11=  np.genfromtxt(f'Input{slash}K.txt', delimiter=',')
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
P25data6=  np.genfromtxt(f'Input{slash}F25.txt', delimiter=',')  
P25data7=  np.genfromtxt(f'Input{slash}G25.txt', delimiter=',')  
P25data8=  np.genfromtxt(f'Input{slash}H25.txt', delimiter=',')
P25data9=  np.genfromtxt(f'Input{slash}I25.txt', delimiter=',')  
P25data10=  np.genfromtxt(f'Input{slash}J25.txt', delimiter=',')
P25data11=  np.genfromtxt(f'Input{slash}K25.txt', delimiter=',')  
P25data12=  np.genfromtxt(f'Input{slash}L25.txt', delimiter=',')  
P25data13=  np.genfromtxt(f'Input{slash}M25.txt', delimiter=',')
P25data14=  np.genfromtxt(f'Input{slash}N25.txt', delimiter=',')
P25data15=  np.genfromtxt(f'Input{slash}O25.txt', delimiter=',')  
P25data16=  np.genfromtxt(f'Input{slash}P25.txt', delimiter=',')  
P25data17=  np.genfromtxt(f'Input{slash}Q25.txt', delimiter=',') 


New_Pdata1 = np.concatenate((Pdata1,P25data1))
New_Pdata2= np.concatenate((Pdata7,P25data17))
New_Pdata3 = np.concatenate((Pdata11,P25data13))
New_Pdata4 = np.concatenate((Pdata10,P25data12))
New_Pdata5 = np.concatenate((Pdata6,P25data16))
New_Pdata6 = np.concatenate((Pdata15,P25data6))
New_Pdata7 = np.concatenate((Pdata16,P25data7))
New_Pdata8 = np.concatenate((Pdata17,P25data8))
New_Pdata9 = np.concatenate((Pdata9,P25data11))
New_Pdata10 = np.concatenate((Pdata8,P25data15))
New_Pdata11 = np.concatenate((Pdata2,P25data2))
New_Pdata12 = np.concatenate((Pdata3,P25data3))
New_Pdata13 = np.concatenate((Pdata4,P25data4))
New_Pdata14 = np.concatenate((Pdata5,P25data5))
New_Pdata15 = np.concatenate((Pdata14,P25data9))
New_Pdata16 = np.concatenate((Pdata13,P25data10))
New_Pdata17 = np.concatenate((Pdata12,P25data14))

"""Modelling the P state"""

Pmodel00 = interp1d(New_Pdata1, New_Pdata2, kind='cubic',bounds_error=False) #F=0 mF=0

Pmodel1_1 = interp1d(New_Pdata1, New_Pdata3, kind='cubic',bounds_error=False) #F=1 mF=-1
Pmodel10 = interp1d(New_Pdata1, New_Pdata4, kind='cubic',bounds_error=False) #F=1 mF=0
Pmodel11 = interp1d(New_Pdata1, New_Pdata5, kind='cubic',bounds_error=False) #F=1 mF=1

Pmodel2_2 = interp1d(New_Pdata1, New_Pdata6, kind='cubic',bounds_error=False) #F=2 mF=-2
Pmodel2_1 = interp1d(New_Pdata1, New_Pdata7, kind='cubic',bounds_error=False) #F=2 mF=-1
Pmodel20 = interp1d(New_Pdata1, New_Pdata8, kind='cubic',bounds_error=False) #F=2 mF=0
Pmodel21 = interp1d(New_Pdata1, New_Pdata9, kind='cubic',bounds_error=False) #F=2 mF=1
Pmodel22 = interp1d(New_Pdata1, New_Pdata10, kind='cubic',bounds_error=False) #F=2 mF=2

Pmodel3_3 = interp1d(New_Pdata1, New_Pdata11, kind='cubic',bounds_error=False) #F=3 mF=-3
Pmodel3_2 = interp1d(New_Pdata1, New_Pdata12, kind='cubic',bounds_error=False) #F=3 mF=-2
Pmodel3_1 = interp1d(New_Pdata1, New_Pdata13, kind='cubic',bounds_error=False) #F=3 mF=-1
Pmodel30 = interp1d(New_Pdata1, New_Pdata14, kind='cubic',bounds_error=False) #F=3 mF=0
Pmodel31 = interp1d(New_Pdata1, New_Pdata15, kind='cubic',bounds_error=False) #F=3 mF=1
Pmodel32 = interp1d(New_Pdata1, New_Pdata16, kind='cubic',bounds_error=False) #F=3 mF=2
Pmodel33 = interp1d(New_Pdata1, New_Pdata17, kind='cubic',bounds_error=False) #F=3 mF=3

"""Modeling the S-state"""

BreitRabiE = np.genfromtxt(f'Input{slash}2S_BreitRabi.csv', delimiter=',') #get breit rabi data for 2S1/2 state
Smodel11 = interp1d(BreitRabiE[:,0], BreitRabiE[:,1], kind='cubic',bounds_error=False) #F=1 mF=1
Smodel10 = interp1d(BreitRabiE[:,0], BreitRabiE[:,2], kind='cubic',bounds_error=False) #F=1 mF=0
Smodel1_1 = interp1d(BreitRabiE[:,0], BreitRabiE[:,3], kind='cubic',bounds_error=False) #F=1 mF=-1

Smodel2_2 = interp1d(BreitRabiE[:,0], BreitRabiE[:,4], kind='cubic',bounds_error=False) #F=2 mF=-2
Smodel2_1 = interp1d(BreitRabiE[:,0], BreitRabiE[:,5], kind='cubic',bounds_error=False) #F=2 mF=-1
Smodel20 = interp1d(BreitRabiE[:,0], BreitRabiE[:,6], kind='cubic',bounds_error=False) #F=2 mF=0
Smodel21 = interp1d(BreitRabiE[:,0], BreitRabiE[:,7], kind='cubic',bounds_error=False) #F=2 mF=1
Smodel22 = interp1d(BreitRabiE[:,0], BreitRabiE[:,8], kind='cubic',bounds_error=False) #F=2 mF=2


class zeeman_shift:
    
    def __init__(self):
        
        self._S2 = [Smodel22, Smodel21, Smodel20, Smodel2_1, Smodel2_2] #array for models for S levels F=2
        self._S1 = [Smodel11, Smodel10, Smodel1_1] #array for models for S levels F=1

        self._P3 = [Pmodel33, Pmodel32, Pmodel31, Pmodel30, Pmodel3_1, Pmodel3_2, Pmodel3_3] #holds models for P levels F=3
        self._P2 = [Pmodel22, Pmodel21, Pmodel20, Pmodel2_1, Pmodel2_2] #holds models for P levels F=2
        self._P1 = [Pmodel11, Pmodel10, Pmodel1_1] #holds models for P levels F=1
        self._P0 = [Pmodel00] #holds models for P levels F=0

    def S_energy(self, F, mF, Bfield):
        if F==2:
            return self._S2[2-mF](Bfield) #return the zeeman shift (2pi*Hz) for 2S1/2 state for the given mF value
        elif F==1:
            return self._S1[1-mF](Bfield)

    def P_adjacent_energies(self, F, mF, Bfield):
        #doesn't work for mF=+-3 but that should never be an input, max input should be mF=+-2
        if F==3:
            return (self._P3[-1 + 3-mF](Bfield), self._P3[3-mF](Bfield), self._P3[1 + 3-mF](Bfield)) #return the zeeman shifts (2pi*Hz) for the 2P3/2 state for adjacent mF levels as to the one specified (Deltamf=+1,0,-1)

        elif F==2:
            if mF==2:
                return (np.nan, self._P2[2-mF](Bfield), self._P2[1 + 2-mF](Bfield)) #+1 transitions is impossible
            elif mF==-2:
                return (self._P2[-1 + 2-mF](Bfield), self._P2[2-mF](Bfield), np.nan) #-1 transition is impossible
            else:
                return (self._P2[-1 + 2-mF](Bfield), self._P2[2-mF](Bfield), self._P2[1 + 2-mF](Bfield))

        elif F==1:
            if mF==2:
                return (np.nan, np.nan, self._P1[1 + 1-mF](Bfield))
            elif mF==-2:
                return (self._P1[-1 + 1-mF](Bfield), np.nan, np.nan)
            elif mF==1:
                return (np.nan, self._P1[1-mF](Bfield), self._P1[1 + 1-mF](Bfield))
            elif mF==-1:
                return (self._P1[-1 + 1-mF](Bfield), self._P1[1-mF](Bfield), np.nan)
            else:
                return (self._P1[-1 + 1-mF](Bfield), self._P1[1-mF](Bfield), self._P1[1 + 1-mF](Bfield))

        elif F==0:
            #should only have to consider max mF=+-1 since this state is only accessed from the F=1 state

            if mF==1:
                return (np.nan, np.nan, self._P0[1 + -mF](Bfield))
            elif mF==-1:
                return (self._P0[-1 + -mF](Bfield), np.nan, np.nan)
            else:
                return (np.nan, self._P0[-mF](Bfield), np.nan)