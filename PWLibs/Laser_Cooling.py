import numpy as np
import scipy.constants as sc
from scipy.interpolate import interp1d
from math import isnan
from sys import platform
from PWLibs.Zeeman_Shift import zeeman_shift

# Location of the field source file, depending on operating platform
if platform == "win32":
	slash = "\\"
elif platform == "linux" or "linux2":
	slash = "/"

gamma = 5.92e6 #natural linewidth of Lithium
omega_L =  2.80738910e+15 #2.8073894195e15
lambda_L = 2*np.pi*(3e8)/omega_L #6.7142624787e-7 # Wavelength of detuned laser
s_L = 2 # 0.1 #I/Isat for the cooling laser
detune = omega_L - (2*np.pi*4.468107035e14 - 2*np.pi*803.5e6) #detuning of the laser
mLi = 6.941*sc.physical_constants["atomic mass constant"][0] #mass of lithium atom in kg
#below is array of k vectors for each laser
kLs = [np.array([2*np.pi/lambda_L,0,0]),np.array([-2*np.pi/lambda_L,0,0]),np.array([0,2*np.pi/lambda_L,0]),np.array([0,-2*np.pi/lambda_L,0]),np.array([0,0,2*np.pi/lambda_L]),np.array([0,0,-2*np.pi/lambda_L])]
Zeeman_Shift = zeeman_shift() #define class used to calculate zeeman shifts for different states.

def M(theta):
    '''This function defines the rotation matrix that enters the local coordinate system of the atom in order to calcaulte the W_j paramter in the probability function'''
    M = 0.5*np.array([[1+np.cos(theta), -np.sqrt(2)*np.sin(theta), 1- np.cos(theta)],
    [np.sqrt(2)*np.sin(theta), 2*np.cos(theta), -np.sqrt(2)*np.sin(theta)],
    [ 1- np.cos(theta), np.sqrt(2)*np.sin(theta), 1+np.cos(theta) ]], dtype = complex)
    return M

def calculate_probability(atom, mF, trap, dt):
    probabilities = np.zeros((6,3)) #array of the probabilities of the 3 transitions (delta mF of 1, 0 ,-1) from each of the 6 lasers

    for i in range(6):

        #calculate W_ij
        B = trap.rQuery(np.array([atom]))[0] #vector of the magnetic field at the location of the atom
        theta = np.arccos(np.dot(B,kLs[i])/(np.linalg.norm(B)*np.linalg.norm(kLs[i]))) #angle in radians between the magnetic field and the laser k vector
        E_0 = np.sqrt(s_L)

        if i/3 <= 1: #create electric field vector for the laser 
            #laser is in x or y direction (the beams are LCP wrt propagation)
            E = np.array([E_0*np.exp(1j*np.dot(kLs[i],atom[:3])),1j*E_0*np.exp(1j*np.dot(kLs[i],atom[:3])),0])
        else:
            #laser is in z direction (the beams are RCP wrt to propagation)
            E = np.array([E_0*np.exp(1j*np.dot(kLs[i],atom[:3])),-1j*E_0*np.exp(1j*np.dot(kLs[i],atom[:3])),0])

        E = np.array( [(-1/np.sqrt(2)) * (E[0] + 1j*E[1]), E[2], (1/np.sqrt(2)) * (E[0] - 1j*E[1])], dtype=complex ) #create electric field vector in spherical basis

        E = np.matmul(M(theta),E) #calculate electric field vector in rotated frame

        W = np.array([abs(E[0]),abs(E[1]), abs(E[2])])**2 #calculate Wij weights for transitions (mf=+1 , 0 , -1)

        Bmod = np.linalg.norm(B)
        lower_state_shift = Zeeman_Shift.S_energy(int(mF),Bmod) #calculate zeeman shift of the current atom state #L=0 J=1/2 F=2

        upper_state_shifts = Zeeman_Shift.P_adjacent_energies(int(mF), Bmod) #calculate zeeman shift of possible upper atoms states (deltamF=+1,0,-1)

        probabilities[i] = (gamma*s_L*dt/2)*np.array([W[0]/(1+s_L*W[0] + (4/gamma**2) * (detune + np.dot(kLs[i],atom[3:]) + lower_state_shift - upper_state_shifts[0])**2), #calculate probability of transition to each possible upper state.
        W[1]/(1+s_L*W[1] + (4/gamma**2) * (detune + np.dot(kLs[i],atom[3:]) + lower_state_shift - upper_state_shifts[1])**2),
        W[2]/(1+s_L*W[2] + (4/gamma**2) * (detune + np.dot(kLs[i],atom[3:]) + lower_state_shift - upper_state_shifts[2])**2)])

    #below samples the probabilities to find whether a transition occurs

    prob_1 = probabilities[0,0]

    random_number = np.random.rand()

    if 0 < random_number < prob_1: #see if the first possible transition occurs (1st laser with deltamF=+1)
        return (0,0)

    for i in range(6): #loop through each index of laser
        for j in range(3): #loop through each index of transition

            if j==2: 
                if i==5:
                    return(np.nan, np.nan) #no transition occurs so return NaN

                prob_2 = prob_1 + probabilities[i+1,0]
                x , y = i + 1, 0 #have to shift indicies since the transition we are examining is on the next line of the array

            else:
                prob_2 = prob_1 + probabilities[i, j+1]
                x , y = i , j + 1 #we examine if the transition described by (i , j + 1) occurs.


            if prob_1 <= random_number < prob_2:
                return x,y #return the transition that occurs x=laser index y=transition index.

            prob_1 = prob_2


def Laser(atom_array, trap,dt):
    indicies = []
    for i, atom in enumerate(atom_array):

        if isnan(atom[0]): #check if atom has left the trap
            indicies.append(i) #add index to list to remove row from array at end of loop
            continue #continue to next atom

        laser_index, transition_index = calculate_probability(atom[:6], atom[6], trap, dt) #check whether a transition occurs for the atom

        if isnan(laser_index):
            continue #if no interaction then skip to next atom

        #if there is an interaction
        k_mod = np.linalg.norm(kLs[laser_index]) #k vector of laser that atom interacted with

        theta = np.random.rand()*np.pi 
        phi = np.random.rand()*2*np.pi

        k_s = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]) #select k vector of photon spontaneously emitted using a random uniform distribution

        atom_array[i,3:6] += ((sc.hbar*k_mod)/mLi) * (kLs[laser_index]/k_mod + k_s) #change velocity of atom due to interaction event
        upper_state = atom[6] + (1 - transition_index) #mF value of the upper state the atom transitions to

        if abs(upper_state) == 3: #atom transitioned to mF=3 state
            atom_array[i,6] = 2 * upper_state/3 #atom returns to either mF=+2,-2 state.
        
        elif abs(upper_state) == 2: #atom transitioned to mF=2 state
            atom_array[i,6] = (upper_state/2) * np.random.randint(1,3) #atom returns to either the mF=+-1 state or mF=+-2 state with a 50/50 chance of either

        else: #atom is in mF=+1,0,-1
            atom_array[i,6] = upper_state + np.random.randint(-1,2) #mF can change by any of +1,0,-1 

    atom_array = np.delete(atom_array,indicies,axis=0) #delete rows of atoms that have left the trap

    return atom_array