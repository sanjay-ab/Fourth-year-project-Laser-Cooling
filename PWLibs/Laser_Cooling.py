import numpy as np
import scipy.constants as sc
from math import isnan
from sys import platform
from PWLibs.Zeeman_Shift import zeeman_shift

# Location of the field source file, depending on operating platform
if platform == "win32":
	slash = "\\"
elif platform == "linux" or "linux2":
	slash = "/"

mLi = 6.941*sc.physical_constants["atomic mass constant"][0] #mass of lithium atom in kg
gamma = 2*np.pi * 5.92e6 #natural linewidth of Lithium (2pi*hz)
const1 = 4/gamma**2 
omega_0 = 2.807389399e+15 #frequency of 2S1/2 -> 2P3/2 transition excluding hyperfine shifts
omega = 2.807387462245734e15 #frequency of F=2 -> F'=3 laser transtition including the hyperfine shifts at zero field
omega_r = 2.80739257e15 #frequency of F=1 -> F'=2 repump transition including the hyperfine shifts at zero field
laser_origins = [np.array([-1e-2,0,0]),np.array([1e-2,0,0]),np.array([0,-1e-2,0]),np.array([0,1e-2,0]),np.array([0,0,-1e-2]),np.array([0,0,1e-2])] #origin of lasers
laser_vectors = [np.array([1,0,0]),np.array([-1,0,0]),np.array([0,1,0]),np.array([0,-1,0]),np.array([0,0,1]),np.array([0,0,-1])] #directions of lasers
Zeeman_Shift = zeeman_shift() #define class used to calculate zeeman shifts for different states.

def mF_state(F,mF):
    """Calculates the mF state the atom falls to depending on the mF value of the upper state it decays from and the F value of the ground state it falls to"""
    if F==2: #atom return to F=2 state
        if mF == 3: #atom transitioned to mF=3 state
            return 2  #atom returns to either mF=+2,-2 state.
        elif mF == -3:
            return -2
        elif mF == 2: #atom transitioned to mF=2 state
            return  np.random.randint(1,3) #atom returns to either the mF=+-1 state or mF=+-2 state with a 50/50 chance of either
        elif mF == -2:
            return - np.random.randint(1,3)
        else:
            return mF + np.random.randint(-1,2) #mF can change by any of +1,0,-1

    else: #atom returns to F=1 state
        if mF == 2: #atom transitioned to mF=2 state
            return  1 #atom returns to mF=+1
        elif mF == -2:
            return - 1
        elif mF == 1:
            return np.random.randint(0,2)
        elif mF == -1:
            return - np.random.randint(0,2)
        else: 
            return np.random.randint(-1,2) #mF can be any of +1,0,-1

def calculate_probability(atom, mF, F, B, dt, omega_L, S_0, W_0, k_mod):
    probabilities = np.zeros((6,3,3)) #array of the probabilities of the 3 transitions (delta mF of 1, 0 ,-1) from each of the 6 lasers and each F level (F+1, F, F-1)
    
    for k in range(3):
        #loop through each F value 
        for i in range(6): 
            #loop through each laser
            #calculate W_ij
            Bmod = np.linalg.norm(B)
            theta = np.arccos(np.dot(B,laser_vectors[i])/Bmod) #angle in radians between the magnetic field and the laser k vector
            k_L = k_mod * laser_vectors[i] #k vector of laser
            atom[:3] -= laser_origins[i] #shift atom by distance to laser

            if i in (0,1): #x lasers
                #x and y direction lasers are LCP wrt propagation direction
                index1, index2, index3 = 0, 1, 2 #define indicies for calculating waist of laser
                pol = 1 # define laser as LCP polarized (LCP photon possesses +hbar of momentum so effects the mF+1 transition?)

            elif i in (2,3): #+y laser
                index1, index2, index3 = 1, 2, 0
                pol = 1 # define laser as LCP polarized

            else: #+z laser
                #z direction lasers are RCP wrt propagation direction
                index1, index2, index3 = 2, 0, 1
                pol = 1 # define laser as RCP polarized 

            W = W_0 * np.sqrt(1 + ( 2*atom[index1] / (k_mod * W_0**2) )**2 ) #calculate waist. 
            S = S_0 * (W_0/W)**2 * np.exp( -2 * (atom[index2]**2 + atom[index3]**2 ) / W**2 ) #calculate I/Isat using eqn for gaussian beam intensity

            E = 0.5 * np.array([pol - np.cos(theta), -np.sqrt(2) * np.sin(theta), pol + np.cos(theta)]) #rotated electric field vector in spherical basis

            Wi = np.array([abs(E[0]),abs(E[1]), abs(E[2])])**2 #calculate Wij weights for transitions (mf=+1 , 0 , -1)

            lower_state_shift = Zeeman_Shift.S_energy(int(F),int(mF),Bmod) #calculate zeeman shift of the current atom state #L=0 J=1/2 F=2

            upper_state_shifts = Zeeman_Shift.P_adjacent_energies(int(F-k+1),int(mF), Bmod) #calculate zeeman shift of possible upper atoms states (deltamF=+1,0,-1)

            const2 = (omega_L - omega_0) - np.dot(k_L,atom[3:]) #Delta - dot(k,v)

            for j in range(3):
                if isnan(upper_state_shifts[j]):
                    probabilities[i,j,k] = 0 #if the transition is impossible
                else:
                    probabilities[i,j,k] = (gamma*S*dt/2) * np.array(Wi[j]/(1+S*Wi[j] + const1 * (const2 + 2*np.pi*(lower_state_shift - upper_state_shifts[j]))**2)) #calculate probability of transition to each possible upper state.

    #below samples the probabilities to find whether a transition occurs
    random_number = np.random.rand()

    prob_1 = probabilities[0,0,0]

    if 0 < random_number < prob_1: #see if the first possible transition occurs (1st laser with deltamF=+1)
        return (0,0,0)

    for k in range(3): #loop through each index of F transition
        for i in range(6): #loop through each index of laser
            for j in range(3): #loop through each index of mF transition

                if j==2: 
                    if i==5:
                        if k==2:
                            return (np.nan, np.nan, np.nan)
                        prob_2 = prob_1 + probabilities[0 ,0 ,k+1]
                        x , y , z = 0, 0 , k+1
                    else:
                        prob_2 = prob_1 + probabilities[i+1 ,0 ,k]
                        x , y , z = i + 1, 0 , k  #have to shift indicies since the transition we are examining is on the next line of the array

                else:
                    prob_2 = prob_1 + probabilities[i, j+1, k]
                    x , y , z= i , j + 1 , k  #we examine if the transition described by (i , j + 1) occurs.


                if prob_1 <= random_number < prob_2:
                    return x,y,z #return the transition that occurs x=laser index y=transition index.

                prob_1 = prob_2


def Laser(atom_array, trap, dt, detuning, detuning_r, S_0, W_0):
    indicies = []
    B = trap.rQuery(atom_array) #vector magnetic field at the position of each atom
    omega_pump = omega + detuning
    k_L = omega_pump/sc.c #modulus of detuned lasers k vectors

    omega_repump = omega_r + detuning_r
    k_r = omega_repump/sc.c #modulus of detuned lasers k vectors

    for i, atom in enumerate(atom_array):

        if isnan(atom[0]): #check if atom has left the trap
            indicies.append(i) #add index to list to remove row from array at end of loop
            continue #continue to next atom

        if atom[7] == 2: #if atom in F=2 state
            omega_L = omega_pump #set frequency to pump laser frequency as atom only interacts with this laser
            k_mod = k_L
        else: #atom in F=1 state
            omega_L = omega_repump #frequency of repump laser as atom only interacts with repump laser
            k_mod = k_r

        laser_index, transition_index_mF, transition_index_F = calculate_probability(atom[:6], atom[6], atom[7], B[i], dt, omega_L, S_0, W_0, k_mod) #check whether a transition occurs for the atom

        if isnan(laser_index):
            continue #if no interaction then skip to next atom

        #if there is an interaction

        theta = np.random.rand()*np.pi 
        phi = np.random.rand()*2*np.pi

        k_s = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]) #select k vector of photon spontaneously emitted using a random uniform distribution

        atom_array[i,3:6] += ((sc.hbar*k_mod)/mLi) * (laser_vectors[laser_index] + k_s) #change velocity of atom due to interaction event
        upper_state_mF = atom[6] + (1 - transition_index_mF) #mF value of the upper state the atom transitions to
        upper_state_F = atom[7] + (1- transition_index_F)

        if upper_state_F == 3: #if atom transitioned to F=3 state
            atom_array[i,6] = mF_state(2, upper_state_mF)
        elif upper_state_F == 0:
            atom_array[i,6] = mF_state(1, upper_state_mF) #don't have to set F state of atoms if it excites to either F=3 (0) since it must have been in F=2 (1) so it does not change 
        else: #F=1,2 states
            if np.random.rand()<(1/3):
                atom_array[i,7] = 2 #1/3 chance to return to F=2 state
                atom_array[i,6] = mF_state(2, upper_state_mF)
            else:
                atom_array[i,7] = 1 #2/3 chance to return to F=1 state. From branching ratios 
                atom_array[i,6] = mF_state(1, upper_state_mF)


    atom_array = np.delete(atom_array,indicies,axis=0) #delete rows of atoms that have left the trap

    return atom_array