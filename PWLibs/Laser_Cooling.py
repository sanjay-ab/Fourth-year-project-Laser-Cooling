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
const3 = sc.hbar/mLi 
gamma = 2*np.pi * 5.92e6 #natural linewidth of Lithium (2pi*hz)
const1 = 4/gamma**2 
omega_0 = 2.807389399e+15 #frequency of 2S1/2 -> 2P3/2 transition excluding hyperfine shifts
omega = 2.807387462245734e15 #frequency of F=2 -> F'=3 laser transtition including the hyperfine shifts at zero field
omega_r = 2.80739257e15 #frequency of F=1 -> F'=2 repump transition including the hyperfine shifts at zero field
laser_origins = [np.array([-1e-2,0,0]),np.array([1e-2,0,0]),np.array([0,-1e-2,0]),np.array([0,1e-2,0]),np.array([0,0,-1e-2]),np.array([0,0,1e-2])] #origin of lasers
laser_vectors = [np.array([1,0,0]),np.array([-1,0,0]),np.array([0,1,0]),np.array([0,-1,0]),np.array([0,0,1]),np.array([0,0,-1])] #directions of lasers
Zeeman_Shift = zeeman_shift() #define class used to calculate zeeman shifts for different states.

def mF_state(mF, F):
    """Returns the mF, F state the atom returns to given the mF, F value of its upper state"""

    if mF != 0:
        sn = mF/abs(mF)

    mF = abs(mF)
    rn = np.random.rand() #create random number for sampling 

    if F==3:
        if mF == 3:
            return sn * 2, 2
        elif mF == 2:
            if rn < 5/15:
                return sn * 2, 2
            else:
                return sn * 1, 2
        elif mF == 1:
            if rn < 1/15:
                return sn * 2, 2
            elif rn < 9/15:
                return sn * 1, 2
            else:
                return 0, 2
        else:
            if rn < 3/15:
                return -1, 2
            elif rn < 12/15:
                return 0, 2
            else:
                return 1, 2

    if F==2:
        if mF == 2 :
            if rn < 4/12:
                return sn * 2, 2
            elif rn < 2/12:
                return sn * 1, 2
            else:
                return sn * 1, 1
        if mF == 1:
            if rn < 2/12:
                return sn * 2, 2
            elif rn < 3/12:
                return sn * 1, 2
            elif rn < 6/12:
                return 0, 2
            elif rn < 9/12:
                return sn * 1, 1
            else:
                return 0 ,1
        else:
            if rn < 3/12:
                return -1, 2
            elif rn < 6/12:
                return 1, 2
            elif rn < 7/12:
                return 1, 1
            elif rn < 11/12:
                return 0, 1
            else:
                return -1, 1
    
    if F==1:
        if mF == 1:
            if rn < 6/60:
                return sn * 2, 2
            elif rn < 9/60:
                return sn * 2, 2
            elif rn < 10/60:
                return 0, 2
            elif rn < 35/60:
                return sn * 1, 1
            else:
                return 0, 1
        elif mF == 0:
            if rn < 3/60:
                return  1, 2
            elif rn < 7/60:
                return 0, 2
            elif rn < 10/60:
                return -1, 2
            elif rn < 35/60:
                return 1, 1
            else:
                return -1, 1

    if F==0:
        return np.random.randint(-1, 2), 1


def calculate_probability(atom, mF, F, B, dt, omega_L, S_0, W_0, k_mod, pol):
    random_number = np.random.rand() #create random number for sampling probabilities
    prob2 = 0
    #pol is the polarization of the lasers +1 means it carries -hbar angular momentum and -1 means it carries +hbar angular momentum
    for k in [1, 0, -1]: #begin looking at deltaF=0 transition as this should be much more likely than other transitions
        #loop through each F value 
        for i in range(6): 
            #loop through each laser
            #calculate W_ij
            Bmod = np.linalg.norm(B)
            theta = np.arccos(np.dot(B,laser_vectors[i])/Bmod) #angle in radians between the magnetic field and the laser k vector
            k_L = k_mod * laser_vectors[i] #k vector of laser
            pos = atom[:3] - laser_origins[i] #shift atom by distance to laser

            if i in (0,1): #x lasers
                #x and y direction lasers are LCP wrt propagation direction
                index1, index2, index3 = 0, 1, 2 #define indicies for calculating waist of laser

            elif i in (2,3): #+y laser
                index1, index2, index3 = 1, 2, 0

            else: #+z laser
                #z direction lasers are LCP wrt propagation direction
                index1, index2, index3 = 2, 0, 1
                pol = pol*-1

            W = W_0 * np.sqrt(1 + ( 2*pos[index1] / (k_mod * W_0**2) )**2 ) #calculate waist. 
            S = S_0 * (W_0/W)**2 * np.exp( -2 * (pos[index2]**2 + pos[index3]**2 ) / W**2 ) #calculate I/Isat using eqn for gaussian beam intensity

            E = 0.5 * np.array([pol - np.cos(theta), -np.sqrt(2) * np.sin(theta), pol + np.cos(theta)]) #rotated electric field vector in spherical basis

            Wi = np.array([abs(E[0]),abs(E[1]), abs(E[2])])**2 #calculate Wij weights for transitions (mf=+1 , 0 , -1)

            lower_state_shift = Zeeman_Shift.S_energy(int(F),int(mF),Bmod) #calculate zeeman shift of the current atom state #L=0 J=1/2 F=2

            upper_state_shifts = Zeeman_Shift.P_adjacent_energies(int(F+k),int(mF), Bmod) #calculate zeeman shift of possible upper atoms states (deltamF=+1,0,-1)

            const2 = (omega_L - omega_0) - np.dot(k_L,atom[3:]) #Delta - dot(k,v)

            for j in range(3):
                prob1 = prob2
                if isnan(upper_state_shifts[j]):
                    continue #transition is impossible so move onto next one
                else:
                    prob2 += ((gamma/(2*np.pi))*S*dt/2) * np.array(Wi[j]/(1+S*Wi[j] + const1 * (const2 + 2*np.pi*(lower_state_shift - upper_state_shifts[j]))**2)) #calculate proabability

                if prob1 <= random_number < prob2: #sample probability to see what transition occurs
                    return i, 1-j, k #laser index, change in mF, change in F

    return np.nan, np.nan, np.nan #if no transition occurs


def Laser(atom_array, trap, dt, detuning, detuning_r, S_0p, S_0r, W_0, pol_p, pol_r):
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
            S_0 = S_0p
            pol = pol_p
        else: #atom in F=1 state
            omega_L = omega_repump #frequency of repump laser as atom only interacts with repump laser
            k_mod = k_r
            S_0 = S_0r
            pol = pol_r

        laser_index, transition_mF, transition_F = calculate_probability(atom[:6], atom[6], atom[7], B[i], dt, omega_L, S_0, W_0, k_mod, pol) #check whether a transition occurs for the atom

        if isnan(laser_index):
            continue #if no interaction then skip to next atom

        #if there is an interaction

        theta = np.random.rand()*np.pi 
        phi = np.random.rand()*2*np.pi

        k_s = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]) #select k vector of photon spontaneously emitted using a random uniform distribution

        atom_array[i,3:6] += const3 * k_mod * (laser_vectors[laser_index] + k_s) #change velocity of atom due to interaction event

        upper_state_mF = atom[6] + transition_mF #mF value of the upper state the atom transitions to
        upper_state_F = atom[7] + transition_F

        atom_array[i,6], atom_array[i,7] = mF_state(upper_state_mF, upper_state_F)

    atom_array = np.delete(atom_array,indicies,axis=0) #delete rows of atoms that have left the trap

    return atom_array