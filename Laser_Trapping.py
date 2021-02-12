#!/usr/bin/env python3
import concurrent.futures
import itertools
import time
from os import cpu_count
from sys import platform

import numpy as np
import scipy.constants as sc

from PWLibs.Li_Plotting import plotting
from PWLibs.Trap_Dist import makeLi
from PWLibs.TrapVV import rVV
from PWLibs.Li_GS_E import Lizeeman
from PWLibs.ARBInterp import tricubic
from PWLibs.Laser_Cooling import Laser

mLi = 6.941*sc.physical_constants["atomic mass constant"][0] #mass of lithium in kg
timer = time.perf_counter()
dt = 2e-8 #timestep of simulation
t_end = 1e-3 #end of simulation
F = 2 #F number of the ground state

# Location of the field source file, depending on operating platform
if platform == "win32":
	slash = "\\"
elif platform == "linux" or "linux2":
	slash = "/"


trapfield=np.genfromtxt(f"Input{slash}SmCo28.csv", delimiter=',') # Load MT-MOT magnetic field
trapfield[:,:3]*=1e-3 # Modelled the field in mm for ease, make ;it m
# This is a vector field but the next piece of code automatically takes the magnitude of the field

LiEnergies = Lizeeman(trapfield)	# this instantiates the H Zeeman energy class with the field
# This class instance now contains 4 Zeeman energy fields for the different substates of H
# For now we are only interested in the F=1, mF=1 state, HEnergies.U11
# Others are U00, U1_1, U10

trap = tricubic(trapfield,'quiet') #creates an interpolator for the trapfield so that we can find the magnetic field vector at any point in the trap.

interpolators = []
i = F
while i>=-F: #loops through all mF values for F=2 in 2S1/2 state in 7Li

	if i<0:
		name = f"_{str(abs(i))}" 
	else:
		name = str(i)

	interpolators.append(tricubic(getattr(LiEnergies, f"U2{name}"),"quiet")) #adds all interpolators required into the list

	i -= 1

#"tricubic" creates an instance of the tricubic interpolator for this energy field
# We can now query arbitrary coordinates within the trap volume and get the interpolated Zeeman energy and its gradient
#We create interpolators for each other mF value in the F=2 2S1/2 state of 7Li.





def iterate(atom_array, index, n_chunks):

	state_index = [] #hold the indices of atoms depending on state. i.e. state_index[0] holds the indicies of atoms in state F=2 mF=2 in the array atom_array.
	for _ in range(2*F + 1): #loop through the total number of states available for atoms to create array
		state_index.append([])
	state_index[2] = list(np.arange(0,len(atom_array), 1)) #initially set all atoms to state mF=0 F=2

	t = 0
	loop = time.perf_counter()

	while t < t_end:

		for x in range(2*F + 1): #propagate each atom. #Loop through each array for each state
			if len(state_index[x]) != 0:
				atom_array[np.array(state_index[x])] = rVV(interpolators[x], np.take(atom_array, state_index[x], axis=0), dt, mLi) #propagate each atom through the trap - change interpolator depending on what state the atoms are in

		atom_array, state_index = Laser(atom_array, state_index, trap, dt) #this can be called no matter the state of the atom as it can always interact with the laser.

		t += dt

		if time.perf_counter()- loop > 60:	# every 60 seconds this will meet this criterion and run the status update code below
			print(f'Chunk {index} loop ' + '{:.1f}'.format(100 * (t / t_end)) + ' % complete. Time elapsed: {}s'.format(int(time.perf_counter()-timer)))	# percentage complete
			loop = time.perf_counter()	# reset status counter

	print("\n Chunk {} of {} complete. \n Time elapsed: {}s.".format(index, n_chunks, int(time.perf_counter()-timer))) #print index to get a sense of how far through the iteration we are
	return atom_array #return the chunk




def main():
	#N = 1000 #starting number of lithium particles
	#T = 0.01 #starting temperature of lithium cloud
	#rsd = 1e-3

	#LiRange_init = makeLi(N, T, rsd) #create initial lithium distribution
	#np.savetxt("Li_init N=1000 T=0.01 rsd=1e-3.csv", LiRange_init, delimiter=',')
	LiRange_init = np.genfromtxt(f"Input{slash}Li_init N=1000 T=0.01 rsd=1e-3.csv", delimiter=',')

	print("Solving particle motion")

	chunks_per_workers = 1 #number of chunks per worker - vary between 1 and 40 for efficient execution depending on the length of the simulation
	n_chunks = chunks_per_workers * cpu_count() #number of chunks (cpu_count yields the number of logical processors on the system)
	indicies = np.arange(1,n_chunks+1) #used to show which chunk we are currently on
	chunks = [LiRange_init[slice(int((len(LiRange_init)/n_chunks)*i),int((len(LiRange_init)/n_chunks)*(i+1)))] for i in range(n_chunks)] #split array into chunks to be computed in parallel.
	
	with concurrent.futures.ProcessPoolExecutor() as executor: #use with statement so that executor automatically closes after completion
		"""executor.map maps the iterables in the second parameter to the function "iterate" similar to the inbuilt map function although this allows multiple versions of the function
		to run in parallel with each other. It yields the return value of the function iterate which can be extracted from "results" by iterating through. Above, we create the chunks 
		we want to execute with rather than using the chunksize parameter. This reduces the overhead in starting and stopping parallel processes by controlling the number of processes created."""

		results = executor.map(iterate, *(chunks,indicies,itertools.repeat(n_chunks,n_chunks))) #result of simulation
	
	LiRange = []

	for result in results:
		#loop through the array of arrays of atoms
		for res in result:
			#loop through each atom in each array of atoms
			LiRange.append(res) #append each atom to list

	LiRange = np.asarray(LiRange) #turn list into 2d numpy array

	print('Done')
	print(f"Time elapsed: {int(time.perf_counter()-timer)}s")  #can't just use time.perf_counter() since this is not equal to the simulation time when executing on the supercomputer

	print("Saving output")
	np.savetxt(f"Output{slash}Li_init.csv", LiRange_init, delimiter=',')
	np.savetxt(f"Output{slash}Li_end t={t_end} dt={dt} Nli=1000.csv", LiRange, delimiter=',')
	print("Done")

	#print("Graphing Output")
	#plotting(LiRange, LiRange_init)
	#print("Done")



if __name__ == '__main__':
	main()
