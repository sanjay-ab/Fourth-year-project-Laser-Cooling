#!/usr/bin/env python3

import concurrent.futures
import itertools
import time
from os import cpu_count
from sys import platform
from csv import writer

import numpy as np
import scipy.constants as sc

from PWLibs.Li_Plotting import plotting
from PWLibs.TrapVV import rVV
from PWLibs.Li_GS_E import Lizeeman
from PWLibs.ARBInterp import tricubic
from PWLibs.Laser_Cooling import Laser

gamma = 2*np.pi * 5.92e6 #natural linewidth of Lithium (2pi*hz)
mLi = 6.941*sc.physical_constants["atomic mass constant"][0] #mass of lithium in kg
timer = time.perf_counter()

dt = 2e-8 #timestep of simulation (s)
t_start = 0 #start time of simulation
t_end = 2e-3 #duration of simulation (s)
times_to_save = [5e-4,1e-3,1.5e-3] #list times in the simulation in which to save the data
prev_det = "" #previous detunings written in pairs of (detuning, time til changed) e.g. (-7, 1ms)
det_times = [] #times at which to change the detunings. Array is one smaller than the "det" array below
det = [-7] #different detunings to change to 
det_r = [-5] #detuning for the repump laser
S_0p = 5 #ratio of I0/Isat for lasers
S_0r = 20
W_0 = 1e-3 #waist of laser beams
pol_p = 1 #polarizations of pump laser beams +1 means it carries -hbar ang mom, -1 means it carries +hbar ang mom
pol_r = 1 #polarizations of the repump laser beams

# Location of the field source file, depending on operating platform
if platform == "win32":
	slash = "\\"
elif platform == "linux" or "linux2":
	slash = "/"

LiRange_init = np.genfromtxt(f"Input{slash}Li_init N=1000 T=0.05 rsd=1e-3.csv", delimiter=',') #get initial distribution of lithium 

trapfield=np.genfromtxt(f"Input{slash}SmCo28.csv", delimiter=',') # Load MT-MOT magnetic field
trapfield[:,:3]*=1e-3 # Modelled the field in mm for ease, make ;it m
# This is a vector field but the next piece of code automatically takes the magnitude of the field

LiEnergies = Lizeeman(trapfield)	# this instantiates the H Zeeman energy class with the field
# This class instance now contains 4 Zeeman energy fields for the different substates of H
# For now we are only interested in the F=1, mF=1 state, HEnergies.U11
# Others are U00, U1_1, U10

trap = tricubic(trapfield,'quiet') #creates an interpolator for the trapfield so that we can find the magnetic field vector at any point in the trap.

#"tricubic" creates an instance of the tricubic interpolator for this energy field
# We can now query arbitrary coordinates within the trap volume and get the interpolated Zeeman energy and its gradient
#We create interpolators for each other mF value in the F=2 2S1/2 state of 7Li.

interpolators2 = []
interpolators1 = []

for i in range(2,-3,-1): #loops through all mF values for F=2 in 2S1/2 state in 7Li

	if i<0:
		name = f"_{str(abs(i))}" 
	else:
		name = str(i)

	interpolators2.append(tricubic(getattr(LiEnergies, f"U2{name}"),"quiet")) #adds all interpolators required into the list

for i in range(1,-2,-1): #loops through all mF values for F=1 in 2S1/2 state in 7Li

	if i<0:
		name = f"_{str(abs(i))}" 
	else:
		name = str(i)

	interpolators1.append(tricubic(getattr(LiEnergies, f"U1{name}"),"quiet")) #adds all interpolators required into the list
	
interpolators = [interpolators1, interpolators2]

def iterate(array, index, n_chunks, prev_dets):

	atom_array = np.zeros((len(array),8)) #create array to hold atoms and their states
	if len(array[0])==8:#if array already holds the atom's states 
		atom_array = array
	else: #if the atoms havea an unspecified state then put them all in mF=0 F=2
		atom_array[:,:6] = array
		atom_array[:,7] = 2 #set into F=2 state

	t = 0
	loop = time.perf_counter()
	pointer1 = 0 #for tracking when to save file
	pointer2 = 0 #for tracking when to change detuning

	while t < t_end:

		arr = atom_array[:,:6] #create new array excluding column for index
		for F in [1,2]:
			mask1 = atom_array[:,7] == F #mask of which elements are in state F

			for x in range(F, -F -1, -1): #propagate each atom. 
				mask2 = atom_array[:,6] == x #create mask of which elements are in state x
				mask = mask1 & mask2 #combine masks 

				if len(mask)!=0:
					arr[mask] = rVV(interpolators[F-1][F-x], arr[mask], dt, mLi) #propagate each atom through the trap - change interpolator depending on what state the atoms are in

			atom_array[:,:6] == arr #assign new speeds to full atom array

		atom_array = Laser(atom_array, trap, dt, det[pointer2]*gamma, det_r[pointer2]*gamma, S_0p, S_0r, W_0, pol_p, pol_r) #this can be called no matter the state of the atom as it can always interact with the laser.

		t += dt

		if time.perf_counter()- loop > 60:	# every 60 seconds this will meet this criterion and run the status update code below
			print(f'Chunk {index} loop ' + '{:.1f}'.format(100 * (t / t_end)) + ' % complete. Time elapsed: {}s'.format(int(time.perf_counter()-timer)))	# percentage complete
			loop = time.perf_counter()	# reset status counter

		if pointer1 != len(times_to_save):
			if t>times_to_save[pointer1]: #save data to a file at specified time through iteration
				time_str = format(times_to_save[pointer1] + t_start, ".1e")
				with open(f"Output{slash}Li_end dt={dt} (detuning, time)={prev_dets}({det[pointer2]},{time_str}) Sp={S_0p} Sr={S_0r} W_0={W_0*1000}mm pol(p,r)=({pol_p},{pol_r}).csv",'a+',newline='') as outfile:
					csv_writer = writer(outfile)
					for row in atom_array:
						csv_writer.writerow(row)

				pointer1 += 1

		if pointer2 != len(det_times): #change variable to help in naming files
			if t > det_times[pointer2]:
				time_str = format(det_times[pointer2]+t_start, ".1e")
				prev_dets = prev_dets + f"({det[pointer2]}, {time_str})"
				pointer2 += 1
				

	print("\n Chunk {} of {} complete. \n Time elapsed: {}s.".format(index, n_chunks, int(time.perf_counter()-timer))) #print index to get a sense of how far through the iteration we are
	return atom_array #return the chunk




def main(prev_det, LiRange_init):

	print("Solving particle motion")

	chunks_per_workers = 1 #number of chunks per worker - vary between 1 and 40 for efficient execution depending on the length of the simulation
	n_chunks = chunks_per_workers * cpu_count() #number of chunks (cpu_count yields the number of logical processors on the system)
	indicies = np.arange(1,n_chunks+1) #used to show which chunk we are currently on
	chunks = [LiRange_init[slice(int((len(LiRange_init)/n_chunks)*i),int((len(LiRange_init)/n_chunks)*(i+1)))] for i in range(n_chunks)] #split array into chunks to be computed in parallel.
	
	with concurrent.futures.ProcessPoolExecutor() as executor: #use with statement so that executor automatically closes after completion
		"""executor.map maps the iterables in the second parameter to the function "iterate" similar to the inbuilt map function although this allows multiple versions of the function
		to run in parallel with each other. It yields the return value of the function iterate which can be extracted from "results" by iterating through. Above, we create the chunks 
		we want to execute with rather than using the chunksize parameter. This reduces the overhead in starting and stopping parallel processes by controlling the number of processes created."""

		results = executor.map(iterate, *(chunks,indicies,itertools.repeat(n_chunks,n_chunks),itertools.repeat(prev_det,n_chunks))) #result of simulation
	
	LiRange = []

	for result in results:
		#loop through the array of arrays of atoms
		for res in result:
			#loop through each atom in each array of atoms
			LiRange.append(res) #append each atom to list

	LiRange = np.asarray(LiRange) #turn list into 2d numpy array

	for x in range(len(det_times)): #create variable to help in naming the file
		time_str = format(det_times[x]+t_start, ".1e")
		prev_det += f"({det[x]}, {time_str})"
	time_str = format(t_end+t_start, ".1e")
	prev_det += f"({det[len(det)-1]},{time_str})"

	print('Done')
	print(f"Time elapsed: {int(time.perf_counter()-timer)}s")  #can't just use time.perf_counter() since this is not equal to the simulation time when executing on the supercomputer

	print("Saving output")
	np.savetxt(f"Output{slash}Li_init.csv", LiRange_init, delimiter=',')
	np.savetxt(f"Output{slash}Li_end dt={dt} (detuning, time)={prev_det} Sp={S_0p} Sr={S_0r} W_0={W_0*1000}mm pol(p,r)=({pol_p},{pol_r}).csv", LiRange, delimiter=',')
	print("Done")

	#print("Graphing Output")
	#plotting(LiRange, LiRange_init)
	#print("Done")



if __name__ == '__main__':
	main(prev_det, LiRange_init)
