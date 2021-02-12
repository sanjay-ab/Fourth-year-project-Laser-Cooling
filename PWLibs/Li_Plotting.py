import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # allows graphing in 3D
import numpy as np
import scipy.constants as sc


def plotting(LiRange, LiRange_init):
	"""Function takes care of graphing the result of the simulation"""
	mLi = 6.941*sc.physical_constants["atomic mass constant"][0] # mass H

	init_speeds = [np.linalg.norm(LiRange_init[i,3:]) for i in range(len(LiRange_init))] #initial speeds of Lithium atoms
	speeds = [np.linalg.norm(LiRange[i,3:]) for i in range(len(LiRange))] #final speeds of Lithium atoms

	kinetic_energy = [0.5* mLi* s**2 for s in speeds] #final kinetic energy of Lithium atoms
	init_kinetic_energy = [0.5* mLi* s**2 for s in init_speeds] #initial kinetic energy of Lithium atoms

	colour = colours(speeds,max(speeds)) #colour array to demonstrate final speeds of particles graphically
	init_colour = colours(init_speeds,max(init_speeds)) #colour array used to demonstrate initial speeds of particles graphically

	print(f"Number of Lithium left in trap: {len(LiRange)}")

	plot2 = plt.figure('3D Final Positions of Particles')
	ax2 = plot2.add_subplot(111, projection='3d') #plot final positions of particles in 3d while colour coding speeds 
	ax2.set_xlabel("x(mm)", fontsize = 10)
	ax2.set_ylabel("y(mm)", fontsize = 10)
	ax2.set_zlabel("z(mm)", fontsize = 10)
	ax2.set_xlim3d([-10,10])
	ax2.set_ylim3d([-10,10])
	ax2.set_zlim3d([-10,10])
	ax2.scatter(LiRange[:,0]*1000,LiRange[:,1]*1000,LiRange[:,2]*1000, c=colour, s=0.5) #plots positions of final particles and colour codes them according to their speeds.

	_ = plt.figure('Final Speeds of Lithium particles')
	plt.xlabel("Speed ($ms^{-1}$)", fontsize = 10)
	plt.ylabel("Normalized Counts", fontsize = 10)
	plt.hist(speeds, bins= "auto", zorder = 1,density = True) #histogram of final speeds of Lithium atoms.

	_ = plt.figure('Initial Speeds of Lithium particles')
	plt.xlabel("Speed ($ms^{-1}$)", fontsize = 10)
	plt.ylabel("Normalized Counts", fontsize = 10)
	plt.hist(init_speeds, bins= "auto", zorder = 1,density = True) #histogram of initial speeds of Lithium atoms.

	_ = plt.figure('Final Kinetic Energy of Lithium particles')
	plt.xlabel("$log_{10}$(Kinetic Energy (J))", fontsize = 10)
	plt.ylabel("Normalized Counts", fontsize = 10)
	plt.hist(np.log10(kinetic_energy), bins= "auto", zorder = 1,density = True) #histogram of final kinetic energy of Lithium atoms.

	_ = plt.figure('Initial Kinetic Energy of Lithium particles')
	plt.xlabel("$log_{10}$(Kinetic Energy (J))", fontsize = 10)
	plt.ylabel("Normalized Counts", fontsize = 10)
	plt.hist(np.log10(init_kinetic_energy), bins= "auto", zorder = 1,density = True) #histogram of initial kinetic energy of Lithium atoms.

	_ = plt.figure('Final Positions of Particles')
	plt.xlabel("x(mm)", fontsize = 10)
	plt.ylabel("y(mm)", fontsize = 10)
	plt.xlim([-10,10])
	plt.ylim([-10,10])
	plt.scatter(LiRange[:,0]*1000,LiRange[:,1]*1000, c=colour, s=0.5) #plots positions of final particles and colour codes them according to their speeds.

	_ = plt.figure('Initial Positions of Particles')
	plt.xlabel("x(mm)", fontsize = 10)
	plt.ylabel("y(mm)", fontsize = 10)
	plt.xlim([-10,10])
	plt.ylim([-10,10])
	plt.scatter(LiRange_init[:,0]*1000,LiRange_init[:,1]*1000, c=init_colour, s=0.5) #plots positions of initial particles and colour codes them according to their speeds.

	_ = plt.figure('Final Phase - Space Distribution of Particles')
	plt.xlabel("z(mm)", fontsize = 10)
	plt.ylabel("Velocity ($ms^{-1}$)", fontsize = 10)
	plt.scatter(LiRange[:,2]*1000,LiRange[:,5],s=1) #plots final phase space distn of atoms

	_ = plt.figure('Initial Phase - Space Distribution of Particles')
	plt.xlabel("z(mm)", fontsize = 10)
	plt.ylabel("Velocity ($ms^{-1}$)", fontsize = 10)
	plt.scatter(LiRange_init[:,2]*1000,LiRange_init[:,5],s=1) #plots initial phase space distn of atoms.

	plt.show()

def colours(speeds,max_speed):
	"""function that creates a simple colour key based on speeds for showing speed in plotting"""
	colour = []

	for  item in speeds: #create an array for colour coding graphs.
		colour.append([float(item/max_speed), 0 ,float(1-item/max_speed)]) #colour moves from blue (cold) to red (hot)

	return colour 