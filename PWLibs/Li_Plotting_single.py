import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # allows graphing in 3D
import numpy as np
import scipy.constants as sc
from matplotlib.lines import Line2D


def plotting(LiRange, loc, name, save):
	"""Function takes care of graphing the result of the simulation"""
	mLi = 6.941*sc.physical_constants["atomic mass constant"][0] # mass H

	speeds = [np.linalg.norm(LiRange[i,3:6]) for i in range(len(LiRange))] #final speeds of Lithium atoms

	colour = colours(speeds,max(speeds)) #colour array to demonstrate final speeds of particles graphically
	legend_elements = [Line2D([0], [0], marker='o', color='w', label='F=2', markerfacecolor='r', markersize=8),Line2D([0], [0], marker='o', color='w', label='F=1', markerfacecolor='b', markersize=8)]
	col = []

	for item in LiRange:
		if item[7] == 2:
			col.append('r')
		else:
			col.append('b')

	plot2 = plt.figure('3D Positions of Particles')
	ax2 = plot2.add_subplot(111, projection='3d') #plot final positions of particles in 3d while colour coding speeds 
	ax2.set_xlabel("x(mm)", fontsize = 10)
	ax2.set_ylabel("y(mm)", fontsize = 10)
	ax2.set_zlabel("z(mm)", fontsize = 10)
	ax2.set_xlim3d([-10,10])
	ax2.set_ylim3d([-10,10])
	ax2.set_zlim3d([-10,10])
	ax2.scatter(LiRange[:,0]*1000,LiRange[:,1]*1000,LiRange[:,2]*1000, c=colour, s=0.5) #plots positions of final particles and colour codes them according to their speeds.

	plt.figure()#'Positionx - Time'
	plt.xlabel("Time", fontsize = 10)
	plt.ylabel("x(mm)", fontsize = 10)
	plt.scatter(range(0,len(LiRange)),LiRange[:,0]*1000, c=colour, s=0.5) #plots positions of final particles and colour codes them according to their speeds.
	if save: plt.savefig(f"{loc}Positionx {name}.png")

	plt.figure()#'Positiony - Time'
	plt.xlabel("Time", fontsize = 10)
	plt.ylabel("y(mm)", fontsize = 10)
	plt.scatter(range(0,len(LiRange)),LiRange[:,1]*1000, c=colour, s=0.5) #plots positions of final particles and colour codes them according to their speeds.
	if save: plt.savefig(f"{loc}Positiony {name}.png")

	plt.figure()#'Positionz - Time'
	plt.xlabel("Time", fontsize = 10)
	plt.ylabel("z(mm)", fontsize = 10)
	plt.scatter(range(0,len(LiRange)),LiRange[:,2]*1000, c=colour, s=0.5) #plots positions of final particles and colour codes them according to their speeds.
	if save: plt.savefig(f"{loc}Positionz {name}.png")

	plot = plt.figure()#f'Position - Time{name}'
	ax = plot.add_subplot(111)
	ax.set_xlabel("Time", fontsize = 10)
	ax.set_ylabel("x(mm)", fontsize = 10)
	ax.scatter(range(0,len(LiRange)),LiRange[:,0]*1000, c=col, s=0.5) #plots positions of final particles and colour codes them according to their speeds.
	ax.legend(handles=legend_elements,loc='upper right')
	if save: plt.savefig(f"{loc}Position2 {name}.png")

	plot3 = plt.figure()#f'Speed - Time
	ax3 = plot3.add_subplot(111)
	ax3.set_xlabel("Time", fontsize = 10)
	ax3.set_ylabel("Speed($ms^{-1}$)", fontsize = 10)
	ax3.scatter(range(0,len(LiRange)),speeds, c=col,s=0.5) #plots positions of final particles and colour codes them according to their speeds.
	ax3.legend(handles=legend_elements,loc='lower left')
	if save: plt.savefig(f"{loc}Speed {name}.png")

	plt.figure()#'Velocity - Position Distribution'
	plt.xlabel("x(mm)", fontsize = 10)
	plt.ylabel("Velocity ($ms^{-1}$)", fontsize = 10)
	plt.scatter(LiRange[:,0]*1000,LiRange[:,3],s=1) #plots final phase space distn of atoms
	if save: plt.savefig(f"{loc}xVel-Pos {name}.png")

	plt.figure()#'Velocity - Position Distribution'
	plt.xlabel("y(mm)", fontsize = 10)
	plt.ylabel("Velocity ($ms^{-1}$)", fontsize = 10)
	plt.scatter(LiRange[:,1]*1000,LiRange[:,4],s=1) #plots final phase space distn of atoms
	if save: plt.savefig(f"{loc}yVel-Pos {name}.png")

	plt.figure()#'Velocity - Position Distribution'
	plt.xlabel("z(mm)", fontsize = 10)
	plt.ylabel("Velocity ($ms^{-1}$)", fontsize = 10)
	plt.scatter(LiRange[:,2]*1000,LiRange[:,5],s=1) #plots final phase space distn of atoms
	if save: plt.savefig(f"{loc}zVel-Pos {name}.png")

	plt.figure()#'mF States of atom'
	plt.xlabel("Time", fontsize = 10)
	plt.ylabel("mF", fontsize = 10)
	plt.scatter(range(0,len(LiRange)),LiRange[:,6],s=1) #plots final phase space distn of atoms
	if save: plt.savefig(f"{loc}mF {name}.png")

	plt.figure()#'F States of atom'
	plt.xlabel("Time", fontsize = 10)
	plt.ylabel("F", fontsize = 10)
	plt.scatter(range(0,len(LiRange)),LiRange[:,7],s=1) #plots final phase space distn of atoms
	if save: plt.savefig(f"{loc}F {name}.png")

	#plt.show()

def colours(speeds,max_speed):
	"""function that creates a simple colour key based on speeds for showing speed in plotting"""
	colour = []

	for  item in speeds: #create an array for colour coding graphs.
		colour.append([float(item/max_speed), 0 ,float(1-item/max_speed)]) #colour moves from blue (cold) to red (hot)

	return colour 
