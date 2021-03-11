import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # allows graphing in 3D
import numpy as np
import scipy.constants as sc


def plotting(LiRange):
	"""Function takes care of graphing the result of the simulation"""
	mLi = 6.941*sc.physical_constants["atomic mass constant"][0] # mass H

	speeds = [np.linalg.norm(LiRange[i,3:6]) for i in range(len(LiRange))] #final speeds of Lithium atoms

	colour = colours(speeds,max(speeds)) #colour array to demonstrate final speeds of particles graphically

	plot2 = plt.figure('3D Positions of Particles')
	ax2 = plot2.add_subplot(111, projection='3d') #plot final positions of particles in 3d while colour coding speeds 
	ax2.set_xlabel("x(mm)", fontsize = 10)
	ax2.set_ylabel("y(mm)", fontsize = 10)
	ax2.set_zlabel("z(mm)", fontsize = 10)
	ax2.set_xlim3d([-10,10])
	ax2.set_ylim3d([-10,10])
	ax2.set_zlim3d([-10,10])
	ax2.scatter(LiRange[:,0]*1000,LiRange[:,1]*1000,LiRange[:,2]*1000, c=colour, s=0.5) #plots positions of final particles and colour codes them according to their speeds.

	_ = plt.figure('Positions of Particles')
	plt.xlabel("x(mm)", fontsize = 10)
	plt.ylabel("y(mm)", fontsize = 10)
	plt.scatter(LiRange[:,0]*1000,LiRange[:,1]*1000, c=colour, s=0.5) #plots positions of final particles and colour codes them according to their speeds.

	_ = plt.figure('Position - Time')
	plt.xlabel("Time", fontsize = 10)
	plt.ylabel("x(mm)", fontsize = 10)
	plt.scatter(range(0,len(LiRange)),LiRange[:,0]*1000, c=colour, s=0.5) #plots positions of final particles and colour codes them according to their speeds.

	_ = plt.figure('Speed - Time')
	plt.xlabel("Time", fontsize = 10)
	plt.ylabel("Speed($ms^{-1}$)", fontsize = 10)
	plt.scatter(range(0,len(LiRange)),speeds, c=colour, s=0.5) #plots positions of final particles and colour codes them according to their speeds.

	_ = plt.figure('Phase - Space Distribution')
	plt.xlabel("z(mm)", fontsize = 10)
	plt.ylabel("Velocity ($ms^{-1}$)", fontsize = 10)
	plt.scatter(LiRange[:,0]*1000,LiRange[:,3],s=1) #plots final phase space distn of atoms

	_ = plt.figure('mF States of atom')
	plt.xlabel("Time", fontsize = 10)
	plt.ylabel("mF", fontsize = 10)
	plt.scatter(range(0,len(LiRange)),LiRange[:,6],s=1) #plots final phase space distn of atoms

	_ = plt.figure('F States of atom')
	plt.xlabel("Time", fontsize = 10)
	plt.ylabel("F", fontsize = 10)
	plt.scatter(range(0,len(LiRange)),LiRange[:,7],s=1) #plots final phase space distn of atoms

	plt.show()

def colours(speeds,max_speed):
	"""function that creates a simple colour key based on speeds for showing speed in plotting"""
	colour = []

	for  item in speeds: #create an array for colour coding graphs.
		colour.append([float(item/max_speed), 0 ,float(1-item/max_speed)]) #colour moves from blue (cold) to red (hot)

	return colour 
if __name__ == '__main__':
	LiRange = np.genfromtxt(r"C:\Users\Sanjay\Documents\Uni stuff\Physics\Project\Laser Cooling Trap Code\MyCode\Output\total detuning=-7gamma S=20.csv",delimiter=',')
	plotting(LiRange)