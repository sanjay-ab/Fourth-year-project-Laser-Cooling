# Masters-project (Laser cooling)

Some code I've written for my masters project on "Simulating the sympathetic cooling of atomic hydrogen with ultra-cold atomic lithium". This repo holds the code for the laser cooling side of the simulation. The code for the sympathetic cooling side is held in another repo.

This was a collaborative project and only the files that I produced are held here. The additional libraries and data required for the program to run have not been included.

The program runs step-wise with a defineable time-step. Every time-step, the program calculates the effect of the lasers on the lithium atoms and it solves the equation of motion of the atoms in the trap by numerically integrating using a velocity verlet algorithm.
