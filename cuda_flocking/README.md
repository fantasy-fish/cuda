## CUDA Flocking

In this project, I implemented a flocking simulation based on the Reynolds Boids algorithm, along with two levels
of optimization: a uniform grid, and a uniform grid with semi-coherent memory access. The source code is based on Upenn's
GPU programming course project.

In the Boids flocking simulation, particles representing birds or fish
(boids) move around the simulation space according to three rules:

1. cohesion - boids move towards the perceived center of mass of their neighbors
2. separation - boids avoid getting to close to their neighbors
3. alignment - boids generally try to move with the same direction and speed as
their neighbors

Below are some ressults:
