Name: Gagandeep
Roll no: 102303349
Group no:3C25
Lab: LAB1

----------------------------------------

Q1: DAXPY

Threads and Time:

2 Threads  : 0.002247 sec
4 Threads  : 0.034476 sec
8 Threads  : 0.004417 sec

Observation:
For 2 threads, execution time was minimum.
With 4 threads, time increased due to thread overhead.
With 8 threads, performance improved slightly but was not optimal.

This happened due to limited CPU cores and thread management overhead.

----------------------------------------

Q2: Matrix Multiplication (1D)

Threads and Time:

2 Threads  : 0.357360 sec
4 Threads  : 0.409395 sec
8 Threads  : 0.420742 sec

Observation:
In 1D parallelization, only the outer loop was parallelized.
Performance improvement was limited due to poor load balancing.
Optimal performance was achieved with 2 threads.

----------------------------------------

Q2: Matrix Multiplication (2D)

Threads and Time:

2 Threads  : 0.170162 sec
4 Threads  : 0.226810 sec
8 Threads  : 0.216421 sec

Observation:
In 2D parallelization, both row and column loops were parallelized using collapse(2).
Workload was distributed more evenly among threads.
2D version was significantly faster than 1D version.

----------------------------------------

Q3: PI Calculation

Threads, PI Value and Time:

2 Threads  : PI = 3.141691 , Time = 0.076509 sec
4 Threads  : PI = 3.141784 , Time = 0.096210 sec
8 Threads  : PI = 3.142550 , Time = 0.094454 sec

Observation:
The calculated value of PI was close to actual value (3.14159).
Reduction avoided race conditions.
Best performance was achieved with 2 threads.
Increasing threads caused overhead.

----------------------------------------

Conclusion:

OpenMP improved program performance by using multiple threads.
However, maximum speedup was achieved only till optimal number of threads.
Beyond that, overhead and limited CPU cores reduced performance.

2D parallelization performed better than 1D in matrix multiplication.

