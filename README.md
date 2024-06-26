# There are the following scripts available:

## Typical workflow (samples initial conditions, simulates, learns, simulates learned, and compares):

    python comparison.py --generate --steps=100 --implicit --soft --without --model=RB --folder_name=TEST
    python plot_compare.py --plot_RB_errors --GT --without --implicit --soft --folder_name=TEST

## It is also possible to compare just training an validation losses

    python3 compare_train_errors.py

## Alternatively, you can do that step by step

First generate dataset for training with:

    python3 simulate.py --generate --steps=50000 --model=RB

for the rigid body (or HT for heavy top, or P3D for the particle in three dimensions)

Then we train implicit and soft networks:

    python3 learn.py --method=without --model=RB

(or implicit or soft).

Then choose a different initial conditions and see how well our network fits the evolution. If the initial condition is too different we will not get a good fit. If it is the same we will fit perfectly.

    python3 simulate.py --steps=500 --generate
    python3 simulate.py --steps=500 --implicit
    python3 simulate.py --steps=500 --soft
    python3 simulate.py --steps=500 --without

## And we can plot and see:

    python3 plot_compare.py --plot_m --plot_E --plot_L

Check training error and errors while learning.

## Typical arguments used for the training can be found in folder

    typical_args

More detailed documentation can be found [here](https://www.karlin.mff.cuni.cz/~pavelka/direct-poisson-neural-networks/)

Please cite as [M. Šípka, M. Pavelka, O. Esen, and M. Grmela, Direct Poisson neural networks: learning non-symplectic mechanical systems, Journal of Physics A: Mathematical and Theoretical 56(49), 2023.](https://iopscience.iop.org/article/10.1088/1751-8121/ad0803)
