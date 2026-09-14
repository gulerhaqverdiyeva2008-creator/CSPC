# CSPC - Computer Science for Physics and Chemistry

My coursework repository. Each practical is under PW<n>/Lab <X>/.

## Setup

conda env create -f PW<n>/Lab\ <X>/environment.yml
conda activate cspc

---

## PW1 - Lab A: Reproducible Foundations

**What I built:**
- Created a reproducible Conda environment.
- Implemented and tested a radioactive decay simulation.
- Added tests for negative decay rates and the analytical decay law.
- Compared the performance of a pure-Python loop and NumPy.

**Speed comparison (loop vs NumPy):**
- loop : 3.7118 s
- numpy : 0.0004 s
- speed-up: 10132.09 x faster

**Tests:** all passing? (yes)

**Conclusion:**
- The NumPy version is much faster than the pure-Python loop.
- The tests confirm that the simulation starts with the correct number of atoms, rejects negative decay rates, and follows the analytical decay law on average.s