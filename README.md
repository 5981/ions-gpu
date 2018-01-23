# Calculating ion dynamics using GPU

This repository was created mostly for fun and learning purposes.

It is convinient that it also has direct connection with author's work at [LPI RAS](http://www.lebedev.ru) and [MEPhI](https://mephi.ru/).

Also, check out [this project](http://www.ytterbium.space/)

If you want to use your GPU for calculations and play with ions, fork/clone/download and open "Ion dynamics.ipynb" using jupyter.

You need these:
1) CUDA (means you need nVidia for that)
2) [numba with cuda](http://numba.pydata.org/numba-doc/dev/cuda/overview.html) (this requires you to have gpu with compute capability at least 2.0)

I have found that 'numba' is the easiest python module to start using GPU for calculations. It is also slower than pycuda. nVidia and Continuum seems to promote it, maybe it'll get better

If you want to learn how to use numba, go [here](https://nyu-cds.github.io/python-numba/05-cuda/). Beside 'numba' itself, you'll also learn some GPU basics. If you want to do serious GPU calculations, just use C/C++ (all python modules do that). It is fast and [well documented](http://docs.nvidia.com/cuda/). If you have AMD, you have to use [OpenCL](https://www.khronos.org/opencl/) (nVidia users can also use OpenCL).

In this repository we use GPU to solve 'nbody' problem for ions in a trap. It is possible to use octree (same performance) or octree on GPU (too hard to implement), but we won't.
1) Launch 'jupyter', open .ipynb file
2) Execute cells one by one, you should see 3d animation
3) Try changing trap parameters, use YOUR trap! (currently trap parameters are somewhat similar to Mg trap at LPI RAS, but I may be wrong here).

If you want to make this little project better, you can:
1) add 'galaxy.py' (Naturally, interaction between ions is similar to that of stars, so this code could be easily transformed to galaxy calculations.)
2) solve weird bug which generally occurs after a lot of ions leave the trap at once (sometimes there's some kind of memory glitch, and GPU just throws all ions away. It goes away if we let GPU be for a while)
3) make the 'viscosity' model better by changing it to the true Doppler cooling model
4) use different ions in one trap (should be easy)
5) using YOUR trap parameters find crystals that live forever


Thanks for reading! Now look at them go:
