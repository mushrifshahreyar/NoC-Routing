This is the gem5 simulator.

The main website can be found at http://www.gem5.org

A good starting point is http://www.gem5.org/about, and for
more information about building the simulator and getting started
please see http://www.gem5.org/documentation and
http://www.gem5.org/documentation/learning_gem5/introduction.

To build gem5, you will need the following software: g++ or clang,
Python (gem5 links in the Python interpreter), SCons, SWIG, zlib, m4,
and lastly protobuf if you want trace capture and playback
support. Please see http://www.gem5.org/documentation/general_docs/building
for more details concerning the minimum versions of the aforementioned tools.

Once you have all dependencies resolved, type 'scons
build/<ARCH>/gem5.opt' where ARCH is one of ARM, NULL, MIPS, POWER, SPARC,
or X86. This will build an optimized version of the gem5 binary (gem5.opt)
for the the specified architecture. See
http://www.gem5.org/documentation/general_docs/building for more details and
options.

The basic source release includes these subdirectories:
   - configs: example simulation configuration scripts
   - ext: less-common external packages needed to build gem5
   - src: source code of the gem5 simulator
   - system: source for some optional system software for simulated systems
   - tests: regression tests
   - util: useful utility programs and files

To run full-system simulations, you will need compiled system firmware
(console and PALcode for Alpha), kernel binaries and one or more disk
images.

If you have questions, please send mail to gem5-users@gem5.org

Enjoy using gem5 and please share your modifications and extensions.

# NoC 

## Running gem5 simulator

```
 ./build/X86/gem5.opt -d m5out/ configs/example/garnet_synth_traffic.py --num-cpus=64 --num-dirs=64 --network=garnet --topology=Mesh_XY --mesh-rows=8 --sim-cycles=100000 --vcs-per-vnet=8 --injectionrate=0.4 --synthetic=uniform_random   --routing-algorithm=8 
```

For routing algorithm: `--routing-algorithm=x`
- x = 1; lookupRoutingTable (default)
- x = 2; XY (default)
- x = 3; Odd Even
- x = 4; Q Routing training done in C++
- x = 5; Q Routing testing done in C++
- x = 6; Q Rouing testing done in python using Python.h, `Python filename: Q_Routing_test.py` 
- x = 7; DQN done in Python, `Python filename: DQN.py`
- x = 8; DQN Testing Python, `Python filename: DQNTest.py`
- x = 9; DQN Virtual buffer Training, `Python filename: DQN_vc.py`
- x = 10; DQN hops buffer Training, `Python filename: DQN_hops.py`
- x = 11; DQN Virtual buffer Testing, `Python filename: DQNvcTesting.py`
- x = 12; DQN hops buffer Testing, `Python filename: DQNHopsTesting.py`

## How to run

1. Open two terminals [Only required from routing algorithm number=6 else one terminal in enough]
    a. In first terminal run the above command with the required algorithm number
    b. In 2nd terminal run do, Ex
```
python3 *Required filename*
python3 DQN.py
```
2. Programs in both terminals will execute simultaneously.

![Example 1](./ex1.jpg)
![Example 2](./ex2.jpg)
## Some Extra files:
1. Python file ```Run.py```
-   This file is used for training the model. Intead of running ./build... in 1st terminal, run `python run.py`. You can specify the number of iterations in the run.py file and also the routing algorithm number.

2. Python file ```readStats.py```
-   This is a script used for simplifying the work used to get the required output from the ```/m5out/stats.txt``` file. You can specify the required routing algorithms, different injection rates, traffic types and the required keywords to get from stats.txt file in the list. All the outputs will be printed in Results.txt file.


