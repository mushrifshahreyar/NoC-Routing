import os
import progressbar
import time
import subprocess

ITERATIONS = 1000
#i = 0

#print("Deleting existing Q-Table")
#os.system('python3 DQN.py &')
#return_code = os.system('scons ./build/X86/gem5.opt')
return_code = 0

synthetic = ['uniform_random', 'tornado', 'bit_complement', 'bit_reverse', 'bit_rotation', 'neighbor', 'shuffle', 'transpose']

if return_code == 0:
    print("Starting Training Phase")
#    bar = progressbar.ProgressBar(maxval=ITERATIONS, widgets = [' [', progressbar.Timer(), '] ', progressbar.Bar('=', ' [','] '), ' ', progressbar.Percentage()]).start()

    for i in range(ITERATIONS):
#        bar.update(i)
#        result = os.system('python3 DQN.py &')
        print("Iteration Number:", i)
        result = os.system('./build/X86/gem5.opt -d m5out/ configs/example/garnet_synth_traffic.py --num-cpus=64 --num-dirs=64 --network=garnet --topology=Mesh_XY --mesh-rows=8 --sim-cycles=100000 --inj-vnet=-1 --vcs-per-vnet=8 --injectionrate=0.1 --synthetic=uniform_random   --routing-algorithm=10')

    print("Training Phase Ended")
    #os.system('pkill -9 -f DQN.py')

else:
    print('Error occured while compiling!')
