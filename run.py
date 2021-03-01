import os
import progressbar
import time
import subprocess

ITERATIONS = 30
#i = 0

#print("Deleting existing Q-Table")
#os.system('rm Q_Table.txt')
#return_code = os.system('scons ./build/X86/gem5.opt')
return_code = 0

synthetic = ['uniform_random', 'tornado', 'bit_complement', 'bit_reverse', 'bit_rotation', 'neighbor', 'shuffle', 'transpose']

if return_code == 0:
    print("Starting Training Phase")
#    bar = progressbar.ProgressBar(maxval=ITERATIONS, widgets = [' [', progressbar.Timer(), '] ', progressbar.Bar('=', ' [','] '), ' ', progressbar.Percentage()]).start()

    for i in range(ITERATIONS):
#        bar.update(i)
#        result = os.system('python3 DQN.py &')
        result = os.system('./build/X86/gem5.opt -d m5out/ configs/example/garnet_synth_traffic.py --num-cpus=16 --num-dirs=16 --network=garnet --topology=Mesh_XY --mesh-rows=4 --sim-cycles=100000 --inj-vnet=-1 --vcs-per-vnet=8 --injectionrate=0.1 --synthetic=shuffle   --routing-algorithm=5')

    print("Training Phase Ended")

else:
    print('Error occured while compiling!')
