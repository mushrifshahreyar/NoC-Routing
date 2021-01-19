import os
import progressbar
import time
import subprocess

ITERATIONS = 1000
#i = 0

print("Deleting existing Q-Table")
os.system('rm Q_Table.txt')
return_code = os.system('scons ./build/X86/gem5.opt')



if return_code == 0:
    print("Starting Training Phase")
    bar = progressbar.ProgressBar(maxval=ITERATIONS, widgets = [' [', progressbar.Timer(), '] ', progressbar.Bar('=', ' [','] '), ' ', progressbar.Percentage()]).start()

    for i in range(ITERATIONS):
        bar.update(i)
        result = os.system('./build/X86/gem5.opt -d m5out/ configs/example/garnet_synth_traffic.py --num-cpus=16 --num-dirs=16 --network=garnet --topology=Mesh_XY --mesh-rows=4 --sim-cycles=100000 --inj-vnet=-1 --vcs-per-vnet=8 --injectionrate=0.1 --synthetic=uniform_random   --routing-algorithm=3 > /dev/null 2>&1')

    print("Training Phase Ended")

else:
    print('Error occured while compiling!')
