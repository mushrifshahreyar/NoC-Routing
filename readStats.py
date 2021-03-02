import os


keywords = ['average_packet_latency','average_packet_network_latency','average_packet_queueing_latency','packets_injected::total', 'packets_received::total']
traffic_type = ['shuffle', 'uniform_random']
injection_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

routing_algorithms = {
        "Q_Routing_CPP": 3
        }

def readFile():
    l = []
    with open('m5out/stats.txt') as f:
        l = f.readlines()
        l = [x.strip() for x in l]
    return l

def getValues(lines):
    vals = []
    for key in keywords:
        vals += [l for l in lines if key in l]    
    return vals

def print_results(results, traffic, algorithm, inj_rate):
    with open("Results.txt", "a+") as f:
        f.write("Traffic: {}\t Algorithm: {}\t Injection rate: {}\n".format(traffic,algorithm, inj_rate))
        for res in results:
            res = res.split()
            k = res[0].split('.')
            f.write(k[3] + " " + res[1] + "\n")
            f.write("\n")


if __name__ == "__main__":
#    return_code = os.system('scons ../build/X86/gem5.opt -j12')
    return_code = 0
    if(os.path.exists("Results.txt")):
        os.system("rm Results.txt")
    if(return_code == 0):
        for traffic in traffic_type:
            for routing_algo in routing_algorithms:
                for rate in injection_rate:
                    code = './build/X86/gem5.opt -d m5out/ configs/example/garnet_synth_traffic.py --num-cpus=16 --num-dirs=16 --network=garnet --topology=Mesh_XY --mesh-rows=4 --sim-cycles=100000 --inj-vnet=-1 --vcs-per-vnet=8 --injectionrate={} --synthetic={}   --routing-algorithm={}'.format(rate,traffic,routing_algorithms[routing_algo])
                    result = os.system(code)
                    if(result == 0):
                        lines = readFile()
                        results = getValues(lines)
                        print_results(results,traffic, routing_algo, rate)


