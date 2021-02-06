keywords = ['average_packet_latency','average_packet_network_latency','average_packet_queueing_latency','packets_injected::total', 'packets_received::total']

def readFile():
    l = []
    with open('../m5out/stats.txt') as f:
        l = f.readlines()
        l = [x.strip() for x in l]
    return l

def getValues(l):
    vals = []
    for key in keywords:
        if key in l:
            print(l)
            vals += l
    print(vals)

lines = readFile()
getValues(lines)
