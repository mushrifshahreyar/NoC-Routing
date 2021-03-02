import numpy as np

#--- Constants
NROUTERS = 16
NACTIONS = 4
WEIGHT = 1

def read_QTableFile():
    file_path = '../../../../../'
    file_name = 'Q_Table.txt'
    Q_Table = np.zeros((NROUTERS,NROUTERS,NACTIONS))
    
    with open(file_name,'r') as f:
        i = 0
        j = 0
        k = 0
        for line in f:
            if(line == "\n"):
                continue
            for val in line.split():
                Q_Table[i][j][k] = float(val)
                k += 1
            j += 1
            k = 0
            if(j == 16):
                i += 1
                j = 0
    return Q_Table

def read_QTable_hopsFile():
    file_path = '../../../../../'
    file_name = 'Q_Table_hops.txt'
    Q_Table = np.zeros((NROUTERS,NROUTERS,NACTIONS))
    
    with open(file_name,'r') as f:
        i = 0
        j = 0
        k = 0
        for line in f:
            if(line == "\n"):
                continue
            for val in line.split():
                Q_Table[i][j][k] = float(val)
                k += 1
            j += 1
            k = 0
            if(j == 16):
                i += 1
                j = 0
    return Q_Table

def get_action(Q_Table, Q_Table_hops, my_id, dest_id):
    Q_Temp = Q_Table[my_id][dest_id]
    Q_Temp1 = Q_Table_hops[my_id][dest_id]
    print(Q_Temp)
    print("----")
    print(Q_Temp1)
    print(np.amin(Q_Temp))
    print(np.amin(Q_Temp1))

    res = np.array(Q_Temp1)
    print()
    print(res)
    action = np.argmin(res)
#    print("Q_Table Values " + str(Q_Table[my_id][dest_id]))
    
    return action

def start(my_Router_id, dest_Router_id):
    print("---------$$$$$$$$--------_$$$$$$$$$")   
    #--- Getting input about the router id's
#   my_Router_id = input()
#   dest_Router_id =  input()
#   src_Router_id = input()
#	print("I was here")
    #--- Reading Q Table from the file
    print(str(my_Router_id) + " " + str(dest_Router_id))
    Q_Table = read_QTableFile()
    Q_Table_hops = read_QTable_hopsFile()
    #print("I was")    
    #--- get action from Q-Table and return the action
    action = get_action(Q_Table, Q_Table_hops, my_Router_id, dest_Router_id)
    print("Python-Action " + str(action))
    return action


start(3, 5)
