import numpy as np

#--- Constants
NROUTERS = 16
NACTIONS = 4

def read_QTableFile():
    file_path = '../../../../../'
    file_name = 'Q_Table.txt'
    Q_Table = np.zeros((NROUTERS,NROUTERS,NACTIONS))
    
    with open(file_path + file_name,'r') as f:
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

def get_action(Q_Table, my_id, dest_id):
    action = np.where(Q_Table[my_id][dest_id] == np.amin(Q_Table[my_id][dest_id]))

    return action[0][0]

def start(my_Router_id, dest_Router_id):

    #--- Getting input about the router id's
#    my_Router_id = input()
#    dest_Router_id =  input()
#    src_Router_id = input()

    #--- Reading Q Table from the file
    print("from python")
    Q_Table = read_QTableFile()
    
    #--- get action from Q-Table and return the action
    action = get_action(Q_Table, my_Router_id, dest_Router_id)
    return action
