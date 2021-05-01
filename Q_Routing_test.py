import numpy as np
from os import path
import os
#--- Constants
NROUTERS = 16
NACTIONS = 4

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

def get_action(Q_Table, my_id, dest_id):
    action = np.where(Q_Table[my_id][dest_id] == np.amin(Q_Table[my_id][dest_id]))
    return action[0][0]

def start(my_Router_id, dest_Router_id):

    #--- Getting input about the router id's
#   my_Router_id = input()
#   dest_Router_id =  input()
#   src_Router_id = input()
#	print("I was here")
    #--- Reading Q Table from the file
    Q_Table = read_QTableFile()
    #--- get action from Q-Table and return the action
    action = get_action(Q_Table, my_Router_id, dest_Router_id)
    # print("Python-Action " + str(action)) 
    return action

if(__name__ == "__main__"):
    iterations = 0
    while(1):
        iterations += 1
        print('Iteration:', iterations)
        while(1):
            if(path.exists("input.txt")):
                print("exists")
                break
        my_id = -1
        dest_id = 0

        while(True):
            #print("here")
            with open('input.txt','r') as f:
                lines = f.readlines()
                print(lines)
                if(len(lines) == 2):
                    print("break")
                    break

        my_id = int(lines[0])
        dest_id = int(lines[1])
        print(my_id)
        print(dest_id)
        os.remove("input.txt")
        action = start(my_id,dest_id)
        print("Python action",action)
        f = open("action.txt","w")
        f.write(str(action))
        f.close()


