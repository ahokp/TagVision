# Author: Paul Ahokas
# This file handles saving RGB-D object pose data and reading it

import pickle

def saveData(data_dict, dir):
    # This function saves RGB-D object pose data into given file.
    
    output = open(dir, 'wb')
    pickle.dump(data_dict, output)
    output.close()

def readData(dir):
    # This function reads and returns RGB-D object pose data from given file.
    
    pkl_file = open(dir, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()

    return data