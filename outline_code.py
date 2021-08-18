import numpy as np
import yaml 

"""
Notes : 

1. To handle loopless transaction forwarding we can store the 
sender's id in the event for receiving the message in the event queue

2. Configuration parameters stored in a yaml file for easier recreation of experiments.
Also helps avoid clutter in code

3. Consider maintaining a list of end_of_chain blocks for easier access to blockchain ends.

"""

class simulator:

    def __init__(self, cfg_file):
        ## cfg_file contains all the parameters for the simulation experiment
        ## Must initialise all fields such as graph for the network and peers
        pass 
    
    def calc_latency(self, idx, idy, data_size):
        ## Returns the time taken for transfer of data
        pass

    def add_peer(self, new_idx):
        pass

    def start_world(self):
        pass


class peer:

    def __init__(self, idx, txn_inter_arrival_mean, type):
        ## idx is the index of the peer 
        ## Type here means fast or slow
        pass 

    def get_next_transaction_args(self):
        # Samples from the exponential distribution
        # Returns Id_y, C, and the time of transaction
        pass 
    
    def create_block(self):
        ## Take transactions from list of maintained transactions
        ## Sample duration of creation from exponential distribution
        pass

    def validate_block(self, block):
        ## Validate a given block
        ## Remember to remove common transactions from transactions list if block is correct
        ## Remember to cancel block creation event if new longest chain appears
        pass

class block:

    def __init__(self, id, parent, list_of_transactions, gen_peer_id):
        ## Initialize block
        ## Remember to store length of current chain upto this block as well
        ## One of the transactions will be the coinbase
        pass 

    def check_size(self):
        ## Checks the size of the block is <= 1 MB
        pass
        


        