import numpy as np
import yaml 
import heapq 
from uuid import uuid1

"""
Notes : 

1. To handle loopless transaction forwarding we can store the 
sender's id in the event for receiving the message in the event queue

2. Configuration parameters stored in a yaml file for easier recreation of experiments.
Also helps avoid clutter in code

3. Consider maintaining a list of end_of_chain blocks for easier access to blockchain ends.

4. Consider attackers of only one type : They can make transactions that allow someone's balance to become negative

"""
def calc_latency(type_s, type_r, data_size):
        ## Returns the time taken for transfer of data
        pass

class Simulator:

    def __init__(self, cfg_file):
        ## cfg_file contains all the parameters for the simulation experiment
        ## Must initialise all fields such as graph for the network and peers
        pass 

    def add_peer(self, new_idx):
        pass

    def start_world(self):
        pass


class Peer:

    def __init__(self, idx, txn_inter_arrival_mean, mean_mining_time, peer_type, mining_fee, simulator):
        ## idx is the index of the peer 
        ## Type here means fast or slow
        self.idx = idx 
        self.Ttx = txn_inter_arrival_mean
        self.mean_mining_time = mean_mining_time
        self.peer_type = peer_type 
        self.pending_txns = set() 
        self.seen_txns = set()
        self.pending_blocks = []
        self.simulator = simulator
        self.blocktree = [Block(None,None,None,None,True)]
        self.next_block_creation_event = None ## Time at which our next block will be created
        self.current_chain_end = self.blocktree[0]
        self.mining_fee = mining_fee

    def initialise_neighbours(self, neighbours):
        self.neighbours = {nei : set() for nei in neighbours} ## dict of peers : msg sent from us to them

    def broadcast(self, msg):
        for neighbour, sent in self.neighbours.items():
            if msg not in sent and msg not in neighbour.neighbours[self]:
                latency = calc_latency(self.peer_type, neighbour.peer_type, msg.size)
                receiving_time = self.simulator.current_time + latency 
                if isinstance(msg, Transaction):
                    func = neighbour.receive_transaction
                    args = {"txn" : msg}
                else:
                    func = neighbour.receive_block
                    args = {"block" : msg}
                self.simulator.add_event(Event(func,args), receiving_time)
        
    def create_transaction(self, args):
        # Samples from the exponential distribution
        # Returns Id_y, C, and the time of transaction
        current_peers = self.simulator.peer_ids.copy()
        current_peers.remove(self.idx)
        idy = np.random.choice(current_peers)
        txn_id = uuid1()
        coins = None ## Discuss how to generate coins. We do 
        ##              have our estimate of balance depending on the block we are mining on
        new_txn = Transaction(txn_id, self.idx, idy, coins)
        self.pending_txns.add(new_txn)
        self.broadcast(new_txn)
        next_txn_time = self.simulator.current_time+round(np.random.exponential(self.Ttx))
        create_txn_event = Event(self.create_transaction,{})
        self.simulator.add_event(create_txn_event, next_txn_time)


    def mine_block(self):
        coinbase_id = uuid1()
        coinbase = Transaction(coinbase_id,None,self.idx,self.mining_fee)
        while True:
            num_of_transactions = np.random.randint(0,1024)
            curr_block_txns = list(np.random.choice(list(self.pending_txns),num_of_transactions))
            curr_block_txns = [coinbase]+curr_block_txns
            checkpoint = self.current_chain_end.checkpoint.copy()
            checkpoint = self.get_new_checkpoint(checkpoint, curr_block_txns)
            if checkpoint:
                break
        blkid = uuid1()
        args = {"block": Block(blkid, self.current_chain_end, curr_block_txns, self.idx, False)}
        creation_time = self.simulator.current_time + round(np.random.exponential(self.mean_mining_time))
        block_create_event = Event(self.create_block,args) 
        self.simulator.add_event(block_create_event, creation_time)
        

    def create_block(self, args):
        block = args["block"]
        self.blocktree.append(block)
        self.current_chain_end = block
        self.pending_txns -= set(block.txns)
        self.seen_txns |= set(block.txns)
        self.broadcast(block)
        self.mine_block()
    
    def update_checkpoint(self, txn, checkpoint): ## Done only after checking for txn id uniqueness
        if txn.sender is not None:
            if txn.sender in checkpoint:
                checkpoint[txn.sender] -= txn.coins
                if txn.receiver in checkpoint:
                    checkpoint[txn.receiver] += txn.coins
                else:
                    checkpoint[txn.receiver] = txn.coins
                if checkpoint[txn.sender] < 0:
                    return None 
            else:
                return None 
        else:
            ## Coinbase txn
            if txn.receiver in checkpoint:
                checkpoint[txn.receiver] += txn.coins
            else:
                checkpoint[txn.receiver] = txn.coins
            
        return checkpoint
        
    def get_new_checkpoint(self, checkpoint, txns):
        for txn in txns:
            checkpoint = self.update(txn, checkpoint)
            if checkpoint is None:
                return None
        return checkpoint

    def receive_transaction(self, args):
        txn = args["txn"]
        self.broadcast(txn)
        if txn not in self.seen_txns:
            self.pending_txns.add(txn)
    
    def add_pending_blocks(self, new_block):
        remove_blocks = set()
        max_chain_length_block=new_block
        for block in self.pending_blocks:
            if block.parent in self.blocktree:
                checkpoint = (block.parent).checkpoint.copy()
                checkpoint = self.get_new_checkpoint(checkpoint, block.txns)
                if checkpoint:
                    block.store_checkpoint(checkpoint)
                    self.blocktree.append(block)
                    if block.chain_length > max_chain_length_block.chain_length:
                        max_chain_length_block = block
                remove_blocks.add(block)
        self.pending_blocks = list(set(self.pending_blocks) - remove_blocks)
        return max_chain_length_block

    def receive_block(self, args):
        block = args["block"]
        if block.parent not in self.blocktree:
            self.pending_blocks.append(block)
            self.broadcast(block)
        else:
            checkpoint = (block.parent).checkpoint.copy()
            checkpoint = self.get_new_checkpoint(checkpoint, block.txns)
            if checkpoint:
                # broadcast here 
                block.store_checkpoint(checkpoint)
                self.blocktree.append(block)
                if block.parent == self.current_chain_end:
                    self.next_block_creation_event.execute = False
                    self.seen_txns |= set(block.txns)
                    self.pending_txns -= set(block.txns)
                    self.current_chain_end = block
                    temp_chain_end=add_pending_blocks(block)
                    if temp_chain_end != self.current_chain_end:
                        pointer = temp_chain_end
                        while pointer != self.current_chain_end:
                            self.pending_txns -= set(pointer.txns)
                            self.seen_txns |= set(pointer.txns)
                            pointer = pointer.parent
                    self.current_chain_end = temp_chain_end
                    self.mine_block()
                else:
                    temp_chain_end=add_pending_blocks(block)
                    if temp_chain_end.chain_length > self.current_chain_end.chain_length:
                        self.next_block_creation_event.execute = False
                        pointer = temp_chain_end
                        self.seen_txns = set()
                        while pointer.blkid != "GENESIS":
                            self.seen_txns |= set(pointer.txns)
                            pointer = pointer.parent
                        self.pending_txns -= self.seen_txns
                        self.current_chain_end = temp_chain_end
                        self.mine_block()
                
class Block:

    def __init__(self, blkid, parent, list_of_transactions, gen_peer_id, genesis_block):
        ## Initialize block
        ## Remember to store length of current chain upto this block as well
        ## One of the transactions will be the coinbase
        if genesis_block:
            self.blkid = "GENESIS"
            self.checkpoint = {}
            self.chain_length = 1
            self.size = 1
        else:
            self.parent  = parent 
            self.parent_id = self.parent.blkid
            self.blkid = blkid 
            self.txns = list_of_transactions
            self.creator_id = gen_peer_id
            self.chain_length = self.parent.chain_length + 1
            self.size = self.txns

    def store_checkpoint(self, checkpoint):
        self.checkpoint = checkpoint
        

class Event:

    def __init__(self, func, args):
        self.func = func
        self.args = args
        self.execute = True
    
    def execute_event(self):
        if self.execute:
            func(args)

class Transaction:

    def __init__(self, txn_id, sender, receiver, coins):
        self.txn_id = txn_id ## Must be unique
        assert sender >=0 and sender 
        self.sender = sender ## When None it means its the mining fee (coin base transaction)
        self.receiver = receiver ## Receiver can't be none (invalid transaction)
        assert coins >= 0, "Negative Transactions disallowed"
        self.coins = coins
        self.size = 1 ## in KB
