import numpy as np
import yaml 
import heapq 
from uuid import uuid1
import argparse
import networkx as nx
"""
Notes : 

1. To handle loopless transaction forwarding we can store the 
sender's id in the event for receiving the message in the event queue

2. Configuration parameters stored in a yaml file for easier recreation of experiments.
Also helps avoid clutter in code

3. Consider maintaining a list of end_of_chain blocks for easier access to blockchain ends.

4. Consider attackers of only one type : They can make transactions that allow someone's balance to become negative

"""

class Simulator:

    def __init__(self, cfg_file):
        ## cfg_file contains all the parameters for the simulation experiment
        ## Must initialise all fields such as graph for the network and peers
        with open(cfg_file,'r') as fp:
            self.cfg = yaml.safe_load(fp) 
        self.event_queue = {}
        self.current_time = 0
        self.rho = np.random.uniform(self.cfg["low_rho"],self.cfg["high_rho"],size=(self.cfg["num_peers"],self.cfg["num_peers"]))
        self.total_events = 0

    def calc_latency(self,type_s, type_r, data_size, rho_val):
        ## Returns the time taken for transfer of data
        if type_s == "slow" or type_r == "slow":
            c = self.cfg["slow_cij_val"]
        else:
            c = self.cfg["high_cij_val"]
        d = np.random.exponential(self.cfg["dij_cij_factor"]/c)*1000
        return rho_val + d + (data_size/c)*1000
        

    def get_graph(self):
        n = self.cfg["num_peers"]
        perm = list(np.random.permutation(n))
        in_net = [perm[0]]
        graph = [[]]*n
        for peer_id in perm[1:]:
            num_connections = np.random.randint(1,len(in_net)+1)
            connections = list(np.random.choice(in_net, size = num_connections, replace = False))
            graph[peer_id] = connections
            for conn in connections:
                graph[conn].append(peer_id)
            in_net.append(peer_id)
        return graph
    
    def create_peers(self):
        n = self.cfg["num_peers"]
        slow_peers = round(n*self.cfg["slow_fraction"])
        temp = list(np.random.permutation(n))
        peer_types=[(idx, "slow") for idx in temp[:slow_peers]]+[(idx, "fast") for idx in temp[slow_peers:]]
        peer_types.sort()
        graph = self.get_graph()
        self.peer_graph = graph
        self.peer_list=[]
        for idx, peer_type in peer_types:
            self.peer_list.append(Peer(idx, self.cfg["txn_inter_arrival_time"], self.cfg["mean_mining_time"],peer_type,self.cfg["mining_fee"],self))
        for peer in self.peer_list:
            idx = peer.idx
            conns = list(np.array(self.peer_list)[graph[idx]])
            peer.initialise_neighbours(conns)

    def add_event(self,event,time):
        if time in self.event_queue:
            (self.event_queue[time]).append(event)
        else:
            self.event_queue[time] = [event]

    def initialise_event_queue(self):
        initial_events = []
        for peer in self.peer_list:
            mining_event = Event(peer.start_mining, None)
            txn_event = Event(peer.create_transaction, None)
            initial_events+=[mining_event, txn_event]
        self.event_queue[self.current_time] = initial_events
        
    def run_world(self):
        while self.current_time <= self.cfg["stop_time"]:
            for event in self.event_queue[self.current_time]:
                event.execute_event()
            self.total_events+=len(self.event_queue[self.current_time])
            self.event_queue.pop(self.current_time)
            print("Executed all events at time {}".format(self.current_time))
            try:
                self.current_time = min(self.event_queue)
            except:
                break
        ## Final graph calculations etc
        print("Simulation over\nTotal events executed = {}".format(self.total_events))


    def start_world(self):
        self.create_peers()
        self.initialise_event_queue()
        self.run_world()
        


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
        self.all_received_blocks=set()

    def initialise_neighbours(self, neighbours):
        self.neighbours = {nei : set() for nei in neighbours} ## dict of peers : msg sent from us to them

    def broadcast(self, msg):
        for neighbour, sent in self.neighbours.items():
            if msg not in sent and msg not in neighbour.neighbours[self]:
                latency = self.simulator.calc_latency(self.peer_type, neighbour.peer_type, msg.size,self.simulator.rho[self.idx,neighbour.idx])
                receiving_time = round(self.simulator.current_time + latency,2)
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
        current_peers = list(range(self.simulator.cfg["num_peers"]))
        current_peers.remove(self.idx)
        idy = np.random.choice(current_peers)
        txn_id = uuid1()
        if self.idx in self.current_chain_end.checkpoint:
            coins = np.random.uniform(0, self.current_chain_end.checkpoint[self.idx])
        else:
            coins = 0 ## COIN GENERATION
        new_txn = Transaction(txn_id, self.idx, idy, coins)
        self.pending_txns.add(new_txn)
        self.broadcast(new_txn)
        next_txn_time = round(self.simulator.current_time+np.random.exponential(self.Ttx),2)
        create_txn_event = Event(self.create_transaction,{})
        self.simulator.add_event(create_txn_event, next_txn_time)

    
    def mine_block(self):
        coinbase_id = uuid1()
        coinbase = Transaction(coinbase_id,None,self.idx,self.mining_fee)
        while True:
            num_of_transactions = np.random.randint(0,min(1024,len(self.pending_txns)+1))
            curr_block_txns = list(np.random.choice(list(self.pending_txns),size=num_of_transactions,replace=False))
            curr_block_txns = [coinbase]+curr_block_txns
            checkpoint = self.current_chain_end.checkpoint.copy()
            checkpoint = self.get_new_checkpoint(checkpoint, curr_block_txns)
            if checkpoint:
                break
        blkid = uuid1()
        args = {"block": Block(blkid, self.current_chain_end, curr_block_txns, self.idx, False)}
        creation_time = round(self.simulator.current_time + np.random.exponential(self.mean_mining_time),2)
        block_create_event = Event(self.create_block,args) 
        self.simulator.add_event(block_create_event, creation_time)
        

    def create_block(self, args):
        block = args["block"]
        checkpoint = (block.parent).checkpoint.copy()
        checkpoint = self.get_new_checkpoint(checkpoint, block.txns)
        block.store_checkpoint(checkpoint)
        self.blocktree.append(block)
        self.current_chain_end = block
        self.pending_txns -= set(block.txns)
        self.seen_txns |= set(block.txns)
        self.broadcast(block)
        self.mine_block()
    
    def start_mining(self,args):
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
            checkpoint = self.update_checkpoint(txn, checkpoint)
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
        if block in self.all_received_blocks:
            return
        self.all_received_blocks.add(block)
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
                    temp_chain_end=self.add_pending_blocks(block)
                    if temp_chain_end != self.current_chain_end:
                        pointer = temp_chain_end
                        while pointer != self.current_chain_end:
                            self.pending_txns -= set(pointer.txns)
                            self.seen_txns |= set(pointer.txns)
                            pointer = pointer.parent
                    self.current_chain_end = temp_chain_end
                    self.mine_block()
                else:
                    temp_chain_end=self.add_pending_blocks(block)
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
            self.size = len(self.txns)

    def store_checkpoint(self, checkpoint):
        self.checkpoint = checkpoint
        

class Event:

    def __init__(self, func, args):
        self.func = func
        self.args = args
        self.execute = True
    
    def execute_event(self):
        if self.execute:
            self.func(self.args)

class Transaction:

    def __init__(self, txn_id, sender, receiver, coins):
        self.txn_id = txn_id ## Must be unique
        self.sender = sender ## When None it means its the mining fee (coin base transaction)
        self.receiver = receiver ## Receiver can't be none (invalid transaction)
        assert coins >= 0, "Negative Transactions disallowed"
        self.coins = coins
        self.size = 1 ## in Kb

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    simul = Simulator(args.config)
    simul.start_world()
    

