import numpy as np
import yaml 
from queue import PriorityQueue
from uuid import uuid1
import argparse
import networkx as nx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os
import gc

class PriorityEntry(object):

    def __init__(self, priority, data):
        self.data = data
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority

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
        self.event_queue = PriorityQueue(0)
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
        self.genesis_block = Block(None,None,None,None,True,self.cfg["num_peers"])
        mining_times = self.cfg["mean_mining_time"]
        if self.cfg["too_many_peers"]:
            mining_times = [mining_times[0]+ i*((mining_times[1]-mining_times[0])/(n-1)) for i in range(n)]
        self.alpha = 1/np.sum(np.reciprocal(mining_times,dtype=float))
        slow_peers = round(n*self.cfg["slow_fraction"])
        temp = list(np.random.permutation(n))
        peer_types=[(idx, "slow") for idx in temp[:slow_peers]]+[(idx, "fast") for idx in temp[slow_peers:]]
        peer_types.sort()
        graph = self.get_graph()
        self.peer_graph = graph
        self.peer_list=[]
        for idx, peer_type in peer_types:
            self.peer_list.append(Peer(idx, self.cfg["txn_inter_arrival_time"], mining_times[idx],peer_type,self.cfg["mining_fee"],self,self.genesis_block))
        for peer in self.peer_list:
            idx = peer.idx
            conns = list(np.array(self.peer_list)[graph[idx]])
            peer.initialise_neighbours(conns)

    def add_event(self,event,time):
        self.event_queue.put(PriorityEntry(time,event))

    def initialise_event_queue(self):
        initial_events = []
        for peer in self.peer_list:
            mining_event = Event(peer.start_mining, None)
            txn_event = Event(peer.create_transaction, None)
            self.add_event(mining_event,self.current_time)
            self.add_event(txn_event,self.current_time)
        
    def run_world(self):
        completed_events=0
        while (not self.event_queue.empty()) and completed_events < self.cfg["max_events"]:
            entry = self.event_queue.get()
            self.current_time = entry.priority
            entry.data.execute_event()
            completed_events+=1
            print(completed_events)
        print("Simulation is over with total {} events executed at simulation time {} sec".format(completed_events,self.current_time/1000))


    def start_world(self):
        self.create_peers()
        self.initialise_event_queue()
        self.run_world()
        
    def show_txns(self):
        print("Transactions :")
        for peer in self.peer_list:
            print("Peer ID {} : {}".format(peer.idx,peer.total_txns))

    def show_blocks(self):
        print("Blocks :")
        for peer in self.peer_list:
            print("Peer ID {} : {}".format(peer.idx,peer.total_blocks))
    
    def show_peer_graph(self):
        graph = nx.DiGraph()
        for i in range(self.cfg["num_peers"]):
            graph.add_node(i,status="peer")
        for i in range(self.cfg["num_peers"]):
            for x in self.peer_graph[i]:
                graph.add_edge(i,x)
        nx.draw_networkx(graph, with_labels=False, node_size=10, width=0.5, arrowsize=5)
        plt.savefig('p2p_graph.png', dpi=300, bbox_inches='tight')
        plt.show()
        



class Peer:

    def __init__(self, idx, txn_inter_arrival_mean, mean_mining_time, peer_type, mining_fee, simulator, genesis_block):
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
        self.blocktree = [genesis_block]
        self.next_block_creation_event = None ## Time at which our next block will be created
        self.current_chain_end = self.blocktree[0]
        self.mining_fee = mining_fee
        self.all_received_blocks=set()
        self.total_blocks=0
        self.total_txns=0
        self.block_arrival_text=""
        self.blockchain_txns=0
        self.pending_txn_max_size=0
        self.pending_txn_option_size = 0
        self.number_of_mines=0

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
                sent.add(msg)
        
    def create_transaction(self, args):
        # Samples from the exponential distribution
        # Returns Id_y, C, and the time of transaction
        current_peers = list(range(self.simulator.cfg["num_peers"]))
        current_peers.remove(self.idx)
        idy = np.random.choice(current_peers)
        txn_id = uuid1()
        if self.idx in self.current_chain_end.checkpoint:
            coins = np.random.uniform(0, self.current_chain_end.checkpoint[self.idx]/100000)
        else:
            coins = 0 ## COIN GENERATION
        # coins=0
        new_txn = Transaction(txn_id, self.idx, idy, coins)
        self.pending_txns.add(new_txn)
        self.pending_txn_max_size = max(self.pending_txn_max_size,len(self.pending_txns))
        self.broadcast(new_txn)
        self.total_txns+=1
        next_txn_time = round(self.simulator.current_time+np.random.exponential(self.Ttx),2)
        create_txn_event = Event(self.create_transaction,{})
        self.simulator.add_event(create_txn_event, next_txn_time)

    
    def mine_block(self):
        coinbase_id = uuid1()
        coinbase = Transaction(coinbase_id,None,self.idx,self.mining_fee)
        # while True:
        num_of_transactions = min(1024,len(self.pending_txns)+1)-1#np.random.randint(0,min(1024,len(self.pending_txns)+1))
        self.pending_txn_option_size +=len(self.pending_txns)
        self.number_of_mines+=1
        curr_block_txns = list(np.random.choice(list(self.pending_txns),size=num_of_transactions,replace=False))
        curr_block_txns = [coinbase]+curr_block_txns
        checkpoint = self.current_chain_end.checkpoint.copy()
        checkpoint, txn_idx = self.get_new_checkpoint(checkpoint, curr_block_txns,mining=True)
        # if checkpoint:
        #     break
        blkid = uuid1()
        args = {"block": Block(blkid, self.current_chain_end, curr_block_txns[:txn_idx], self.idx, False,self.simulator.cfg["num_peers"])}
        creation_time = round(self.simulator.current_time + np.random.exponential(self.mean_mining_time),2)
        block_create_event = Event(self.create_block,args)
        self.next_block_creation_event = block_create_event 
        self.simulator.add_event(block_create_event, creation_time)
        

    def create_block(self, args):
        block = args["block"]
        checkpoint = (block.parent).checkpoint.copy()
        checkpoint = self.get_new_checkpoint(checkpoint, block.txns)
        block.store_checkpoint(checkpoint)
        self.blocktree.append(block)
        self.blockchain_txns+=len(block.txns)
        self.current_chain_end = block
        self.pending_txns -= set(block.txns)
        # self.seen_txns |= set(block.txns)
        self.broadcast(block)
        self.total_blocks+=1
        self.mine_block()
        print("Block created at time {} with id : {} by Peer ID : {}".format(self.simulator.current_time,block.blkid,self.idx))
    
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
        
    def get_new_checkpoint(self, checkpoint, txns, mining=False):
        txn_idx = 0
        for txn in txns:
            temp_checkpoint = checkpoint.copy()
            checkpoint = self.update_checkpoint(txn, checkpoint)
            if checkpoint is None:
                if mining:
                    return temp_checkpoint, txn_idx
                return None
            txn_idx+=1
        if mining:
            return checkpoint, txn_idx
        return checkpoint

    def receive_transaction(self, args):
        # print("Received Transaction at Peer ID : {}".format(self.idx))
        txn = args["txn"]
        self.broadcast(txn)
        if txn not in self.current_chain_end.seen_txns:
            self.pending_txns.add(txn)
            self.pending_txn_max_size = max(self.pending_txn_max_size,len(self.pending_txns))
    
    def add_pending_blocks(self, new_block):
        remove_blocks = {}
        foliage = set(self.pending_blocks).add(new_block)
        max_chain_length_block=new_block
        check_blocks = {new_block}
        visited_adding = [(0,0)]*len(self.pending_blocks)
        blk_idx = {blk : i for i, blk in enumerate(self.pending_blocks)}
        p_idx = 0
        while p_idx<len(self.pending_blocks):
            if visited_adding[p_idx][0]!=1:
                temp=[]
                judgement = None 
                cur_block = self.pending_blocks[p_idx]
                invalid = False
                while True:
                    if cur_block==new_block:
                        judgement=True
                        break
                    elif visited_adding[blk_idx[cur_block]]==(1,0):
                        judgement=False
                        break
                    elif visited_adding[blk_idx[cur_block]]==(1,-1):
                        judgement=True 
                        invalid=True
                        break
                    elif visited_adding[blk_idx[cur_block]]==(1,1):
                        judgement=True
                        break
                    elif cur_block.parent in foliage:
                        temp.append(blk_idx[cur_block])
                        cur_block = cur_block.parent
                    else:
                        temp.append(blk_idx[cur_block])
                        judgement = False 
                        break
                if judgement:
                    for idx in temp[::-1]:
                        if invalid:
                            visited_adding[idx] = (1,-1)
                            continue
                        blk = self.pending_blocks[idx]
                        checkpoint = blk.parent.checkpoint.copy()
                        checkpoint = self.get_new_checkpoint(checkpoint,blk.txns)
                        if checkpoint:
                            blk.store_checkpoint(checkpoint)
                            visited_adding[idx] = (1,1)
                            if blk.chain_length > max_chain_length_block.chain_length:
                                max_chain_length_block = blk
                        else:
                            visited_adding[idx] = (1,-1)
                            invalid=True
                else:
                    for idx in temp:
                        visited_adding[idx]=(1,0)
            p_idx+=1
        
        for i, dec in enumerate(visited_adding):
            if dec==(1,1):
                self.blocktree.append(self.pending_blocks[i])
                (self.pending_blocks[i]).parent.seen_its_child(self.idx)
        self.pending_blocks = [self.pending_blocks[i] for i, dec in enumerate(visited_adding) if dec == (1,0)]
        return max_chain_length_block

    def receive_block(self, args):
        # print("Received block at Peer ID : {}".format(self.idx))
        block = args["block"]
        if block in self.all_received_blocks:
            return
        self.all_received_blocks.add(block)
        self.block_arrival_text+="{} : {} ms\n\n".format(block.blkid,self.simulator.current_time)
        if block.parent not in self.blocktree:
            self.pending_blocks.append(block)
            self.broadcast(block)
        else:
            checkpoint = (block.parent).checkpoint.copy()
            checkpoint = self.get_new_checkpoint(checkpoint, block.txns)
            if checkpoint:
                self.broadcast(block)
                block.store_checkpoint(checkpoint)
                self.blocktree.append(block)
                self.blockchain_txns+=len(block.txns)
                block.parent.seen_its_child(self.idx)
                if block.parent == self.current_chain_end:
                    if self.next_block_creation_event is not None:
                        self.next_block_creation_event.execute = False
                    # self.seen_txns |= set(block.txns)
                    self.pending_txns -= set(block.txns)
                    self.current_chain_end = block
                    temp_chain_end=self.add_pending_blocks(block)
                    if temp_chain_end != self.current_chain_end:
                        pointer = temp_chain_end
                        while pointer != self.current_chain_end:
                            self.pending_txns -= set(pointer.txns)
                            # self.seen_txns |= set(pointer.txns)
                            pointer = pointer.parent
                    self.current_chain_end = temp_chain_end
                    self.mine_block()
                else:
                    temp_chain_end=self.add_pending_blocks(block)
                    if temp_chain_end.chain_length > self.current_chain_end.chain_length:
                        if self.next_block_creation_event is not None:
                            self.next_block_creation_event.execute = False
                        self.pending_txns -= temp_chain_end.seen_txns
                        self.current_chain_end = temp_chain_end
                        self.mine_block()
    
    def get_stats(self):
        blocks_per_peer = [0]*(self.simulator.cfg["num_peers"])
        cur_block = self.current_chain_end
        while cur_block.blkid!="GENESIS":
            blocks_per_peer[cur_block.creator_id]+=1
            cur_block=cur_block.parent
        # print(blocks_per_peer)
        return blocks_per_peer
        

    def show_blocktree(self):  
        tree = nx.DiGraph()  
        for block in self.blocktree:
            tree.add_node(block.blkid,status="Branch_block")
        for block in self.blocktree:
            if block.blkid!="GENESIS":
                tree.add_edge(block.blkid,block.parent.blkid)
        cur_block = self.current_chain_end
        while cur_block.blkid!="GENESIS":
            tree.nodes[cur_block.blkid]["status"] = "Longest_chain_block"
            cur_block = cur_block.parent
        tree.nodes["GENESIS"]["status"] = "GENESIS_block"
        node_color = []
        genesis_color, longest_chain_color, branch_color = 'yellow', 'red', 'blue'
        for node in tree.nodes(data=True):
            if 'Branch_block' == node[1]['status']:
                node_color.append(branch_color)
            elif 'Longest_chain_block' == node[1]['status']:
                node_color.append(longest_chain_color)
            elif 'GENESIS_block' == node[1]['status']:
                node_color.append(genesis_color)
        nx.draw_networkx(tree, with_labels=False, node_size=10, node_color=node_color, width=0.5, arrowsize=5)
        genesis_patch = mpatches.Patch(color=genesis_color, label='Genesis block')
        longest_patch = mpatches.Patch(color=longest_chain_color, label='Longest Chain block')
        branch_patch = mpatches.Patch(color=branch_color, label='Branch block')
        plt.legend(handles=[genesis_patch, longest_patch, branch_patch], loc="upper right")        
        plt.savefig('blockchain_{}.png'.format(self.idx), dpi=300, bbox_inches='tight')
        plt.show()

    def show_fraction_of_chain(self):
        blocks_per_peer = self.get_stats()
        y = [num/self.current_chain_end.chain_length for num in blocks_per_peer]
        x = ["peer_{}\nHp:{}\n{}".format(idx,round(self.simulator.alpha/self.simulator.peer_list[idx].mean_mining_time,3),self.simulator.peer_list[idx].peer_type) for idx in range(self.simulator.cfg["num_peers"])]
        fig = plt.figure(figsize = (10, 5))
        plt.bar(x, y, color ='blue',width = 0.4)
        for index, value in enumerate(y):
            value = round(value,3)
            plt.text(index , 1.02*value,'%f' % value, ha='center', va='bottom')
        plt.xticks(x)
        plt.xlabel("Peer ID")
        plt.ylabel("Fraction of Longest chain")
        plt.title("Fraction of longest chain produced per peer")
        plt.show()

    def show_fraction_of_total_blocks(self):
        blocks_per_peer = self.get_stats()
        y=[]
        for i, num in enumerate(blocks_per_peer):
            blocks_produced = self.simulator.peer_list[i].total_blocks
            if blocks_produced!=0:
                y.append(num/blocks_produced)
            else:
                y.append(0)
        x = ["peer_{}\nHp:{}\n{}".format(idx,round(self.simulator.alpha/self.simulator.peer_list[idx].mean_mining_time,3),self.simulator.peer_list[idx].peer_type) for idx in range(self.simulator.cfg["num_peers"])]
        fig = plt.figure(figsize = (10, 5))
        plt.bar(x, y, color ='blue',width = 0.4)
        for index, value in enumerate(y):
            value = round(value,3)
            plt.text(index , 1.02*value,'%f' % value, ha='center', va='bottom')
        plt.xticks(x)
        plt.xlabel("Peer ID")
        plt.ylabel("Fraction of total blocks")
        plt.title("Fraction of Total blocks that went into Longest chain per peer")
        plt.show()
    
    def write_block_arrival_time(self):
        folder = self.simulator.cfg["text_files_folder"]
        if not os.path.isdir(folder):
            os.mkdir(folder)
        with open(os.path.join(folder,"peer_{}.txt".format(self.idx)),'w') as fp:
            fp.write("Peer ID : {}\nMean Block Creation Time : {} ms\n\n\n".format(self.idx,self.simulator.alpha)+self.block_arrival_text)
    
    def get_branch_lengths(self):
        is_parent = {block : 0 for block in self.blocktree}
        for block in self.blocktree:
            if block!=self.simulator.genesis_block:
                is_parent[block.parent]=1
        longest_chain_idx = 0
        y=[]
        for block, val in is_parent.items():
            if val==0:
                y.append(block.chain_length)
                if block == self.current_chain_end:
                    longest_chain_idx = len(y)-1
                
        x=["SB_{}".format(i) for i in range(longest_chain_idx)]+["MB"]+["SB_{}".format(i) for i in range(longest_chain_idx,len(y)-1)]
        fig = plt.figure(figsize = (10, 5))
        plt.bar(x, y, color ='blue',width = 0.4)
        for index, value in enumerate(y):
            plt.text(index , 1.02*value,'%d' % int(value), ha='center', va='bottom')
        plt.xticks(x)
        plt.xlabel("Branches (SB : Side Branch ; MB : Main Branch)")
        plt.ylabel("Length in number of blocks")
        plt.title("Length of Blockchain branches")
        plt.show()
        
    def show_final_stats(self):
        print("".join(['*']*30)+"Peer ID {}".format(self.idx)+"".join(['*']*30))
        self.show_blocktree()
        self.show_fraction_of_chain() 
        self.show_fraction_of_total_blocks()
        self.get_branch_lengths()
        self.write_block_arrival_time()
        print("Average Number of Transactions per block in entire blockchain for Peer ID {} : {}".format(self.idx,self.blockchain_txns/len(self.blocktree)))
        print("Max Size of pending txn pool for Peer ID {} : {}".format(self.idx,self.pending_txn_max_size))
        print("Average size of transaction pool at time of choosing txns is {}\n".format(self.pending_txn_option_size/self.number_of_mines))

        
            
class Block:

    def __init__(self, blkid, parent, list_of_transactions, gen_peer_id, genesis_block, number_of_peers):
        ## Initialize block
        ## Remember to store length of current chain upto this block as well
        ## One of the transactions will be the coinbase
        if genesis_block:
            self.blkid = "GENESIS"
            self.checkpoint = {i : 0 for i in range(number_of_peers)}
            self.chain_length = 1
            self.size = 1
            self.received_by = 0
            self.seen_txns = set()
        else:
            self.parent  = parent 
            self.parent_id = self.parent.blkid
            self.blkid = blkid 
            self.txns = list_of_transactions
            self.creator_id = gen_peer_id
            self.chain_length = self.parent.chain_length + 1
            self.size = len(self.txns)
            self.received_by = 1
            self.seen_txns = self.parent.seen_txns | set(self.txns)
        self.peers_who_saw_child = [False]*number_of_peers
        self.sum_chpeers = 0

    def store_checkpoint(self, checkpoint):
        self.checkpoint = checkpoint
    
    def seen_its_child(self, peer_idx):
        self.sum_chpeers += (self.peers_who_saw_child[peer_idx] ^ True)
        self.peers_who_saw_child[peer_idx] = True
        if self.sum_chpeers == len(self.peers_who_saw_child):
            try:
                del self.seen_txns
                gc.collect()
            except:
                pass

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
    simul.show_txns()
    simul.show_blocks()
    simul.show_peer_graph()
    for i in range(simul.cfg["num_peers"]):
        simul.peer_list[i].show_final_stats()
        
    


    

