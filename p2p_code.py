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
    """The event queue used is a min priority queue and this is the class used 
    to record events in it. Priority is given to the time of execution of event

    Args:
        priority (float): Time of execution of event
        data (Event): Event object representing the event to be executed

    Attributes:
        priority (float): Denotes time of execution. Events with lower time of execution will be executed earlier.
        data (Event): Event class object. Contains the event to be exccuted in the form of a function.
        
    """

    def __init__(self, priority, data):
        self.data = data
        self.priority = priority

    def __lt__(self, other):
        """Comparision method 

        Args:
            other (PriorityEntry): A different PriorityEntry object to compare with
        
        Returns:
            True if self has lesser priority than other

        """
        return self.priority < other.priority


class Simulator:
    """Simulator will create all peers and initialize them. It will also execute events at the next time step 
    by iterating over the  event queue in increasing order of time

    Args:
        cfg_file (str): .yaml file containing all parameters for the simulation 

    Attributes:
        cfg_filename (str): .yaml file containing all simulation configuration parameters
        cfg (dict of str to float or int or str): Contains all parameters from cfg_filename in dictionary format {param_name : value}
        event_queue (PriorityQueue): A min PriorityQueue which stores PriorityEntry objects
        current_time (float): The time correspnding to the current event being executed
        rho (float): Numpy array containing delay times corresponding to speed of light propagation for every pair of peers
        alpha (float): alpha/Tk gives fraction of hashing power for a peer where Tk is the mean mining time of the peer
        genesis_block (Block): The genesis block that is supplied to every peer at the start of the simulation
        peer_graph (list of list of int): P2P network between peers stored in the format of adjacency lists using peer IDs
        peer_list (list of Peer): List of Peer objects representing peers in P2P network
        
    """

    def __init__(self, cfg_file):
        self.cfg_filename = cfg_file
        with open(cfg_file,'r') as fp:
            self.cfg = yaml.safe_load(fp) 
        self.event_queue = PriorityQueue(0)
        self.current_time = 0
        self.rho = np.random.uniform(self.cfg["low_rho"],self.cfg["high_rho"],size=(self.cfg["num_peers"],self.cfg["num_peers"]))

    def calc_latency(self,type_s, type_r, data_size, rho_val):
        """Calculates latency between two peers in network

        Args:
            type_s (str): Takes values 'slow' or 'fast'. Network speed type of sender peer
            type_s (str): Takes values 'slow' or 'fast'. Network speed type of receiver peer
            data_size (int): Memory size of message to be sent
            rho_val (float): Delay due to speed of light propagation between sender and receiver peer

        Returns:
            Float value representing time taken send message from sender to receiver

        """
        if type_s == "slow" or type_r == "slow":
            c = self.cfg["slow_cij_val"]
        else:
            c = self.cfg["high_cij_val"]
        d = np.random.exponential(self.cfg["dij_cij_factor"]/c)*1000
        return rho_val + d + (data_size/c)*1000
        

    def get_graph(self):
        """Creates a random connected P2P graph between peers at the start of simulation
        
        Args:
            None

        Returns:
            A list of list of int representing the graph in adjacency list format

        """
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
        """Creates, connects and initialises the peers 

        Args:
            None

        Returns:
            None

        """
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
        """Adds a PriorityEntry to event_queue

        Args:
            event (Event): Event class object containing the event to be executed
            time (float): Time at which to execute event
        
        Returns:
            None

        """
        self.event_queue.put(PriorityEntry(time,event))

    def initialise_event_queue(self):
        """Initialises event_queue so as start the mining and transactions creation process at each peer

        Args:
            None

        Returns:
            None

        """
        initial_events = []
        for peer in self.peer_list:
            mining_event = Event(peer.start_mining, None)
            txn_event = Event(peer.create_transaction, None)
            self.add_event(mining_event,self.current_time)
            self.add_event(txn_event,self.current_time)
        
    def run_world(self):
        """Starts executing events in event_queue starting at time 0

        Args:
            None
        
        Returns:
            None

        """
        completed_events=0
        while (not self.event_queue.empty()) and completed_events < self.cfg["max_events"]:
            entry = self.event_queue.get()
            self.current_time = entry.priority
            entry.data.execute_event()
            completed_events+=1
            print(completed_events)
        print("Simulation is over with total {} events executed at simulation time {} sec".format(completed_events,self.current_time/1000))


    def start_world(self):
        """Starts the simulation by creating peers, initialising event_queue and commencing execution of events

        Args:
            None

        Returns:
            None

        """
        self.create_peers()
        self.initialise_event_queue()
        self.run_world()
        
    def show_txns(self):
        """Displays total transactions created by each peer at the end of simulation

        Args:
            None

        Returns:
            None

        """
        print("Transactions :")
        for peer in self.peer_list:
            print("Peer ID {} : {}".format(peer.idx,peer.total_txns))

    def show_blocks(self):
        """Displays total blocks created by each peer at the end of simulation

        Args:
            None

        Returns:
            None
            
        """
        print("Blocks :")
        for peer in self.peer_list:
            print("Peer ID {} : {}".format(peer.idx,peer.total_blocks))
    
    def show_peer_graph(self):
        """Displays the connected P2P graph network

        Args:
            None

        Returns:
            None
            
        """
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
    """Peer class instants represent a peer in the network

    Args:
        idx (int): ID of the peer
        txn_inter_arrival_mean (float): Mean time between creation of two transactions
        mean_mining_time (float): Mean time between creation of two blocks
        peer_type (str): Takes values 'slow' or 'fast'
        mining_fee (float): Bitcoins paid to a miner on generation of a block
        simulator (Simulator): Reference to the simulator running the simulation 
        genesis_block (Block): Genesis block created by the simulator at the start of simulation

    Attributes:
        idx (int): Peer ID in the network
        Ttx (float): Mean time between creation of transactions
        mean_mining_time (float): Mean time between creation of blocks
        peer_type (str): Takes values 'slow' or 'fast' 
        pending_txns (set of Transaction): Set of transactions that can be included in a block being 
            mined on the longest chain
        pending_blocks (list of Block): List of received blocks whose parent has not been received yet
        simulator (Simulator): Reference to the simulator running the simulation
        blocktree (list of Block): List of blocks that have been confirmed and added to this peer's 
            version of the blockchain
        next_block_creation_event (Event): Event class object denoting the next event of creation of a 
            block. Useful for cancelling block creation event that creates a block on a shorter chain
        current_chain_end (Block): The block being mined on currently 
        mining_fee (float): Money generated due to creation of one block
        all_received_blocks (set of Block): Set of blocks that have been received to avoid re-addition to blockchain
        total_blocks (int): Total blocks generated by peer
        total_txns (int): Total transactions generated by peer
        block_arrival_text (str): Records the block hash, block number, time of arrival and parent block hash of
            every received block
        blockchain_txns (int): Total number of transactions in the blocktree
        pending_txn_max_size (int): Maximum size of the pending_txns pool
        pending_txn_option_size (int): Sums up the length of pending_txns at the time of choosing a set of
            transactions to include in a block being mined. Used to find average size of pending_txns pool
            at time of mining
        number_of_mines (int): Number of times a set of transactions was chosen for a block to be mined
        neighbours (dict of Peer to set of Block or Transaction): Dictionary recording what messages have been 
            sent to each of its immediate neighbours or peers

    """

    def __init__(self, idx, txn_inter_arrival_mean, mean_mining_time, peer_type, mining_fee, simulator, genesis_block):
        self.idx = idx 
        self.Ttx = txn_inter_arrival_mean
        self.mean_mining_time = mean_mining_time
        self.peer_type = peer_type 
        self.pending_txns = set() 
        self.pending_blocks = []
        self.simulator = simulator
        self.blocktree = [genesis_block]
        self.next_block_creation_event = None 
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
        """Initialises the immediate neighbours and the set of messages sent to each of them to empty set

        Args:
            neighbours (list of Peer): List of Peer objects representing immediate neighbours in network

        Returns:
            None

        """
        self.neighbours = {nei : set() for nei in neighbours} ## dict of peers : msg sent from us to them

    def broadcast(self, msg):
        """Given a message, forwards it to neighbours who have not received it from this peer 
        and who have not sent it to this peer

        Args:
            msg (Block or Transaction): A Block object or Transaction object 

        Returns:
            None
            
        """
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
        """Creates a transaction 

        Args:
            args (None): Added for uniformity

        Returns:
            None
            
        """
        # Samples from the exponential distribution
        # Returns Id_y, C, and the time of transaction
        current_peers = list(range(self.simulator.cfg["num_peers"]))
        current_peers.remove(self.idx)
        idy = np.random.choice(current_peers)
        txn_id = uuid1()
        if self.idx in self.current_chain_end.checkpoint:
            coins = np.random.uniform(0, self.current_chain_end.checkpoint[self.idx]/100000)
        else:
            coins = 0 
        new_txn = Transaction(txn_id, self.idx, idy, coins)
        self.pending_txns.add(new_txn)
        self.pending_txn_max_size = max(self.pending_txn_max_size,len(self.pending_txns))
        self.broadcast(new_txn)
        self.total_txns+=1
        next_txn_time = round(self.simulator.current_time+np.random.exponential(self.Ttx),2)
        create_txn_event = Event(self.create_transaction,{})
        self.simulator.add_event(create_txn_event, next_txn_time)

    
    def mine_block(self):
        """Creates a block creation event and adds it to the simulator's event_queue so as to 
        create a block in the future

        Args:
            None

        Returns:
            None
            
        """
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
        blkid = uuid1()
        args = {"block": Block(blkid, self.current_chain_end, curr_block_txns[:txn_idx], self.idx, False,self.simulator.cfg["num_peers"])}
        creation_time = round(self.simulator.current_time + np.random.exponential(self.mean_mining_time),2)
        block_create_event = Event(self.create_block,args)
        self.next_block_creation_event = block_create_event 
        self.simulator.add_event(block_create_event, creation_time)
        

    def create_block(self, args):
        """Creates a block 

        Args:
            args (dict of str to Block): Dictionary containing the Block object that needs to be added to the blockchain
                and also broadcasted

        Returns:
            None
            
        """
        block = args["block"]
        checkpoint = (block.parent).checkpoint.copy()
        checkpoint = self.get_new_checkpoint(checkpoint, block.txns)
        block.store_checkpoint(checkpoint)
        self.blocktree.append(block)
        self.blockchain_txns+=len(block.txns)
        self.current_chain_end = block
        self.pending_txns -= set(block.txns)
        self.broadcast(block)
        self.total_blocks+=1
        self.mine_block()
        print("Block created at time {} with id : {} by Peer ID : {}".format(self.simulator.current_time,block.blkid,self.idx))
    
    def start_mining(self,args):
        """Starts the mining process. Used by simulator's event_queue to initialise the event_queue

        Args:
            args (None): Added for uniformity

        Returns:
            None
            
        """
        self.mine_block()

    def update_checkpoint(self, txn, checkpoint): 
        """Given a transaction and a dictionary of balances for each pair it updates the balances 
        of the peers involved in the transaction

        Args:
            txn (Transaction): The transaction which states the sender, receiver peers and the amount of 
                bitcoins involved
            checkpoint (dict of int to int): Dictionary storing balances in the form of {peer_id : balance} format
        
        Returns:
            Updated checkpoint with changed balances or None if any balance becomes negative due to the transaction

        """
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
        """Given a list of transactions and an initial checkpoint, generates the updated checkpoint after 
        processing all transactions. 

        Args:
            checkpoint (dict of int to int): Conatins balances for each peer in the format {peer_id : balance}
            txns (list of Transaction): List of Transaction objects 
            mining (boolean): Boolean value to control output. Refer to Returns section

        Returns:
            If mining is True, then it returns the index of transactions upto which the balances remain non-
            negative and the resulting updated checkpoint.
            If mining is False, then it checks through all transactions and returns updated checkpoint only
            if balances remain non-negative else None
            
        """
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
        """Receives a transaction from a neighbour peer

        Args:
            args (dict of str to Transaction): Dictionary containing the Transaction object.

        Returns:
            None

        """
        txn = args["txn"]
        self.broadcast(txn)
        if txn not in self.current_chain_end.seen_txns:
            self.pending_txns.add(txn)
            self.pending_txn_max_size = max(self.pending_txn_max_size,len(self.pending_txns))
    
    def add_pending_blocks(self, new_block):
        """Adds pending blocks (whose parents haven't arrived) to the blockchain on addition of a 
        new block to the blockchain

        Args:
            new_block (Block): Block object that has been received and added to the blockchain. It can trigger 
                addition of some pending blocks
        
        Returns:
            Amongst the newly added blocks, it returns the one with maximum chain length (or height) in the blockchain

        """
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
                self.blockchain_txns+=len(self.pending_blocks[i].txns)
        self.pending_blocks = [self.pending_blocks[i] for i, dec in enumerate(visited_adding) if dec == (1,0)]
        return max_chain_length_block

    def receive_block(self, args):
        """Receives a block, validates and adds it to the blockchain if it is a valid block

        Args:
            args (dict of str to Block): Dictionary containing the received block

        Returns:
            None

        """
        block = args["block"]
        if block in self.all_received_blocks:
            return
        self.all_received_blocks.add(block)
        self.block_arrival_text+="{}, {}, {} sec, {}\n".format(block.blkid,block.chain_length-1,self.simulator.current_time/1000,block.parent.blkid)
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
        """Calculates the number of blocks each peer has in the longest chain

        Args:
            None

        Returns:
            None

        """
        blocks_per_peer = [0]*(self.simulator.cfg["num_peers"])
        cur_block = self.current_chain_end
        while cur_block.blkid!="GENESIS":
            blocks_per_peer[cur_block.creator_id]+=1
            cur_block=cur_block.parent
        return blocks_per_peer
        

    def show_blocktree(self):  
        """Creates a graphical representation of the blockchain for users to visualize

        Args:
            None

        Returns:
            None

        """
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
        # plt.savefig('blockchain_{}.png'.format(self.idx), dpi=300, bbox_inches='tight')
        plt.show()

    def show_fraction_of_chain(self):
        """Displays a bar graph showing the fraction of blocks created by each peer in this peer's
        version of the blockchain

        Args:
            None
        
        Returns:
            None

        """
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
        """Displays a bar graph showing the fraction of blocks created by each peer that got into 
        this peer's version of the blockchain

        Args:
            None
        
        Returns:
            None
            
        """
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
        """Writes the block arrival data to a text file

        Args:
            None
        
        Returns:
            None
            
        """
        folder = self.simulator.cfg["text_files_folder"]
        target = os.path.join(folder,os.path.basename(self.simulator.cfg_filename))
        if not os.path.isdir(target):
            os.mkdir(target)
        with open(os.path.join(target,"peer_{}.txt".format(self.idx)),'w') as fp:
            fp.write("blockHash, blockNum, TimeOfArrival, parentBlockHash\n"+self.block_arrival_text)
    
    def get_branch_lengths(self):
        """Displays a bar graph showing the lengths of all the side branches (orphaned chains) 
        and the longest chain in this peers version of the blockchain

        Args:
            None
        
        Returns:
            None
            
        """
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
        """Calls the plotting functions all together. Also displays some data regarding the average number of 
        transactions included in each block in the blockchain, the maximum size of the pending transactions pool
        and the average size of the pending transactions pool when mining a block

        Args:
            None

        Returns:
            None

        """
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
    """Represents the block in a blockchain

    Args:
        blkid (str): ID for the block
        parent (Block): Parent block 
        list_of_transactions (list of Transaction): List of transactions to be included in this block
        gen_peer_id (int): ID of the creator peer
        genesis_block (boolean): True if genesis block needs to be created False otherwise
        number_of_peers (int): Number of peers in the network

    Attributes:
        blkid (str): ID identifying each block uniquely
        checkpoint (dict of int to int): Dictioanry storing balances of each peer as calculated along the 
            chain this block was mined on, from the genesis block to this block
        chain_length (int): Denotes the length of the chain that this block was mined on. Chain length 
            for genesis block is considered 1
        size (int): Memory size in Kb
        seen_txns (set of Transaction): Transactions that have been included in all the blocks (including
            this block) on the chain that this block was mined on
        parent (Block): Parent block for this block
        txns (list of Transaction): Transactions that have been included in this block
        creator_id (int): ID of the peer whoc created this block
        peers_who_saw_child (list of boolean): peers_who_saw_child[idx] is True if peer with ID = idx
            has received a block that is a child of this block (that mines on this block)
        sum_chpeers (int): Number of True values in peers_who_saw_child

    """

    def __init__(self, blkid, parent, list_of_transactions, gen_peer_id, genesis_block, number_of_peers):
        if genesis_block:
            self.blkid = "GENESIS"
            self.checkpoint = {i : 0 for i in range(number_of_peers)}
            self.chain_length = 1
            self.size = 1
            self.seen_txns = set()
        else:
            self.parent  = parent 
            self.blkid = blkid 
            self.txns = list_of_transactions
            self.creator_id = gen_peer_id
            self.chain_length = self.parent.chain_length + 1
            self.size = len(self.txns)
            self.seen_txns = self.parent.seen_txns | set(self.txns)
        self.peers_who_saw_child = [False]*number_of_peers
        self.sum_chpeers = 0

    def store_checkpoint(self, checkpoint):
        """Records the checkpoint at this block

        Args:
            checkpoint (dict of int to int): Dictioanry storing balances of each peer as calculated along the 
                chain this block was mined on, from the genesis block to this block
        
        Returns:
            None

        """
        self.checkpoint = checkpoint
    
    def seen_its_child(self, peer_idx):
        """Registers that a particular peer has received a block that is a child of this block
        (that was mined on this block). This function prunes the blockchain by deleting seen_txns
        for blocks on which mining is guaranteed to not take place at any time in the future

        Args:
            peer_idx (int): ID of the peer who received a child block of this block

        Returns:
            None

        """
        self.sum_chpeers += (self.peers_who_saw_child[peer_idx] ^ True)
        self.peers_who_saw_child[peer_idx] = True
        if self.sum_chpeers == len(self.peers_who_saw_child):
            try:
                del self.seen_txns
                gc.collect()
            except:
                pass

class Event:
    """Event class instants represent an event to be executed in the event_queue

    Args:
        func (function): Function to be executed which simulates event execution
        args (dict of str to Block or Transaction): Arguments for the function to be executed

    Attributes:
        func (function): A receive_transaction, receive_block, create_transaction or create_block function 
            corresponding to a particular peer
        args (dict of str to Block or Transaction): Dictionary contains the arguments needed for the function 
            to be called
        execute (boolean): A boolean value to toggle between executing and not executing the event. True by 
            default

    """

    def __init__(self, func, args):
        self.func = func
        self.args = args
        self.execute = True
    
    def execute_event(self):
        """Executes the stored function with the stored arguments if execute is True

        Args:
            None
        
        Returns:
            None

        """
        if self.execute:
            self.func(self.args)

class Transaction:
    """Transaction class instances represent a transaction in the simulation

    Args:
        txn_id (str): Transaction ID for this transaction
        sender (int): ID of peer who will pay bitcoins
        receiver (int): ID of peer who will receive bitcoins
        coins (float): Bitcoins involved in the transfer

    Attributes:
        txn_id (str): The unique ID of the transaction
        sender (int): The ID of the peer who will pay the bitcoin
        receiver (int): The ID of the peer who will receive the bitcoin
        coins (float): Bitcoins involved in the transfer of money
        size (int): Memory size of the transaction in Kb
        
    """

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
        
    


    

