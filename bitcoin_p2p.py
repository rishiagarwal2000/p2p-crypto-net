import numpy as np
from queue import PriorityQueue
import uuid
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class PriorityEntry(object):

    def __init__(self, priority, data):
        self.data = data
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority
    
np.random.seed(765)

class node:
    def __init__(self, id, is_slow, genesis_block, Tk=10):
        self.id = id
        self.is_slow = is_slow
        self.peers = []
        self.txn_pool = {}
        self.longest_chain_block = genesis_block
        self.history = {}
        self.last_received_block = -1
        self.block_chain = {genesis_block.id: 0}
        self.err_epsilon = 1e-6
        self.Tk = Tk
    def add_peer(self, other):
        if other not in self.peers:
            self.peers.append(other)
    def is_valid(self, txn):
        return txn.c >= 0 and self.longest_chain_block.balances[txn.idx] >= txn.c
    def get_random_coins(self):
        if np.random.rand() < self.err_epsilon:
            return self.longest_chain_block.balances[self.id] + 10
        return np.random.uniform(0, self.longest_chain_block.balances[self.id])
    def get_pow_delay(self):
        return np.random.exponential(self.Tk)
class Transaction:
    def __init__(self, idx, idy, c):
        self.id = str(uuid.uuid4())
        self.idx = idx
        self.idy = idy
        self.c = c
    def print(self):
        print(f"idx: {self.idx}, idy: {self.idy}, c: {self.c}")
class Block:
    def __init__(self, parent, txns, balances, miner, length=1):
        self.id = str(uuid.uuid4())
        self.parent = parent
        self.txns = txns
        self.balances = balances
        self.num_txns = len(txns)
        self.length = length
        self.miner = miner
    def contains(self, txn):
        if self.parent is None:
            return txn in self.txns.keys()
        else:
            return txn in self.txns.keys() or self.parent.contains(txn)
    def print(self):
        if self.parent is not None:
            print(f"id: {self.id}, parent: {self.parent.id}, length: {self.length}")
        else:
            print(f"id: {self.id}, GENESIS, length: {self.length}")
        return self.txns.keys()
class Simulator:
    def __init__(self, n, z, Ttx, Tks):
        self.n = n
        self.z = z
        self.Ttx = Ttx
        self.Tks = Tks
        self.genesis_block = Block(None, {}, {i:0 for i in range(self.n)}, None)
        self.block_chain = {self.genesis_block.id:self.genesis_block}
        num_slow = int(round(n * z))
        self.nodes = [node(i, i<=num_slow, self.genesis_block, Tk=Tks[i]) for i in range(n)]
        self.form_network()
        self.txns = {}
        self.event_queue = PriorityQueue(0)
        self.event_space = {"gen_block": self.gen_block, "send_block": self.send_block, "receive_block": self.receive_block, "gen_transaction": self.gen_transaction, "receive_transaction": self.receive_transaction}
        self.time = 0
        self.MAX_TIME = 100
        self.MAX_NUM_EVENTS = 5000
        self.MAX_NUM_TRANSACTIONS = 1000
        self.MINING_REWARD = 50
        self.eps = 1e-7
        for i in range(self.n):
            self.event_queue.put(PriorityEntry(0, ("gen_block", i)))
            self.event_queue.put(PriorityEntry(0, ("gen_transaction", i)))
    def form_network(self):
        perm = list(np.random.permutation(self.n))
        added = [perm[0]]
        for i in perm[1:]:
            peer = np.random.choice(added)
            self.nodes[peer].add_peer(self.nodes[i])
            self.nodes[i].add_peer(self.nodes[peer])
        deg = 3
        perm.reverse()
        for i in perm:
            peers = np.random.choice(perm, size=deg, replace=False)
            for peer in peers:
                self.nodes[peer].add_peer(self.nodes[i])
                self.nodes[i].add_peer(self.nodes[peer])
    def simulate(self):
        count_events = 0
        while not self.event_queue.empty() and count_events < self.MAX_NUM_EVENTS:
            count_events += 1
            entry = self.event_queue.get()
            time, event = entry.priority, entry.data
            event_func, event_params = event
            print(f"Executing {event_func} at {time}, {event_params}")
            new_events = self.event_space[event_func](event_params, time)
            for new_time, new_event in new_events:
                print(f"Adding new {new_event[0]} for {new_time}, {new_event[1]}")
                self.event_queue.put(PriorityEntry(new_time, new_event))
            self.time = time
        for block_id, block in self.block_chain.items():
            print(f"{block_id}")
            txn_ids = block.print()
            for txn_id in txn_ids:
                self.txns[txn_id].print()
    def gen_transaction(self, params, time):
        i = params
        idx, node = i, self.nodes[i]
        idy = np.random.randint(self.n)
        c = node.get_random_coins()
        txn = Transaction(idx, idy, c)
        self.txns[txn.id] = txn
        if node.is_valid(txn):
            node.txn_pool[txn.id] = False
        new_events = []
        for peer in node.peers:
            if peer.id not in node.history.get(txn.id, []):
                node.history[txn.id] = node.history.get(txn.id, []) + [peer.id]
                new_time = time + self.get_latency(i, peer.id, 1)
                new_params = peer.id, txn, i
                new_event = ('receive_transaction', new_params)
                new_events.append((new_time, new_event))
        new_time = time + self.get_inter_arrival()
        new_params = i
        new_event = ('gen_transaction', new_params)
        new_events.append((new_time, new_event))
        return new_events
    def receive_transaction(self, params, time):
        i, txn, sender = params
        node = self.nodes[i]
        new_events = []
        if node.is_valid(txn):
            node.history[txn.id] = node.history.get(txn.id, []) + [sender]
            for peer in node.peers:
                if peer.id not in node.history.get(txn.id, []):
                    node.history[txn.id] = node.history.get(txn.id, []) + [peer.id]
                    new_time = time + self.get_latency(i, peer.id, 1)
                    new_params = peer.id, txn, i
                    new_event = ('receive_transaction', new_params)
                    new_events.append((new_time, new_event))
        return new_events
    def gen_block(self, params, time):
        i = params
        node = self.nodes[i]
        new_events = []
        txns = {}
        count = 0
        prune = []
        new_balances = node.longest_chain_block.balances.copy()
        for txn_id in node.txn_pool.keys():
            if count == self.MAX_NUM_TRANSACTIONS - 1:
                break
            if not node.longest_chain_block.contains(txn_id) and node.is_valid(self.txns[txn_id]):
                count += 1
                prune.append(txn_id)
            txns[txn_id] = True
            txn = self.txns[txn_id]
            new_balances[txn.idx] -= txn.c
            new_balances[txn.idy] += txn.c
        for txn_id in prune:
            node.txn_pool.pop(txn_id)
        coinbase_txn = Transaction(None, i, self.MINING_REWARD)
        new_balances[i] += self.MINING_REWARD
        self.txns[coinbase_txn.id] = coinbase_txn
        txns[coinbase_txn.id] = True
        new_block = Block(node.longest_chain_block, txns, new_balances, node, node.longest_chain_block.length+1)
        new_time = time + node.get_pow_delay()
        node.send_block = True
        new_params = i, new_block.id, time
        self.block_chain[new_block.id] = new_block
        new_event = ('send_block', new_params)
        new_events.append((new_time, new_event))
        return new_events
    def send_block(self, params, time):
        i, block_id, gen_time = params
        block = self.block_chain[block_id]
        node = self.nodes[i]
        new_events = []
        if node.last_received_block < gen_time:
            for peer in node.peers:
                if peer.id not in node.history.get(block_id, []):
                    print(f"{i} sent block {block.id} to {peer.id}")
                    node.history[block_id] = node.history.get(block_id, []) + [peer.id]
                    new_time = time + self.get_latency(i, peer.id, block.num_txns)
                    new_params = peer.id, block_id, i
                    new_event = ('receive_block', new_params)
                    new_events.append((new_time, new_event))
            new_time = time + 0
            new_params = i
            new_event = ('gen_block', new_params)
            new_events.append((new_time, new_event))
        else:
            self.block_chain.pop(block_id)
        node.longest_chain_block = block
        node.block_chain[block_id] = time
        return new_events
    def receive_block(self, params, time):
        i, block_id, sender = params
        node = self.nodes[i]
        new_events = []
        block = self.block_chain[block_id]
        print(f"{i} received block {block.id} from {sender}")
        if self.is_valid_block(block_id):
            if block_id not in node.block_chain.keys():
                node.block_chain[block_id] = time
            if node.longest_chain_block.length < block.length:
                node.longest_chain_block = block
                node.last_received_block = time
                new_time = time + self.eps
                new_params = i
                new_event = ('gen_block', new_params)
                new_events.append((new_time, new_event))
            node.history[block_id] = node.history.get(block_id, []) + [sender]
            for peer in node.peers:
                if peer.id not in node.history.get(block_id, []):
                    node.history[block_id] = node.history.get(block_id, []) + [peer.id]
                    new_time = time + self.get_latency(i, peer.id, block.num_txns)
                    new_params = peer.id, block_id, i
                    new_event = ('receive_block', new_params)
                    new_events.append((new_time, new_event))
        return new_events
    def get_latency(self, i, j, num_txn):
        pij = np.random.uniform(10,500)
        cij = 5 if self.nodes[i].is_slow or self.nodes[j].is_slow else 100
        dij = np.random.exponential(96 / cij / 1000)
        latency = pij / 1000 + dij + num_txn / cij / 1000 # (in seconds)
        return latency
    def is_valid_block(self, block_id):
        return True
    def get_inter_arrival(self):
        return 20 #np.random.exponential(self.Ttx)
    def find_confirm_chain(self):
        node_wise_last_blocks_temp = []
        min_length = len(self.block_chain)
        for node in self.nodes:
            node_wise_last_blocks_temp.append(node.longest_chain_block)
            min_length = min(node.longest_chain_block.length, min_length)
        node_wise_last_blocks = []
        for block in node_wise_last_blocks_temp:
            cur_block = block
            while cur_block.length > min_length:
                cur_block = cur_block.parent
            node_wise_last_blocks.append(cur_block)
        backtrack = True
        while backtrack:
            node_wise_last_blocks_temp = node_wise_last_blocks
            node_wise_last_blocks = []
            if not all(block.id == node_wise_last_blocks_temp[0].id for block in node_wise_last_blocks_temp):
                backtrack = True
                for block in node_wise_last_blocks_temp:
                    cur_block = block.parent
                node_wise_last_blocks.append(cur_block)
            else:
                backtrack = False
                confirm_block = node_wise_last_blocks_temp[0]
        return confirm_block
    def show_blockchain(self):
        confirm_block = self.find_confirm_chain()
        tree = nx.DiGraph()
        for block_id, block in self.block_chain.items():
            tree.add_node(block_id, status='unconfirmed')
        for block_id, block in self.block_chain.items():
            if block.parent is not None:
                tree.add_edge(block_id, block.parent.id)   
        cur_block = confirm_block
        while True:
            tree.nodes[cur_block.id]['status'] = 'confirmed'
            if cur_block.parent is not None:
                cur_block = cur_block.parent
            else:
                break
        tree.nodes[self.genesis_block.id]['status'] = 'genesis'
        node_color = []
        genesis_color, confirmed_color, unconfirmed_color = 'blue', 'red', 'yellow'
        for node in tree.nodes(data=True):
            if 'unconfirmed' == node[1]['status']:
                node_color.append(unconfirmed_color)
            elif 'confirmed' == node[1]['status']:
                node_color.append(confirmed_color)
            elif 'genesis' == node[1]['status']:
                node_color.append(genesis_color)

        nx.draw_networkx(tree, with_labels=False, node_size=10, node_color=node_color, width=0.5, arrowsize=5)
        genesis_patch = mpatches.Patch(color=genesis_color, label='Genesis block')
        confirmed_patch = mpatches.Patch(color=confirmed_color, label='Confirmed block')
        unconfirmed_patch = mpatches.Patch(color=unconfirmed_color, label='Unconfirmed block')
        plt.legend(handles=[genesis_patch, confirmed_patch, unconfirmed_patch], loc="upper right")        
        plt.savefig('blockchain.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    Tks = [100, 90, 80, 70, 60, 55, 50, 45, 40, 30]
    # Tks = [100, 100, 100, 100, 100, 100, 100, 100, 100, 10]
    np.random.shuffle(Tks)
    sim = Simulator(10, 0, 10, Tks)
    sim.simulate()
    sim.show_blockchain()