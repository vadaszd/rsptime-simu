
from collections import defaultdict
from itertools import count
import numpy as np


class Barrier(object):
    def __init__(self, name):
        self.name = name
        self.in_edges = list()
        self.out_edges = list()

    def __str__(self):
        return "Barrier('{}')".format(self.name)
    __repr__ = __str__
    
    class BarrierDict(defaultdict):
        def __missing__(self, key):
            barrier = self[key] = Barrier(key)
            return barrier
                
class TrxBp(object):
    """ A transaction blue-print
    """
    def __init__(self, name, fr, to, loc=0., scale=1., alpha=-1.0, beta=.10):
        fr.out_edges.append(self)
        to.in_edges.append(self)
        self.fr, self.to = fr, to
        self.name, self.loc, self.scale, self.alpha, self.beta = \
             name,      loc,      scale,      alpha,      beta
        
    def __str__(self):
        if self.loc != 0 or self.scale != 1.:
            return "TrxBp('{}', {:3.1f}, {:3.1f})".format(self.name, self.loc, self.scale)
        else:
            return "TrxBp('{}')".format(self.name)
        
    __repr__ = __str__
        
    def collect_events(self, events, X, refTime=0):
        start  = refTime + np.exp(np.random.normal(self.loc + self.alpha, self.scale * self.beta))
        finish = start   + np.exp(np.random.normal(self.loc             , self.scale            ))
        events.extend([(start, "{" + self.name), (finish, self.name + "}")])
        X.append(finish - start)
        return finish
    
class FSBGraph(object):
    
    def __init__(self):
        self.nodes = Barrier.BarrierDict()
        self.edges = list()
        self.node_count = count(0)
        
    @property
    def num_children(self):
        return len(self.edges)
        
    def next_node_id(self):
        return "B{}".format(next(self.node_count))
        
    def add_edge(self, from_id, to_id, loc=0., scale=1., alpha=-1.0, beta=.10):
        name = "{}-->{}".format(from_id, to_id)
        self.edges.append(TrxBp(name, self.nodes[from_id], self.nodes[to_id], 
                                loc, scale, alpha, beta)
                         )
        
    def walk_in_topological_order(self, start_node_id, edge_func, node_func):
        """ Visit each edge and non-start node in a deterministic topological order
        """
        pending_edges = { node: set(node.in_edges) for node in self.nodes.values() }
        pending_nodes = [self.nodes[start_node_id]]
        while pending_nodes:
            from_node = pending_nodes.pop(-1)
            for edge in from_node.out_edges:
                to_node = edge.to
                pending_edges[to_node].discard(edge)
                edge_func(edge)
                if not pending_edges[to_node]:
                    pending_nodes.append(to_node)
                    node_func(to_node)
        if any(pending_edges.values()):
            raise ValueError('Cyclic graph!')

    def collect_events(self, start_node_id, end_node_id, events, X, refTime=0):
        finish_times = dict()
        finish_times[self.nodes[start_node_id]] = refTime
        
        def edge_func(trx):
            #print trx
            finish_times[trx] = trx.collect_events(events, X, finish_times[trx.fr])
    
        def node_func(barrier):
            #print barrier
            finish_times[barrier] = max(finish_times[trx] for trx in barrier.in_edges)
            
        self.walk_in_topological_order(start_node_id, edge_func, node_func)
        return finish_times[self.nodes[end_node_id]] - refTime
    
class FSBSampler(object):
    
    def __init__(self, graph, num_children, training_set_size=0, validation_set_size=0, test_set_size=0):
        self.graph = graph
        self.num_children = num_children
        print ("Generating {} training, {} validation and {} test samples "
               "using the following concurrency structure:" \
               .format(training_set_size, validation_set_size, test_set_size))
        self.train_X_samples     , self.train_RT_samples      = self.sample(training_set_size  ,) 
        self.validation_X_samples, self.validation_RT_samples = self.sample(validation_set_size,)  
        self.test_X_samples      , self.test_RT_samples       = self.sample(test_set_size      ,)

    def sample(self, N, ):
        """ @param N: number of samples to generate
        """
        X_samples = list()
        RT_samples = list()
        for i in range(N):
            events = list()
            X = list()
            RT = self.graph.collect_events(self.graph.start_node, 
                                           self.graph.end_node, 
                                           events, X)
            X_samples.append(X)
            RT_samples.append(RT)
            
        X_samples = np.array(X_samples)
        RT_samples = np.array(RT_samples)
        return X_samples, RT_samples
    
    def get_training_batch(self, batch_size):
        idx = np.random.randint(self.train_X_samples.shape[0], size=batch_size)
        return self.train_X_samples[idx, :], self.train_RT_samples[idx]
    
class _NonSeq(object):
    def __sub__(self, other):
        if isinstance(other, Sequential):
            other.append(self)
            return other
        else:
            return Sequential((self, other))

class _NonPar(object):
    def __or__(self, other):
        if isinstance(other, Parallel):
            other.append(self)
            return other
        else:
            return Parallel((self, other))

class Sequential(list, _NonPar):
    
    def __str__(self):
        return "(" + " - ".join(str(x) for x in self) + ")"
    __repr__ = __str__
        
    def __sub__(self, other):
        if isinstance(other, Sequential):
            other.extend(self)
            return other
        else:
            self.append(other)
            return self
        
    def generate_graph(self, graph, first=None, last=None):
        if first is None: 
            first = graph.next_node_id()
        fr = first
        for structure in self[:-1]:
            to = graph.next_node_id()
            structure.generate_graph(graph, fr, to)
            fr = to
        if last is None:
            last = graph.next_node_id()
        structure.generate_graph(graph, fr, last)
        return first, last
        
class Parallel(list, _NonSeq):

    def __str__(self):
        return "(" + " | ".join(str(x) for x in self) + ")"
    __repr__ = __str__
    
    def __or__(self, other):
        if isinstance(other, Parallel):
            other.extend(self)
            return other
        else:
            self.append(other)
            return self

    def generate_graph(self, graph, first=None, last=None):
        if first is None: 
            first = graph.next_node_id()
        if last is None:
            last = graph.next_node_id()
        for structure in self:
            structure.generate_graph(graph, first, last)
        return first, last
        
class Trx(_NonSeq, _NonPar):
    
    def __init__(self, name, loc=0., scale=1., alpha=-1.0, beta=.10):
        self.name, self.loc, self.scale, self.alpha, self.beta = \
             name,      loc,      scale,      alpha,      beta
        
    def __str__(self):
        if self.loc != 0 or self.scale != 1.:
            return "Trx('{}', {:3.1f}, {:3.1f})".format(self.name, self.loc, self.scale)
        else:
            return "Trx('{}')".format(self.name)
    __repr__ = __str__

    def generate_graph(self, graph, first=None, last=None):
        if first is None: 
            first = graph.next_node_id()
        if last is None:
            last = graph.next_node_id()
        graph.add_edge(first, last, self.loc, self.scale, self.alpha, self.beta)
        return first, last    