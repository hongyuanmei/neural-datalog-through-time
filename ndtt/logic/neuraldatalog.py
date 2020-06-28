"""
Neural Datalog does
(1) create params for rules (e.g. transform params, LSTM params)
variant of time, i.e. for each state of Datalog database
(2) create provable terms (and their dependency dags)
(3) create cell blocks
(4) kill cell blocks as needed 
(5) disprove terms when their dependencies aren't true
"""
import os 
import gzip
import pickle

import numpy
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F

from ndtt.logic.datalogplus import DatalogPlus
from ndtt.logic.database import DatabaseFast as Database
from ndtt.neural.linear import LinearZero, LinearCell
from ndtt.neural.aggr import Aggregation, DecayWeight, AggregationDecay
#from ndtt.neural.gate import CTLSTMGate
#from ndtt.neural.cell import CTLSTMCell
from ndtt.neural.act import Tanh, Softplus

class NeuralDatalog(object): 

    def __init__(self, lstmpool, updatemode, device=None): 
        print("init neural datalog database")
        self.datalog = DatalogPlus()
        self.lstmpool = lstmpool
        self.updatemode = updatemode
        if lstmpool == 'full': 
            self.compute_new_cell = self.compute_new_cell_full
        elif lstmpool == 'simp': 
            self.compute_new_cell = self.compute_new_cell_simple
        else: 
            raise Exception(f"Unknown LSTM pooling method : {lstmpool}")
        if updatemode == 'sync': 
            self.update_cells = self.update_cells_sync
        elif updatemode == 'async': 
            self.update_cells = self.update_cells_async
        else: 
            raise Exception(f"Unknown update mode : {updatemode}")
        self.tdb = dict() # tdb stands for temporal database 
        device = device or 'cpu'
        self.device = torch.device(device)

    def load_rules_from_database(self, path_database): 
        self.path_database = path_database
        with open(path_database) as f: 
            rules = f.read().split('\n')
        if rules[-1] == '': 
            rules.pop()
        self.load_rules(rules)

    def load_rules(self, rules): 
        """
        rules : list of rules 
        """
        self.datalog.load_rules(rules)

    """
    params may change on the fly
    init_params only creates containers for these params
    then we update params given (loaded) tdb
    note : params are not discarded even if whoever used it dies
    so as to avoid cold start (if that node block comes to being again)
    """
    def init_params(self): 
        """
        create transform params (:- or transform rule)
        create LSTM params (<- or see rule)
        """
        self.transform_param = dict() 
        self.transform_param_intensity = dict() # for extra-dimension of intensity
        self.see_param = dict()
        self.see_param_intensity = dict() # for extra-dimension of intensity
        self.aggregation = dict()
        self.aggregation['#_aggregate_decay_#'] = AggregationDecay()
        self.decay_weight = dict()
        self.activation = dict()
        self.activation['tanh'] = Tanh()
        self.activation['softplus'] = Softplus()
        self.transform_idx_and_dim = set()
        self.see_idx_and_dim = set()

    def update_params_given_tdb(self): 
        for s, tdbs_s in self.tdb.items(): 
            for tdb in tdbs_s: 
                for i, cdb_pack in tdb.items(): 
                    cdb = cdb_pack['db_after_create']
                    if cdb is not None: 
                        self.update_params(cdb)
                    cdb = cdb_pack['db_after_kill']
                    if cdb is not None: 
                        self.update_params(cdb)

    def update_params(self, cdb): 
        """
        update params with current database, i.e., cdb
        if cdb has any params that we haven't yet, create them
        """
        self.update_transform_params(cdb)
        self.update_see_params(cdb)
    
    def update_transform_params(self, cdb): 
        """
        update transform params given current database
        """
        cdb_idx_and_dim = cdb.transform_idx_and_dim
        new_idx_and_dim = cdb_idx_and_dim - self.transform_idx_and_dim
        # NOT loop over all of them, track the set difference
        for one_idx_and_dim in new_idx_and_dim: 
            self.update_transform_params_per_idx_and_dim(one_idx_and_dim)
        self.transform_idx_and_dim = self.transform_idx_and_dim.union(new_idx_and_dim)

    def update_transform_params_per_idx_and_dim(self, idx_and_dim): 
        betaparam_i, mparam_i, in_dim, in_z, out_dim, out_z, head_is_event = idx_and_dim
        if betaparam_i not in self.aggregation: 
            #print(f"create agg : {betaparam_i}")
            self.aggregation[betaparam_i] = Aggregation(betaparam_i)
        if mparam_i not in self.transform_param: 
            #print(f"create transform : {mparam_i}")
            self.transform_param[mparam_i] = LinearZero(
                mparam_i, in_dim, out_dim, in_z, out_z)
        if head_is_event and mparam_i not in self.transform_param_intensity: 
            #print(f"create transform intensity : {mparam_i}")
            self.transform_param_intensity[mparam_i] = LinearZero(
                mparam_i, in_dim, 1, in_z, [])

    def update_see_params(self, cdb): 
        """
        update see params given current database
        """
        cdb_idx_and_dim = cdb.see_idx_and_dim
        new_idx_and_dim = cdb_idx_and_dim - self.see_idx_and_dim
        for one_idx_and_dim in new_idx_and_dim: 
            self.update_see_params_per_idx_and_dim(one_idx_and_dim)
        self.see_idx_and_dim = self.see_idx_and_dim.union(new_idx_and_dim)

    def update_see_params_per_idx_and_dim(self, idx_and_dim): 
        betaparam_i, mparam_i, in_dim, in_z, cell_dim, cell_z, head_is_event = idx_and_dim
        if betaparam_i not in self.aggregation: 
            #print(f"create agg : {betaparam_i}")
            self.aggregation[betaparam_i] = Aggregation(betaparam_i)
        if betaparam_i not in self.decay_weight: 
            self.decay_weight[betaparam_i] = DecayWeight(betaparam_i)
        if mparam_i not in self.see_param: 
            #print(f"create see : {mparam_i}")
            self.see_param[mparam_i] = LinearCell(
                mparam_i, in_dim, cell_dim, in_z, cell_z)
        if head_is_event and mparam_i not in self.see_param_intensity: 
            #print(f"create see intensity : {mparam_i}")
            self.see_param_intensity[mparam_i] = LinearCell(
                mparam_i, in_dim, 1, in_z, [])

    def load_params(self, idx_and_dim, params): 
        """
        first load the idx and dim of each param to create the containers
        """
        self.transform_idx_and_dim = idx_and_dim['transform']
        self.see_idx_and_dim = idx_and_dim['see']
        for one_idx_and_dim in self.transform_idx_and_dim: 
            self.update_transform_params_per_idx_and_dim(one_idx_and_dim)
        for one_idx_and_dim in self.see_idx_and_dim: 
            self.update_see_params_per_idx_and_dim(one_idx_and_dim)
        """
        then load the prev saved states/values of params
        NOTE : we check existence of params in saved model
        to accomodate models trained by old code
        """
        for k, v in self.map_params().items():
            if k not in params: 
                continue
            for kk, vv in v.items(): 
                if kk not in params[k]: 
                    continue
                vv.load_state_dict(params[k][kk])

    def save_params(self, loc):
        """
        save params and their idx and dim
        """
        rst = dict() 
        for k, v in self.map_params().items(): 
            rst[k] = dict()
            for kk, vv in v.items(): 
                rst[k][kk] = vv.state_dict()
        torch.save(rst, loc)
        rst = {
            'transform': self.transform_idx_and_dim, 
            'see': self.see_idx_and_dim
        }
        with open(loc+'_idx_and_dim.pkl', 'wb') as f: 
            pickle.dump(rst, f)
        # finished

    def chain_params(self): 
        rst = []
        for _, p in self.map_params().items(): 
            if isinstance(p, dict): 
                for k, v in p.items(): 
                    rst.append(v.parameters())
            else: 
                rst.append(p.parameters())
        return chain.from_iterable(rst)

    def check_params(self): 
        for _, p in self.map_params().items(): 
            if isinstance(p, dict): 
                for k, v in p.items(): 
                    for kk, vv in v.named_parameters(): 
                        for i in torch.flatten(vv.data): 
                            if torch.isnan(i): 
                                m = f"param nan : {k}, {v}, {kk}, {vv}"
                                raise Exception(m)

    def check_grads(self): 
        for _, p in self.map_params().items(): 
            if isinstance(p, dict): 
                for k, v in p.items(): 
                    for kk, vv in v.named_parameters(): 
                        if vv.grad is not None: 
                            for i in torch.flatten(vv.grad.data): 
                                if torch.isnan(i): 
                                    m = f"grad nan : {k}, {v}, {kk}, {vv}\n"
                                    m += f"grad is : {vv.grad}"
                                    raise Exception(m)

    def print_params(self): 
        for _, p in self.map_params().items(): 
            if isinstance(p, dict): 
                for k, v in p.items(): 
                    self.print_param_(k,v)
            else: 
                self.print_param_(k,v)
    
    def print_param_(self, k, v): 
        print(f"params of {k}")
        for n, p in v.named_parameters(): 
            if p.requires_grad: 
                print(f"param {n} is : {p.data}")

    def map_params(self): 
        return {
            'transform_param': self.transform_param, 
            'transform_param_intensity': self.transform_param_intensity, 
            'see_param': self.see_param, 
            'see_param_intensity': self.see_param_intensity, 
            'aggregation': self.aggregation, 
            'decay_weight': self.decay_weight, 
            'activation': self.activation
        }

    def count_params(self): 
        cnt = 0
        for _, param_dict in self.map_params().items(): 
            for _, param in param_dict.items(): 
                cnt += param.count_params()
        return cnt

    def cuda(self): 
        # put all tensors on cuda
        # self.params[k].cuda()
        # self.terms[k].cuda()
        raise NotImplementedError


    def load_tdb(self, db, s, num, loc): 
        file_name = f'db-{db}_s-{s}'
        path_db = os.path.join(loc, f'{file_name}.pkl')
        assert os.path.exists(path_db), f"not exist : {path_db}"
        # already been cached 
        print(f"load previously cached temporal databases")
        with gzip.GzipFile(path_db, 'r') as f:
            self.tdb[s] = pickle.load(f)
        total_num = len(self.tdb[s])
        assert total_num >= num, "not enough cached tdb"
        del self.tdb[s][num : ] # clear for memory
        print(f"only first {num} are useful, {total_num - num} (of {total_num}) are discarded")
    """
    create variables that may CHANGE as events happen
    e.g. provable terms, valid events, cell blocks
    as querying might be slow
    we will cache all the query results 
    s.t. when we train/dev/test on the same data with the same database
    we can directly read the cache
    """
    def create_tdb(self, db, s, seqs, loc, per=1): 
        """
        temporal database is a dict that stores: 
        db : database e.g. structd32
        s : split e.g. train/dev/test...
        split : {0 : seq of databases of seq-0, 1 : ... }
        e.g. 
        train : {0 : seq of dbs, 1 : seq of dbs, ... }
        split : train or dev or test ...
        seqs : seqs of this split to make tdb
        loc : where to save the cached database
        form of loc : PATHDOMAIN/tdbcache
        """
        if not os.path.exists(loc): 
            os.makedirs(loc)
        
        #file_name = f'db-{db}_s-{s}_r-{r}'
        file_name = f'db-{db}_s-{s}'
        path_db = os.path.join(loc, f'{file_name}.pkl')
        assert not os.path.exists(path_db), f"already exist : {path_db}"
        # this assertion avoids mistaken rewriting
        self.tdb[s] = list()
        total_num = len(seqs)

        """
        tdb_seq[-1], tdb_seq[0] : same for all seqs 
        we don't repeat their creation
        """
        #print(f"create tdb for prefix")
        prefix = self.create_tdb_seq_prefix()

        #print(f"create tdb for seqs")
        for i, seq in enumerate(seqs): 
            self.tdb[s].append( self.create_tdb_seq(seqs[i], prefix) )
            if ( i + 1 ) % per == 0: 
                print(f"finish {i}-th seq tdb creation")
 
        print(f"save newly created tdb for {total_num} seqs")
        with gzip.GzipFile(path_db, 'w') as f:
            # cache the temporal database
            # use gzip to save space
            pickle.dump(self.tdb[s], f)

    def create_tdb_seq_prefix(self): 
        #print(f"create tdb for prefix")
        return self.create_tdb_seq([{'name': 'bos', 'time': 0.0}])

    #@profile
    def create_tdb_seq(self, seq, prefix=None): 
        """
        create a dict of databases 
        idx : database (stored as Python dicts)
        meaning the database used for >= idx-th event token in seq
        e.g. 0 : database 
        meaning that database used after 0-th event, i.e. bos
        """
        self.datalog.clear_asserted_facts() # clear seq-asserted facts
        # in case bos does nothing (which is not probable but possible)
        # we should store the database properties with index -1 
        tdb_seq = dict()

        if prefix is not None and -1 in prefix: 
            # if this has been pre-created
            tdb_seq[-1] = prefix[-1]
        else: 
            tdb_seq[-1] = {
                'time': None, 'name': None, 
                'killed': False, 'killed_cells': [], 
                'created': False, 'created_cells': [], 
                'db_after_create': self.get_current_database(), 
                'db_after_kill': None
            }
        # save the original database before even bos happens
        cdb = tdb_seq[-1]['db_after_create']

        for i, e in enumerate(seq): 
            #print(f"\n\n\nmake tdb for {i}-th : {e}")
            t = e['time']
            k = e['name']

            if prefix is not None and i in prefix: 
                # if this has been pre-created
                tdb_seq[i] = prefix[i]
                #print(f"use prefix for index {i}")
                """
                MUST update database even if we use cached prefix
                otherwise, future queries are incorrect
                """
                self.datalog.create_cells(tdb_seq[i]['created_cells'])
                if tdb_seq[i]['created']: 
                    cdb = tdb_seq[i]['db_after_create']
                    assert cdb is not None, "really created new things?"
                self.datalog.kill_cells(tdb_seq[i]['killed_cells'])
                if tdb_seq[i]['killed']: 
                    cdb = tdb_seq[i]['db_after_kill']
                    assert cdb is not None, "really killed anything?"
                """
                NOTE : database updating is very tricky 
                if we ever again find any bug related to temporal change of database
                we should revisit this part and investigate 
                whether the database is updated and tracked properly
                """

            else: 
                #print(f"\n\n\nget current database for i = {i} : {e}")
                #print(f"new info for index {i}")
                #print(f"{k} at {t}")
                #print(f"tdb current : {tdb_seq[i]}")
                killed, who_to_kill = self.will_kill_cells(k, cdb)
                created, new_cells = self.will_create_cells(k, cdb)
                """
                # NOTE : don't use methods in datalog---try to reduce # of calls to datalog
                killed, who_to_kill = self.datalog.will_kill_cells(k)
                created, new_cells = self.datalog.will_create_cells(k)
                """
                #print(f"killed={killed}, killed_cells={who_to_kill}")
                #print(f"created={created}, created_cells={new_cells}")
                tdb_seq[i] = {
                    'time': t, 'name': k, 
                    'killed': killed, 'killed_cells': who_to_kill, 
                    'created': created, 'created_cells': new_cells, 
                }
                """
                update database : MUST happen before getting updated database
                """
                if created: 
                    self.datalog.create_cells(tdb_seq[i]['created_cells'])
                    #print(f"\n\nget cdb after create ... i = {i} : {e}")
                    #print(f"\n\ncreated : {tdb_seq[i]['created_cells']}")
                    tdb_seq[i]['db_after_create'] = self.get_current_database()
                    cdb = tdb_seq[i]['db_after_create']
                else: 
                    tdb_seq[i]['db_after_create'] = None
                
                if killed:
                    self.datalog.kill_cells(tdb_seq[i]['killed_cells'])
                    #print(f"\n\nget cdb after kill ... i = {i} : {e}")
                    #print(f"\n\nkilled : {tdb_seq[i]['killed_cells']}")
                    tdb_seq[i]['db_after_kill'] = self.get_current_database()
                    cdb = tdb_seq[i]['db_after_kill']
                else: 
                    tdb_seq[i]['db_after_kill'] = None
        """
        we clear asserted facts twice to be safe 
        one to clear left-over of last seq, one for next seq
        although it is more than necessary 
        """
        self.datalog.clear_asserted_facts() # clear for next seq
        return tdb_seq

    #@profile 
    def will_kill_cells(self, k, cdb): 
        event_term = self.datalog.aug_term(k)
        """
        events keys are those will create, be seen by and/or kill something
        so it is possible that something in the sequence 
        not exist in keys cuz they do nothing but jut existing there...
        """
        who_to_kill = cdb.events[event_term].who_it_kill if event_term in cdb.events else []
        who_to_kill_set = cdb.events[event_term].who_it_kill_set if event_term in cdb.events else set()
        exist_cells = cdb.cell_names_set
        assert who_to_kill_set.issubset(exist_cells), "kill things that don't exist?!"
        will_kill = True if len(who_to_kill) > 0 else False
        return will_kill, who_to_kill

    #@profile
    def will_create_cells(self, k, cdb): 
        event_term = self.datalog.aug_term(k)
        """
        events keys are those will create, be seen by and/or kill something
        so it is possible that something in the sequence 
        not exist in keys cuz they do nothing but jut existing there...
        """
        who_see_it = cdb.events[event_term].who_see_it if event_term in cdb.events else []
        who_see_it_set = cdb.events[event_term].who_see_it_set if event_term in cdb.events else set()
        exist_cells = cdb.cell_names_set
        #print(f"\n who sees this event : {who_see_it_set}")
        #print(f"\n exist cells : {exist_cells}")
        new_cells = who_see_it_set.difference(exist_cells)
        will_create = True if len(new_cells) > 0 else False
        return will_create, sorted(list(new_cells))

    #@profile
    def get_current_database(self): 
        return Database(self.datalog)
    
    #@profile
    def create_cache(self, times): 
        """
        cache the computed embeddings
        cache is times-specific (i.e. interval-specific)
        """
        assert isinstance(times, torch.Tensor), "it has to be a tensor"
        self.cache_times = times
        self.cache_emb = {
            'true(specialaug)': torch.zeros(*times.size(), 1, device=self.device), 
            'bos(specialaug)': torch.zeros(*times.size(), 1, device=self.device)
        }
        self.cache_inten = {} # cache intensities
        #self.cache_inten_bound = {} # cache intensity bounds
        #bound computation doesn't need time or embedding values

    #@profile
    def clear_cache(self): 
        # current: clear all cache
        # todo: we want to do smart clearing: only clear things affected
        del self.cache_times
        del self.cache_emb
        del self.cache_inten
        #del self.cache_inten_bound
        #bound computation doesn't need time or embedding values

    #@profile
    def compute_intensities(self, event_types, cdb, active): 
        """
        compute intensities for event types at times
        using the current database, i.e. cdb
        """
        for i, e in enumerate(event_types): 
            if e not in self.cache_inten: 
                self.compute_intensity_(e, cdb, active)
        intensities = [self.cache_inten[e] for e in event_types]
        return torch.stack(intensities, dim=-1)

    def compute_intensity_bounds(self, event_types, cdb, active): 
        """
        compute intensity bounds for event types at times 
        using the current database, i.e. cdb
        """
        self.cache_inten_bound = dict()
        for i, e in enumerate(event_types): 
            if e not in self.cache_inten_bound: 
                self.compute_intensity_bound_(e, cdb, active)
        intensity_bounds = [self.cache_inten_bound[e] for e in event_types]
        del self.cache_inten_bound
        return torch.stack(intensity_bounds, dim=-1) # 1 * len(event_types)

    #@profile
    def compute_intensity_(self, event_type, cdb, active): 
        """
        compute intensity of a given event type
        """
        event_type = self.datalog.aug_term(event_type)
        provable = event_type in cdb.terms
        hascell = event_type in cdb.cells
        assert provable, \
            f"event type {event_type} not provable? no way cuz it must be head of :- rule"
        intensity = 0.0
        t = cdb.terms[event_type]
        for r_i, b_facts_i in t.transform_edges.items(): 
            # maybe chech r_i b_i 1-to-1 correspondence?
            for b_i, facts_i in b_facts_i.items():
                to_aggr = []
                for body_i, m_i in facts_i: 
                    to_cat = []
                    for subgoal in body_i: 
                        if subgoal not in self.cache_emb: 
                            self.compute_embedding_(subgoal, cdb, active)
                        to_cat.append(self.cache_emb[subgoal])
                    output = self.transform_param_intensity[m_i](torch.cat(to_cat, dim=-1))
                    #assert output.size(-1) == 1, "more than one output dimension?"
                    to_aggr.append(output)
                aggred = self.aggregation[b_i](torch.stack(to_aggr, dim=0))
                # after aggregation, sum them up
                # r_i and b_i is 1-to-1
                intensity += aggred
        if hascell: # if it is an active LSTM cell 
            intensity += active['intensity_cells'][event_type].decay(self.cache_times)
        self.cache_inten[event_type] = \
            self.activation['softplus'](intensity).squeeze(-1) # sequeeze 1

    def compute_intensity_bound_(self, event_type, cdb, active): 
        """
        compute intensity bound of a given event type 
        no need to compute embeddings
        """
        event_type = self.datalog.aug_term(event_type)
        provable = event_type in cdb.terms
        hascell = event_type in cdb.cells
        assert provable, \
            f"event type {event_type} not provable? no way cuz it must be head of :- rule"
        intensity = 0.0 
        t = cdb.terms[event_type]
        for r_i, b_facts_i in t.transform_edges.items(): 
            for b_i, facts_i in b_facts_i.items(): 
                to_aggr = []
                for body_i, m_i in facts_i: 
                    """
                    IMPORTANT : embedding entries must be in [-1, +1]
                    thus no matter how it changes over time 
                    (no matter whether it even has a time-varying entry)
                    set entries to -1 and +1 according to parameters 
                    can get its upper bound
                    """
                    output = self.transform_param_intensity[m_i].get_upperbound() # 1-dim
                    to_aggr.append(output)
                aggred = self.aggregation[b_i](torch.stack(to_aggr, dim=0))
                intensity += aggred
        if hascell: # if it is an active LSTM cell 
            intensity += active['intensity_cells'][event_type].get_upperbound()
        self.cache_inten_bound[event_type] = \
            self.activation['softplus'](intensity) # size : 1

    #@profile
    def compute_embeddings(self, terms, cdb, active): 
        # cdb for current database
        for i, t in enumerate(terms): 
            if t not in self.cache_emb: 
                self.compute_embedding_(t, cdb, active)
        embs = [self.cache_emb[t] for t in terms]
        # can't stack embs cuz they may have different dimensions
        return embs

    #@profile
    def compute_embedding_(self, term, cdb, active): 
        provable = term in cdb.terms
        hascell = term in cdb.cells
        assert provable or hascell, \
            f"why compute embedding of {term} if not provable and no cell?"
        embedding = 0.0
        if provable: 
            t = cdb.terms[term]
            for r_i, b_facts_i in t.transform_edges.items(): 
                # maybe chech r_i b_i 1-to-1 correspondence?
                for b_i, facts_i in b_facts_i.items(): 
                    to_aggr = []
                    for body_i, m_i in facts_i: 
                        to_cat = []
                        for subgoal in body_i: 
                            if subgoal not in self.cache_emb: 
                                self.compute_embedding_(subgoal, cdb, active)
                            to_cat.append(self.cache_emb[subgoal])
                        output = self.transform_param[m_i](torch.cat(to_cat, dim=-1))
                        to_aggr.append(output)
                    aggred = self.aggregation[b_i](torch.stack(to_aggr, dim=0))
                    # after aggregation, sum them up
                    # r_i and b_i is 1-to-1
                    embedding += aggred
        if hascell: # if it is an active LSTM cell 
            embedding += active['cells'][term].decay(self.cache_times)
        self.cache_emb[term] = self.activation['tanh'](embedding)

    #@profile
    def update_cells_sync(self, event, cdb, active):
        event_type = event['name']
        assert event_type is not 'eos', "EOS doesn't update cells"
        event_type = self.datalog.aug_term(event_type)
        if event_type in cdb.events: 
            """
            events keys are those will create, be seen by and/or kill something
            so it is possible that something in the sequence 
            not exist in keys cuz they do nothing but jut existing there...
            """
            new_cell, new_cell_intensity = dict(), dict()
            for c in cdb.events[event_type].who_see_it: 
                # c is head of <- rule with event_type as condition0
                new_cell[c] = self.compute_new_cell(
                    c, event_type, cdb, active, event['time'], False)
                if cdb.cells[c].is_event: 
                    new_cell_intensity[c] = self.compute_new_cell(
                        c, event_type, cdb, active, event['time'], True)
            for c in new_cell: 
                active['cells'][c].update(event['time'], new_cell[c])
                if cdb.cells[c].is_event: 
                    active['intensity_cells'][c].update(
                        event['time'], new_cell_intensity[c])


    #@profile
    def update_cells_async(self, event, cdb, active):
        event_type = event['name']
        assert event_type is not 'eos', "EOS doesn't update cells"
        event_type = self.datalog.aug_term(event_type)
        if event_type in cdb.events: 
            """
            events keys are those will create, be seen by and/or kill something
            so it is possible that something in the sequence 
            not exist in keys cuz they do nothing but jut existing there...
            """
            """
            this version updates in async way : update as it is computed
            """
            for c in cdb.events[event_type].who_see_it: 
                # t is the head of the <- rule with event_type as subgoal0
                new_cell = self.compute_new_cell(
                    c, event_type, cdb, active, event['time'], False)
                active['cells'][c].update(event['time'], new_cell)
                if cdb.cells[c].is_event: 
                    # if it is an event, it has an extra cell to be updated
                    new_cell_intensity = self.compute_new_cell(
                        c, event_type, cdb, active, event['time'], True)
                    active['intensity_cells'][c].update(
                        event['time'], new_cell_intensity)

    def compute_new_cell_full(self, c, e, cdb, active, time, for_intensity_dim=False): 
        """
        compute new cell values by updating with LSTM
        use intensity related params if this cell is also an event 
        c : cell that is affected by event e
        """
        #print(f"use LSTM FULL")

        e = self.datalog.aug_term(e)
        if e not in self.cache_emb: 
            self.compute_embedding_(e, cdb, active)
        
        if for_intensity_dim: 
            see_param = self.see_param_intensity
            old_cell = active['intensity_cells'][c]
            gate = active['intensity_gates'][c]
        else: 
            see_param = self.see_param
            old_cell = active['cells'][c]
            gate = active['gates'][c]
        
        c_old = old_cell.decay(float(time))
        cb_old = old_cell.retrospect()['target']
        
        """
        NEW POOLING 
        for rule r: 
            for instantiation m: 
                make projection and compute c_rm, cb_rm, d_rm
                compute dc_rm, dcb_rm
            compute weight w_rm
        pool to get dc and dcb
        get new c and new cb
        pool d 
        """

        dcell, dcell_target = 0.0, 0.0 
        to_aggr_fac, to_aggr_w, to_aggr_d = [], [], []
        for r_i, b_facts_i in cdb.cells[c].see_edges[e].items(): 
            for b_i, facts_i in b_facts_i.items(): 
                to_aggr_c, to_aggr_cb  = [], []
                for body_i, m_i in facts_i: 
                    to_cat = [self.cache_emb[e]]
                    for subgoal in body_i: 
                        if subgoal not in self.cache_emb: 
                            self.compute_embedding_(subgoal, cdb, active)
                        to_cat.append(self.cache_emb[subgoal])
                    cated = torch.cat(to_cat, dim=-1)
                    cated = self.filter_emb(cated)
                    output = see_param[m_i](cated)
                    output = gate(output, c_old, cb_old)
                    c_rm, cb_rm, d_rm = output['start'], output['target'], output['decay']
                    to_aggr_c.append(c_rm)
                    to_aggr_cb.append(cb_rm)
                    to_aggr_d.append(d_rm)
                
                to_aggr_c = torch.stack(to_aggr_c, dim=0)
                to_aggr_cb = torch.stack(to_aggr_cb, dim=0)
                dcell += self.aggregation[b_i](to_aggr_c - c_old)
                dcell_target += self.aggregation[b_i](to_aggr_cb - cb_old)
                """
                compute w_m
                """
                fac, w = self.decay_weight[b_i](
                    c_old - cb_old, to_aggr_c - c_old, to_aggr_cb - cb_old
                )
                to_aggr_fac.append(fac)
                to_aggr_w.append(w)
        
        to_aggr_fac = torch.cat(to_aggr_fac, dim=0)
        to_aggr_w = torch.cat(to_aggr_w, dim=0)
        to_aggr_d = torch.stack(to_aggr_d, dim=0)
        decay_gate = self.aggregation['#_aggregate_decay_#'](
            to_aggr_fac, to_aggr_w, to_aggr_d )
        
        return {
            'start': c_old + dcell, 'target': cb_old + dcell_target, 
            'decay': decay_gate
        }

    def compute_new_cell_simple(self, c, e, cdb, active, time, for_intensity_dim=False): 
        """
        compute new cell values by updating with LSTM 
        use intensity related params if this cell is also an event 
        c : cell that is affected by event e 
        NOTE : much simplified from the version in paper 
        not as expressive in theory
        fast and can give almost the same results in practice
        """
        #print(f"use LSTM SIMPLE")

        e = self.datalog.aug_term(e)
        if e not in self.cache_emb: 
            self.compute_embedding_(e, cdb, active)
        
        if for_intensity_dim: 
            see_param = self.see_param_intensity
            old_cell = active['intensity_cells'][c].retrospect()
            gate = active['intensity_gates'][c]
        else: 
            see_param = self.see_param
            old_cell = active['cells'][c].retrospect()
            gate = active['gates'][c]

        """
        SIMPLE POOLING 
        for rule r: 
            for instantiation m: 
                make projection 
            aggrate projections
        make proj to gate
        """
        
        embedding = 0.0
        for r_i, b_facts_i in cdb.cells[c].see_edges[e].items():
            for b_i, facts_i in b_facts_i.items(): 
                to_aggr = []
                for body_i, m_i in facts_i: 
                    to_cat = [self.cache_emb[e]]
                    for subgoal in body_i: 
                        if subgoal not in self.cache_emb: 
                            self.compute_embedding_(subgoal, cdb, active)
                        to_cat.append(self.cache_emb[subgoal]) 
                    cated = torch.cat(to_cat, dim=-1)
                    cated = self.filter_emb(cated)
                    output = self.see_param[m_i](cated)
                    to_aggr.append(output)
                aggred = self.aggregation[b_i](torch.stack(to_aggr, dim=0))
                embedding += aggred
        
        return gate(embedding, old_cell['start'], old_cell['target'])

    #@profile
    def filter_emb(self, x): 
        """
        embedding size : # times * 6D 
        """
        d = x.dim()
        if d == 1: 
            return x 
        elif d == 2: 
            # in our setting, most recent time is the actual event time
            return x[-1, :]
        else: 
            raise Exception(f"what embedding is? {x}")