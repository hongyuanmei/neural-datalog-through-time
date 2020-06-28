import torch
import torch.nn as nn
import torch.nn.functional as F

from ndtt.logic.cell import Cell 
from ndtt.logic.term import Term
from ndtt.logic.event import Event


class DatabaseFast(object): 

    #@profile
    def __init__(self, datalog): 
        """
        get useful attributes of current datalog database inluding: 
        provable terms, valid event types, active cell blocks, etc...
        """
        """
        query datalog once for all the facts to avoid overhead
        """
        #print(f"finding all provable terms...")
        all_terms = datalog.find_all_provable_terms()
        #print(f"finding all events...")
        all_events = datalog.find_all_events()
        #print(f"finding all cells...")
        all_cells = datalog.find_all_cells()
        #print(f"finding all transform facts...")
        all_transform_facts = datalog.find_all_transform_facts()
        #print(f"finding all see facts...")
        all_see_facts = datalog.find_all_see_facts()
        #print(f"finding all killcell facts...")
        all_killcell_facts = datalog.find_all_killcell_facts()
        #print(f"finding all functor dims...")
        all_functor_dims = datalog.find_all_functor_dims()

        """
        functor dims must be ready before all others 
        because all others need to know dims of terms
        """
        self.functor_dims = self.create_functor_dims(all_functor_dims)

        """
        save the names 
        """
        self.term_names = all_terms
        self.term_names_set = set(all_terms) # easy to check existence
        self.event_names = all_events
        self.event_names_set = set(all_events) # easy to check existence
        self.cell_names = all_cells
        self.cell_names_set = set(all_cells) # easy to check existence
        """
        loop over facts once and save 
        1. transform/see edges and who sees/be_killed_by each event
        2. collect valid transform/see params idx and dim of current database
        """
        # for 1. 
        transform_edges = dict()
        who_see_them = dict()
        see_edges = dict()
        who_they_kill = dict()
        # for 2. 
        transform_idx_and_dim = set()
        see_idx_and_dim = set()
        # collect!
        for f in all_transform_facts: 
            self.make_transform_edge(f, transform_edges)
            self.make_transform_idx_and_dim(
                f, transform_idx_and_dim, datalog)
        for f in all_see_facts: 
            self.make_who_see_it(f, who_see_them)
            self.make_see_edge(f, see_edges)
            self.make_see_idx_and_dim(
                f, see_idx_and_dim, datalog)
        for f in all_killcell_facts: 
            self.make_who_it_kill(f, who_they_kill)
        """
        create terms, events, and cells
        """
        self.terms = self.create_terms(all_terms, transform_edges, datalog)
        self.events = self.create_events(all_events, who_see_them, who_they_kill, datalog)
        self.cells = self.create_cells(all_cells, see_edges, datalog)
        """
        store transform/see param idx and dim for current database
        """
        self.transform_idx_and_dim = transform_idx_and_dim
        self.see_idx_and_dim = see_idx_and_dim
    
    #@profile
    def make_transform_edge(self, fact, edges): 
        """
        each fact has form : term, subgoals, rule_i, betaparam_i, mparam_i
        for a given term, each rule_i corresponds to only one betaparam_i
        """
        term, r_i, b_i, m_i = fact[0], fact[-3], fact[-2], fact[-1]
        if term not in edges: 
            edges[term] = dict()
        if r_i not in edges[term]: 
            edges[term][r_i] = {b_i : [] }
        edges[term][r_i][b_i].append( (fact[1:-3], m_i) )

    #@profile
    def make_transform_idx_and_dim(self, fact, transform_idx_and_dim, datalog): 
        """
        each fact has form : term, subgoals, rule_i, betaparam_i, mparam_i
        for a given term, each rule_i corresponds to only one betaparam_i
        """
        temp = [
            fact[-2], fact[-1] # betaparam and mparam
        ]
        # get dim and zero
        in_dim, in_zero = self.aug_dim( fact[1:-3], datalog )
        out_dim, out_zero = self.aug_dim( fact[:1], datalog )
        # store all info
        temp += [
            in_dim, in_zero, 
            out_dim, out_zero, 
            datalog.get_functor(fact[0]) in datalog.auxilary['event_functors'] 
            # event or not (i.e. intensity exist)
        ]
        transform_idx_and_dim.add(tuple(temp))

    #@profile
    def create_terms(self, all_terms, transform_edges, datalog): 
        rst = dict()
        for i, t in enumerate(all_terms): 
            t = datalog.aug_term(t)
            d, z = self.aug_dim((t,), datalog)
            is_event = t in self.event_names_set
            term_transform_edges = transform_edges[t]
            for r, v in term_transform_edges.items(): 
                assert len(v) == 1, "each rule has only one aggregator"
            rst[t] = Term(t, d, z, term_transform_edges, is_event)
        return rst

    #@profile
    def make_who_see_it(self, fact, who): 
        """
        each fact has form : term, event, subgoals, rule_i, betaparam_i, mparam_i
        for a given term, each rule_i corresponds to only one betaparam_i
        """
        term, event = fact[0], fact[1]
        if event not in who: 
            who[event] = set()
        who[event].add(term)
    
    #@profile
    def make_who_it_kill(self, fact, who_they_kill): 
        """
        each fact has form : term, event, subgoals, etc...
        """
        term, event = fact[0], fact[1]
        if event not in who_they_kill: 
            who_they_kill[event] = set()
        who_they_kill[event].add(term)

    #@profile
    def create_events(self, all_events, who_see_them, who_they_kill, datalog): 
        rst = dict()
        for i, e in enumerate(all_events): 
            e = datalog.aug_term(e)
            d, z = self.aug_dim((e,), datalog)
            if e in who_see_them: 
                # not every event is seen 
                # e.g., we may have no cell block 
                # so nothing gets updated, everything is static
                who_see_it = sorted(list(who_see_them[e]))
            else: 
                who_see_it = list()
            if e in who_they_kill: # not every event kills
                who_it_kill = sorted(list(who_they_kill[e]))
            else: 
                who_it_kill = list()
            rst[e] = Event(e, d, z, who_see_it, who_it_kill)
        """
        all_events are valid event types that have intensities
        rst may have more than that: e.g. 
        1. bos s.t. we know who it updates
        """
        aug_bos = datalog.aug_term('bos')
        if aug_bos in who_see_them: 
            who_see_it = sorted(list(who_see_them[aug_bos]))
        else: 
            who_see_it = list()
        if aug_bos in who_they_kill: 
            who_it_kill = sorted(list(who_they_kill[aug_bos]))
        else: 
            who_it_kill = list()
        rst[aug_bos] = Event(aug_bos, 1, [0], who_see_it, who_it_kill)
        """
        2. eos s.t. when we INDEX the database with eos, we won't get KeyError
        """
        aug_eos = datalog.aug_term('eos')
        rst[aug_eos] = Event(aug_eos, 1, [0], [], [])
        """
        3. anything that is seen by (thus create) or kills something
        e.g. in tv domain, release(p1) will create prog(p1) cell 
        but release(p1) doesn't have an intensity cuz we don't predict it
        """
        for e in who_see_them: 
            if e not in rst: 
                who_it_kill = who_they_kill[e] if e in who_they_kill else []
                rst[e] = Event(e, 1, [0], who_see_them[e], who_it_kill)
        for e in who_they_kill: 
            if e not in rst: 
                who_see_it = who_see_them[e] if e in who_see_them else []
                rst[e] = Event(e, 1, [0], who_see_it, who_they_kill[e])
        return rst

    #@profile
    def make_see_edge(self, fact, edges): 
        """
        each fact has form : term, event, subgoals, rule_i, betaparam_i, mparam_i
        for a given term, each rule_i corresponds to only one betaparam_i
        """
        term, r_i, b_i, m_i = fact[0], fact[-3], fact[-2], fact[-1]
        e = fact[1]
        if term not in edges: 
            edges[term] = dict()
        if e not in edges[term]: 
            edges[term][e] = dict()
        if r_i not in edges[term][e]: 
            edges[term][e][r_i] = {b_i : [] }
        edges[term][e][r_i][b_i].append( (fact[2:-3], m_i) )

    #@profile
    def make_see_idx_and_dim(self, fact, see_idx_and_dim, datalog): 
        """
        each fact has form : term, event, subgoals, rule_i, betaparam_i, mparam_i
        for a given term, each rule_i corresponds to only one betaparam_i
        """
        temp = [
            fact[-2], fact[-1] # betaparam and mparam
        ]
        # get dim and zero
        in_dim, in_zero = self.aug_dim( fact[1:-3], datalog )
        out_dim, out_zero = self.aug_dim( fact[:1], datalog )
        # store all info 
        temp += [
            in_dim, in_zero, 
            out_dim, out_zero, 
            datalog.get_functor(fact[0]) in datalog.auxilary['event_functors'] 
            # event or not (i.e. intensity exist)
        ]
        see_idx_and_dim.add(tuple(temp))

    #@profile
    def create_cells(self, all_cells, see_edges, datalog): 
        rst = dict()
        for i, c in enumerate(all_cells): 
            c = datalog.aug_term(c)
            d, z = self.aug_dim((c,), datalog)
            is_event = c in self.event_names_set
            """
            it is possible that c is a valid cell 
            but there is no see fact about it 
            because all events that can be seen by it 
            may turn out invalid under current database 
            e.g. turnover(purpl10, pink8) will 
            kill ballat(purple10) and create ballat(pink8)
            although ballat(pink8) should be updated by turnover(purple10, pink8)
            that already happened after creation and before killing 
            therefore, after killing, turnover(purple10, pink8) not valid anymore 
            (cuz ball not at purple10 anymore)
            so we get no facts about turnover(purple10, pink8)
            including see facts about ballat(pink8) 
            however, this doesn't affect correctness 
            cuz ballat(pink8) is already updated after creation (but before killing)
            no need to update it again
            and should NOT be updated again after killing 
            db_after_kill should be used by next event, not event before it...
            """
            if c in see_edges: 
                cell_see_edges = see_edges[c]
            else: 
                cell_see_edges = dict()
            for e, v in cell_see_edges.items(): 
                for r, vv in v.items(): 
                    assert len(vv) == 1, "each rule has only one aggregator"
            rst[c] = Cell(c, d, z, cell_see_edges, is_event)
        return rst

    #@profile
    def create_functor_dims(self, all_functor_dims): 
        rst = dict()
        for functor, dim in all_functor_dims: 
            assert functor not in rst, "only one dim value for each functor"
            rst[functor] = int(dim)
        return rst

    #@profile 
    def aug_dim(self, terms, datalog): 
        """
        try use this but not the aug_dim in datalogplus
        to reduce count of queries
        """
        d, z = 0, []
        for s in terms:
            s_func = datalog.get_functor(s)
            s_dim = self.get_functor_dim(s_func) 
            if s_dim > 0: 
                z += []
                d += s_dim
            else: 
                z += [d]
                d += 1
        return d, tuple(z)

    #@profile
    def get_functor_dim(self, functor): 
        if functor in self.functor_dims: 
            return self.functor_dims[functor]
        else: 
            return 0

    def __repr__(self): 
        rst = f'Database : \n'
        rst += f'Provable Terms : \n'
        rst += f'Names : {self.term_names}\n'
        for k, v in self.terms.items(): 
            rst += f'{str(v)}\n'
        rst += f'Valid Events : \n'
        rst += f'Names : {self.event_names}\n'
        for k, v in self.events.items(): 
            rst += f'{str(v)}\n'
        rst += f'Active Cells : \n'
        rst += f'Names : {self.cell_names}\n'
        for k, v in self.cells.items(): 
            rst += f'{str(v)}\n'
        rst += f'Functor Dims : \n'
        for k, v in self.functor_dims.items(): 
            rst += f'{k} : {v}\n'
        return rst
