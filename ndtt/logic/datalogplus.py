"""
Datalog Plus adds features to standard Datalog
(1) query proof DAG
(2) define events
(3) assign dimensions to functors
(4) istrue(head) if hasproofdag(head) OR hascell(head)
# head is proved if it is a goal of a proof DAG :-
# hascell(head) if it was created by a <- rule
(5) see : head <- event, subgoals 
# update head if event happens and subgoals are true
# create head cell if not yet
(6) killcell : !head <- event, subgoals
# retract hascell(head). 
# neuraldatalog will destroy the associated cell blocks
"""

from ndtt.logic.datalog import Datalog


class DatalogPlus(object):

    def __init__(self): 
        print("init datalog plus database...")
        self.datalog = Datalog()
        """
        auxilary information for querying the database 
        """
        self.auxilary = {
            'keywords': {
                'embed': set(), 'event': set(), 
                'transform': set(), 
                'see': set(), 'killcell': set()
            },
            'functors': set(), 
            # all outermost functors
            # maybe useful
            'auto_logic_rules': [
                'istrue(bos(specialaug)).', # make bos true
                #'event(bos(specialaug)).', # make bos valid event 
                'istrue(X) :- hasproofdag(X).', # X is true if X hasproofdag
                'istrue(X) :- hascell(X).', # X is true if X hascell
                'hasproofdag(registerpredicte).', # to register predicate hasproofdag
                'hascell(registerpredicte).' # to register predicate hascell
            ], 
            'exclude': set( ['registerpredicte'] ), 
            # exclude these when querying for all things of a category
            # because they are only used to register predicates
            'event_functors': set(), 
            # terms with these outermost functors can be events 
            # so they have intensity, so extra params for them
            # collected while writing event rules...
            'transform_idx_and_functor': set(), 
            'see_idx_and_functor': set(), 
            # asserted facts 
            'asserted_facts' : set()
        }

    def load_rules(self, rules): 
        """
        rules : list of rules 
        """
        """
        temporally save all terms that may be head or condit1
        because they may be events 
        this is useful when we convert rule in new format (paper) to rule in old format (code)
        e.g., when we convert old-format event declaration to new-format event declaration
        we need to loop over all possible events to find the vars of event functor
        """
        self.temp_all_head_condit1 = self.extrac_all_head_condit1(rules)
        for i_r, rule in enumerate(rules): 
            self.load_one_rule(i_r, rule)
        del self.temp_all_head_condit1
        """
        load auto logic rules
        """
        for rule in self.auxilary['auto_logic_rules']: 
            self.datalog(rule)

    def extrac_all_head_condit1(self, rules): 
        """
        extract head and condit1 terms in all rules 
        e.g., head :- condit1, ..., conditN. 
        => [head, condit1]
        """
        ans = []
        for r in rules: 
            if ':-' in r: 
                if ':-' == r[:2]: 
                    # declaration 
                    pass 
                else: 
                    # :- rule 
                    head_bname, body_mname = r.split(':-')
                    if ':' in head_bname: 
                        head = head_bname.split(':')[0]
                    else:
                        head = head_bname
                    if '::' in body_mname: 
                        body = body_mname.split('::')[0]
                    else: 
                        body = body_mname[:-1] # no . 
                    condit1 = self.extract_nth_term(body, 0)
                    ans += [head.replace(' ', ''), condit1]
            elif '<-' in r: 
                # <- rule 
                head_bname, body_mname = r.split('<-')
                if '!' in head_bname:
                    # only keep term, so no ! symbol
                    head_bname = head_bname.replace('!', '')
                if ':' in head_bname: 
                    head = head_bname.split(':')[0]
                else:
                    head = head_bname
                if '::' in body_mname: 
                    body = body_mname.split('::')[0]
                else: 
                    body = body_mname[:-1] # no . 
                condit1 = self.extract_nth_term(body, 0)
                ans += [head.replace(' ', ''), condit1]
            else: 
                assert ':-' not in r and '<-' not in r, f"not a fact?! : {r}"
                ans += [r[:-1].replace(' ', '')]
        return ans 

    def load_one_rule(self, i_r, rule): 
        """
        load one neural datalog rule to database 
        Note: one neural datalog rule may correspond to 
        more than one datalog rule because of logic rules
        e.g. head :- subgoals corresponds to 
        transform(head, subgoals) :- istrue(subgoals). # how neural emb transformed
        hasproofdag(head) :- istrue(subgoals) # regular datalog logic proof rule
        e.g. head <- event, subgoals corresponds to 
        see(head, event, subgoals) :- istrue(subgoals). 
        it doesn't add any logic proof rule
        but when head cell is created, a hascell(head). fact will be asserted. 
        e.g. !head <- event, subgoals corresponds to 
        killcell(head, event, subgoals) :- istrue(subgoals). 
        it doesn't add any logic proof rule
        but when event happens, the hascell(head). fact will be retracted. 
        """
        """
        each rule may have :: followd by rule_idx, betaparam_idx, mparam_idx
        e.g. head :- subgoals :: rule_idx, betaparam_idx, mparam_idx. 
        """
        rule, rule_idx, betaparam_idx, mparam_idx = self.extract_idx(i_r, rule)
        #print(f"\nraw rule is {rule}, with rule idx {rule_idx} and param idx {betaparam_idx} and {mparam_idx}")
        to_be_loaded = self.rewrite_rule(rule, rule_idx, betaparam_idx, mparam_idx)
        #print(f"{len(to_be_loaded)} rules to load for this raw rule")
        for r in to_be_loaded: 
            #print(f"loading\n{r}")
            self.datalog(r)
            #print(f"done")

    def aug_terms_declaration(self, r): 
        """
        aug terms to expression inside
        :- event(exp).
        NOT aug for embed declaration 
        :- embed(exp).
        cuz its 0-th arg must be a functor 
        (i.e. must not have arg)
        """
        i_left = r.index('(')
        i_right = r.rindex(')') # reverse order 
        auged = self.aug_terms_exp(r[i_left+1 : i_right])
        new_rule = f'{r[:i_left]}({auged}).'
        return new_rule

    def collect_functors_declaration(self, r): 
        """
        collect functors in declaration
        :- event(exp).
        NO need to do for embed declaration 
        :- embed(exp).
        cuz functors declared by embed must be also used somewhere else
        """
        i_left = r.index('(')
        i_right = r.rindex(')') # reverse order 
        self.collect_functors_exp(r[i_left+1 : i_right])

    def aug_terms_rule(self, r): 
        r = r.replace(' ', '')
        if ':-' in r: 
            head, body, _ = self.separate_rule(r, ':-')
            delim = ':- '
        elif '<-' in r: 
            head, body, _ = self.separate_rule(r, '<-')
            delim = '<- '
        else: 
            raise Exception(f"what a rule?! : {r}")
        head = self.aug_terms_exp(head)
        body = self.aug_terms_exp(body)
        return f'{head}{delim}{body}.'

    def collect_functors_rule(self, r): 
        r = r.replace(' ', '')
        if ':-' in r: 
            head, body, _ = self.separate_rule(r, ':-')
        elif '<-' in r: 
            head, body, _ = self.separate_rule(r, '<-')
        else: 
            raise Exception(f"what a rule?! : {r}")
        self.collect_functors_exp(head)
        self.collect_functors_exp(body)

    def aug_terms_fact(self, r): 
        i_dot = r.rindex('.') # reverse order
        auged = self.aug_terms_exp(r[:i_dot])
        return f'{auged}.'

    def collect_functors_fact(self, r): 
        i_dot = r.rindex('.') # reverse order
        self.collect_functors_exp(r[:i_dot])

    def aug_terms_exp(self, exp): 
        """
        aug terms in each given expression
        pyDatalog can't handle rules with bodies that are not expressions
        e.g. e(0,0) :- globe. raises 
        pyDatalog.util.DatalogError: Invalid body for clause
        we aug such terms with (specialaug)
        s.t. globe(specialaug)
        if we want to print any rule out, we remove (specialaug)
        so users are not even aware of this
        """
        rst = [self.aug_term(t) for t in self.extract_terms(exp)]
        return ', '.join(rst)

    def collect_functors_exp(self, exp): 
        """
        collect outermost functors of each term 
        """
        for t in self.extract_terms(exp): 
            t = t.replace('!', '') # exclude any possible ! from !head 
            self.auxilary['functors'].add(self.get_functor(t))

    def aug_term(self, t): 
        # aug term t if it doesn't have ()
        if '(' not in t and ')' not in t: 
            t += '(specialaug)'
        return t

    def rewrite_rule(self, rule, rule_idx, betaparam_idx, mparam_idx): 
        """
        rewrite rules such that we can easily query proof DAG 
        """
        """
        augmentation must happen before rewriting!!!
        otherwise, globe inside keyword (e.g. transform) won't be augmented
        e.g. email(S,R) :- person(S), person(R), globe.
        will have a translated rule: 
        transform(email(S,R), person(S), person(R), globe) :- ... 
        the globe won't be augmented because it is not (but inside) a term!!!
        """
        rule_idx = self.aug_terms_exp(rule_idx) # aug rule idx e.g. rule(i) 
        betaparam_idx = self.aug_terms_exp(betaparam_idx) 
        mparam_idx = self.aug_terms_exp(mparam_idx)
        # aug param idx i.e. betaparam(i), mparam(i)
        # it should do nothing for rule idx and param idx, but we keep it for safe
        declare = ':-' == rule[:2]
        if declare: 
            # is a declaration  
            declare_event = 'event' in rule
            # declare intensity of an event 
            declare_dim = 'embed' in rule
            # defines dimension of a functor 
            if declare_event: 
                """
                two ways to declare event type 
                old => :- event(watch(U, P)). => it has vars
                new => :- event(watch, 8). or :- event(watch). => it has no vars
                """
                has_vars = False 
                has_digit = False 
                for c in rule: 
                    if c.isupper(): 
                        has_vars = True
                    if c.isdigit(): 
                        has_digit = True
                
                ans = []
                
                if has_vars: 
                    """
                    old way => :- event(watch(U, P)). 
                    if watch needs an embedding
                    then there must exist e.g. :- embed(watch, 8) in rules 
                    """
                    rule = self.aug_terms_declaration(rule)
                    self.collect_functors_declaration(rule)
                    ans += [ 
                        self.rewrite_event_rule_logic(rule),
                        self.rewrite_event_rule(
                            rule, rule_idx, betaparam_idx, mparam_idx) 
                    ]
                else: 
                    """
                    new way => :- event(watch, 8). or :- event(watch).
                    it does 2 things: 
                    (1) it declares :- embed(watch, D). except if no D specified
                    (2) it declares :- event(watch(U, P)). 
                    so we need to create :- event(watch(U, P)). in old format to use
                    """
                    if has_digit: 
                        # add embed declaration
                        embed_from_event = rule.replace('event', 'embed')
                        ans += [ self.rewrite_dim_rule(embed_from_event) ]
                    # now we make :- event(watch(U, P)). 
                    rules_old_way = self.convert_rule_format_for_event(rule)
                    for r_old in rules_old_way: 
                        r_old = self.aug_terms_declaration(r_old)
                        self.collect_functors_declaration(r_old)
                        ans += [
                            self.rewrite_event_rule_logic(r_old),
                            self.rewrite_event_rule(
                                r_old, rule_idx, betaparam_idx, mparam_idx)
                        ]
                
                return ans

            elif declare_dim:
                # don't augment to functors inside embed declaration
                # no need to collect functors in :- embed rule either
                # cuz they will be collected from somewhere else anyway
                return [ self.rewrite_dim_rule(rule) ]
            else: 
                raise Exception(f"what it declares? {rule}")
        else: 
            define_dynamic = '<-' in rule 
            # defines which term sees which terms as the given event happens
            define_static = ':-' in rule 
            # defines proof DAG 
            is_fact = not ( define_dynamic or define_static )
            define_leaf = is_fact
            # defines a leaf node in the proof DAG
            if define_dynamic: 
                rule = self.aug_terms_rule(rule)
                self.collect_functors_rule(rule)
                return [ self.rewrite_dynamic_rule(rule, rule_idx, betaparam_idx, mparam_idx) ]
            elif define_static: 
                rule = self.aug_terms_rule(rule)
                self.collect_functors_rule(rule)
                """
                write logic rule for logic inference
                e.g. email(S,R) :- person(S), person(R).
                addition to rewritten transform rule, 
                we need to keep a logic rule: 
                e.g. hasproofdag(email(S,R)) :- istrue(person(S)), istrue(person(R)). 
                """
                return [
                    self.rewrite_static_rule_logic(rule),
                    self.rewrite_static_rule(rule, rule_idx, betaparam_idx, mparam_idx) ]
            elif define_leaf: 
                rule = self.aug_terms_fact(rule)
                self.collect_functors_fact(rule)
                # write logic rule as well
                return [
                    self.rewrite_leaf_rule_logic(rule),
                    self.rewrite_leaf_rule(rule, rule_idx, betaparam_idx, mparam_idx) ]
            else: 
                raise Exception(f"what does this rule do? {rule}")

    def convert_rule_format_for_event(self, event_declare): 
        """
        convert event declaration from new way in paper to old way in implementation
        e.g., 
        new => :- event(watch, 8) or :- event(watch). 
        old => :- event(watch(X0, X1)). 
        """
        i_left = event_declare.index('(')
        suffix = event_declare[i_left + 1: ]
        i = 0 
        while suffix[i] != ',' and suffix[i] != ')': 
            i += 1
        event_functor = suffix[:i]
        counts = set()
        for term in self.temp_all_head_condit1: 
            if event_functor == self.get_functor(term): 
                # count # of vars in this term with this functor 
                count = term.count(',') + 1
                counts.add(count)
        ans = []
        for count in counts: 
            variables = []
            for c in range(count): 
                variables += [f'X{c}']
            variables = ','.join(variables)
            ans += [f':- event({event_functor}({variables})).']
        return ans

    def extract_idx(self, i_r, rule): 
        """
        extract rule, betaparam, mparam idx
        for params, cases include: 
        betaparams(something), mparams(something)
        allparams(something)
        """
        rule_i = f'rule({i_r})'
        
        if 'allparams' in rule or 'mparams' in rule or 'betaparams' in rule: 
            # written in old format
            pass 
        else: 
            # otherwise, maybe in new format
            # convert it to old format
            rule = self.convert_rule_format_for_idx(rule)

        if '::' in rule: 
            # explicit parameter index
            rule, params = rule.split('::')
            rule += '.'
            # split params
            params = params.replace('.', '') # extract expression
            params = params.replace(' ', '') # squeeze spaces
            if 'allparams' in params: 
                betaparam_i = params.replace('allparams', 'betaparams')
                mparam_i = params.replace('allparams', 'mparams')
            elif ('betaparams' in params) and ('mparams' in params): 
                betaparam_i, mparam_i = self.extract_terms(params)
            elif ('betaparams' in params) and ('mparams' not in params): 
                betaparam_i = params 
                mparam_i = f'mparams({rule_i})'
            elif ('betaparams' not in params) and ('mparams' in params): 
                betaparam_i = f'betaparams({rule_i})'
                mparam_i = params
            else: 
                raise Exception(f"Unknown params : {params}")
        else: 
            betaparam_i = f'betaparams({rule_i})'
            mparam_i = f'mparams({rule_i})'
        
        return rule, rule_i, betaparam_i, mparam_i

    def convert_rule_format_for_idx(self, rule): 
        """
        convert rule from new format in paper to old format in implementation 
        case-(1) 
        head : beta :-/<- condit1, ..., conditN :: full_matrix.
        head :-/<- condit1, ..., conditN :: mparams(full_matrix), betaparams(beta). 
        case-(2)
        head :-/<- condit1, ..., conditN :: full_matrix 
        head :-/<- condit1, ..., conditN :: mparams(full_matrix). 
        case-(3)
        head : beta :-/<- condit1, ..., conditN. 
        head :-/<- condit1, ..., conditN :: betaparams(beta). 
        case-(4)
        head :-/<- condit1, ..., conditN. 
        """
        if ':-' not in rule and '<-' not in rule: 
            # no separator, must be fact
            # no params shared 
            return rule

        if ':-' in rule: 
            sep = ':-'
        elif '<-' in rule: 
            sep = '<-'
        else: 
            raise Exception(f"No separater?! rule : {rule}")
        
        head_bname, body_mname, _ = self.separate_rule(rule, sep)
        # separate_rule doesn't remove spaces
        if ':' in head_bname and '::' in body_mname: 
            head, bname = head_bname.split(':')
            bname = bname.replace(' ', '')
            body, mname = body_mname.split('::')
            mname = mname.replace(' ', '')
            ans = f"{head}{sep}{body}:: betaparams({bname}), mparams({mname})."
        elif ':' not in head_bname and '::' in body_mname: 
            body, mname = body_mname.split('::')
            mname = mname.replace(' ', '')
            ans = f"{head_bname}{sep}{body}:: mparams({mname})."
        elif ':' in head_bname and '::' not in body_mname: 
            head, bname = head_bname.split(':')
            bname = bname.replace(' ', '')
            ans = f"{head}{sep}{body_mname} :: betaparams({bname})."
        else: 
            # case-(4)
            return rule
        
        return ans 

    def rewrite_dynamic_rule(self, rule, rule_idx, betaparam_idx, mparam_idx): 
        """
        rewrite dynamic rule 
        head <- event, subgoals. 
        OR 
        !head <- event, subgoals.
        """
        head, body, num_terms = self.separate_rule(rule, '<-')
        #head_exist = head in body
        to_killcell = '!' in head
        
        if to_killcell: 
            # to destroy cell block of head
            new_rule = self.rewrite_dynamic_rule_killcell(
                head, body, num_terms, rule_idx, betaparam_idx, mparam_idx )
        else: 
            # to define which sees which as which event happens
            new_rule = self.rewrite_dynamic_rule_see(
                head, body, num_terms, rule_idx, betaparam_idx, mparam_idx )

        return new_rule
    
    def rewrite_dynamic_rule_killcell(
        self, head, body, num_terms, rule_idx, betaparam_idx, mparam_idx):
        """
        killcell(head, event, subgoals) :- hascell(head), istrue(event), istrue(subgoals).
        """
        keyword = f'killcell{num_terms + 4}' # head and param idx
        #subgoal0 = self.extract_nth_term(body, 0)
        head = head.replace('!', '')
        new_rule = f'{keyword}({head}, {body}, {rule_idx}, {betaparam_idx}, {mparam_idx})'
        """
        may consider adding another condition: subgoal0 should be an event
        but not sure if it is necessary
        """
        istrue_conds = [f'istrue({t})' for t in self.extract_terms(body) ]
        assert len(istrue_conds) >= 1, f"no terms in body?! : {body}"
        concat = ', '.join(istrue_conds)
        new_rule += f' :- hascell({head}), {concat}.'
        self.auxilary['keywords']['killcell'].add(keyword)
        return new_rule

    def rewrite_dynamic_rule_see(
        self, head, body, num_terms, rule_idx, betaparam_idx, mparam_idx):
        """
        use see keyword to rewrite dynamic rule
        head <- event, subgoals. 
        see(head, event, subgoals, rule_idx, betaparam_idx, mparam_idx) :- istrue(event), istrue(subgoals).
        meaning: 
        cell of head is updated when event happens and all subgoals are true 
        if head doesn't have cell yet, create it 
        when updated, also see (that's why to use this keyword) embeddings of subgoals
        may consider adding another condition: subgoal0 should be an event
        but not sure if it is necessary
        """
        keyword = f'see{num_terms + 4}' # head and param idx
        #subgoal0 = self.extract_nth_term(body, 0)
        new_rule = f'{keyword}({head}, {body}, {rule_idx}, {betaparam_idx}, {mparam_idx})'
        istrue_conds = [f'istrue({t})' for t in self.extract_terms(body) ]
        assert len(istrue_conds) >= 1, f"no terms in body?! : {body}"
        concat = ', '.join(istrue_conds)
        new_rule += f' :- {concat}.'
        self.auxilary['keywords']['see'].add(keyword)
        """
        collect betaparam_idx, mparam_idx and associated functors
        for future use of creating LSTM params
        """
        temp = [
            'see', betaparam_idx, mparam_idx, 
            tuple( [self.get_functor(t) for t in self.extract_terms(body) ] ), 
            tuple( [self.get_functor(t) for t in self.extract_terms(head) ] )
        ]
        self.auxilary['see_idx_and_functor'].add(tuple(temp))
        return new_rule

    def rewrite_static_rule(self, rule, rule_idx, betaparam_idx, mparam_idx): 
        """
        use transform to rewrite static rule 
        head :- subgoals.
        transform(head, subgoals, rule_idx, param_idx) :- istrue(subgoals).
        """
        head, body, num_terms = self.separate_rule(rule, ':-')
        keyword = f'transform{num_terms + 4}'
        new_rule = f'{keyword}({head}, {body}, {rule_idx}, {betaparam_idx}, {mparam_idx})'
        istrue_conds = [f'istrue({t})' for t in self.extract_terms(body) ]
        assert len(istrue_conds) >= 1, f"no terms in body?! : {body}"
        concat = ', '.join(istrue_conds)
        new_rule += f' :- hasproofdag({head}), {concat}.'
        self.auxilary['keywords']['transform'].add(keyword)
        """
        collect betaparam_idx, mparam_idx and associated functors
        for future use of creating transform params
        """
        temp = [
            'transform', betaparam_idx, mparam_idx, 
            tuple( [self.get_functor(t) for t in self.extract_terms(body) ] ), 
            tuple( [self.get_functor(t) for t in self.extract_terms(head) ] )
        ]
        self.auxilary['transform_idx_and_functor'].add(tuple(temp))
        return new_rule

    def rewrite_static_rule_logic(self, rule): 
        """
        use hasproofdag to rewrite static rule 
        head :- subgoals.
        hasproofdag(head) :- istrue(subgoals).
        """
        head, body, num_terms = self.separate_rule(rule, ':-')
        keyword = f'hasproofdag'
        istrue_conds = [f'istrue({t})' for t in self.extract_terms(body) ]
        assert len(istrue_conds) >= 1, f"no terms in body?! : {body}"
        concat = ', '.join(istrue_conds)
        new_rule = f'{keyword}({head}) :- {concat}.'
        return new_rule

    def rewrite_leaf_rule(self, rule, rule_idx, betaparam_idx, mparam_idx): 
        """
        use transform to rewrite leaf rule 
        head.
        transform(head, true(specialaug), rule_idx, betaparam_idx, mparam_idx).
        NO variable allowed in a fact
        """
        head = rule[ : rule.rindex('.')] # reverse is safer
        body = f'true(specialaug)'
        keyword = f'transform{1 + 4}'
        new_rule = f'{keyword}({head}, {body}, {rule_idx}, {betaparam_idx}, {mparam_idx}) :- hasproofdag({head}).'
        self.auxilary['keywords']['transform'].add(keyword)
        """
        collect betaparam_idx, mparam_idx and associated functors
        for future use of creating transform params
        """
        temp = [
            'transform', betaparam_idx, mparam_idx, 
            tuple( [self.get_functor(t) for t in self.extract_terms(body) ] ), 
            tuple( [self.get_functor(t) for t in self.extract_terms(head) ] )
        ]
        self.auxilary['transform_idx_and_functor'].add(tuple(temp))
        return new_rule

    def rewrite_leaf_rule_logic(self, rule): 
        """
        use hasproofdag to rewrite leaf rule 
        head.
        hasproofdag(head).
        NO variable allowed in a fact
        """
        head = rule[ : rule.rindex('.')] # reverse is safer
        new_rule = f'hasproofdag({head}).'
        return new_rule

    def rewrite_event_rule(self, rule, rule_idx, betaparam_idx, mparam_idx): 
        """
        use event{4} keyword 
        :- event(term).
        event{4}(term, rule_idx, betaparam_idx, mparam_idx) :- term.
        """
        i_left = rule.index('(')
        i_right = rule.rindex(')') # reverse order 
        term = rule[i_left+1 : i_right]
        keyword = f'event{0 + 4}'
        new_rule = f'{keyword}({term}, {rule_idx}, {betaparam_idx}, {mparam_idx}) :- istrue({term}).'
        self.auxilary['keywords']['event'].add(keyword)
        """
        collect functors that have intensities
        for future use of creating transform intensity params
        """
        self.auxilary['event_functors'].add( self.get_functor(term) )
        return new_rule

    def rewrite_event_rule_logic(self, rule): 
        """
        use the event keyword to declare logic constraints
        not sure if we really need event logic rule 
        in the end, :- event(something) just declares an intensity
        it doesn't seem to need anything else
        but we keep it for safe for now
        """
        #rule = rule.replace(' ', '')[2:-1]
        i_left = rule.index('(')
        i_right = rule.rindex(')') # reverse order 
        term = rule[i_left+1 : i_right]
        return f"event({term}) :- istrue({term})."

    def rewrite_dim_rule(self, rule): 
        """
        use embed keyword to rewrite rules about dimensionality 
        :- embed(functor, dim).
        embed(functor, dim).
        """
        i_left = rule.index('(')
        i_right = rule.rindex(')') # reverse order 
        content = rule[i_left+1 : i_right]
        keyword = f'embed{0 + 2}'
        new_rule = f'{keyword}({content}).'
        self.auxilary['keywords']['embed'].add(keyword)
        return new_rule

    def separate_rule(self, rule, separator=':-'): 
        i_arrow = rule.index(separator)
        i_period = rule.rindex('.') # reverse is safer
        head = rule[ : i_arrow]
        body = rule[ i_arrow + len(separator) : i_period ]
        num_terms = self.count_terms(body)
        return head, body, num_terms

    def extract_terms(self, exp): 
        # extract all terms of an expression
        rst = []
        beg, dep, l = 0, 0, len(exp)
        for i, c in enumerate(exp): 
            if c == ',' and dep == 0: 
                rst.append(exp[beg : i])
                beg = i + 1 
            if c == '(': 
                dep += 1 
            if c == ')': 
                dep -= 1 
        if beg < l: 
            rst.append(exp[beg : l])
        for t in rst: 
            t.replace(' ', '') # squeeze spaces ' ' in terms
        return rst

    def count_terms(self, exp):
        """
        count # of terms in an expression 
        3 = count_terms('term1, term2, term31(term32(term33))')
        """
        return len(self.extract_terms(exp))

    def extract_nth_term(self, exp, n):
        return self.extract_terms(exp)[n]

    #@profile
    def find_all_transform_facts(self): 
        results = []
        for k in self.auxilary['keywords']['transform']: 
            assert 'transform' in k, "not a transform keyword?"
            num_arg = int(k[len('transform'):])
            args = [f'X{i}' for i in range(num_arg)]
            results += self.query_rule(k, args)
        results = sorted(list(set(results)))
        return results

    def query_transform(self, term): 
        """
        query the proof DAG of term e.g. 
        transform3(term, X0, X1)?
        transform4(term, X0, X1, X2)?
        """
        term = self.aug_term(term)
        results = []
        for k in self.auxilary['keywords']['transform']: 
            assert 'transform' in k, "not a transform keyword?"
            num_arg = int(k[len('transform'):])
            args = [f'X{i}' for i in range(num_arg - 1)]
            args.insert(0, term)
            results += self.query_rule(k, args)
        results = sorted(list(set(results)))
        return results

    #@profile
    def find_all_see_facts(self): 
        results = []
        for k in self.auxilary['keywords']['see']: 
            assert 'see' in k, "not a see keyword?"
            num_arg = int(k[len('see'):])
            args = [f'X{i}' for i in range(num_arg)]
            results += self.query_rule(k, args)
        results = sorted(list(set(results)))
        return results

    def query_see_event(self, event_term): 
        """
        query the see DAG of event_term e.g. 
        see3(X0, event_term, X1)?
        see4(X0, event_term, X1, X2)?
        """
        event_term = self.aug_term(event_term)
        results = []
        for k in self.auxilary['keywords']['see']: 
            assert 'see' in k, "not a see keyword?"
            num_arg = int(k[len('see'):])
            args = [f'X{i}' for i in range(num_arg - 1)]
            args.insert(1, event_term)
            results += self.query_rule(k, args)
        results = sorted(list(set(results)))
        return results

    #@profile
    def find_all_killcell_facts(self): 
        results = []
        for k in self.auxilary['keywords']['killcell']: 
            assert 'killcell' in k, "not a killcell keyword?"
            num_arg = int(k[len('killcell'):])
            args = [f'X{i}' for i in range(num_arg)]
            results += self.query_rule(k, args)
        results = sorted(list(set(results)))
        return results

    def query_killcell_event(self, event_term): 
        """
        query killcell DAG of event term
        """
        event_term = self.aug_term(event_term)
        results = []
        for k in self.auxilary['keywords']['killcell']: 
            assert 'killcell' in k, "not a killcell keyword?"
            num_arg = int(k[len('killcell'):])
            args = [f'X{i}' for i in range(num_arg - 1)]
            args.insert(1, event_term)
            results += self.query_rule(k, args)
        results = sorted(list(set(results)))
        return results

    def query_see_cell(self, cell_term): 
        """
        query the see DAG of cell_term e.g. 
        see3(cell_term, X0, X1)?
        see4(cell_term, X1, X1, X2)?
        """
        cell_term = self.aug_term(cell_term)
        results = []
        for k in self.auxilary['keywords']['see']: 
            assert 'see' in k, "not a see keyword?"
            num_arg = int(k[len('see'):])
            args = [f'X{i}' for i in range(num_arg - 1)]
            args.insert(0, cell_term)
            results += self.query_rule(k, args)
        results = sorted(list(set(results)))
        return results

    def create_cell(self, term): 
        hascell_fact = f'hascell({term}).'
        self.datalog(hascell_fact)
        self.auxilary['asserted_facts'].add(hascell_fact)

    #@profile
    def will_create_cells(self, event_term): 
        """
        check if given event will create any cell
        according to head <- event, subgoals
        """
        event_term = self.aug_term(event_term) # in case smt like bos
        #print(f"event is {event_term}")
        who_see_it = [
            f[0] for f in self.query_see_event(event_term)
        ]
        #print(f"who sees it : {who_see_it}")
        exist_cells = self.find_all_cells()
        #print(f"exist cells : {exist_cells}")
        new_cells = set(who_see_it) - set(exist_cells)
        #print(f"new cells : {new_cells}")
        will_create = True if len(new_cells) > 0 else False
        return will_create, sorted(list(new_cells))

    def create_cells(self, new_cells): 
        """
        as event happens, according to head <- event, subgoals
        assert hascell(head). logic fact to database 
        """
        for term in new_cells: 
            self.create_cell(term)
    
    def kill_cell(self, term): 
        hascell_fact = f'hascell({term}).'
        self.datalog.retract_fact(hascell_fact)
        self.auxilary['asserted_facts'].remove(hascell_fact) # must already exist

    #@profile
    def will_kill_cells(self, event_term): 
        """
        check if given event will kill any cell 
        according to !head <- event, subgoals
        """
        #print(f"\n\n\n{event_term} may kill things\n\n\n")
        event_term = self.aug_term(event_term) # in case smt like bos
        who_to_kill = [
            f[0] for f in self.query_killcell_event(event_term)
        ]
        exist_cells = self.find_all_cells()
        assert set(who_to_kill).issubset(set(exist_cells)), "kill things that don't exist?!"
        will_kill = True if len(who_to_kill) > 0 else False
        return will_kill, sorted(list(who_to_kill))

    def kill_cells(self, who_to_kill): 
        """
        as event happens, according to !head <- event, subgoals
        retract hascell(head). logic fact from database
        """
        for term in who_to_kill: 
            self.kill_cell(term)

    def clear_asserted_facts(self): 
        to_retract = list(self.auxilary['asserted_facts'])
        for f in to_retract: 
            self.datalog.retract_fact(f)
            self.auxilary['asserted_facts'].remove(f)

    #@profile
    def find_all_functor_dims(self): 
        """
        find all the functor : dim pairs
        """
        k = list(self.auxilary['keywords']['embed'])
        assert len(k)==1, "not 1 embed keyword?"
        k = k[0]
        assert 'embed' in k, "not an embed keyword?"
        return self.query_rule(k, ['X0', 'X1'])

    def query_functor_dim(self, functor): 
        """
        query dim of functor
        embed(functor, D)?
        """
        k = list(self.auxilary['keywords']['embed'])
        assert len(k)==1, "not 1 embed keyword?"
        k = k[0]
        assert 'embed' in k, "not an embed keyword?"
        ds = self.query_rule(k, [functor, 'D'])
        assert len(ds)<=1, "more than 1 possible dimension value?!"
        if len(ds) == 1:
            return int(ds[0][1])
        else: 
            # if no dim declared, take 0
            return 0

    #@profile
    def query_rule(self, keyword, args):
        """
        query the rule with given query 
        """
        variables, terms = [], []
        for a in args: 
            if a[0].isupper(): 
                variables.append(a)
            else: 
                terms.append(a)
        inter = {}
        assert len(variables) > 0, "why query with no var?"
        query = f"{keyword}({', '.join(args)})?"
        #print(f"query is : {query}")
        self.datalog(query)
        #print("query done")
        for v in variables: 
            #print(f"grab {v}")
            D = self.datalog(f"{v}.data.")
            #print(f"finish grabbing {v}")
            inter[v] = [
                ''.join(self.datalog.unwrap_flattened_cmd(e)) for e in D 
            ]
            #print(f"finish processing {v}")
        l = len(inter[variables[0]])
        for t in terms: 
            inter[t] = [t] * l 
        res = set()
        for i in range(l): 
            temp = []
            for a in args: 
                temp.append(inter[a][i])
            res.add(tuple(temp))
        return sorted(list(res))

    #@profile
    def find_all_provable_terms(self): 
        """
        each provable term has a neural embedding 
        that is computed based on its subgoals (maybe true(specialaug))
        this can be done by looping over all transform facts 
        and collecting their heads 
        OR 
        by querying hasproofdag(X)?
        """
        results = set()
        #fans = self.find_all_transform_fans()
        #for f in fans: 
        #    results.add(f[0])
        temp = self.query_rule('hasproofdag', ['X0'])
        for i in temp: 
            if i[0] not in self.auxilary['exclude']:
                results.add(i[0])
        return sorted(list(results))

    def aug_dim(self, terms): 
        """
        try not use this very often 
        use aug_dim in database instead 
        to reduce count of queries
        """
        d, z = 0, []
        for s in terms: 
            s_dim = self.get_term_dim(s)
            if s_dim > 0: 
                z += []
                d += s_dim
            else: 
                z += [d]
                d += 1
        return d, tuple(z)

    #@profile
    def find_all_events(self): 
        """
        find all terms that are also events 
        """
        results = set()
        k, num_arg = f'event{4}', 4 # not use event logic rule s.t we exclude bos
        args = [f'X{i}' for i in range(num_arg)]
        temp = self.query_rule(k, args)
        for i in temp: 
            # only keep event name, rule idx and parameter idx useless
            results.add(i[0])
        return sorted(list(results))

    #@profile
    def find_all_cells(self):
        """
        each head of <- rule has a cell
        """
        results = set()
        #fans = self.find_all_see_fans()
        #for f in fans: 
        #    results.add(f[0])
        temp = self.query_rule('hascell', ['X0'])
        for i in temp: 
            if i[0] not in self.auxilary['exclude']: 
                results.add(i[0])
        return sorted(list(results))

    def get_functor(self, term): 
        # get outer most functor 
        # email == get_functor(email(ceo(google), cfo(google)))
        term = term.replace(' ', '')
        if '(' not in term: 
            return term 
        i = term.index('(')
        return term[:i]

    def get_term_dim(self, term): 
        return self.query_functor_dim(self.get_functor(term))
