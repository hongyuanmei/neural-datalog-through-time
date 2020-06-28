"""
PyDatalog is a very useful package but very hard to use.
To deal with this, Guanghui design this wrapper to interact with it.
With this wrapper, you could add terms or execute logic expressions in multiple places with the
same 'database' (pls point it out is it's a misnomer) and without messing your local environment.

To implement this, Guanghui design a dict to simulate a local environment. Every time you call the
create_term function, it will create a term and then add it into the dict. Every time you
execute a command, the local dict will simulate a local environment for this command.

Additionally, this Datalog wrapper enables nested terms

Example: (suppose the following codes are in another file)
    >>> from domains.pydatalog_wrapper import Datalog
    >>> # The following line is optional. If a term mentioned in the expression
    >>> # has not been created yet, it will be created automatically.
    >>> dl = Datalog()
    >>> dl('+ friend(alice,bob)')
    >>> dl('+ friend(carl,bob)')
    >>> dl('friend(X,Y) <= friend(Y,X)')
    >>> dl('friend(X,bob)')
    >>> bob_friends = dl('X.data')
    >>> # bob_friends is ['carl', 'alice']
"""


from pyDatalog.pyParser import Term
import string
from functools import partial

class PredicateError(Exception): 
    pass

class Datalog(object):
    def __init__(self):
        self.fake_locals = dict()
        self.func_mark = 'func_mark'
        self.create_term(self.func_mark)

    def create_term(self, term_name):
        """
        Create terms to use. The created terms will be stored in fake_local.
        Note: I will NOT check the validity of the terms. Not all the term names are allowed,
        but I assume you have known the nomination rules.
        :param term_name: Either a list or a string. If it's a list, I will create all the terms inside.
        """
        assert type(term_name) in [str, list], "Please pass in either a string or a list."
        if isinstance(term_name, list):
            for term_name_ in term_name:
                self.create_term(term_name_)
            return
        if term_name not in self.fake_locals and not term_name[0].isdigit():
            # only create undefined terms
            self.fake_locals[term_name] = Term(term_name)

    @staticmethod
    def split_cmd(cmd):
        """
        Convert a cmd in string to list of terms.
        E.g.
        Input: 'able_to_send_email(X, Y) <= (person(X) & person(Y))'
        Output: ['able_to_send_email', '(', 'X', ',', 'Y', ')', '<=', '(', 'person', '(', 'X', ')', '&', 'person', '(',
        'Y', ')', ')']
        :param str cmd: Command in string
        :return: A list of terms.
        :rtype: list
        """
        two_char_marks_mappings = [['<=', '_LE_'], ['==', '_EQ_'], ['>=', '_GE_'], [':-', '_CD_']]
        for mark_, rep_ in two_char_marks_mappings:
            cmd = cmd.replace(mark_, rep_)
        punctuations = string.punctuation.replace('_', '')
        for punc_ in punctuations:
            cmd = cmd.replace(punc_, ' '+punc_+' ')
        for mark_, rep_ in two_char_marks_mappings:
            cmd = cmd.replace(rep_, mark_)
        return cmd.split()

    @staticmethod
    def is_variable(term):
        return term[0].isupper()

    @staticmethod
    def is_constant(term):
        return term[0].islower()

    def create_unseen_terms(self, cmd_list):
        """
        Create the terms appeared in the command.
        :param list cmd_list: Returned by the function split_cmd.
        :return: Nothing
        """
        for term in cmd_list:
            if term[0].isalpha():
                self.create_term(term)

    def flatten_nested_helper(self, cmd_list):
        assert len(cmd_list) >= 4, 'The command should be at least 4 terms/punctuation marks.'
        assert self.is_constant(cmd_list[0]) or self.is_variable(cmd_list[0]), 'The first term should be constant or variable.'
        assert cmd_list[1] == '(' and cmd_list[-1] == ')', 'The brackets should appear in the command.'
        func_head = cmd_list[0]
        func_body_terms = list()
        depth = 0
        beg_idx = 2
        for term_idx in range(2, len(cmd_list)):
            term = cmd_list[term_idx]
            if term == '(':
                depth += 1
            elif term == ')':
                depth -= 1
            if (depth == 0 and term == ',') or term_idx == len(cmd_list)-1:
                func_body_terms.append(cmd_list[beg_idx:term_idx])
                beg_idx = term_idx + 1

        for func_term_idx, func_term in enumerate(func_body_terms):
            if '(' in func_term:
                func_body_terms[func_term_idx] = self.flatten_nested_helper(func_term)

        ret_list = ['[', self.func_mark, ',', func_head]
        for func_term in func_body_terms:
            ret_list = ret_list + [','] + func_term
        ret_list.append(']')

        return ret_list

    def _unwrap_flattened_cmd(self, cmd_list):
        """
        atomic internal method recursively called by unwrap_flattened_cmd 
        """
        if (not isinstance(cmd_list, tuple)) or (len(cmd_list) == 0) or (cmd_list[0] != self.func_mark):
            return cmd_list

        assert len(cmd_list) >= 3, 'Length error: ' + ' '.join(cmd_list)

        func_head = cmd_list[1]
        body_terms = list(map(self._unwrap_flattened_cmd, cmd_list[2:]))
        ret_list = [func_head, '(']
        for body_term in body_terms:
            ret_list.append(body_term)
            ret_list.append(',')
        ret_list[-1] = ')'
        return ret_list

    def flatten_nested_left(self, cmd_list):
        if '(' not in cmd_list:
            return cmd_list
        flattened_cmd = self.flatten_nested_helper(cmd_list)
        return flattened_cmd[3:4] + ['('] + flattened_cmd[5:-1] + [')']

    def flatten_nested_right(self, cmd_list):
        if len(cmd_list) == 0:
            return list()
        if cmd_list[0] == '<=':
            return ['<='] + self.flatten_nested_right(cmd_list[1:])
        depth = 0
        beg_idx = 0
        last_idx = len(cmd_list) - 1
        if cmd_list[0] == '(':
            depth -= 1
            beg_idx += 1
            last_idx -= 1
        rst = list()
        is_first = True
        for item_idx, item in enumerate(cmd_list):
            if item_idx == last_idx:
                if not is_first:
                    rst.append('&')
                rst.extend(
                    self.flatten_nested_left(cmd_list[beg_idx: item_idx+1])
                )
                break
            elif item == '(':
                depth += 1
            elif item == ')':
                depth -= 1
            elif item == '&' and depth == 0:
                if is_first:
                    is_first = False
                else:
                    rst.append('&')
                rst.extend(
                    self.flatten_nested_left(cmd_list[beg_idx: item_idx])
                )
                beg_idx = item_idx + 1
        return ['('] + rst + [')']

    #@profile
    def __call__(self, cmd, is_py_datalog=False):
        """
        To execute a command with the fake environment.
        :param str cmd: The command to execute.
        """
        #print(f"cmd = {cmd}")
        cmd_list = self.split_cmd(cmd)
        if not is_py_datalog:
            cmd_list = self.datalog_to_pydatalog(cmd_list)

        self.create_unseen_terms(cmd_list)
        beg_idx = 0
        if cmd_list[0] == '+':
            beg_idx = 1
        end_idx = len(cmd_list)
        if '<=' in cmd_list:
            end_idx = cmd_list.index('<=')

        cmd_list = cmd_list[:beg_idx] \
                   + self.flatten_nested_left(cmd_list[beg_idx:end_idx])\
                   + self.flatten_nested_right(cmd_list[end_idx:])
        cmd = ' '.join(cmd_list)

        #print("cmd to eval is {}".format(cmd))
        rst = eval(cmd, None, self.fake_locals)
        """
        # NOTE : str(rst) is very slow and takes upto 90% time of __call__
        # so we don't use it 
        # to keep safe, when we use __call__, we only query with known keywords
        # so that the rst is valid 
        # moreover, we try to only query datalog once in DatabaseFast class 
        # and do all the other 'fake' queries in that class 
        # to minimize # of calls of __call__
        try: 
            str(rst)
        except AttributeError as e: 
            print(f"error message is : {e.args[0]}")
            err = PredicateError(e.args[0])
            err.predicate_info = e.args[0]
            raise err
        """
        return rst

    @staticmethod
    def datalog_to_pydatalog(cmd_list):
        assert cmd_list[-1] in ['.', '?']
        if ':-' in cmd_list:
            assert cmd_list[-1] == '.'
            split_idx = cmd_list.index(':-')
            left_part, right_part = cmd_list[:split_idx], cmd_list[split_idx+1:-1]
            conditions = list()
            beg_idx = 0
            depth = 0
            for term_idx, term in enumerate(right_part):
                if depth == 0 and term == ',':
                    conditions.append(right_part[beg_idx: term_idx])
                    beg_idx = term_idx + 1
                if term == '(':
                    depth += 1
                elif term == ')':
                    depth -= 1
            r"""
            Hongyuan: without following line, the last condition will be lost
            so if the rule has only one condition, the entire condition is lost
            """
            conditions.append(right_part[beg_idx:])

            r"""
            Hongyuan: the commented line adds another layer of () that does not seem useful
            """
            #conditions = [['('] + condition + [')'] for condition in conditions]

            right_part = ['(']
            for condition in conditions:
                right_part += condition
                r"""
                Hongyuan: add the & symbol that is important in pyDatalog
                """
                right_part += ['&']
            right_part.pop()
            right_part += [')']
            cmd_list = left_part + ['<='] + right_part
        else:
            if cmd_list[-1] == '.':
                cmd_list.pop()
                if '(' in cmd_list:
                    cmd_list.insert(0, '+')
            else:
                cmd_list.pop()
        #print("translated pyDatalog is {}".format(cmd_list))
        return cmd_list

    def unwrap_flattened_cmd(self, cmd):

        def recursive(item):
            if not (isinstance(item, list) or isinstance(item, tuple)):
                return str(item)
            if (isinstance(item, list) or isinstance(item, tuple)) and item[0] == self.func_mark:
                item = self._unwrap_flattened_cmd(item)
            if isinstance(item, list) or isinstance(item, tuple):
                item = [recursive(sub_item) for sub_item in item]
                item = ''.join(item)
            return str(item)

        return recursive(cmd)

    def retract_fact(self, fact, is_py_datalog=False):
        """
        Retract a fact.
        Retracting a rule is not supported now.
        Return True is the fact is executed. Note: Being executed doesn't imply that the fact existed.
        Return False if the syntax of the fact is wrong.
        :param str fact: The fact to be retracted.
        :param bool is_py_datalog: If True, treat it as py datalog clause
        """
        cmd_list = self.split_cmd(fact)
        if not is_py_datalog:
            cmd_list = self.datalog_to_pydatalog(cmd_list)
        self.create_unseen_terms(cmd_list)

        if cmd_list[0] != '+':
            return False
        cmd_list = ['-'] + self.flatten_nested_left(cmd_list[1:])
        cmd = ' '.join(cmd_list)

        eval(cmd, None, self.fake_locals)
        return True
