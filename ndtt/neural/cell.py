import torch
import torch.nn as nn
import torch.nn.functional as F

class CTLSTMCell(object):
    """CTLSTMCell: :Mei and Eisner 2017
    each cell is basically (cell memories, decay gate)
    NOTE : not inherent nn.Module can make creation and copy faster
    """
    def __init__(self, name, cell_dim, cell_zero, time_created, device=None):
        super(CTLSTMCell, self).__init__()
        device = device or 'cpu'
        self.device = torch.device(device)
        self.name = name
        self.cell_dim = cell_dim
        assert cell_dim >= 1, "cell dim must > 0"
        self.cell_zero = cell_zero
        self.cell_mask = torch.ones(cell_dim, device=device)
        for c in self.cell_zero: 
            self.cell_mask[c] = 0.0
        self.time_created = time_created
        self.max = torch.finfo().max
        self.min = torch.finfo().min
        self.reset()

    def reset(self):
        """
        time of last time the states are updated
        """
        #print(f"{self.name} is reset")
        self.time_updated = self.time_created
        self.history = [
            {
                'start': torch.zeros(size=[self.cell_dim], dtype=torch.float32, device=self.device),
                'target': torch.zeros(size=[self.cell_dim], dtype=torch.float32, device=self.device),
                'decay': torch.ones(size=[self.cell_dim], dtype=torch.float32, device=self.device),
            }
        ]

    def detach(self): 
        """
        detach the history graph 
        """
        self.history[-1]['start'] = self.history[-1]['start'].detach()
        self.history[-1]['target'] = self.history[-1]['target'].detach()
        self.history[-1]['decay'] = self.history[-1]['decay'].detach()

    def update(self, time, cell, append=False):
        """
        this function updates the configuration of this partition
        append = False to save memory
        """
        assert cell['start'].size(0) == self.cell_dim, \
            f"Dimension mismatch : {cell['start'].size(0)} vs. {self.cell_dim}"
        self.time_updated = time
        """
        new cells are aggregated values so might be too large or too small
        """
        for k, v in cell.items(): 
            v[v>self.max] = self.max
            v[v<self.min] = self.min        

        if append:
            self.history.append(cell)
        else:
            self.history[-1]['start'] = cell['start']
            self.history[-1]['target'] = cell['target']
            self.history[-1]['decay'] = cell['decay']

    def retrospect(self, idx=-1):
        """
        this function pops most recent memory
        """
        return self.history[idx]

    def decay(self, time):
        """
        float or FloatTensor time: absolute time
        """
        dtime = time - self.time_updated
        #print("decay : {}".format(self.nodes[-1]['decay']))
        #print("time updated {:.4f}".format(self.time_updated))
        #print("time and dtime : {} and {}".format(time, dtime))
        """
        NOTE : such assertions may be too aggressive 
        e.g., suppose a :- b, a <- c, a
        then if c happens at time 1, a gets a cell at time 1 
        then at time 1, a's cell wil be updated considering a's embedding 
        what's the problem then? --- it is very tricky !!!
        the ``current'' database will be the one after creating a's cell 
        so under current database, a's cell exists
        so to compute a'emb, 
        it try to find a's cell at 1^- --- before a actually got a cell!!!
        then the dtime will be a negative value!!!
        SOLUTION : check creation time when decaying 
        if we find any < 0 dtime, we consult its creation time 
        if current time > creation time, then it is an error or bug 
        if current time < creation time, then it is the tricky case 
        in this case, we return 0-vector---equal to not have a cell at all!!!
        NOTE : why we haven't revised it yet? 
        having dtime < 0 in general is dangerous 
        so to make safe programs, we rather not have that bad example-ish rules...
        here are 2 alternatives to get the same behavior but still make program safe
        (1) if a :- b and a <- c, ... 
        then a better be asserted by bos as well : a <- bos. 
        (2) we can make a time-varying by making b a cell 
        a :- b, b <- bos, b <-c 
        s.t. when c happens, b gets updated and a is only affected indirectly...
        """
        if isinstance(time, float):
            assert dtime >= 0.0, "dtime negative? dtime is {}".format(dtime)
        elif isinstance(time, torch.Tensor):
            dtime = dtime.unsqueeze(
                dtime.dim()).expand(*dtime.size(), self.cell_dim)
            assert dtime.min() >= 0.0, \
                f"dtime negative? min is {dtime.min()}. name is {self.name}. time updated is {self.time_updated}. time created is {self.time_created}. history is {self.history}"
        else:
            raise Exception(f"time is neither float nor torch.Tensor? type : {type(time)}, value : {time}")

        assert self.history[-1]['decay'].min() >= 0.0, f"decay is negative? {self.history[-1]['decay']}"
        discount = torch.exp( -self.history[-1]['decay'] * dtime )
        res = self.history[-1]['start'] * discount \
            + self.history[-1]['target'] * (1-discount)
        return res * self.cell_mask

    def get_upperbound(self): 
        """
        return a value that can't be exceeded no matter what time it is 
        i.e., for each dim, get max(start, target) 
        """
        temp = torch.stack( [self.history[-1]['start'], self.history[-1]['target']], dim=0 )
        return temp.max(0)[0].detach()

    def cuda(self, device=None):
        device = device or 'cuda:0'
        self.device = torch.device(device)
        assert self.device.type == 'cuda'
        super().cuda(self.device)

    def cpu(self):
        self.device = torch.device('cpu')
        super().cuda(self.device)

    def copy(self):
        """
        between shallow and deep copy : only copy initial values 
        """
        cp_obj = CTLSTMCell(
            self.name, self.cell_dim, self.cell_zero, self.time_created, self.device)
        return cp_obj

    def deepcopy(self): 
        """
        deep copy : copy current (i.e. most recent) values 
        """
        cp_obj = self.copy()
        cp_obj.time_updated = self.time_updated
        for k, v in self.history[-1].items(): 
            """
            .detach but not .clone 
            cuz we do not want the graph info behind it
            why no graph? cp_obj might be passed into other processes
            """
            cp_obj.history[-1][k] = v.detach()
        return cp_obj

    def __repr__(self):
        s0 = 'CTLSTMCell:\n'
        s0 += '{} : cell_dim={}, cell_zero={}, time_created={}'.format(
            self.name, self.cell_dim, self.cell_zero, self.time_created
        )
        s1 = f'\ntime_updated={self.time_updated}, history={self.history}'
        return s0+s1

    