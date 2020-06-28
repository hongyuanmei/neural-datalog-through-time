import os
import pickle

class DataLoader(object):

    def __init__(self):
        """
        seq: seqs of train, dev, test, test1, etc
        e.g. seq = {'train': train_data, 'dev': dev_data, 'test': test_data}
        train/dev/test_data is a list
        mask: for each seq, we may not compute its log-lik for all the events 
        so mask indicates the starting time for computation 
        e.g. we have a seq with T=100, and we want the seq by 80 for training
        80 - 90 for dev, 90 - 100 for test 
        in this case, mask['train'] = 0, mask['dev'] = 80, mask['test'] = 90
        sometimes, mask files do not exis, indicating entire seq is used 
        in that case, its mask is automatically set as 0
        """
        self.seq = {}
        self.mask = {}
        """
        sizes: # of seqs of train, dev, test, test1, etc
        e.g. sizes = {'train': 1000, 'dev': 100, 'test': 100}
        """
        self.sizes = {}
        self.lens = {}

    def load_data(self, loc, data_splits): 
        for split, split_ratio in data_splits: 
            """
            regardless of raw data format
            we will get
            :seq[split] a list of sequences
            :sizes[split] # of sequences
            """
            print(f"load {split_ratio*100.0:.6f}% of {split} data")
            self.seq[split] = self.load_seq(loc, split)
            self.sizes[split] = len(self.seq[split])
            self.mask[split] = self.load_mask(loc, split)
            """
            may not use all the data 
            """
            actual_size = int(self.sizes[split] * split_ratio)
            assert actual_size > 0, f"no data left for {split}? ratio is {split_ratio}"
            assert actual_size <= self.sizes[split], f"more data than available? ratio is {split_ratio}"
            del self.seq[split][actual_size:]
            self.sizes[split] = actual_size
            del self.mask[split][actual_size:]
            print(f"{actual_size} sequences left for {split}")

    def load_seq(self, loc, split):
        with open(os.path.join(loc, '{}.pkl'.format(split)), 'rb') as f:
            seq = pickle.load(f)
        return seq

    def load_mask(self, loc, split):
        path_mask = os.path.join(loc, '{}_mask.pkl'.format(split))
        if os.path.exists(path_mask):
            with open(path_mask, 'rb') as f:
                mask = pickle.load(f)
            assert len(mask) == self.sizes[split]
        else:
            mask = [ 0.0 ] * self.sizes[split]
        return mask

    def __repr__(self): 
        s = ''
        for k, v in self.sizes.items(): 
            s += f'{v} {k} seqs, '
        return s[:-2]