import numpy as np


class XVal(object):
    def __init__(self, N, num_folds=5, shuffle=True):
        self.N = N
        self.num_folds = num_folds
    
        if shuffle:
            self.inds = np.random.permutation(N)
        else:
            self.inds = np.arange(N)
        
        self.current_fold = 0
        
    def next(self):
        per_fold = int(self.N / self.num_folds)
        
        # build test index set based on what current fold is
        if self.current_fold >= self.num_folds:
            raise StopIteration
        elif self.current_fold < self.num_folds - 1:
            test_inds = self.inds[self.current_fold * per_fold:((self.current_fold + 1) * per_fold)]
        elif self.current_fold == self.num_folds - 1:
            test_inds = self.inds[self.current_fold * per_fold:]
            
        train_inds = np.setdiff1d(self.inds, test_inds, assume_unique=True)
        
        self.current_fold += 1
        return np.setdiff1d(train_inds, test_inds, assume_unique=True), test_inds
    
    def __iter__(self):
        return self