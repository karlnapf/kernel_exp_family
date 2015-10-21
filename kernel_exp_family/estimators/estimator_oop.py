class EstimatorBase(object):
    def __init__(self, D):
        self.D = D
    
    def fit(self, X):
        raise NotImplementedError()
    
    def log_pdf(self, X):
        raise NotImplementedError()
    
    def log_pdf_single(self, x):
        raise NotImplementedError()
    
    def grad_single(self, x):
        raise NotImplementedError()
    
    def objective(self, X):
        raise NotImplementedError()
    