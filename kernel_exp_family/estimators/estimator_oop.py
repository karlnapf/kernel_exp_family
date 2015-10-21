class EstimatorBase(object):
    def __init__(self, D):
        self.D = D
    
    def fit(self, X):
        raise NotImplementedError()
    
    def log_pdf_multiple(self, X):
        raise NotImplementedError()
    
    def log_pdf(self, x):
        raise NotImplementedError()
    
    def grad(self, x):
        raise NotImplementedError()
    
    def objective(self, X):
        raise NotImplementedError()
    