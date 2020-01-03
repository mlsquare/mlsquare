from sklearn.base import BaseEstimator

class tpm(BaseEstimator):
    def __init__(self,data=None, typ = "latent.trait", constraint = 'NULL', 
    max_guessing = 1, IRT_param = True, start_val = 'NULL', na_action = 'NULL', control = list()):
        self.data=data
        self.type = typ
        self.constraint = constraint
        self.max_guessing= max_guessing
        self.start_val= start_val
        self.na_action= na_action
        self.control=control

class twoPl(BaseEstimator):
    def __init__(self, data=None, constraint = 'NULL', IRT_param = True, start_val = 'NULL', 
    na_action = 'NULL', control = list(), Hessian = True):
        self.data=data
        self.constraint = constraint
        self.IRT_param= IRT_param
        self.start_val= start_val
        self.na_action= na_action
        self.control=control
        self.Hessian=Hessian

class rasch(BaseEstimator):
    def __init__(self, data=None, constraint = 'NULL', IRT_param = True, start_val = 'NULL', 
    na_action = 'NULL', control = list(), Hessian = True):
        self.data=data
        self.constraint = constraint
        self.IRT_param= IRT_param
        self.start_val= start_val
        self.na_action= na_action
        self.control=control
        self.Hessian=Hessian
