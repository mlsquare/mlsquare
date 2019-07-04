from abc import ABC, abstractmethod

class Registry(object): ## Move to commons.decorator?

    def __init__(self):
        self.data = {} ##variable name options -- model_data or model_info
    
    def register(self, model): ##variable name options -- register_model
        model = model()
        wrapper = model.wrapper
        module_name = model.module_name
        # Update this flow with the model versions option
        self.data[(module_name, model.__class__.__name__)] = {'default':[model, wrapper]}


    def __getitem__(self, key):
       return self.data[key]


class BaseModel(ABC):

	@abstractmethod
	def create_model(self):
		raise NotImplementedError('Needs to be implemented!')

	@abstractmethod
	def set_params(self):
		raise NotImplementedError('Needs to be implemented!')

	@abstractmethod
	def get_params(self):
		raise NotImplementedError('Needs to be implemented!')

	@property
	@abstractmethod
	def wrapper(self):
		raise NotImplementedError('Needs to be implemented!')

	@wrapper.setter
	def wrapper(self, obj):
		self._wrapper = obj

registry = Registry()
