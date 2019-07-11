from abc import ABC, abstractmethod

class Registry(object): ## Move to commons.decorator?

    def __init__(self):
        self.data = {} ##variable name options -- model_data or model_info
    # @staticmethod ## static or class - crosscheck
    def register(self, model): ##variable name options -- register_model
        model = model()
        adapter = model.adapter
        module_name = model.module_name
        # Update this flow with the model versions option
        self.data[(module_name, model.__class__.__name__)] = {'default':[model, adapter]}


    def __getitem__(self, key):
       return self.data[key]


class BaseModel(ABC):

	enc = None

	@abstractmethod
	def create_model(self):
		raise NotImplementedError('Needs to be implemented!')

	@abstractmethod
	def set_params(self):
		raise NotImplementedError('Needs to be implemented!')

	@abstractmethod
	def get_params(self):
		raise NotImplementedError('Needs to be implemented!')

	@abstractmethod
	def update_params(self):
		raise NotImplementedError('Needs to be implemented!')

	@property
	@abstractmethod
	def adapter(self):
		raise NotImplementedError('Needs to be implemented!')

	@adapter.setter
	def adapter(self, obj):
		self._adapter = obj

	def transform_data(self, X, y, y_pred):
		return X, y, y_pred

registry = Registry()


'''
Add versioning option for models
1) module is available and mapping is available. Better version of mapping.
2) Module is avialbale, no mapping.
3) Module is not available. Implement adapter, model etc from scratch
4) uuid for versions
'''