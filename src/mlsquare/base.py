from abc import ABC, abstractmethod

## Move to commons.decorator?
class Registry(object):

    def __init__(self):
        # Variable name options -- model_data or model_info
        self.data = {}
    # @staticmethod -- static or class - crosscheck
    # Using static/class isn't feasible since the whole registry framework
    # relies on an instance Registry class.
    def register(self, model):
        model = model()
        adapter = model.adapter
        module_name = model.module_name
        model_name = model.name
        version = model.version
        try: ## Improve this flow.
            self.data[(module_name, model_name)].update({version:[model, adapter]})
        except KeyError:
            self.data[(module_name, model_name)] = {version:[model, adapter]}


    def __getitem__(self, key):
        return self.data[key]


class BaseModel(ABC):

	enc = None

	@abstractmethod
	def create_model(self, **kwargs):
		raise NotImplementedError('Needs to be implemented!')

	@abstractmethod
	def set_params(self, **kwargs):
		raise NotImplementedError('Needs to be implemented!')

	@abstractmethod
	def get_params(self):
		raise NotImplementedError('Needs to be implemented!')

	@abstractmethod
	def update_params(self, params):
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
