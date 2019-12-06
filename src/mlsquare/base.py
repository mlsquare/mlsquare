from abc import ABC, abstractmethod



class Registry(object):
    """
	This class is used to maintain a registry.

    Parameters
    ----------
    data : dict
        This variable holds the registry details.


    Methods
    -------
	register(model)
        Use this method to register a model in registry

    """


    def __init__(self):
        # Variable name options -- model_data or model_info
        self.data = {}

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

	"""
	This class acts as 'Base' class for all models created in mlsquare.

    Parameters
    ----------
    enc : sklearn.preprocessing.OneHotEncoder
        The variable to hold OneHotEncoder instances passed by the model.


    Methods
    -------
	create_model(**kwargs)
        Method to return final dnn model.

	set_params(**kwargs)
        Method to set model parameters of proxy model.

	get_params()
        Method to read model parameters

	update_params(params)
        Method to update model parameters

	adapter()
        Getter and setter methods for adapter property.

	transform_data(X, y, y_pred)
        Method to transform data provided by the user.

    """

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

class BaseTransformer(ABC):
    """
	A base class for matrix decomposition models.

    This class can be used as a base class for any dimensionality reduction models.
    While implementing ensure all required methods are implemented or over written
    Please refer to sklearn decomposition module for more details.

    Methods
    -------
	fir_transform(input_args)
        fits the model to output input values with reduced dimensions.
    """
    @abstractmethod
    def fit_transform(self, X, y=None, **kwargs):
        raise NotImplementedError('Needs to be implemented!')

registry = Registry()
