from tensorflow.python.keras.layers import Layer

class Bin(Layer):
    """
    A custom keras layer to create bins in
    DNDT implementation(https://arxiv.org/pdf/1806.06988.pdf).
    The input for this layer should always be a Keras 'Input' layer.
    The layer then accepts a feature of your choice(index_of_feature) and
    the number of bins(num_of_cuts + 1) you want the selected feature
    to be split into.
    For a given input of shape (None, features) and cut points = D, this layer
    returns a tensor of shape (None, D+1).

    # Arguments
        index_of_feature: Integer, The position of your choice of feature to be split into bins.
        num_of_cuts: Integer, The number of bins you want the feature to be split into.

    Note: This layer was initially created with the intention to be used in DecisionTree layer.
    We later realized that the DecisionTree layer was better off with a `binning` function rather than
    a layer. Hence, it's not used in DecisionTree layer anymore.
    The same applies to KronProd layer as well.

    """

    def __init__(self, index_of_feature, num_of_cuts, temperature = 0.1, **kwargs):
        self.index_of_feature = index_of_feature
        self.num_of_cuts = num_of_cuts
        self.temperature = temperature
        super(Bin, self).__init__(**kwargs)

    def build(self, input_shape):
        self.cut_points = self.add_weight(name='cut_points',
                                    shape=(self.num_of_cuts,),
                                    initializer='uniform',
                                    trainable=True)
        super(Bin, self).build(input_shape)

    def call(self, x):
        from tensorflow.python.keras import backend as K # Fix
        import tensorflow as tf

        X = x[:, self.index_of_feature : self.index_of_feature + 1]
        D = self.num_of_cuts
        W = K.reshape(tf.linspace(1.0, D + 1.0, D + 1), [1, -1])
        self.cut_points = tf.contrib.framework.sort(self.cut_points)
        b = K.cumsum(tf.concat([K.constant(0.0, shape=[1]), -self.cut_points], 0))
        h = tf.matmul(X, W) + b
        output = tf.nn.softmax(h / self.temperature)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_of_cuts+1)


class KronProd(Layer):

    """
    A layer to compute the Kron product.
    This layer expects a list of tensors as input. Kronecker product
    operation is then applied in sequence to each element in the list.

    """

    def __init__(self, **kwargs):
        super(KronProd, self).__init__(**kwargs)

    def build(self, input_shape):
        super(KronProd, self).build(input_shape)

    def call(self, x):
        # from tensorflow.python.keras import backend as K # Fix
        import tensorflow as tf
        # from functools import reduce
        input_tensor_list = x

        def kron_prod(a,b):
          res = tf.einsum('ij,ik->ijk', a, b)
          res = tf.reshape(res, [-1, tf.reduce_prod(res.shape[1:])])
          return res

        output = reduce(kron_prod, input_tensor_list)
        self.output_dim = output.get_shape().as_list()

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.output_dim[1])

from functools import reduce

class DecisionTree(Layer):

    """
    A custom Keras layer that implements the
    Deep Neural Decision Tree(https://arxiv.org/pdf/1806.06988.pdf).
    On initialization the layer expects the number of cuts you
    would like to make per feature.

    Note: The input expected is of rank 2.

    Parameters
    ----------
        cuts_per_feature: list, int
        This variable can either be a list of integers or a single integer.
        If passed as a list, each integer is the number of cuts to be made on
        it's corresponding feature.
        If passed as an integer, that many number of cuts are made, consistently
        for each feature.


    Examples
    --------

        # Creating a model with DecisionTree layer using
        # Keras functional API
        >>> visible = Input(shape=(4,))
        >>> hidden_layer = DecisionTree(cuts_per_feature=[1,1,1,1])(visible)
        >>> output = Dense(3, activation='sigmoid')(hidden_layer)
        >>> model = Model(inputs=visible, ouputs=output)

        # The above model accepts an input with (None, 4) shape. The Decsion tree layer
        # then provides an output of shape (None, 16) - 2 bins per feature in this scenario.
        # The model then applies a regular 'Dense' layer to this output.

    Input
    -----
        2D tensor with shape: `(batch_size, num_of_feature)`.

    Output
    ------
        2D tensor with shape: `(batch_size, num_of_bins)`.
        num_of_bins = prod(cuts_per_feature + 1)
    """

    def __init__(self, cuts_per_feature, **kwargs):
        self.cuts_per_feature = [1 if i < 1 else int(i) for i in cuts_per_feature]
        if not all(isinstance(n, int) for n in self.cuts_per_feature):
            print([type(n) for n in self.cuts_per_feature])
            print(self.cuts_per_feature)
            raise ValueError('All elements in `cuts_per_feature` should be of type `int`.')
        super(DecisionTree, self).__init__(**kwargs)

    def build(self, input_shape):
        self._trainable_cutpoints = []
        for i in self.cuts_per_feature:
          self._trainable_cutpoints.append(self.add_weight(name='cut_points'+'_'+str(i),
                                        shape=(i,),
                                        initializer='uniform',
                                        trainable=True))
        super(DecisionTree, self).build(input_shape)


    def binning_fn(self,index_of_feature, num_of_cuts, x, temperature = 0.1):
        from tensorflow.python.keras import backend as K # Fix
        import tensorflow as tf
        X = x[:, index_of_feature - 1 : index_of_feature]
        D = num_of_cuts
        W = K.reshape(tf.linspace(1.0, D + 1.0, D + 1), [1, -1])
        cutpoints_value = tf.contrib.framework.sort(self._trainable_cutpoints[index_of_feature - 1])
        b = K.cumsum(tf.concat([K.constant(0.0, shape=[1]), -cutpoints_value], 0))
        h = tf.matmul(X, W) + b
        output = tf.nn.softmax(h / temperature)
        return output

    def call(self, x):

      def kron_prod(a,b):
          import tensorflow as tf
          res = tf.einsum('ij,ik->ijk', a, b)
          res = tf.reshape(res, [-1, tf.reduce_prod(res.shape[1:])])
          return res

      bin_result = []
      for i, value in enumerate(self.cuts_per_feature):
        bin_result.append(
            self.binning_fn(index_of_feature=i+1, num_of_cuts=value, x=x)
        )

      output = reduce(kron_prod, bin_result)
      self.output_dim = output.get_shape().as_list()

      return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim[1])

## TODO
# Default cutpoints - ceiling operation
# Error handling in layers
# Fix multiple imports
## Memory issue Inform user- Raise error - Reduce cuts or opt for higher memory