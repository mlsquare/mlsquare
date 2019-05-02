from keras.layers import Layer 

class Bin(Layer):
    """
    A custom keras layer to create bins in DNDT implementation(https://arxiv.org/pdf/1806.06988.pdf).
    The input for this layer should always be the Keras 'Input' layer.
    The layer then accepts a feature of your choice(index_of_feature) and the number of bins(num_of_cuts + 1)
    you want the selected feature to be split into.
    For a given input of shape (None, features) and cut points = D, this layer returns a tensor
    of shape (None, D+1).

    # Properties
        index_of_feature: Integer, The position of your choice of feature to be split into bins.
        num_of_cuts: Integer, The number of bins you want the feature to be split into.

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
        from keras import backend as K # Fix
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

    def __init__(self,cut_points_list, **kwargs):
        self.cut_points_list = cut_points_list
#         self.num_class = num_class
        super(KronProd, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
#         self.cut_points = self.add_weight(name='cut_points', # Validate this choice of trainable weight.
#                                     shape=(self.num_cut,), # Or (self.num_cut,)
#                                     initializer='uniform',
#                                     trainable=True)
        super(KronProd, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        from keras import backend as K # Fix
        import tensorflow as tf
        from functools import reduce
        input_tensor_list = x
#         input_tensor_list = []
#         start_point = 0
#         for i in self.cut_points_list:
#           end_point = start_point + i + 1
#           input_tensor_list.append(x[:, start_point:end_point])
#           start_point = end_point
          
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

class DecisionTrees2(Layer):

    def __init__(self, num_cut, **kwargs):
        self.num_cut = num_cut
        self.cut_points_list = []
        super(DecisionTrees2, self).__init__(**kwargs)

    def build(self, input_shape):
        for i in self.num_cut:
          self.cut_points_list.append(self.add_weight(name='cut_points'+'_'+str(i), # Validate this choice of trainable weight.
                                        shape=(i,), # Or (self.num_cut,)
                                        initializer='uniform',
                                        trainable=True))
        super(DecisionTrees2, self).build(input_shape)  # Be sure to call this at the end
        
        
    def bin_fn(self,feature_num, num_cut_value, x):
        from keras import backend as K # Fix
        import tensorflow as tf
        temperature = 0.1
        X = x[:, feature_num - 1 : feature_num]
        # D = self.cut_points.get_shape().as_list()[0]
        D = num_cut_value
        W = K.reshape(tf.linspace(1.0, D + 1.0, D + 1), [1, -1]) # find 'K' equivalent of linspace
        cut_points_value = tf.contrib.framework.sort(self.cut_points_list[feature_num - 1])  # change tf
        b = K.cumsum(tf.concat([K.constant(0.0, shape=[1]), -cut_points_value], 0))
        h = tf.matmul(X, W) + b
        output = tf.nn.softmax(h / temperature)
        return output

    def call(self, x):
      
      def kron_prod(a,b):
          import tensorflow as tf
          res = tf.einsum('ij,ik->ijk', a, b)
          res = tf.reshape(res, [-1, tf.reduce_prod(res.shape[1:])])
          return res
      
      binning_layer_list = []
      for i, value in enumerate(self.num_cut):
        binning_layer_list.append(
            self.bin_fn(feature_num=i+1, num_cut_value=value, x=x)
        )

#       output = KronProd(cut_points_list=self.num_cut)(binning_layer_list) # Remove layer, replace as logic
      print(len(binning_layer_list))
      output = reduce(kron_prod, binning_layer_list)
      self.output_dim = output.get_shape().as_list()

      return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim[1])