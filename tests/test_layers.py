from mlsquare.layers import Bin, KronProd
import numpy as np
from keras.layers import Input
from keras.models import Model

def test_bin_layer():
    visible = Input(shape=(4,))
    output = Bin(index_of_feature=1, num_of_cuts=2)(visible)
    model = Model(inputs=visible, outputs=output)

    model.compile(optimizer='adam', loss='mse')
    model.fit(x=np.random.random((5,4)), y=np.random.random((5,3)))
    pred = model.predict(np.random.random((5,4)))
    assert pred.shape == (5,3)
