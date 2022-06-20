import tensorflow as tf
from tensorflow.python.keras.initializers import glorot_normal
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
class NoMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)
    def build(self, input_shape):
        super(NoMask, self).build(input_shape)
    def call(self, x, mask=None, **kwargs):
        return x
    def compute_mask(self, inputs, mask):
        return None
def concat_func(inputs, axis=-1, mask=False):
    if not mask:
        inputs = list(map(NoMask(), inputs))
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)
def reduce_mean(input_tensor,
                axis=None,
                keep_dims=False,
                name=None,
                reduction_indices=None):
    try:
        return tf.reduce_mean(input_tensor,
                              axis=axis,
                              keep_dims=keep_dims,
                              name=name,
                              reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_mean(input_tensor,
                              axis=axis,
                              keepdims=keep_dims,
                              name=name)
class SEST(Layer):
    def __init__(self, reduction_ratio=3, seed=1024, **kwargs):
        self.reduction_ratio = reduction_ratio
        self.seed = seed
        super(SEST, self).__init__(**kwargs)
    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `SEST` layer should be called '
                             'on a list of at least 2 inputs')
        self.filed_size = len(input_shape)
        self.embedding_size = input_shape[0][-1]
        reduction_size = max(1, self.filed_size // self.reduction_ratio)
        self.W_1 = self.add_weight(shape=(
            self.filed_size, reduction_size), initializer=glorot_normal(seed=self.seed), name="W_1")
        self.W_2 = self.add_weight(shape=(
            reduction_size, self.filed_size), initializer=glorot_normal(seed=self.seed), name="W_2")
        self.W_3 = self.add_weight(shape=(
            self.filed_size, reduction_size), initializer=glorot_normal(seed=self.seed), name="W_3")
        self.W_4 = self.add_weight(shape=(
            reduction_size, self.filed_size), initializer=glorot_normal(seed=self.seed), name="W_4")
        self.tensordot = tf.keras.layers.Lambda(
            lambda x: tf.tensordot(x[0], x[1], axes=(-1, 0)))
        super(SEST, self).build(input_shape)
    def call(self, inputs, training=None, **kwargs):
        if K.ndim(inputs[0]) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))
        inputs = concat_func(inputs, axis=1)
        Z = reduce_mean(inputs, axis=-1, )
        A_1 = tf.nn.relu(self.tensordot([Z, self.W_1]))
        A_2 = tf.nn.relu(self.tensordot([A_1, self.W_2]))
        V = tf.multiply(inputs, tf.expand_dims(A_2, axis=2))
        inputs_abs = tf.abs(inputs)
        abs_mean=reduce_mean(inputs_abs,axis=-1, )
        A_1_ = tf.nn.relu(self.tensordot([abs_mean, self.W_3]))
        A_2_ = tf.nn.relu(self.tensordot([A_1_, self.W_4]))
        thres = tf.multiply(tf.expand_dims(abs_mean, axis=2), tf.expand_dims(A_2_, axis=2))
        sub = tf.keras.layers.subtract([inputs_abs, thres])
        zeros = tf.keras.layers.subtract([sub, sub])
        n_sub = tf.keras.layers.maximum([sub, zeros])
        R = tf.keras.layers.multiply([tf.sign(inputs), n_sub])
        Result = tf.concat([V, R], -1)
        return tf.split(Result, self.filed_size, axis=1)
    def compute_output_shape(self, input_shape):
        return input_shape
    def compute_mask(self, inputs, mask=None):
        return [None] * self.filed_size
    def get_config(self, ):
        config = {'reduction_ratio': self.reduction_ratio, 'seed': self.seed}
        base_config = super(SEST, self).get_config()
        base_config.update(config)
        return base_config