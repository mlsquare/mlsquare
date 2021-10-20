import theano.tensor as T
import theano
import numpy as np
from theano.compile.ops import as_op
# from theano.compile.nanguardmode import NanGuardMode
# from theano.compile.debugmode import DebugMode


@as_op(itypes=[theano.tensor.ivector], otypes=[theano.tensor.ivector])
def numpy_unique(a):
    return np.unique(a)


def lda_theano_loss(n_components, margin):
    """
    The main loss function (inner_lda_objective) is wrapped in this function due to
    the constraints imposed by Keras on objective functions
    """
    def inner_lda_objective(y_true, y_pred):
        """
        It is the loss function of LDA as introduced in the original paper.
        It is adopted from the the original implementation in the following link:
        https://github.com/CPJKU/deep_lda
        Note: it is implemented by Theano tensor operations, and does not work on Tensorflow backend
        """
        r = 1e-4

        # init groups
        yt = T.cast(y_true.flatten(), "int32")
        groups = numpy_unique(yt)

        def compute_cov(group, Xt, yt):
            Xgt = Xt[T.eq(yt, group).nonzero()[0], :]
            Xgt_bar = Xgt - T.mean(Xgt, axis=0)
            m = T.cast(Xgt_bar.shape[0], 'float32')
            return (1.0 / (m - 1)) * T.dot(Xgt_bar.T, Xgt_bar)

        # scan over groups
        covs_t, _ = theano.scan(fn=compute_cov, outputs_info=None,
                                      sequences=[groups], non_sequences=[y_pred, yt],
                                    #   mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True),
                                      mode='DebugMode'
                                      )

        # compute average covariance matrix (within scatter)
        Sw_t = T.mean(covs_t, axis=0)

        # compute total scatter
        Xt_bar = y_pred - T.mean(y_pred, axis=0)
        m = T.cast(Xt_bar.shape[0], 'float32')
        St_t = (1.0 / (m - 1)) * T.dot(Xt_bar.T, Xt_bar)

        # compute between scatter
        Sb_t = St_t - Sw_t

        # cope for numerical instability (regularize)
        Sw_t += T.identity_like(Sw_t) * r

        # return T.cast(T.neq(yt[0], -1), 'float32')*T.nlinalg.trace(T.dot(T.nlinalg.matrix_inverse(St_t), Sb_t))

        # compute eigenvalues
        evals_t = T.slinalg.eigvalsh(Sb_t, Sw_t)

        # get eigenvalues
        top_k_evals = evals_t[-n_components:]

        # maximize variance between classes
        # (k smallest eigenvalues below threshold)
        thresh = T.min(top_k_evals) + margin
        top_k_evals = top_k_evals[(top_k_evals <= thresh).nonzero()]
        costs = T.mean(top_k_evals)

        return -costs

    return inner_lda_objective


def mse_in_theano(y_true, y_pred):
    return T.mean(T.square(y_pred - y_true), axis=-1)


def quantile_loss(quantile=0.5):
    def loss(y_true, y_pred,quantile=quantile):
        from tensorflow.python.keras import backend as K
        e = y_pred-y_true
        Ie = (K.sign(e)+1)/2
        return K.mean(e*(Ie-quantile),axis=-1)
    return loss

# https://stats.stackexchange.com/questions/249874/the-issue-of-quantile-curves-crossing-each-other
def quantile_ensemble_loss(quantile=0.5,margin=0,alpha=0):

    def loss(y_true, y_pred, q=quantile,margin=margin,alpha=alpha):
        from tensorflow.python.keras import backend as K
        error = y_true - y_pred
        quantile_loss = K.mean(K.maximum(q*error, (q-1)*error))
        diff = y_pred[:, 1:] - y_pred[:, :-1]
        penalty = K.mean(K.maximum(0.0, margin - diff)) * alpha
        return quantile_loss + penalty
    return loss

def ordinal_loss(margin=0,alpha=0):

    def loss(y_true,y_pred, margin=margin,alpha=alpha):
        from tensorflow.python.keras import backend as K
        diff = y_pred[:, 1:] - y_pred[:, :-1]
        penalty = K.mean(K.maximum(0.0, margin - diff)) * alpha
        return penalty
    return loss

def lda_loss(n_components, margin=0,method='raleigh_coeff'):
    """
    The main loss function (inner_lda_objective) is wrapped in this function due to
    the constraints imposed by Keras on objective functions
    deep LDA paper: https://arxiv.org/pdf/1704.06305.pdf
    """
    # sourced from https://github.com/tchaton/DeepLDA
    # and which inturn is inspired from https://github.com/VahidooX/DeepLDA.
    # refs
    # https://arxiv.org/pdf/1903.11240.pdf
    # https://arxiv.org/pdf/1906.02590.pdf
    # https://papers.nips.cc/paper/1210-self-organizing-and-adaptive-algorithms-for-generalized-eigen-decomposition.pdf
    def inner_lda_objective(y_true, y_pred):
        """
        It is the loss function of LDA as introduced in the original paper.
        It is adopted from the the original implementation in the following link:
        https://github.com/CPJKU/deep_lda
        Note: it is implemented by Theano tensor operations, and does not work on Tensorflow backend
        """
        r = 1e-4
        locations = tf.where(tf.equal(y_true, 1))
        indices = locations[:, 1]
        y, idx = tf.unique(indices)


        def fn(unique, indexes, preds):
            u_indexes = tf.where(tf.equal(unique, indexes))
            u_indexes = tf.reshape(u_indexes, (1, -1))
            X = tf.gather(preds, u_indexes)
            X_mean = X - tf.reduce_mean(X, axis=0)
            m = tf.cast(tf.shape(X_mean)[1], tf.float32)
            return (1 / (m - 1)) * tf.matmul(tf.transpose(X_mean[0]), X_mean[0])

        # scan over groups
        covs_t = tf.map_fn(lambda x: fn(x, indices, y_pred), y, dtype=tf.float32)

        # compute average covariance matrix (within scatter)
        Sw_t = tf.reduce_mean(covs_t, axis=0)

        # compute total scatter
        Xt_bar = y_pred - tf.reduce_mean(y_pred, axis=0)
        m = tf.cast(tf.shape(Xt_bar)[1], tf.float32)
        St_t = (1 / (m - 1)) * tf.matmul(tf.transpose(Xt_bar), Xt_bar)

        # compute between scatter
        dim = tf.shape(y)[0]
        Sb_t = St_t - Sw_t

        # cope for numerical instability (regularize)
        Sw_t += tf.eye(dim) * r

        ''' START : COMPLICATED PART WHERE TENSORFLOW HAS TROUBLE'''
        #cho = tf.eye(dim)
        # look at page 383
        # http://perso.ens-lyon.fr/patrick.flandrin/LedoitWolf_JMA2004.pdf

        if method == 'raleigh_coeff':
            # minimize the -ve of Raleigh coefficient
            r = 1e-3
            cho = tf.cholesky(St_t + tf.eye(dim) * r)
            inv_cho = tf.matrix_inverse(cho)
            evals_t = tf.linalg.eigvalsh(tf.transpose(inv_cho) * Sb_t * inv_cho)  # Sb_t, St_t # SIMPLIFICATION OF THE EQP USING cholesky
            top_k_evals = evals_t[-n_components:]

            index_min = tf.argmin(top_k_evals, 0)
            thresh_min = top_k_evals[index_min] + margin
            mask_min = top_k_evals < thresh_min
            cost_min = tf.boolean_mask(top_k_evals, mask_min)
            cost -tf.reduce_mean(cost_min)
            return
        elif method == 'trace':
            # maximize the trace(Sw^-1 * Sb)
            cost = -tf.linalg.trace(tf.matrix_inv(Sw_t)*Sb_t)

        elif method == 'det_ratio':
            # maximze the ratio of det of between/ within scatter
            cost = -tf.math.divide(tf.linalg.det(Sb_t),tf.linalg.det(Sw_t))


        elif method == 'trace_ratio':
            # minimize the -ve of ratio of trace of betwwen to witin scatter
            cost = -tf.math.divide(tf.linalg.trace(Sb_t),tf.linalg.trace(Sw_t))
        elif method == 'trace_diff':
            # minimize with variation, maximze between variation
            cost = tf.linalg.trace(Sw_t)-tf.linalg.trace(Sb_t)
        else:
            # minimize within variation
            cost = tf.linalg.trace(Sw_t)

        return cost

    return inner_lda_objective