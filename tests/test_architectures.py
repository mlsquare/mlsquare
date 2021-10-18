import pytest

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.utils import to_categorical

from mlsquare.base import registry
from mlsquare.adapters import SklearnKerasClassifier
from mlsquare.architectures.sklearn import GeneralizedLinearModel, KernelGeneralizedLinearModel, CART
from datasets import _load_diabetes, _load_iris, _load_boston, _load_simIrt, _load_1PL_IrtData
from sklearn.decomposition import TruncatedSVD
import tensorflow as tf
from mlsquare.models.embibe import rasch

def _load_irt_data():
    col_name = ['question_code', 'user_id', 'difficulty', 'ability', 'response']
    X,y =_load_simIrt()
    xtrain, xtest, ytrain, ytest= train_test_split(X, y, test_size=0.5, random_state=0)
    x_train_u= xtrain[[col_name[1]]]
    x_train_q= xtrain[[col_name[0]]]
    y_train = ytrain

    x_test_u= xtest[[col_name[1]]]
    x_test_q= xtest[[col_name[0]]]
    pij_true= xtest[[col_name[-1]]]
    t_abilities= sorted({_.user_id:_.ability for _ in X.itertuples(index=True)}.items())
    t_abilities= np.array(list(dict(t_abilities).values()))
    return x_train_u, x_train_q, y_train, x_test_u, x_test_q, pij_true, t_abilities

def _load_1PL_data_params_combinations():
    xtrain, y_train, x_train_user, x_train_questions = _load_1PL_IrtData()

    layer_name= ['latent_trait/ability','difficulty_level', 'disc_param', 'guessing_param', 'slip_param']
    model_param_keys= ['ability_params', 'diff_params', 'disc_params', 'guess_params', 'slip_params']
    params_layer_dict = dict(zip(model_param_keys,layer_name))

    x= np.zeros((32,5))#Placeholder for all possible layer combinations
    for comb in range(x.shape[0]):
        rep = np.array(list(np.binary_repr(comb)), dtype= np.int8)#binary analog
        np.copyto(x[comb][: rep.shape[0]], rep)

    df =pd.DataFrame(x, columns= model_param_keys)
    for col in df.columns:
        df[col]= df[col].apply(lambda x: col if x==1 else 0)
    df.drop_duplicates()

    return xtrain, y_train, x_train_user, x_train_questions, df, params_layer_dict

def _load_decomposition_data():
    X, Y = _load_boston()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=0)
    return x_train

def _load_regression_data():
    X, Y = _load_diabetes()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=0)
    return x_train, x_test, y_train, y_test

def _load_classification_data():
    X, Y = _load_iris()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=0)
    return x_train, x_test, y_train, y_test

def _mock_dope(primal_model):
    model_skeleton, adapt = registry[('sklearn', primal_model.__class__.__name__)]['default']
    final_model = adapt(model_skeleton, primal_model)
    return final_model

def _predict_primal_and_proxy(proxy, primal, x_test):
    pred1 = proxy.predict(x_test)
    pred2 = primal.predict(x_test)
    return pred1, pred2

def _run_ttest(primal_model_class, epochs):
    x_train, x_test, y_train, _ = _load_regression_data()
    primal_model = primal_model_class()
    proxy_model = _mock_dope(primal_model)
    proxy_model.fit(x_train,y_train, epochs=epochs)

    primal_model.fit(x_train, y_train)
    pred1, pred2 = _predict_primal_and_proxy(proxy_model, primal_model, x_test)

    _, p_value = stats.ttest_rel(pred2,pred1)
    return p_value

def _run_chisquare(primal_model_class, epochs, cuts_per_feature=None):
    x_train, x_test, y_train, _ = _load_classification_data()
    primal_model = primal_model_class()
    if isinstance(primal_model, DecisionTreeClassifier):
        y_train = to_categorical(y_train)
    proxy_model = _mock_dope(primal_model)
    proxy_model.fit(x_train,y_train, epochs=epochs, cuts_per_feature=cuts_per_feature)

    primal_model.fit(x_train, y_train)
    pred1, pred2 = _predict_primal_and_proxy(proxy_model, primal_model, x_test)
    if isinstance(primal_model, DecisionTreeClassifier):
        pred2 = np.argmax(pred2, axis=1)
    pred1 = pred1.reshape(pred2.shape)
    pred2 += 1
    pred1 += 1
    _, p_value = stats.chisquare(pred2,pred1)
    return p_value

def _prepare_mock_model(parent_class, adapter, module_name, model_name, version, model_params, primal_model=None):
    class MockModel(parent_class):
        def __init__(self):
            self.cuts_per_feature = None
            self.adapter = adapter
            self.module_name = module_name
            self.name = model_name
            self.version = version

            self.set_params(params=model_params, set_by='model_init')

    mock_model = MockModel()
    mock_model.y = np.array([[0,1], [0,1]])
    mock_model.X = np.array([[0,1], [0,1]])
    if isinstance(primal_model, DecisionTreeClassifier):
        x_train, _, y_train, _ = _load_classification_data()
        y_train = to_categorical(y_train)
        mock_model.primal = DecisionTreeClassifier().fit(x_train, y_train)
    mock_proxy_model = mock_model.create_model()
    return mock_proxy_model

def _run_decomposition_test(primal_model_class, num_components):
    X = _load_decomposition_data()
    primal_model = primal_model_class(n_components= num_components)
    proxy_model = _mock_dope(primal_model)
    sess= tf.Session()

    tf_trans_x = proxy_model.fit_transform(X)
    tf_sigma = proxy_model.singular_values_
    tf_U = tf_trans_x/tf_sigma
    tf_V= proxy_model.components_
    tf_approx_recon =  sess.run(tf.matmul(tf_U, tf.matmul(tf.linalg.diag(tf_sigma), tf_V)))#, adjoint_b=True) v.T in arch-line#161
    #tf_approx_recon= np.dot(tf_U, np.dot(np.diag(tf_sigma), tf_V))#Since tf_U/tf_V are not tensors anymore

    skl_trans_x = primal_model.fit_transform(X)
    skl_sigma = primal_model.singular_values_
    skl_U = skl_trans_x/skl_sigma
    skl_V= primal_model.components_
    skl_approx_recon = np.dot(skl_U, np.dot(np.diag(skl_sigma), skl_V))

    result = np.allclose(tf_approx_recon, skl_approx_recon)
    _, p_value = stats.ttest_rel(tf_sigma, skl_sigma)

    return result, p_value

def _irt_1PL_bias_update_test(primal_model_class):
    xtrain, y_train, x_train_user, x_train_questions, df, params_layer_dict = _load_1PL_data_params_combinations()
    primal_model = primal_model_class()
    model_skeleton, adapt = registry[('mlsquare', primal_model.__class__.__name__)]['default']

    results= []
    input_bias_values= []
    obtained_bias_values=[]
    for val in df.values[:8]:
        layer_keys = val[np.where(val!=0)]
        random_bias = np.random.randint(2,7, len(layer_keys))
        input_bias_values.append(random_bias.tolist())
        bias_list = [{'bias_param':v} for v in random_bias]
        params_ = dict(zip(layer_keys, bias_list))

        proxy_model = adapt(model_skeleton, primal_model)
        proxy_model.fit(x_user= x_train_user, x_questions= x_train_questions, y_vals= y_train, batch_size= 30, epochs=1, params=params_)

        for k1, v1 in params_layer_dict.items():
            for idx, layer in enumerate(proxy_model.model.layers):
                if layer.name==v1:
                    params_layer_dict[k1]= idx
        res= []
        obtained_bias_=[]
        for keys, vals in params_.items():
            obtained_bias = proxy_model.model.layers[params_layer_dict[keys]].get_config()['bias_initializer']['config']['value']
            res.append(vals['bias_param']==obtained_bias)
            obtained_bias_.append(obtained_bias)

        results.append(res)
        obtained_bias_values.append(obtained_bias_)
    return input_bias_values, obtained_bias_values, results


def _run_irt_ttest(primal_model_class, epochs):
    x_user, x_question, y, x_test_user, x_test_quest, pij_true, true_abiltities = _load_irt_data()
    primal_model = primal_model_class()
    model_skeleton, adapt = registry[('mlsquare', primal_model.__class__.__name__)]['default']
    proxy_model = adapt(model_skeleton, primal_model)

    x_user = to_categorical(x_user.values, num_classes =x_user.nunique()[0])
    x_quest = to_categorical(x_question.values, num_classes =x_question.nunique()[0])
    proxy_model.fit(x_user= x_user, x_questions= x_quest, y_vals= y, batch_size=64, epochs= epochs)

    x_test_user = to_categorical(x_test_user.values, num_classes =x_test_user.nunique()[0])
    x_test_quest = to_categorical(x_test_quest.values, num_classes =x_test_quest.nunique()[0])
    pij_est = proxy_model.predict(x_test_user, x_test_quest)

    pij_est= pij_est.reshape(-1)
    pij_true = pij_true.values.reshape(-1)
    est_abilities= [layer.get_weights()[0] for layer in proxy_model.model.layers if layer.name== 'latent_trait/ability']
    est_abilities= est_abilities[0].reshape(-1)

    _, p_val_abl= stats.ttest_rel(est_abilities, true_abiltities)
    _, p_value_pred = stats.ttest_rel(pij_est, pij_true)#est vs. true
    _, p_value_dist = stats.kstest(est_abilities, 'norm')#Kolmogorov-Smirnov

    return p_val_abl, p_value_pred, p_value_dist

def test_1PL_bias_updation_functionality():
    input_b, obtained_b, comp_results = _irt_1PL_bias_update_test(rasch)
    #print('\ninputs:',input_b,'\noutputs:',obtained_b)
    assert input_b==obtained_b

def test_svd_reconstruction():
    result, _ = _run_decomposition_test(TruncatedSVD, 10)
    assert result is True

def test_svd_sigma_vals():
    _, p_value = _run_decomposition_test(TruncatedSVD, 10)
    assert p_value > 1e-01

@pytest.mark.xfail()
def test_irt_ability_dist_prediction_abilities():
    pval_abl, pval_pred, pval = _run_irt_ttest(rasch, 300)
    #_ , _, pval = _run_irt_ttest(rasch, 600)
    dist_flag= pval>0.05
    pred_flag= pval_pred<0.1
    abl_comp_flag= pval_abl<0.1
    flag_msg_di = dict(zip(['dist_flag', 'pred_flag', 'abl_comp_flag'], [
        'Abilities are NOT distributed normally.',
        'True Vs. Estimated predictions differ.',
        'True Vs. Estimated abilities differ.']))
    for flag, message in flag_msg_di.items():
        assert eval(flag), message
    #assert pval>0.05, "abilities are NOT distributed normally."

#@pytest.mark.xfail()
#def test_irt_prediction_abilities():
#    pval_abl, pval_pred, _ = _run_irt_ttest(rasch, 600)
#    assert pval_pred<0.1
#    assert pval_abl<0.1

@pytest.mark.xfail()
def test_linear_regression_ttest():
    p_value = _run_ttest(LinearRegression, 300)
    assert p_value[0] < 0.1

@pytest.mark.xfail()
def test_logistic_regression_chisquare():
    p_value = _run_chisquare(LogisticRegression, 300)
    assert p_value > 0.8

@pytest.mark.xfail()
def test_ridge_ttest():
    p_value = _run_ttest(Ridge, 300)
    assert p_value[0] < 0.1

@pytest.mark.xfail()
def test_lasso_ttest():
    p_value = _run_ttest(Lasso, 300)
    assert p_value[0] < 0.1


@pytest.mark.xfail()
def test_elasticnet_ttest():
    p_value = _run_ttest(ElasticNet, 300)
    assert p_value[0] < 0.1

@pytest.mark.xfail()
def test_linear_svc_chisquare():
    p_value = _run_chisquare(LinearSVC, 300)
    assert p_value > 0.8

@pytest.mark.xfail()
def test_svc_chisquare():
    p_value = _run_chisquare(SVC, 300)
    assert p_value > 0.8

@pytest.mark.xfail()
def test_decision_trees_using_chisquare():
    p_value = _run_chisquare(DecisionTreeClassifier, 300)
    assert p_value > 0.8

@pytest.mark.xfail()
def test_decision_trees_with_cuts_using_chisquare():
    p_value = _run_chisquare(DecisionTreeClassifier, 300, cuts_per_feature=2)
    assert p_value > 0.8

def test_glm_create_model_method():
    model_params = {'layer_1': {'l1': 0,
                                'l2': 0,
                                'activation': 'sigmoid'},
                    'optimizer': 'adam',
                    'loss': 'binary_crossentropy'
                    }
    mock_proxy_model = _prepare_mock_model(GeneralizedLinearModel, SklearnKerasClassifier,
                                           'sklearn', 'LogisticRegression',
                                           'default', model_params)

    assert mock_proxy_model.get_config()['layers'][0]['config']['activation'] == 'sigmoid'
    assert mock_proxy_model.get_config()['layers'][0]['config']['units'] == 2
    assert mock_proxy_model.optimizer.__class__.__name__ == 'Adam'
    assert mock_proxy_model.loss == 'binary_crossentropy'


def test_kernel_glm_create_model_method():
    model_params = {'layer_1': {'kernel_dim': 10,
                                'activation': 'linear'
                                },
                    'layer_2': {
                                'activation': 'softmax'
                                },
                    'optimizer': 'adam',
                    'loss': 'categorical_hinge'}
    mock_proxy_model = _prepare_mock_model(KernelGeneralizedLinearModel, SklearnKerasClassifier,
                                           'sklearn', 'LogisticRegression',
                                           'default', model_params)

    assert mock_proxy_model.get_config()['layers'][0]['config']['activation'] == 'linear'
    assert mock_proxy_model.get_config()['layers'][0]['config']['units'] == 10
    assert mock_proxy_model.optimizer.__class__.__name__ == 'Adam'
    assert mock_proxy_model.loss == 'categorical_hinge'

def test_cart_create_model_method():
    model_params = {
        'layer_3': {'activation': 'sigmoid'},
        'optimizer': 'adam',
        'loss': 'categorical_crossentropy'
    }

    mock_proxy_model = _prepare_mock_model(CART, SklearnKerasClassifier,
                                           'sklearn', 'LogisticRegression',
                                           'default', model_params, primal_model=DecisionTreeClassifier())

    assert mock_proxy_model.get_config()['layers'][2]['config']['activation'] == 'sigmoid'
    assert mock_proxy_model.optimizer.__class__.__name__ == 'Adam'
    assert mock_proxy_model.loss == 'categorical_crossentropy'

def test_linear_svc_transform_data():
    # Pending
    pass
