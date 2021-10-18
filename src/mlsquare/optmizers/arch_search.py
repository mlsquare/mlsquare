import warnings
import numpy as np
import logging
import os
import ray
from ray import tune
import pandas as pd
from dict_deep import *
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp
import copy
from collections import defaultdict

warnings.filterwarnings("ignore")
import time

def get_opt_model(x_user, x_questions, y_vals, proxy_model, **kwargs):
    kwargs.setdefault('nas_params', None)
    kwargs.setdefault('batch_size', 16)
    kwargs.setdefault('epochs', 64)
    kwargs.setdefault('validation_split', 0.2)
    nas_params= kwargs['nas_params']
    def_nas_params = proxy_model.get_params().get('model_nas_params')

    dict_ = defaultdict(dict, {'search_algo_name':def_nas_params.get('search_algo_name'),
                            'nas_params_':None, 'algo':None,'sch':None, 'union':True,
                            'config':{},'search_space':def_nas_params.get('search_space')})
    dict_.update(kwargs)

    if nas_params!=None:
        if not isinstance(nas_params,dict) and isinstance(nas_params['search_space'], dict):
            raise TypeError("nas_params and underlying key search_space should be of type 'dict'")
        dict_['search_space'].update(nas_params['search_space'])
        if 'union' in nas_params and nas_params['union'] is not dict_['union']:
            dict_['search_space']=nas_params['search_space']
        dict_['search_algo_name']= nas_params['search_algo_name']

    def_nas_dict={'search_methods':
                {
                    'hyperOpt':{
                        'algo':HyperOptSearch(dict_['search_space'], reward_attr="mean_error", max_concurrent=4),
                        'scheduler':AsyncHyperBandScheduler(reward_attr="mean_error"),
                        'config':{}
                                }
                }
            }
    dict_['nas_params_']= def_nas_dict['search_methods'][dict_.get('search_algo_name')]
    dict_['algo']= dict_['nas_params_']['algo']
    dict_['sch'] = dict_['nas_params_']['scheduler']
    dict_['cfg']= dict_['nas_params_']['config']
    ray_verbose = False
    _ray_log_level = logging.INFO if ray_verbose else logging.ERROR
    _local_ray_init = False
    if not ray.is_initialized():
        try:
            ray.init()
            _local_ray_init = True
        except RuntimeError as e:
            print("Couldn't initialize `ray` automatically. Please initialize `ray` using `ray.init()` to search for the best model", e)
            return proxy_model.primal_model, 0, 0


    def process_config(config):
        i=0
        config_params= copy.deepcopy(proxy_model.get_params())
        reg_dict = defaultdict(lambda: {'l1':0,'l2':0})
        for key, vals in config.items():
            i+=1
            key_list= key.split('.')
            if 'regularizers' in key_list:
                known= key_list.pop(-1)#l1 OR l2
                reg_dict[tuple(key_list)].update({known:vals})
                vals = reg_dict[tuple(key_list)]
            if 'group_lasso' in key_list:
                known= key_list.pop(-1)#l1 OR l2
                reg_dict[tuple(key_list)].update({known:vals})
                vals = reg_dict[tuple(key_list)]
            print('\nconfig val {}:\n key_list: {};\n values: {}\n\n'.format(i, key_list, vals))
            deep_set(config_params, key_list, vals)#set l1 &l2 both at once
        return config_params

    def train_model(config, reporter):
        updated_params= process_config(config)
        proxy_model.set_params(params=updated_params, set_by='optimizer')
        print('\nIntitializing fit for {} model. . .\nBatch_size: {}; epochs: {};'.format(
            proxy_model.name, kwargs['batch_size'], kwargs['epochs']))

        model = proxy_model.create_model()
        history = model.fit(x=[x_user, x_questions], y=y_vals, batch_size=dict_['batch_size'],
                                    epochs=dict_['epochs'], verbose=0, validation_split=dict_['validation_split'])
        loss, mae, accuracy = model.evaluate(x=[x_user, x_questions], y=y_vals)
        last_checkpoint="weights_tune_{}.h5".format(list(zip(np.random.choice(10, len(config), replace=False), config)))
        model.save_weights(last_checkpoint)
        reporter(mean_error=loss, mean_accuracy=accuracy,
                    checkpoint=last_checkpoint)

    t1 = time.time()
    trials= tune.run(train_model, name= "{}_optimization".format(dict_['search_algo_name']),
                                    resources_per_trial={"cpu": 4},
                                    stop={"mean_error": 0},
                                    num_samples=dict_['num_samples'],
                                    scheduler=dict_['sch'], search_alg= dict_['algo'], config= dict_['cfg'])

    def get_sorted_trials(trial_list, metric):
        return sorted(trial_list, key=lambda trial: trial.last_result.get(metric, 0))

    metric = "mean_error"
    sorted_trials = get_sorted_trials(trials, metric)

    for best_trial in sorted_trials:
        try:
            print("Creating model...")
            best_params= process_config(best_trial.config)
            proxy_model.set_params(params=best_params, set_by='optimizer')
            best_model = proxy_model.create_model()
            weights = os.path.join(
                best_trial.logdir, best_trial.last_result["checkpoint"])
            print("Loading from", weights)
            best_model.load_weights(weights)

            #loss, mae, accuracy = best_model.evaluate(x=[x_user, x_questions], y=y_vals)
            #los_li = []
            #for l in best_model.losses:
            #    #print(l)
            #    if l!=0:
            #        los_li.append(K.eval(l))
            #los_li= sum(los_li)
            #print('\nFinal Reported total loss: {} and cumulative regualrizer+ network loss:{}'.format(loss,los_li))
            break
        except Exception as e:
            print(e)
            print("Loading failed. Trying next model")
    exe_time = time.time()-t1

    if _local_ray_init:
        ray.shutdown()
    return best_model, trials, exe_time
