from ray import tune
import ray
# from ray.tune.suggest import HyperOptSearch
import os
import numpy as np
from ..utils.functions import _get_model_name
import multiprocessing

## Push this as a class with the package name. Ex - class tune(): pass
def get_best_model(X, y, proxy_model, primal_data, primal_model, **kwargs):
    _num_cpus = multiprocessing.cpu_count()

    if _num_cpus > 2:
        _num_cpus = _num_cpus / 2
    _model_name = _get_model_name(primal_model)

    if _model_name in ['LogisticRegression', 'LinearDiscriminantAnalysis']:
        _measure_metric = 'accuracy'
        _ray_stop_criterion = {"mean_accuracy": 99}
    else:
        _measure_metric = 'mean_squared_error'
        _ray_stop_criterion = {"mean_accuracy": 1}

    # Initialize ray
    _local_ray_init = False
    if not ray.is_initialized():
        try:
            ray.init()
            _local_ray_init = True
        except RuntimeError as e:
            print("Couldn't initialize `ray` automatically. Please initialize `ray` using `ray.init()` to search for the best model", e)
            return proxy_model.primal_model


    y_pred = np.array(primal_data['y_pred'])
    kwargs.setdefault('epochs', 250)
    kwargs.setdefault('batch_size', 40)
    kwargs.setdefault('verbose', 1)

    def train_model(config, reporter): ## Change config name
        '''
        This function is used by Tune to train the model with each iteration variations.

        Args:
            config(dict): A dictionary with the search params passed by Tune.
            Similar to the JSON we already have.
            reporter: A function used by Tune to keep a track of the metric by
            which the iterations should be optimized.
        '''
        proxy_model.set_params(params=config, set_by='optimizer')
        model = proxy_model.create_model(metric=_measure_metric)
        model.fit(X, y_pred, epochs=kwargs['epochs'], batch_size=kwargs['batch_size'], verbose=kwargs['verbose'])
        accuracy = model.evaluate(X, y_pred)[1]
        last_checkpoint = "weights_tune_{}.h5".format(config)
        model.save_weights(last_checkpoint)
        reporter(mean_accuracy=accuracy, checkpoint=last_checkpoint)

    # Define experiment configuration
    configuration = tune.Experiment("mlsquare_dope",
                                    run=train_model,
                                    resources_per_trial={"cpu": _num_cpus},
                                    stop=_ray_stop_criterion,
                                    config=proxy_model.get_params())

    trials = tune.run_experiments(configuration, verbose=2)

    # Restore a model from the best trial.
    sorted_trials = get_sorted_trials(trials, "mean_accuracy")

    for best_trial in sorted_trials:
        try:
            print("Creating model...")
            proxy_model.set_params(params=best_trial.config, set_by='optimizer')
            best_model = proxy_model.create_model(metric=_measure_metric)
            weights = os.path.join(
                best_trial.logdir, best_trial.last_result["checkpoint"])
            print("Loading from", weights)
            # TODO Validate this loaded model.
            best_model.load_weights(weights)
            break
        except Exception as e:
            print(e)
            print("Loading failed. Trying next model")

    if _local_ray_init:
        ray.shutdown()
    return best_model


# Utils from Tune tutorials(Not a part of the Tune package) #

def get_sorted_trials(trial_list, metric):
    return sorted(trial_list, key=lambda trial: trial.last_result.get(metric, 0), reverse=True)

# TODO
# Generalize metric choice.
# Add compatibility for linReg and LDA.
# Validate the loaded model(How?).


        # model_params = { 'units': 1,
        #                 'input_dim': 2,
        #                 'activation': ['sigmoid', 'linear'],
        #                 'optimizer': 'adam',
        #                 'loss': 'binary_crossentropy'
        #                 }

# estimation params
# model params
