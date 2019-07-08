from ray import tune
import ray
from ray.tune.suggest import HyperOptSearch
import os
import numpy as np
# from ..architectures import ModelMiddleware

# Initialize ray
ray.init(ignore_reinit_error=True, redis_max_memory=20*1000*1000*1000, object_store_memory=1000000000,
         num_cpus=4)

## Push this as a class with the package name. Ex - class tune(): pass
def get_best_model(X, y, abstract_model, primal_data): ## dict:{abstract_model, primal_data, data_covariates}
    # y_pred = primal_data['y_pred']
    y_pred = np.array(primal_data['y_pred'])
    # model_name = kwargs['primal_data']['model_name']
    # build_fn constructed earlier is passed as an argument to avoid recomputation of the same again.
    # mapping_instance = kwargs['build_fn'] # Rename mapping_instance dnn_instance
    # kwargs.setdefault('cuts_per_feature', None)
    # kwargs.setdefault('units', None)
    # cuts_per_feature = kwargs['cuts_per_feature'] 
    # units = kwargs['units']

    # kwargs.setdefault('space', False)

    def train_model(config, reporter): ## Change config name
        '''
        This function is used by Tune to train the model with each iteration variations.

        Args:
            config(dict): A dictionary with the search params passed by Tune.
            Similar to the JSON we already have.
            reporter: A function used by Tune to keep a track of the metric by
            which the iterations should be optimized.
        '''
        ## IMP - the y_train for DT should be actual y_train and not y_pred.
        ## As per current implementation it takes y_pred
        # abstract_model.set_params(config)
        # model = abstract_model.create_model()

        # model = mapping_instance.__call__(x_train=x_train, y_train=y_pred, params=config)
        abstract_model.set_params(config)
        print(type(y_pred))
        print(y_pred.shape)
        print(y_pred)
        abstract_model.update_params({'input_dim': X.shape[1], 'units': 2})
        # , 'units': y_pred.shape[1]})
        # print(y_pred)
        # y_pred = np.array(y_pred)
        model = abstract_model.create_model()
        ## Collect training settings(epochs, batch etc..) at fit level. Attach it to the model.
        ## training_config or settings. Should we pass x and y similarly?
        model.fit(X, y_pred, epochs=250, batch_size=50, verbose=0) # Epochs should be configurable
        accuracy = model.evaluate(X, y_pred)[1] # Cross check - y_train or y_pred?
        last_checkpoint = "weights_tune_{}.h5".format(config)
        model.save_weights(last_checkpoint)
        reporter(mean_accuracy=accuracy, checkpoint=last_checkpoint)

    # Define experiment configuration
    configuration = tune.Experiment("experiment_name",
                                    run=train_model,
                                    resources_per_trial={"cpu": 4},
                                    stop={"mean_accuracy": 95},
                                    config=abstract_model.get_params())
                                    # config=kwargs['params'])

    # This validation is to check if the user has opted for hyperopt search method
    # if kwargs['space']:
    #     print('hyperopt choosen') # Remove or replace with a better message
    #     space = kwargs['space']
    #     hyperopt_search = HyperOptSearch(space, reward_attr="mean_accuracy")
    #     # TODO
    #     # Should this wrapper be avoided(instead the user passes the HyperOptSearch).
    #     # Add other args for hyperopt search.
    #     # Add the remaining search_algos if necessary.
    #     trials = tune.run_experiments(configuration,
    #                                   search_alg=hyperopt_search, verbose=2)

    # else:
    trials = tune.run_experiments(configuration, verbose=2)

    metric = "mean_accuracy"

    """Restore a model from the best trial."""
    sorted_trials = get_sorted_trials(trials, metric)
    for best_trial in sorted_trials:
        try:
            print("Creating model...")
            # best_model = mapping_instance.__call__(x_train=x_train, params=best_trial.config,
            #                                         cuts_per_feature=cuts_per_feature, units=units)

            # best_model = mapping_instance.__call__(x_train=x_train, y_train=y_pred,
            #                                         params=best_trial.config)  # TODO Pass config as argument
            best_trial.config.update({'units': y_pred.shape[1]})
            print(best_trial.config)
            abstract_model.set_params(best_trial.config)
            best_model = abstract_model.create_model()
            # best_model = make_model(None)
            weights = os.path.join(
                best_trial.logdir, best_trial.last_result["checkpoint"])
            print("Loading from", weights)
            # TODO Validate this loaded model.
            best_model.load_weights(weights)
            break ## ??
        except Exception as e:
            print(e)
            print("Loading failed. Trying next model")

    return best_model


# Utils from Tune tutorials(Not a part of the Tune package) #

def get_sorted_trials(trial_list, metric):
    return sorted(trial_list, key=lambda trial: trial.last_result.get(metric, 0), reverse=True)

# TODO
# Looks like the get_sorted_trials function is behaving at times.
# It returns the "worst" model instead of the "best". Check why this happens.
# Validate the loaded model(How?).


        # model_params = { 'units': 1,
        #                 'input_dim': 2,
        #                 'activation': ['sigmoid', 'linear'],
        #                 'optimizer': 'adam',
        #                 'loss': 'binary_crossentropy'
        #                 }

# estimation params
# model params