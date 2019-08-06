from ray import tune
import ray
# from ray.tune.suggest import HyperOptSearch
import os
import numpy as np

# Initialize ray
ray.init(ignore_reinit_error=True, redis_max_memory=20*1000*1000*1000, object_store_memory=1000000000,
         num_cpus=4)

## Push this as a class with the package name. Ex - class tune(): pass
def get_best_model(X, y, proxy_model, primal_data, **kwargs):
    y_pred = np.array(primal_data['y_pred'])
    kwargs.setdefault('epochs', 250)
    kwargs.setdefault('batch_size', 40)
    kwargs.setdefault('verbose', 0)

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
        model = proxy_model.create_model()
        model.fit(X, y_pred, epochs=kwargs['epochs'], batch_size=kwargs['batch_size'], verbose=kwargs['verbose'])
        accuracy = model.evaluate(X, y_pred)[1]
        last_checkpoint = "weights_tune_{}.h5".format(config)
        model.save_weights(last_checkpoint)
        reporter(mean_accuracy=accuracy, checkpoint=last_checkpoint)

    # Define experiment configuration
    configuration = tune.Experiment("experiment_name",
                                    run=train_model,
                                    resources_per_trial={"cpu": 4},
                                    stop={"mean_accuracy": 95},
                                    config=proxy_model.get_params())
                                    # config=kwargs['params'])

    trials = tune.run_experiments(configuration, verbose=2)

    metric = "mean_accuracy"

    # Restore a model from the best trial.
    sorted_trials = get_sorted_trials(trials, metric)
    for best_trial in sorted_trials:
        try:
            print("Creating model...")
            proxy_model.set_params(params=best_trial.config, set_by='optimizer')
            best_model = proxy_model.create_model()
            weights = os.path.join(
                best_trial.logdir, best_trial.last_result["checkpoint"])
            print("Loading from", weights)
            # TODO Validate this loaded model.
            best_model.load_weights(weights)
            break
        except Exception as e:
            print(e)
            print("Loading failed. Trying next model")

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
