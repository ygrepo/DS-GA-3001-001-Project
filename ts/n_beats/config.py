from math import sqrt

import torch

from ts.n_beats.model import NBeatsNet


def get_config(interval):
    config = {
        "prod": True,
        "device": ("cuda" if torch.cuda.is_available() else "cpu"),
        "percentile": 50,
        "training_percentile": 45,
        "learning_rate": 1e-3,
        "learning_rates": ((10, 1e-4)),
        "num_of_train_epochs": 15,
        "num_of_categories": 6,  # in data provided
        "batch_size": 1024,
        "gradient_clipping": 20,
        "min_learning_rate": 0.0001,
        "lr_ratio": sqrt(10),
        "lr_tolerance_multip": 1.005,
        "min_epochs_before_changing_lrate": 2,
        "print_train_batch_every": 5,
        "print_output_stats": 3,
        "lr_anneal_rate": 0.5,
        "lr_anneal_step": 5
    }

    if interval == "Quarterly":
        config.update({
            "stack_types": [NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
            "thetas_dims": [7, 8],
            "nb_blocks_per_stack": 3,
            "hidden_layer_units": 128,
            "share_weights_in_stack": True,
            "variable": "Quarterly",
            "seasonality": 4,
            "output_size": 8,
        })
    elif interval == "Monthly":
        config.update({
            #     RUNTIME PARAMETERS
            "stack_types": [NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
            "thetas_dims": [7, 8],
            "nb_blocks_per_stack": 3,
            "hidden_layer_units": 128,
            "share_weights_in_stack": True,
            "chop_val": 72,
            "variable": "Monthly",
            "seasonality": 12,
            "output_size": 18,
        })
    elif interval == "Daily":
        config.update({
            #     RUNTIME PARAMETERS
            "stack_types": [NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
            "thetas_dims": [7, 8],
            "nb_blocks_per_stack": 3,
            "hidden_layer_units": 128,
            "share_weights_in_stack": True,
            "variable": "Daily",
            "seasonality": 7,
            "output_size": 14,
        })
    elif interval == "Yearly":

        config.update({
            #     RUNTIME PARAMETERS
            "stack_types": [NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
            "thetas_dims": [7, 8],
            "nb_blocks_per_stack": 3,
            "hidden_layer_units": 128,
            "share_weights_in_stack": True,
            "variable": "Yearly",
            "seasonality": 1,
            "output_size": 6,
        })
    else:
        print("I dont have that config. :(")

    # config["input_size_i"] = config["input_size"]
    config["output_size_i"] = config["output_size"]
    config["tau"] = config["percentile"] / 100
    config["training_tau"] = config["training_percentile"] / 100

    if not config["prod"]:
        config["batch_size"] = 10
        config["num_of_train_epochs"] = 15

    return config
