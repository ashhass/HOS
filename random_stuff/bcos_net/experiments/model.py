import os

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

# from experiment_utils import get_arguments
from experiments.utils import FinalLayer, MyAdaptiveAvgPool2d
from experiments.tr_utils import start_training
from models.unet import Unet

archs = {
    "unet": Unet,
             }


def load_pretrained(exp_params, network):
    model_path = os.path.join("bcos_pretrained", exp_params["exp_name"])
    model_file = os.path.join(model_path, "state_dict.pkl")

    # if not os.path.exists(model_file):
    #     if not os.path.exists(model_path):
    #         os.makedirs(model_path)
    #     download_url_to_file(exp_params["model_url"], model_file)
    loaded_state_dict = torch.load(model_file, map_location="cpu")
    network.load_state_dict(loaded_state_dict)


def get_model(exp_params):
    # if exp_params.get("pretrained", False):
    #     return get_pretrained_model(exp_params)
    logit_bias = exp_params["logit_bias"]
    logit_temperature = exp_params["logit_temperature"]
    network = archs[exp_params["network"]]
    network_opts = exp_params["network_opts"]
    network_list = [network(**network_opts)]

    network_list += [
        MyAdaptiveAvgPool2d((1, 1)),
        FinalLayer(bias=logit_bias, norm=logit_temperature)
    ]
    network = nn.Sequential(*network_list)
    if exp_params["load_pretrained"]:
        load_pretrained(exp_params, network)
    network.opti = exp_params["opti"]
    network.opti_opts = exp_params["opti_opts"]
    return network


def get_optimizer(model, base_lr):
    optimisers = {"Adam": torch.optim.Adam,
                  "AdamW": torch.optim.AdamW,
                  "SGD": torch.optim.SGD}
    the_model = model if not isinstance(model, (nn.DataParallel, DistributedDataParallel)) else model.module
    opt = optimisers[the_model.opti]
    opti_opts = the_model.opti_opts
    opti_opts.update({"lr": base_lr})
    opt = opt(the_model.parameters(), **opti_opts)
    opt.base_lr = base_lr
    return opt


# if __name__ == "__main__":
#     cmd_args = get_arguments()
#     start_training(cmd_args, get_model, get_optimizer)