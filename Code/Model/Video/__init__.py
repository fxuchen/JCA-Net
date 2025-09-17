from .videomodels.resnet import ResNet, BasicBlock
from .videomodels.shufflenetv2 import ShuffleNetV2
from .videomodels.autoencoder_videomodel import AEVideoModel
from .videomodels.frcnn_videomodel import FRCNNVideoModel, update_frcnn_parameter

__all__ = [
    "ResNet",
    "BasicBlock",
    "ShuffleNetV2",
    "AEVideoModel",
    "FRCNNVideoModel",
    "update_frcnn_parameter",
]


def register_model(custom_model):

    if custom_model.__name__ in globals().keys() or custom_model.__name__.lower() in globals().keys():
        raise ValueError(f"Model {custom_model.__name__} already exists. Choose another name.")
    globals().update({custom_model.__name__: custom_model})


def get(identifier):
    
    if isinstance(identifier, str):
        to_get = {k.lower(): v for k, v in globals().items()}
        cls = to_get.get(identifier.lower())
        if cls is None:
            raise ValueError(f"Could not interpret model name : {str(identifier)}")
        return cls
    raise ValueError(f"Could not interpret model name : {str(identifier)}")
