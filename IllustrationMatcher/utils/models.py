import argparse
import enum
import os
import json
import torch
from torch import nn


def json2arg(path_json):
    json_dict = None
    with open(path_json) as fstream:
        try:
            json_dict = json.load(fstream)
        except json.decoder.JSONDecodeError as exc:
            print('cannot load file : ' + path_json)
            print(exc)
    return argparse.Namespace(**json_dict)


class ResNetConv4(nn.Module):
    def __init__(self, original_model, path2weight=None):
        super(ResNetConv4, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-3])

        if path2weight is not None:
            model_weights = torch.load(path2weight)
            features_weights = self.features.state_dict()

            # print(self.features.state_dict()['0.weight'][0, 0, ...])
            # print(model_weights['backbone.0.weight'][0, 0, ...])

            for key in features_weights.keys():
                mkey = 'backbone.{}'.format(key)
                assert features_weights[key].shape == model_weights[mkey].shape
                features_weights[key] = model_weights[mkey]

            # print(self.features.state_dict()['0.weight'][0, 0, ...])

            self.features.load_state_dict(features_weights)

            # print(self.features.state_dict()['0.weight'][0, 0, ...])

    def forward(self, x):
        x = self.features(x)
        return x


#
# @param weight can be both path, or pretrained-model
#
def get_conv4_model(weight=None):
    arch, path2weight = None, None
    if weight is None:
        arch = 'resnet50'
        path2weight = None
    elif os.path.exists(weight):
        arch = json2arg(os.path.join(weight, 'args.json')).t_backbone_arch
        path2weight = os.path.join(weight, 'checkpoints', 'last.pt')
    else:
        arch = weight
        path2weight = None

    orig_model = torch.hub.load(
        'pytorch/vision:v0.6.0', arch,
        pretrained=(path2weight is None)
    )

    conv4 = ResNetConv4(orig_model, path2weight)
    return conv4


if __name__ == "__main__":
    model0 = get_conv4_model('resnet18')
