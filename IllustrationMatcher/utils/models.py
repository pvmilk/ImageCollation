import enum
import os
import torch
from torch import nn

script_path = os.path.abspath(__file__)


class WeightEnum(str, enum.Enum):
    RESNET18 = 'resnet18'
    RESNET50 = 'resnet50'
    RESNET18FT0 = 'resnet18ft0'
    RESNET18FT1 = 'resnet18ft1'
    RESNET50FT1 = 'resnet50ft1'

    def __str__(self):
        return self.value


weight2tuple = {
    'resnet18' : (None, 'resnet18'),
    'resnet50' : (None, 'resnet50'),
    'resnet18ft0' :
    ('weights/simsiam_20220218-014917/checkpoints/last.pt', 'resnet18'),
    'resnet18ft1' :
    ('weights/simsiam_20220224-160133/checkpoints/last.pt', 'resnet18'),
    'resnet50ft1' :
    ('weights/simsiam_20220224-154343/checkpoints/last.pt', 'resnet50'),
}


class ResNetConv4(nn.Module):
    def __init__(self, original_model, path2weight=None):
        super(ResNetConv4, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-3])

        if path2weight is not None:

            model_path = os.path.join(
                os.path.dirname(script_path),
                path2weight
            )
            model_weights = torch.load(model_path)

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


def get_conv4_model(weight: WeightEnum = 'resnet50'):
    orig_model = torch.hub.load(
        'pytorch/vision:v0.6.0',
        weight2tuple[weight][1],
        pretrained=True
    )

    conv4 = ResNetConv4(orig_model, path2weight=weight2tuple[weight][0])
    return conv4


if __name__ == "__main__":
    get_conv4_model('resnet50')
