import os
import torch
from torch import nn

script_path = os.path.abspath(__file__)

copyweight = False
model_name = 'resnet18ft1'

resnet_50 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)

if copyweight:
    path_model = {
        'resnet18ft0' :
        ('weights/simsiam_20220218-014917/checkpoints/last.pt', 'resnet18'),
        'resnet18ft1' :
        ('weights/simsiam_20220224-160133/checkpoints/last.pt', 'resnet18'),
        'resnet50ft1' :
        ('weights/simsiam_20220224-154343/checkpoints/last.pt', 'resnet50'),
    }

    model_path = os.path.join(
        os.path.dirname(script_path),
        path_model[model_name][0]
    )
    resnet_50 = torch.hub.load(
        'pytorch/vision:v0.6.0',
        path_model[model_name][1],
        pretrained=True
    )


class ResNet50Conv4(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Conv4, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-3])

        if copyweight:
            features_weights = self.features.state_dict()

            model_weights = torch.load(model_path)

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


def get_conv4_model():
    conv4 = ResNet50Conv4(resnet_50)
    return conv4
