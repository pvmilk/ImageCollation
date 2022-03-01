import torch
from torch import nn

copyweight = False

resnet_50 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)

if copyweight:
    model_path = 'weights/simsiam_20220218-014917/checkpoints/last.pt'
    resnet_50 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)


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
