import torch
import torch.nn as nn
import torchvision.models as models


class ModelFactory:
    @staticmethod
    def create_model(model_name: str, num_classes: int = 10) -> nn.Module:
        m = model_name.lower().strip()

        if m in ["squeezenet", "squeezenet1_1"]:
            net = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
            net.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
            net.num_classes = num_classes
            nn.init.normal_(net.classifier[1].weight, std=0.01)
            nn.init.constant_(net.classifier[1].bias, 0)
            return net

        elif m == "mobilenet_v3_small":
            net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            in_features = net.classifier[3].in_features
            net.classifier[3] = nn.Linear(in_features, num_classes)
            nn.init.normal_(net.classifier[3].weight, std=0.01)
            nn.init.constant_(net.classifier[3].bias, 0)
            return net

        elif m == "mobilenet_v2":
            net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            in_features = net.classifier[1].in_features
            net.classifier[1] = nn.Linear(in_features, num_classes)
            nn.init.normal_(net.classifier[1].weight, std=0.01)
            nn.init.constant_(net.classifier[1].bias, 0)
            return net

        elif m in ["efficientnet", "efficientnet_b0"]:
            net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            in_features = net.classifier[1].in_features
            net.classifier[1] = nn.Linear(in_features, num_classes)
            nn.init.normal_(net.classifier[1].weight, std=0.01)
            nn.init.constant_(net.classifier[1].bias, 0)
            return net

        elif m == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, num_classes)
            nn.init.normal_(net.fc.weight, std=0.01)
            nn.init.constant_(net.fc.bias, 0)
            return net

        elif m == "resnet34":
            net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, num_classes)
            nn.init.normal_(net.fc.weight, std=0.01)
            nn.init.constant_(net.fc.bias, 0)
            return net

        else:
            raise ValueError(f"Modelo '{model_name}' no reconocido.")


create_model = ModelFactory.create_model
