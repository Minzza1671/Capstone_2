import torchvision


def vgg16_bn(pretrained=False):
    try:
        from torchvision.models import VGG16_BN_Weights
        weights = VGG16_BN_Weights.DEFAULT if pretrained else None
        return torchvision.models.vgg16_bn(weights=weights)
    except ImportError:
        return torchvision.models.vgg16_bn(pretrained=pretrained)


def vgg16(pretrained=False):
    try:
        from torchvision.models import VGG16_Weights
        weights = VGG16_Weights.DEFAULT if pretrained else None
        return torchvision.models.vgg16(weights=weights)
    except ImportError:
        return torchvision.models.vgg16(pretrained=pretrained)
