import torch
import os
import numpy as np
from attribution_certification.models import models_lrp
import sys
import zennit
import zennit.attribution
import zennit.canonizers
import zennit.composites
import zennit.torchvision
import zennit.types
import zennit.rules

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)


class EpsilonPlusBox(zennit.composites.SpecialFirstLayerMapComposite):
    def __init__(self, low, high, canonizers=None):
        layer_map = zennit.composites.LAYER_MAP_BASE + [
            (zennit.types.Convolution, zennit.rules.ZPlus()),
            (torch.nn.Linear, zennit.rules.Epsilon()),
        ]
        first_map = [
            (zennit.types.Convolution, zennit.rules.ZBox(low, high))
        ]
        super().__init__(layer_map, first_map, canonizers=canonizers)


class Epsilon025PlusBox(zennit.composites.SpecialFirstLayerMapComposite):
    def __init__(self, low, high, canonizers=None):
        layer_map = zennit.composites.LAYER_MAP_BASE + [
            (zennit.types.Convolution, zennit.rules.ZPlus()),
            (torch.nn.Linear, zennit.rules.Epsilon(epsilon=0.25)),
        ]
        first_map = [
            (zennit.types.Convolution, zennit.rules.ZBox(low, high))
        ]
        super().__init__(layer_map, first_map, canonizers=canonizers)

################################################################################

class EpsilonPlusBoxVGG11GridPG(zennit.composites.NameMapComposite):
    def __init__(self, low, high, canonizers=None, num_conv_epsilon=0):
        conv_layers = ['base_model.features.3', 'base_model.features.6', 'base_model.features.8', 'base_model.features.11',
                       'base_model.features.13', 'base_model.features.16', 'base_model.features.18']
        name_map = [
            (['base_model.features.0'], zennit.rules.ZBox(low, high)),
            (['base_model.features.1', 'base_model.features.4', 'base_model.features.7', 'base_model.features.9',
                'base_model.features.12', 'base_model.features.14', 'base_model.features.17', 'base_model.features.19', 'base_model.classifier.1', 'base_model.classifier.4'], zennit.rules.Pass()),
            (['base_model.avgpool', 'avgpool2d'], zennit.rules.Norm()),
            (['conv1', 'conv2', 'conv3'], zennit.rules.Epsilon()),
        ]
        non_epsilon_conv = conv_layers[:len(conv_layers) - num_conv_epsilon]
        epsilon_conv = conv_layers[len(conv_layers) - num_conv_epsilon:]
        if non_epsilon_conv:
            name_map.append((non_epsilon_conv, zennit.rules.ZPlus()))
        if epsilon_conv:
            name_map.append((epsilon_conv, zennit.rules.Epsilon()))
        super().__init__(name_map, canonizers=canonizers)


class Epsilon025PlusBoxVGG11GridPG(zennit.composites.NameMapComposite):
    def __init__(self, low, high, canonizers=None, num_conv_epsilon=0):
        conv_layers = ['base_model.features.3', 'base_model.features.6', 'base_model.features.8', 'base_model.features.11',
                       'base_model.features.13', 'base_model.features.16', 'base_model.features.18']
        name_map = [
            (['base_model.features.0'], zennit.rules.ZBox(low, high)),
            (['base_model.features.1', 'base_model.features.4', 'base_model.features.7', 'base_model.features.9',
                'base_model.features.12', 'base_model.features.14', 'base_model.features.17', 'base_model.features.19', 'base_model.classifier.1', 'base_model.classifier.4'], zennit.rules.Pass()),
            (['base_model.avgpool', 'avgpool2d'], zennit.rules.Norm()),
            (['conv1', 'conv2', 'conv3'], zennit.rules.Epsilon(epsilon=0.25)),
        ]
        non_epsilon_conv = conv_layers[:len(conv_layers) - num_conv_epsilon]
        epsilon_conv = conv_layers[len(conv_layers) - num_conv_epsilon:]
        if non_epsilon_conv:
            name_map.append((non_epsilon_conv, zennit.rules.ZPlus()))
        if epsilon_conv:
            name_map.append((epsilon_conv, zennit.rules.Epsilon(epsilon=0.25)))
        super().__init__(name_map, canonizers=canonizers)


class EpsilonGammaBoxVGG11GridPG(zennit.composites.NameMapComposite):
    def __init__(self, low, high, canonizers=None, num_conv_epsilon=0):
        conv_layers = ['base_model.features.3', 'base_model.features.6', 'base_model.features.8', 'base_model.features.11',
                       'base_model.features.13', 'base_model.features.16', 'base_model.features.18']
        name_map = [
            (['base_model.features.0'], zennit.rules.ZBox(low, high)),
            (['base_model.features.1', 'base_model.features.4', 'base_model.features.7', 'base_model.features.9',
                'base_model.features.12', 'base_model.features.14', 'base_model.features.17', 'base_model.features.19', 'base_model.classifier.1', 'base_model.classifier.4'], zennit.rules.Pass()),
            (['base_model.avgpool', 'avgpool2d'], zennit.rules.Norm()),
            (['conv1', 'conv2', 'conv3'], zennit.rules.Epsilon()),
        ]
        non_epsilon_conv = conv_layers[:len(conv_layers) - num_conv_epsilon]
        epsilon_conv = conv_layers[len(conv_layers) - num_conv_epsilon:]
        if non_epsilon_conv:
            name_map.append((non_epsilon_conv, zennit.rules.Gamma(gamma=0.25)))
        if epsilon_conv:
            name_map.append((epsilon_conv, zennit.rules.Epsilon()))
        super().__init__(name_map, canonizers=canonizers)

################################################################################


class EpsilonPlusBoxVGG11BNGridPG(zennit.composites.NameMapComposite):
    def __init__(self, low, high, canonizers=None, num_conv_epsilon=0):
        conv_layers = ['base_model.features.4', 'base_model.features.8', 'base_model.features.11', 'base_model.features.15',
                       'base_model.features.18', 'base_model.features.22', 'base_model.features.25']
        name_map = [
            (['base_model.features.0'], zennit.rules.ZBox(low, high)),
            (['base_model.features.2', 'base_model.features.6', 'base_model.features.10', 'base_model.features.13',
              'base_model.features.17', 'base_model.features.20', 'base_model.features.24', 'base_model.features.27', 'base_model.classifier.1', 'base_model.classifier.4'], zennit.rules.Pass()),
            (['base_model.avgpool', 'avgpool2d'], zennit.rules.Norm()),
            (['conv1', 'conv2', 'conv3'], zennit.rules.Epsilon()),
        ]
        non_epsilon_conv = conv_layers[:len(conv_layers) - num_conv_epsilon]
        epsilon_conv = conv_layers[len(conv_layers) - num_conv_epsilon:]
        if non_epsilon_conv:
            name_map.append((non_epsilon_conv, zennit.rules.ZPlus()))
        if epsilon_conv:
            name_map.append((epsilon_conv, zennit.rules.Epsilon()))
        super().__init__(name_map, canonizers=canonizers)


class Epsilon025PlusBoxVGG11BNGridPG(zennit.composites.NameMapComposite):
    def __init__(self, low, high, canonizers=None, num_conv_epsilon=0):
        conv_layers = ['base_model.features.4', 'base_model.features.8', 'base_model.features.11', 'base_model.features.15',
                       'base_model.features.18', 'base_model.features.22', 'base_model.features.25']
        name_map = [
            (['base_model.features.0'], zennit.rules.ZBox(low, high)),
            (['base_model.features.2', 'base_model.features.6', 'base_model.features.10', 'base_model.features.13',
              'base_model.features.17', 'base_model.features.20', 'base_model.features.24', 'base_model.features.27', 'base_model.classifier.1', 'base_model.classifier.4'], zennit.rules.Pass()),
            (['base_model.avgpool', 'avgpool2d'], zennit.rules.Norm()),
            (['conv1', 'conv2', 'conv3'], zennit.rules.Epsilon(epsilon=0.25)),
        ]
        non_epsilon_conv = conv_layers[:len(conv_layers) - num_conv_epsilon]
        epsilon_conv = conv_layers[len(conv_layers) - num_conv_epsilon:]
        if non_epsilon_conv:
            name_map.append((non_epsilon_conv, zennit.rules.ZPlus()))
        if epsilon_conv:
            name_map.append((epsilon_conv, zennit.rules.Epsilon(epsilon=0.25)))
        super().__init__(name_map, canonizers=canonizers)


class EpsilonGammaBoxVGG11BNGridPG(zennit.composites.NameMapComposite):
    def __init__(self, low, high, canonizers=None, num_conv_epsilon=0):
        conv_layers = ['base_model.features.4', 'base_model.features.8', 'base_model.features.11', 'base_model.features.15',
                       'base_model.features.18', 'base_model.features.22', 'base_model.features.25']
        name_map = [
            (['base_model.features.0'], zennit.rules.ZBox(low, high)),
            (['base_model.features.2', 'base_model.features.6', 'base_model.features.10', 'base_model.features.13',
              'base_model.features.17', 'base_model.features.20', 'base_model.features.24', 'base_model.features.27', 'base_model.classifier.1', 'base_model.classifier.4'], zennit.rules.Pass()),
            (['base_model.avgpool', 'avgpool2d'], zennit.rules.Norm()),
            (['conv1', 'conv2', 'conv3'], zennit.rules.Epsilon()),
        ]
        non_epsilon_conv = conv_layers[:len(conv_layers) - num_conv_epsilon]
        epsilon_conv = conv_layers[len(conv_layers) - num_conv_epsilon:]
        if non_epsilon_conv:
            name_map.append((non_epsilon_conv, zennit.rules.Gamma(gamma=0.25)))
        if epsilon_conv:
            name_map.append((epsilon_conv, zennit.rules.Epsilon()))
        super().__init__(name_map, canonizers=canonizers)

################################################################################


class EpsilonPlusBoxVGG19GridPG(zennit.composites.NameMapComposite):
    def __init__(self, low, high, canonizers=None):
        name_map = [
            (['base_model.features.0'], zennit.rules.ZBox(low, high)),
            (['base_model.features.2', 'base_model.features.5', 'base_model.features.7', 'base_model.features.10',
              'base_model.features.12', 'base_model.features.14', 'base_model.features.16', 'base_model.features.19', 'base_model.features.21', 'base_model.features.23', 'base_model.features.25', 'base_model.features.28', 'base_model.features.30', 'base_model.features.32', 'base_model.features.34'], zennit.rules.ZPlus()),
            (['base_model.features.1', 'base_model.features.3', 'base_model.features.6', 'base_model.features.8', 'base_model.features.11',
              'base_model.features.13', 'base_model.features.15', 'base_model.features.17', 'base_model.features.20', 'base_model.features.22', 'base_model.features.24', 'base_model.features.26', 'base_model.features.29', 'base_model.features.31', 'base_model.features.33', 'base_model.features.35', 'base_model.classifier.1', 'base_model.classifier.4'], zennit.rules.Pass()),
            (['base_model.avgpool', 'avgpool2d'], zennit.rules.Norm()),
            (['conv1', 'conv2', 'conv3'], zennit.rules.Epsilon()),
        ]
        super().__init__(name_map, canonizers=canonizers)


class Epsilon025PlusBoxVGG19GridPG(zennit.composites.NameMapComposite):
    def __init__(self, low, high, canonizers=None):
        name_map = [
            (['base_model.features.0'], zennit.rules.ZBox(low, high)),
            (['base_model.features.2', 'base_model.features.5', 'base_model.features.7', 'base_model.features.10',
              'base_model.features.12', 'base_model.features.14', 'base_model.features.16', 'base_model.features.19', 'base_model.features.21', 'base_model.features.23', 'base_model.features.25', 'base_model.features.28', 'base_model.features.30', 'base_model.features.32', 'base_model.features.34'], zennit.rules.ZPlus()),
            (['base_model.features.1', 'base_model.features.3', 'base_model.features.6', 'base_model.features.8', 'base_model.features.11',
              'base_model.features.13', 'base_model.features.15', 'base_model.features.17', 'base_model.features.20', 'base_model.features.22', 'base_model.features.24', 'base_model.features.26', 'base_model.features.29', 'base_model.features.31', 'base_model.features.33', 'base_model.features.35', 'base_model.classifier.1', 'base_model.classifier.4'], zennit.rules.Pass()),
            (['base_model.avgpool', 'avgpool2d'], zennit.rules.Norm()),
            (['conv1', 'conv2', 'conv3'], zennit.rules.Epsilon(epsilon=0.25)),
        ]
        super().__init__(name_map, canonizers=canonizers)


class EpsilonGammaBoxVGG19GridPG(zennit.composites.NameMapComposite):
    def __init__(self, low, high, canonizers=None, gamma=0.25):
        name_map = [
            (['base_model.features.0'], zennit.rules.ZBox(low, high)),
            (['base_model.features.2', 'base_model.features.5', 'base_model.features.7', 'base_model.features.10',
              'base_model.features.12', 'base_model.features.14', 'base_model.features.16', 'base_model.features.19', 'base_model.features.21', 'base_model.features.23', 'base_model.features.25', 'base_model.features.28', 'base_model.features.30', 'base_model.features.32', 'base_model.features.34'], zennit.rules.Gamma(gamma=gamma)),
            (['base_model.features.1', 'base_model.features.3', 'base_model.features.6', 'base_model.features.8', 'base_model.features.11',
              'base_model.features.13', 'base_model.features.15', 'base_model.features.17', 'base_model.features.20', 'base_model.features.22', 'base_model.features.24', 'base_model.features.26', 'base_model.features.29', 'base_model.features.31', 'base_model.features.33', 'base_model.features.35', 'base_model.classifier.1', 'base_model.classifier.4'], zennit.rules.Pass()),
            (['base_model.avgpool', 'avgpool2d'], zennit.rules.Norm()),
            (['conv1', 'conv2', 'conv3'], zennit.rules.Epsilon()),
        ]
        super().__init__(name_map, canonizers=canonizers)

################################################################################


class EpsilonPlusBoxVGG19BNGridPG(zennit.composites.NameMapComposite):
    def __init__(self, low, high, canonizers=None):
        name_map = [
            (['base_model.features.0'], zennit.rules.ZBox(low, high)),
            (['base_model.features.3', 'base_model.features.7', 'base_model.features.10', 'base_model.features.14', 'base_model.features.17', 'base_model.features.20', 'base_model.features.23', 'base_model.features.27',
              'base_model.features.30', 'base_model.features.33', 'base_model.features.36', 'base_model.features.40', 'base_model.features.43', 'base_model.features.46', 'base_model.features.49'], zennit.rules.ZPlus()),
            (['base_model.features.2', 'base_model.features.5', 'base_model.features.9', 'base_model.features.12', 'base_model.features.16', 'base_model.features.19', 'base_model.features.22', 'base_model.features.25', 'base_model.features.29', 'base_model.features.32',
              'base_model.features.35', 'base_model.features.38', 'base_model.features.42', 'base_model.features.45', 'base_model.features.48', 'base_model.features.51', 'base_model.classifier.1', 'base_model.classifier.4'], zennit.rules.Pass()),
            (['base_model.avgpool', 'avgpool2d'], zennit.rules.Norm()),
            (['conv1', 'conv2', 'conv3'], zennit.rules.Epsilon()),
        ]
        super().__init__(name_map, canonizers=canonizers)


class Epsilon025PlusBoxVGG19BNGridPG(zennit.composites.NameMapComposite):
    def __init__(self, low, high, canonizers=None):
        name_map = [
            (['base_model.features.0'], zennit.rules.ZBox(low, high)),
            (['base_model.features.3', 'base_model.features.7', 'base_model.features.10', 'base_model.features.14', 'base_model.features.17', 'base_model.features.20', 'base_model.features.23', 'base_model.features.27',
              'base_model.features.30', 'base_model.features.33', 'base_model.features.36', 'base_model.features.40', 'base_model.features.43', 'base_model.features.46', 'base_model.features.49'], zennit.rules.ZPlus()),
            (['base_model.features.2', 'base_model.features.5', 'base_model.features.9', 'base_model.features.12', 'base_model.features.16', 'base_model.features.19', 'base_model.features.22', 'base_model.features.25', 'base_model.features.29', 'base_model.features.32',
              'base_model.features.35', 'base_model.features.38', 'base_model.features.42', 'base_model.features.45', 'base_model.features.48', 'base_model.features.51', 'base_model.classifier.1', 'base_model.classifier.4'], zennit.rules.Pass()),
            (['base_model.avgpool', 'avgpool2d'], zennit.rules.Norm()),
            (['conv1', 'conv2', 'conv3'], zennit.rules.Epsilon(epsilon=0.25)),
        ]
        super().__init__(name_map, canonizers=canonizers)


class EpsilonGammaBoxVGG19BNGridPG(zennit.composites.NameMapComposite):
    def __init__(self, low, high, canonizers=None):
        name_map = [
            (['base_model.features.0'], zennit.rules.ZBox(low, high)),
            (['base_model.features.3', 'base_model.features.7', 'base_model.features.10', 'base_model.features.14', 'base_model.features.17', 'base_model.features.20', 'base_model.features.23', 'base_model.features.27',
              'base_model.features.30', 'base_model.features.33', 'base_model.features.36', 'base_model.features.40', 'base_model.features.43', 'base_model.features.46', 'base_model.features.49'], zennit.rules.Gamma(gamma=0.25)),
            (['base_model.features.2', 'base_model.features.5', 'base_model.features.9', 'base_model.features.12', 'base_model.features.16', 'base_model.features.19', 'base_model.features.22', 'base_model.features.25', 'base_model.features.29', 'base_model.features.32',
              'base_model.features.35', 'base_model.features.38', 'base_model.features.42', 'base_model.features.45', 'base_model.features.48', 'base_model.features.51', 'base_model.classifier.1', 'base_model.classifier.4'], zennit.rules.Pass()),
            (['base_model.avgpool', 'avgpool2d'], zennit.rules.Norm()),
            (['conv1', 'conv2', 'conv3'], zennit.rules.Epsilon()),
        ]
        super().__init__(name_map, canonizers=canonizers)


def get_intermediate_activations(model, x, layer, scale):
    if layer == -1:
        return x
    if model.single_spatial:
        outs, conv_acts = model(
            x, conv_layer_idx=layer, after_maxpool=False)
        return conv_acts
    all_loc_conv_acts = []
    grid_size = scale * scale
    for loc_idx in range(grid_size):
        outs, loc_conv_acts = model(
            x, loc_idx=loc_idx, conv_layer_idx=layer, after_maxpool=False)
        all_loc_conv_acts.append(loc_conv_acts.detach())
    cat_row_conv_acts = []
    for row_idx in range(scale):
        cat_row_conv_acts.append(
            torch.cat(all_loc_conv_acts[row_idx * scale:(row_idx + 1) * scale], dim=3))
    conv_acts = torch.cat(cat_row_conv_acts, dim=2)
    return conv_acts


def get_layer_number(model, layer, every_layer):
    if layer == -1:
        return "test"
    if every_layer:
        return "el" + str(layer + 1)
    return "bilinear_conv" + str(layer + 1)

def get_attributor(config, model, head_idx, composite_map, up_box, low_box, model_name):
    if ("VGG" in model_name) or ("Googlenet" in model_name):
        canonizer = [zennit.torchvision.VGGCanonizer()]
    else:
        canonizer = [zennit.torchvision.ResNetCanonizer()]

    if config.composite == "None":
        attributor = zennit.attribution.Gradient(model=model)
    elif ("Box" in config.composite):
        if ("VGG11" in model_name) and ("PointingGame" in model_name):
            composite = composite_map[config.composite](
                low=low_box.clone(), high=up_box.clone(), canonizers=canonizer, num_conv_epsilon=config.num_conv_epsilon)
        else:
            if ("Gamma" in config.composite) and (config.gamma is not None):
                composite = composite_map[config.composite](
                    low=low_box.clone(), high=up_box.clone(), canonizers=canonizer, gamma=config.gamma)
            else:
                composite = composite_map[config.composite](
                    low=low_box.clone(), high=up_box.clone(), canonizers=canonizer)

        attributor = zennit.attribution.Gradient(model=model, composite=composite)
    else:
        composite = composite_map[config.composite](canonizers=canonizer)
        attributor = zennit.attribution.Gradient(model=model, composite=composite)
    return attributor

def lrp_model_setup(config, scale):
    if config.model == 'vgg11':
        model_name = 'ImagenetContainerVGG11PointingGame'
    if config.model == 'vgg19':
        model_name = 'ImagenetContainerVGG19PointingGame'
    elif config.model == 'resnet18':
        model_name = 'ImagenetContainerResnet18PointingGame'
    elif config.model == 'resnet50_2':
        model_name = 'ImagenetContainerWideResnetPointingGame'
    elif config.model == 'resnet152':
        model_name = 'ImagenetContainerResnet152PointingGame'
    elif config.model == 'resnet101':
        model_name = 'ImagenetContainerResnet101PointingGame'
    elif config.model == 'vit_b_16':
        model_name = 'ImagenetContainerViTB16PointingGame'
    if ("VGG11PointingGame" in model_name):
        composite_map = {"EpsilonGammaBox": EpsilonGammaBoxVGG11GridPG, "EpsilonPlus": zennit.composites.EpsilonPlus, "EpsilonAlpha2Beta1": zennit.composites.EpsilonAlpha2Beta1, "EpsilonPlusFlat": zennit.composites.EpsilonPlusFlat,
                         "EpsilonAlpha2Beta1Flat": zennit.composites.EpsilonAlpha2Beta1Flat, "ExcitationBackprop": zennit.composites.ExcitationBackprop, "EpsilonPlusBox": EpsilonPlusBoxVGG11GridPG, "Epsilon025PlusBox": Epsilon025PlusBoxVGG11GridPG}
    elif ("VGG11BNPointingGame" in model_name):
        composite_map = {"EpsilonGammaBox": EpsilonGammaBoxVGG11BNGridPG, "EpsilonPlus": zennit.composites.EpsilonPlus, "EpsilonAlpha2Beta1": zennit.composites.EpsilonAlpha2Beta1, "EpsilonPlusFlat": zennit.composites.EpsilonPlusFlat,
                         "EpsilonAlpha2Beta1Flat": zennit.composites.EpsilonAlpha2Beta1Flat, "ExcitationBackprop": zennit.composites.ExcitationBackprop, "EpsilonPlusBox": EpsilonPlusBoxVGG11BNGridPG, "Epsilon025PlusBox": Epsilon025PlusBoxVGG11BNGridPG}
    elif ("VGG19PointingGame" in model_name):
        composite_map = {"EpsilonGammaBox": EpsilonGammaBoxVGG19GridPG, "EpsilonPlus": zennit.composites.EpsilonPlus, "EpsilonAlpha2Beta1": zennit.composites.EpsilonAlpha2Beta1, "EpsilonPlusFlat": zennit.composites.EpsilonPlusFlat,
                         "EpsilonAlpha2Beta1Flat": zennit.composites.EpsilonAlpha2Beta1Flat, "ExcitationBackprop": zennit.composites.ExcitationBackprop, "EpsilonPlusBox": EpsilonPlusBoxVGG19GridPG, "Epsilon025PlusBox": Epsilon025PlusBoxVGG19GridPG}
    elif ("VGG19BNPointingGame" in model_name):
        composite_map = {"EpsilonGammaBox": EpsilonGammaBoxVGG19BNGridPG, "EpsilonPlus": zennit.composites.EpsilonPlus, "EpsilonAlpha2Beta1": zennit.composites.EpsilonAlpha2Beta1, "EpsilonPlusFlat": zennit.composites.EpsilonPlusFlat,
                         "EpsilonAlpha2Beta1Flat": zennit.composites.EpsilonAlpha2Beta1Flat, "ExcitationBackprop": zennit.composites.ExcitationBackprop, "EpsilonPlusBox": EpsilonPlusBoxVGG19BNGridPG, "Epsilon025PlusBox": Epsilon025PlusBoxVGG19BNGridPG}
    else:
        assert ("VGG" not in model_name) or ("PointingGame" not in model_name)
        composite_map = {"EpsilonGammaBox": zennit.composites.EpsilonGammaBox, "EpsilonPlus": zennit.composites.EpsilonPlus, "EpsilonAlpha2Beta1": zennit.composites.EpsilonAlpha2Beta1, "EpsilonPlusFlat": zennit.composites.EpsilonPlusFlat,
                         "EpsilonAlpha2Beta1Flat": zennit.composites.EpsilonAlpha2Beta1Flat, "ExcitationBackprop": zennit.composites.ExcitationBackprop, "EpsilonPlusBox": EpsilonPlusBox, "Epsilon025PlusBox": Epsilon025PlusBox}

    assert config.batch_size == 1


    img_means = torch.tensor([0.485, 0.456, 0.406])
    img_stds = torch.tensor([0.229, 0.224, 0.225])
    lower_bounds = -img_means / img_stds
    upper_bounds = (1 - img_means) / img_stds
    if config.use_box_stabilizer:
        lower_bounds = lower_bounds - 0.1
        upper_bounds = upper_bounds + 0.1
    low_box = torch.zeros((1, 3, 1, 1))
    up_box = torch.zeros((1, 3, 1, 1))
    for idx in range(3):
        low_box[:, idx] = lower_bounds[idx]
        up_box[:, idx] = upper_bounds[idx]

    if config.cuda:
        low_box = low_box.cuda()
        up_box = up_box.cuda()

    if (config.head_idx == 1 or config.head_idx == 2) and (config.ignore_non_corners):
        print("Exiting because head", config.head_idx, "is not a corner")
        sys.exit(0)


    model = models_lrp.__dict__[model_name](scale, output_head_idx=0, transform_batchnorm=False)
    if config.cuda:
        model.cuda()
    model.eval()
    attributor = get_attributor(config, model, 0, composite_map, up_box, low_box, model_name)    
    return model, attributor
