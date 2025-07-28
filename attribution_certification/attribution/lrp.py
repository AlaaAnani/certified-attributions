import torch
# import explanation_methods
import argparse
import os
import numpy as np
import models
import operator
from tqdm import tqdm
import sys
import torchvision
import traceback
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


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if ("VGG11PointingGame" in args.model):
        composite_map = {"EpsilonGammaBox": EpsilonGammaBoxVGG11GridPG, "EpsilonPlus": zennit.composites.EpsilonPlus, "EpsilonAlpha2Beta1": zennit.composites.EpsilonAlpha2Beta1, "EpsilonPlusFlat": zennit.composites.EpsilonPlusFlat,
                         "EpsilonAlpha2Beta1Flat": zennit.composites.EpsilonAlpha2Beta1Flat, "ExcitationBackprop": zennit.composites.ExcitationBackprop, "EpsilonPlusBox": EpsilonPlusBoxVGG11GridPG, "Epsilon025PlusBox": Epsilon025PlusBoxVGG11GridPG}
    elif ("VGG11BNPointingGame" in args.model):
        composite_map = {"EpsilonGammaBox": EpsilonGammaBoxVGG11BNGridPG, "EpsilonPlus": zennit.composites.EpsilonPlus, "EpsilonAlpha2Beta1": zennit.composites.EpsilonAlpha2Beta1, "EpsilonPlusFlat": zennit.composites.EpsilonPlusFlat,
                         "EpsilonAlpha2Beta1Flat": zennit.composites.EpsilonAlpha2Beta1Flat, "ExcitationBackprop": zennit.composites.ExcitationBackprop, "EpsilonPlusBox": EpsilonPlusBoxVGG11BNGridPG, "Epsilon025PlusBox": Epsilon025PlusBoxVGG11BNGridPG}
    elif ("VGG19PointingGame" in args.model):
        composite_map = {"EpsilonGammaBox": EpsilonGammaBoxVGG19GridPG, "EpsilonPlus": zennit.composites.EpsilonPlus, "EpsilonAlpha2Beta1": zennit.composites.EpsilonAlpha2Beta1, "EpsilonPlusFlat": zennit.composites.EpsilonPlusFlat,
                         "EpsilonAlpha2Beta1Flat": zennit.composites.EpsilonAlpha2Beta1Flat, "ExcitationBackprop": zennit.composites.ExcitationBackprop, "EpsilonPlusBox": EpsilonPlusBoxVGG19GridPG, "Epsilon025PlusBox": Epsilon025PlusBoxVGG19GridPG}
    elif ("VGG19BNPointingGame" in args.model):
        composite_map = {"EpsilonGammaBox": EpsilonGammaBoxVGG19BNGridPG, "EpsilonPlus": zennit.composites.EpsilonPlus, "EpsilonAlpha2Beta1": zennit.composites.EpsilonAlpha2Beta1, "EpsilonPlusFlat": zennit.composites.EpsilonPlusFlat,
                         "EpsilonAlpha2Beta1Flat": zennit.composites.EpsilonAlpha2Beta1Flat, "ExcitationBackprop": zennit.composites.ExcitationBackprop, "EpsilonPlusBox": EpsilonPlusBoxVGG19BNGridPG, "Epsilon025PlusBox": Epsilon025PlusBoxVGG19BNGridPG}
    else:
        assert ("VGG" not in args.model) or ("PointingGame" not in args.model)
        composite_map = {"EpsilonGammaBox": zennit.composites.EpsilonGammaBox, "EpsilonPlus": zennit.composites.EpsilonPlus, "EpsilonAlpha2Beta1": zennit.composites.EpsilonAlpha2Beta1, "EpsilonPlusFlat": zennit.composites.EpsilonPlusFlat,
                         "EpsilonAlpha2Beta1Flat": zennit.composites.EpsilonAlpha2Beta1Flat, "ExcitationBackprop": zennit.composites.ExcitationBackprop, "EpsilonPlusBox": EpsilonPlusBox, "Epsilon025PlusBox": Epsilon025PlusBox}

    assert args.batch_size == 1
    test_data_dict = torch.load(os.path.join(args.dataset_path, 'test.pt'))

    inv_normalize = torchvision.transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    # add_inverse = bcos.data.data_transforms.AddInverse()
    transform = torch.nn.Identity()
    if ("BCos" in args.model) or ("CoDA" in args.model):
        raise NotImplementedError

    img_means = torch.tensor([0.485, 0.456, 0.406])
    img_stds = torch.tensor([0.229, 0.224, 0.225])
    lower_bounds = -img_means / img_stds
    upper_bounds = (1 - img_means) / img_stds
    if args.use_box_stabilizer:
        lower_bounds = lower_bounds - 0.1
        upper_bounds = upper_bounds + 0.1
    low_box = torch.zeros((1, 3, 1, 1))
    up_box = torch.zeros((1, 3, 1, 1))
    for idx in range(3):
        low_box[:, idx] = lower_bounds[idx]
        up_box[:, idx] = upper_bounds[idx]

    if args.cuda:
        low_box = low_box.cuda()
        up_box = up_box.cuda()

    if args.num_test_images is None:
        test_data = torch.utils.data.TensorDataset(
            transform(test_data_dict["data"]), test_data_dict["labels"])
    else:
        test_data = torch.utils.data.TensorDataset(
            transform(test_data_dict["data"][:args.num_test_images]), test_data_dict["labels"][:args.num_test_images])

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False)
    scale = test_data_dict["scale"]
    # print("Scale:", scale)
    grid_size = scale * scale
    if args.head_idx is not None:
        num_heads = 1
        head_list = [args.head_idx]
    elif args.only_corners:
        num_heads = 2
        head_list = [0, grid_size - 1]
    else:
        num_heads = grid_size
        head_list = np.arange(grid_size)

    print(num_heads, head_list)

    if (args.head_idx == 1 or args.head_idx == 2) and (args.ignore_non_corners):
        print("Exiting because head", args.head_idx, "is not a corner")
        sys.exit(0)

    model_list = []
    for idx in head_list:
        print(idx)
        model = models.__dict__[args.model](
            test_data_dict["scale"], output_head_idx=idx, transform_batchnorm=False)  # CHANGED HERE!!! #######################################
        # if args.randomize_weights_nonsign:
        #     model.randomize(with_sign=False)
        # elif args.randomize_weights_sign:
        #     model.randomize(with_sign=True)
        if args.cuda:
            model.cuda()
        model.eval()
        model_list.append(model)

    explanation_methods_list = ["ZLRP"]
    explanation_methods_configs_list = [
        str(args.composite) + "_" + get_layer_number(args.model, args.layer, False)]

    explanation_attributions = {}
    nm_explanation_attributions = {}

    for idx, (exp, config) in enumerate(zip(explanation_methods_list, explanation_methods_configs_list)):
        print(exp, config)
        continue_flag = 0
        if os.path.exists(os.path.join(args.save_path, 'attributions_imagenet_' +
                                       args.model + '_' + os.path.basename(args.dataset_path) + '_' + exp + '_' + config + '_o' + args.save_suffix + '.pt')):
            print(os.path.join(args.save_path, 'attributions_imagenet_' +
                               args.model + '_' + os.path.basename(args.dataset_path) + '_' + exp + '_' + config + '_o' + args.save_suffix + '.pt'), "already exists.")
            continue_flag = 1
            try:
                torch.load(os.path.join(args.save_path, 'attributions_imagenet_' +
                                        args.model + '_' + os.path.basename(args.dataset_path) + '_' + exp + '_' + config + '_o' + args.save_suffix + '.pt'))
            except:
                print("But could not be loaded.")
                continue_flag = 0
            if continue_flag:
                print("Load successful, continuing.")
                continue
        try:
            # explanation_attributions[exp + config] = torch.zeros(
            # (len(test_data), grid_size, 1, img_dims[0] * scale, img_dims[1] * scale))
            explanation_attributions[exp + config] = []
            nm_explanation_attributions[exp + config] = []
            explainer_list = []
            for head_idx, head_pos_idx in enumerate(head_list):
                if ("VGG" in args.model) or ("Googlenet" in args.model):
                    canonizer = [zennit.torchvision.VGGCanonizer()]
                else:
                    canonizer = [zennit.torchvision.ResNetCanonizer()]
                if args.composite == "None":
                    attributor = zennit.attribution.Gradient(
                        model=model_list[head_idx])
                elif ("Box" in args.composite):
                    # print(low_box.shape, up_box.shape)
                    if ("VGG11" in args.model) and ("PointingGame" in args.model):
                        composite = composite_map[args.composite](
                            low=low_box.clone(), high=up_box.clone(), canonizers=canonizer, num_conv_epsilon=args.num_conv_epsilon)
                    else:
                        if ("Gamma" in args.composite) and (args.gamma is not None):
                            composite = composite_map[args.composite](
                                low=low_box.clone(), high=up_box.clone(), canonizers=canonizer, gamma=args.gamma)
                        else:
                            composite = composite_map[args.composite](
                                low=low_box.clone(), high=up_box.clone(), canonizers=canonizer)

                    attributor = zennit.attribution.Gradient(
                        model=model_list[head_idx], composite=composite)
                else:
                    composite = composite_map[args.composite](
                        canonizers=canonizer)
                    attributor = zennit.attribution.Gradient(
                        model=model_list[head_idx], composite=composite)

                head_explanation_attributions = []
                nm_head_explanation_attributions = []
                for batch_idx, (test_X, test_y) in enumerate(tqdm(test_loader)):
                    if args.cuda:
                        test_X = test_X.cuda().requires_grad_(True)
                        test_y = test_y.cuda()
                    inp = get_intermediate_activations(
                        model_list[head_idx], test_X, args.layer, scale)
                    if args.layer != -1:
                        model_list[head_idx].start_conv_layer_idx = args.layer
                    out = model_list[head_idx](inp)
                    out = out[:, test_y[:, head_pos_idx]].item()
                    with attributor:
                        target = torch.eye(
                            1000)[[test_y[:, head_pos_idx]]] * out
                        if args.cuda:
                            target = target.cuda()
                        # print(inp.shape, target.shape)
                        _, relevance = attributor(inp, target)

                    nm_relevance = relevance.sum(
                        dim=1, keepdim=True).detach().cpu()
                    # relevance = (relevance * inp).sum(dim=1, keepdim=True)
                    # head_explanation_attributions.append(
                    # relevance.unsqueeze(0).detach().cpu())
                    nm_head_explanation_attributions.append(
                        nm_relevance.unsqueeze(0).detach().cpu())
                    model_list[head_idx].start_conv_layer_idx = None

                # explanation_attributions[exp +
                    #  config].append(torch.cat(head_explanation_attributions, dim=0))
                nm_explanation_attributions[exp +
                                            config].append(torch.cat(nm_head_explanation_attributions, dim=0))
            # explanation_attributions[exp +
                #  config] = torch.cat(explanation_attributions[exp + config], dim=1)
            nm_explanation_attributions[exp +
                                        config] = torch.cat(nm_explanation_attributions[exp + config], dim=1)
            # print(explanation_attributions[exp + config].shape)
            print("Saving attributions for:", exp, config, "with dimensions:",
                  nm_explanation_attributions[exp + config].shape)
            # torch.save(explanation_attributions[exp + config], os.path.join(args.save_path, 'attributions_imagenet_' +
            #                                                                 args.model + '_' + os.path.basename(args.dataset_path) + '_' + exp + '_' + config + args.save_suffix + '.pt'))
            save_name = 'attributions_imagenet_' + args.model + '_' + \
                os.path.basename(args.dataset_path) + '_' + \
                exp + '_' + config + "_o_updzennit"
            if (args.gamma is not None) and ("Gamma" in args.composite):
                save_name += "_gamma" + str(args.gamma)
            save_name += args.save_suffix + '.pt'
            torch.save(
                nm_explanation_attributions[exp + config], os.path.join(args.save_path, save_name))
        except Exception as ex:
            print(traceback.format_exc())
            continue


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='data/imagenet')
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--model', type=str, default='ImagenetContainerVGG11PointingGame')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--save_path', type=str, default='outs/')
parser.add_argument('--num_test_images', type=int, default=None)
parser.add_argument('--save_suffix', type=str, default='')
parser.add_argument('--composite', type=str, default='EpsilonPlusBox',
                    choices=["None", "EpsilonGammaBox", "EpsilonPlus", "EpsilonAlpha2Beta1", "EpsilonPlusFlat", "EpsilonAlpha2Beta1Flat", "ExcitationBackprop", "EpsilonPlusBox", "Epsilon025PlusBox"])
parser.add_argument("--layer", type=int, default=7)
parser.add_argument('--only_corners', action='store_true', default=False)
parser.add_argument('--head_idx', type=int, default=None)
parser.add_argument('--ignore_non_corners', action='store_true', default=False)
parser.add_argument('--use_box_stabilizer', action='store_true', default=False)
# parser.add_argument('--last_conv_epsilon', action='store_true', default=False)
parser.add_argument('--num_conv_epsilon', type=int, default=0)
parser.add_argument('--gamma', type=float, default=None)
# parser.add_argument('--every_layer', action='store_true', default=False)
# parser.add_argument('--randomize_weights_nonsign',
#                     action='store_true', default=False)
# parser.add_argument('--randomize_weights_sign',
#                     action='store_true', default=False)
# parser.add_argument('--no_multiply', action='store_true', default=False)

args = parser.parse_args()
main(args)
