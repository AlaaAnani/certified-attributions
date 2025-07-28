"""
Code adapted from Sukrut Rao https://github.com/sukrutrao/Attribution-Evaluation
"""
import torch
import copy
import torchvision
from attribution_certification.models import resnet_for_lrp
from attribution_certification.models import resnet_for_lrp_conv
from attribution_certification.models import googlenet_for_lrp
from captum.attr._utils.lrp_rules import IdentityRule
from captum.attr._utils.custom_modules import Addition_Module
from attribution_certification.models import vgg_for_lrp
# from bcos.interpretability.utils import get_pretrained


def get_augmentation_range(shape, scale, index=4):
    assert 0 <= index and index < scale * scale
    y_pos = index // scale
    x_pos = index % scale
    h = shape[2] // scale
    w = shape[3] // scale
    y = h * y_pos
    x = w * x_pos
    return y, x, h, w


####################################################################################################
####################################################################################################
####################################################################################################
import torch
import torchvision
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ImagenetContainerViTB16PointingGame(torch.nn.Module):
    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=False):
        super(ImagenetContainerViTB16PointingGame, self).__init__()

        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx
        self.num_heads = scale * scale
        assert self.output_head_idx is None or (
            0 <= self.output_head_idx < self.num_heads)

        self.base_model = vit_b_16(weights=ViT_B_16_Weights)
        self.encoder_blocks = self.base_model.encoder.layers
        self.embedding = self.base_model.conv_proj
        self.cls_token = self.base_model.class_token
        self.pos_embedding = self.base_model.encoder.pos_embedding
        self.dropout = self.base_model.encoder.dropout
        self.norm = self.base_model.encoder.ln
        self.head = self.base_model.heads

        # Define pseudo-conv-layer IDs for block-wise access
        self.conv_act_ids = [list(range(len(self.encoder_blocks)))]
        self.conv_layer_ids = list(range(len(self.encoder_blocks)))

        self.single_spatial = True
        self.single_head = True
        self.start_conv_layer_idx = None
        self.use_softmax = False

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx

        B = x.shape[0]
        x = self.embedding(x)  # Patchify + linear proj
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_dim)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        start_idx = 0 if start_conv_layer_idx is None else self.conv_act_ids[after_maxpool][start_conv_layer_idx]
        conv_acts = None

        for i in range(start_idx, len(self.encoder_blocks)):
            x = self.encoder_blocks[i](x)
            if conv_layer_idx is not None and i == self.conv_act_ids[after_maxpool][conv_layer_idx]:
                conv_acts = x

        x = self.norm(x)
        cls_embedding = x[:, 0]
        logits = self.head(cls_embedding)

        if self.use_softmax:
            logits = torch.nn.functional.softmax(logits, dim=1)

        if conv_layer_idx is not None:
            return logits, conv_acts
        else:
            return logits

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def _set_rule_to_block(self, block, rule, rule_param):
        for layer in block.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.rule = rule() if rule_param is None else rule(rule_param)

    def set_input_rule(self, rule, rule_param):
        self.embedding.rule = rule() if rule_param is None else rule(rule_param)

    def set_lower_rule(self, rule, rule_param):
        for idx in range(len(self.encoder_blocks) // 3):
            self._set_rule_to_block(self.encoder_blocks[idx], rule, rule_param)

    def set_middle_rule(self, rule, rule_param):
        for idx in range(len(self.encoder_blocks) // 3, 2 * len(self.encoder_blocks) // 3):
            self._set_rule_to_block(self.encoder_blocks[idx], rule, rule_param)

    def set_upper_rule(self, rule, rule_param):
        for idx in range(2 * len(self.encoder_blocks) // 3, len(self.encoder_blocks)):
            self._set_rule_to_block(self.encoder_blocks[idx], rule, rule_param)

    def disable_inplace(self):
        pass  # ViT does not use inplace ReLU, nothing to disable

class ImagenetContainerVGG11PointingGame(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True, every_layer=False):
        super(ImagenetContainerVGG11PointingGame, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert self.output_head_idx is None or (
            self.output_head_idx >= 0 and self.output_head_idx < self.num_heads)

        self.base_model = torchvision.models.vgg11(pretrained=True)

        self.conv_act_ids = [[1, 4, 7, 9, 12, 14, 17, 19],
                             [2, 5, 7, 10, 12, 15, 17, 20]]

        self.single_spatial = True
        self.single_head = True

        self.start_conv_layer_idx = None
        self.avgpool2d = torch.nn.AvgPool2d(kernel_size=8)
        self.use_softmax = False

        self.conv1 = torch.nn.Conv2d(512, 4096, kernel_size=7, padding=0)
        self.conv2 = torch.nn.Conv2d(4096, 4096, kernel_size=1, padding=0)
        self.conv3 = torch.nn.Conv2d(4096, 1000, kernel_size=1, padding=0)

        self.conv1.weight.data = self.base_model.classifier[0].weight.reshape(
            (4096, 512, 7, 7))
        self.conv1.bias.data = self.base_model.classifier[0].bias
        self.conv2.weight.data = self.base_model.classifier[3].weight.reshape(
            (4096, 4096, 1, 1))
        self.conv2.bias.data = self.base_model.classifier[3].bias
        self.conv3.weight.data = self.base_model.classifier[6].weight.reshape(
            (1000, 4096, 1, 1))
        self.conv3.bias.data = self.base_model.classifier[6].bias

    def randomize(self, layer_idx=-1, only_zplus_layers=False, independent=False):
        conv_layers = [0, 3, 6, 8, 11, 13, 16, 18][::-1]
        if layer_idx == -1:
            return
        if independent:
            if layer_idx == 0:
                torch.nn.init.normal_(self.conv3.weight, 0, 0.01)
                torch.nn.init.constant_(self.conv3.bias, 0)
            if layer_idx == 1:
                torch.nn.init.normal_(self.conv2.weight, 0, 0.01)
                torch.nn.init.constant_(self.conv2.bias, 0)
            if layer_idx == 2:
                torch.nn.init.normal_(self.conv1.weight, 0, 0.01)
                torch.nn.init.constant_(self.conv1.bias, 0)
            for idx, layer_pos in enumerate(conv_layers):
                if idx + 3 == layer_idx:
                    torch.nn.init.kaiming_normal_(
                        self.base_model.features[layer_pos].weight, mode="fan_out", nonlinearity="relu")
                    if self.base_model.features[layer_pos].bias is not None:
                        torch.nn.init.constant_(
                            self.base_model.features[layer_pos].bias, 0)
        else:
            if only_zplus_layers:
                conv_layers = conv_layers[:1]
            else:
                if layer_idx >= 0:
                    torch.nn.init.normal_(self.conv3.weight, 0, 0.01)
                    torch.nn.init.constant_(self.conv3.bias, 0)
                if layer_idx >= 1:
                    torch.nn.init.normal_(self.conv2.weight, 0, 0.01)
                    torch.nn.init.constant_(self.conv2.bias, 0)
                if layer_idx >= 2:
                    torch.nn.init.normal_(self.conv1.weight, 0, 0.01)
                    torch.nn.init.constant_(self.conv1.bias, 0)
            for idx, layer_pos in enumerate(conv_layers):
                if idx + 2 >= layer_idx:
                    break
                torch.nn.init.kaiming_normal_(
                    self.base_model.features[layer_pos].weight, mode="fan_out", nonlinearity="relu")
                if self.base_model.features[layer_pos].bias is not None:
                    torch.nn.init.constant_(
                        self.base_model.features[layer_pos].bias, 0)

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        for fidx in range(start_idx, len(self.base_model.features)):
            y = self.base_model.features[fidx](y)
            if (conv_layer_idx is not None) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                conv_acts = y

        outs = self.conv1(y)
        outs = self.base_model.classifier[1](outs)
        outs = self.base_model.classifier[2](outs)
        outs = self.conv2(outs)
        outs = self.base_model.classifier[4](outs)
        outs = self.base_model.classifier[5](outs)
        classifier_outputs = self.conv3(outs)
        if classifier_outputs.shape[2] == 1:
            outs = classifier_outputs.squeeze(2).squeeze(2)
        else:
            outs = self.avgpool2d(
                classifier_outputs).squeeze(2).squeeze(2)

        if self.use_softmax:
            outs = torch.nn.functional.softmax(outs, dim=1)

        if conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        # return self.__call__(x, *kwargs)
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        for idx, layer in enumerate(self.base_model.features.children()):
            if isinstance(layer, torch.nn.ReLU):
                self.base_model.features[idx].inplace = value
        for idx, layer in enumerate(self.base_model.classifier.children()):
            if isinstance(layer, torch.nn.ReLU):
                self.base_model.classifier[idx].inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def set_lower_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(1, 11):
            layer = self.base_model.features[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.features[idx].rule = rule()
                    else:
                        self.base_model.features[idx].rule = rule(rule_param)

    def set_input_rule(self, rule, rule_param):
        if rule_param is None:
            self.base_model.features[0].rule = rule()
        else:
            self.base_model.features[0].rule = rule(rule_param)

    def set_middle_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(11, 21):
            layer = self.base_model.features[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.features[idx].rule = rule()
                    else:
                        self.base_model.features[idx].rule = rule(rule_param)

    def set_upper_rule(self, rule, rule_param):
        if rule_param is None:
            self.avgpool2d.rule = rule()
            self.conv1.rule = rule()
            self.conv2.rule = rule()
            self.conv3.rule = rule()
        else:
            self.avgpool2d.rule = rule(rule_param)
            self.conv1.rule = rule(rule_param)
            self.conv2.rule = rule(rule_param)
            self.conv3.rule = rule(rule_param)


class ImagenetContainerVGG11Disconnected(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True):
        super(ImagenetContainerVGG11Disconnected, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        self.base_model = torchvision.models.vgg11(pretrained=True)

        self.conv_act_ids = [[1, 4, 7, 9, 12, 14, 17, 19],
                             [2, 5, 7, 10, 12, 15, 17, 20]]

        self.single_spatial = False
        self.single_head = False

        self.start_conv_layer_idx = None
        self.use_softmax = False

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None, loc_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        if conv_layer_idx is not None:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, loc_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.base_model.features)):
                y = self.base_model.features[fidx](y)
                if (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                    conv_acts = y
                    return None, conv_acts
        else:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, self.output_head_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.base_model.features)):
                y = self.base_model.features[fidx](y)

            print(y.shape)

            outs = self.base_model.classifier(y.reshape(y.shape[0], -1))

            if self.use_softmax:
                outs = torch.nn.functional.softmax(outs, dim=1)

            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        for idx, layer in enumerate(self.base_model.features.children()):
            if isinstance(layer, torch.nn.ReLU):
                self.base_model.features[idx].inplace = value
        for idx, layer in enumerate(self.base_model.classifier.children()):
            if isinstance(layer, torch.nn.ReLU):
                self.base_model.classifier[idx].inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def set_lower_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(1, 11):
            layer = self.base_model.features[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.features[idx].rule = rule()
                    else:
                        self.base_model.features[idx].rule = rule(rule_param)

    def set_input_rule(self, rule, rule_param):
        if rule_param is None:
            self.base_model.features[0].rule = rule()
        else:
            self.base_model.features[0].rule = rule(rule_param)

    def set_middle_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(11, 21):
            layer = self.base_model.features[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.features[idx].rule = rule()
                    else:
                        self.base_model.features[idx].rule = rule(rule_param)

    def set_upper_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(0, 7):
            layer = self.base_model.classifier[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.classifier[idx].rule = rule()
                    else:
                        self.base_model.classifier[idx].rule = rule(rule_param)


class ImagenetContainerVGG11Disconnected2(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True):
        super(ImagenetContainerVGG11Disconnected2, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        self.base_model = torchvision.models.vgg11(pretrained=True)

        new_linear = torch.nn.Linear(
            512 * 7 * 7 * self.scale * self.scale, 4096, bias=True)
        new_linear.weight.data.fill_(0.0)
        new_linear.bias.data = self.base_model.classifier[0].bias
        row_idx = output_head_idx // self.scale
        col_idx = output_head_idx % self.scale
        new_linear.weight.data.view((4096, 512, 7 * self.scale, 7 * self.scale))[:, :, row_idx * 7:(
            row_idx + 1) * 7, col_idx * 7:(col_idx + 1) * 7] = self.base_model.classifier[0].weight.view((4096, 512, 7, 7))

        self.base_model.classifier_mod = torch.nn.Sequential(new_linear, self.base_model.classifier[1], self.base_model.classifier[
                                                             2], self.base_model.classifier[3], self.base_model.classifier[4], self.base_model.classifier[5], self.base_model.classifier[6])

        self.base_model.feature_list = torch.nn.ModuleList([self.base_model.features, copy.deepcopy(
            self.base_model.features), copy.deepcopy(self.base_model.features), copy.deepcopy(self.base_model.features)])

        del self.base_model.features
        del self.base_model.classifier

        self.conv_act_ids = [[1, 4, 7, 9, 12, 14, 17, 19],
                             [2, 5, 7, 10, 12, 15, 17, 20]]

        self.single_spatial = False
        self.single_head = False

        self.start_conv_layer_idx = None
        self.use_softmax = False

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None, loc_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        if conv_layer_idx is not None:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, loc_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.base_model.feature_list[0])):
                y = self.base_model.feature_list[0][fidx](y)
                if (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                    conv_acts = y
                    return None, conv_acts
        else:
            combined = torch.zeros(
                (x.shape[0], 512, 7 * self.scale, 7 * self.scale)).to(x.device)
            for head_idx in range(self.scale * self.scale):
                y_coord, x_coord, h, w = get_augmentation_range(
                    x.shape, self.scale, head_idx)
                y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
                for fidx in range(start_idx, len(self.base_model.feature_list[0])):
                    y = self.base_model.feature_list[head_idx][fidx](y)
                yc, xc, hc, wc = get_augmentation_range(
                    combined.shape, self.scale, head_idx)
                combined[:, :, yc:yc + hc, xc:xc + wc] = y
            # print(combined.tolist())

            outs = self.base_model.classifier_mod(
                combined.reshape(combined.shape[0], -1))

            if self.use_softmax:
                outs = torch.nn.functional.softmax(outs, dim=1)

            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)


class ImagenetContainerVGG11Disconnected3(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True):
        super(ImagenetContainerVGG11Disconnected3, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        self.base_model = torchvision.models.vgg11(pretrained=True)

        new_linear1 = torch.nn.Linear(
            512 * 7 * 7 * self.scale * self.scale, 512 * 7 * 7 * self.scale * self.scale + 2, bias=False)
        new_linear1.weight.data.fill_(0.0)
        for idx in range(new_linear1.weight.shape[1]):
            new_linear1.weight.data[idx, idx] = 1.0
        hrow_idx = output_head_idx // self.scale
        hcol_idx = output_head_idx % self.scale
        for head_idx in range(self.scale * self.scale):
            row_idx = head_idx // self.scale
            col_idx = head_idx % self.scale
            if (row_idx == hrow_idx) and (col_idx == hcol_idx):
                continue
            new_linear1.weight.data.view((512 * 7 * 7 * self.scale * self.scale + 2, 512, 7 * self.scale,
                                          7 * self.scale))[-2, :, row_idx * 7:(row_idx + 1) * 7, col_idx * 7:(col_idx + 1) * 7] = 1.0
            new_linear1.weight.data.view((512 * 7 * 7 * self.scale * self.scale + 2, 512, 7 * self.scale,
                                          7 * self.scale))[-1, :, row_idx * 7:(row_idx + 1) * 7, col_idx * 7:(col_idx + 1) * 7] = -1.0

        new_linear2 = torch.nn.Linear(
            512 * 7 * 7 * self.scale * self.scale + 2, 4096, bias=True)
        new_linear2.weight.data.fill_(0.0)
        new_linear2.bias.data = self.base_model.classifier[0].bias
        new_linear2.weight.data[:, :-2].view((4096, 512, 7 * self.scale, 7 * self.scale))[:, :, hrow_idx * 7:(
            hrow_idx + 1) * 7, hcol_idx * 7:(hcol_idx + 1) * 7] = self.base_model.classifier[0].weight.view((4096, 512, 7, 7))
        new_linear2.weight.data[:, -2] = 1.0
        new_linear2.weight.data[:, -1] = 1.0

        self.base_model.classifier_mod = torch.nn.Sequential(new_linear1, new_linear2, self.base_model.classifier[1], self.base_model.classifier[
                                                             2], self.base_model.classifier[3], self.base_model.classifier[4], self.base_model.classifier[5], self.base_model.classifier[6])

        self.base_model.feature_list = torch.nn.ModuleList([self.base_model.features, copy.deepcopy(
            self.base_model.features), copy.deepcopy(self.base_model.features), copy.deepcopy(self.base_model.features)])

        del self.base_model.features
        del self.base_model.classifier

        self.conv_act_ids = [[1, 4, 7, 9, 12, 14, 17, 19],
                             [2, 5, 7, 10, 12, 15, 17, 20]]

        self.single_spatial = False
        self.single_head = False

        self.start_conv_layer_idx = None
        self.use_softmax = False

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None, loc_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        if conv_layer_idx is not None:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, loc_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.base_model.feature_list[0])):
                y = self.base_model.feature_list[0][fidx](y)
                if (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                    conv_acts = y
                    return None, conv_acts
        else:
            combined = torch.zeros(
                (x.shape[0], 512, 7 * self.scale, 7 * self.scale)).to(x.device)
            for head_idx in range(self.scale * self.scale):
                y_coord, x_coord, h, w = get_augmentation_range(
                    x.shape, self.scale, head_idx)
                y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
                for fidx in range(start_idx, len(self.base_model.feature_list[0])):
                    y = self.base_model.feature_list[head_idx][fidx](y)
                yc, xc, hc, wc = get_augmentation_range(
                    combined.shape, self.scale, head_idx)
                combined[:, :, yc:yc + hc, xc:xc + wc] = y

            outs = self.base_model.classifier_mod(
                combined.reshape(combined.shape[0], -1))

            if self.use_softmax:
                outs = torch.nn.functional.softmax(outs, dim=1)

            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)


class ImagenetContainerVGG11Disconnected4(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True):
        super(ImagenetContainerVGG11Disconnected4, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        self.base_model = torchvision.models.vgg11(pretrained=True)

        new_linear1 = torch.nn.Linear(
            512 * 7 * 7 * self.scale * self.scale, 512 * 7 * 7 + 2, bias=False)
        new_linear1.weight.data.fill_(0.0)
        hrow_idx = output_head_idx // self.scale
        hcol_idx = output_head_idx % self.scale
        temp_w = new_linear1.weight.data[:-2].reshape((512 * 7 * 7, 512, 7 * self.scale, 7 * self.scale))[:, :, hrow_idx * 7:(
            hrow_idx + 1) * 7, hcol_idx * 7:(hcol_idx + 1) * 7].reshape((512 * 7 * 7, 512 * 7 * 7))
        # print(temp_w)
        # print(temp_w.shape)
        for idx in range(temp_w.shape[0]):
            temp_w[idx, idx] = 1.0
        # print(temp_w)
        new_linear1.weight.data[:-2].reshape((512 * 7 * 7, 512, 7 * self.scale, 7 * self.scale))[:, :, hrow_idx * 7:(
            hrow_idx + 1) * 7, hcol_idx * 7:(hcol_idx + 1) * 7] = temp_w.reshape((512 * 7 * 7, 512, 7, 7))
        # print(torch.where(new_linear1.weight == 1),
        #   torch.where(new_linear1.weight == 1)[0].shape)
        # import sys
        # sys.exit(0)

        for head_idx in range(self.scale * self.scale):
            row_idx = head_idx // self.scale
            col_idx = head_idx % self.scale
            if (row_idx == hrow_idx) and (col_idx == hcol_idx):
                continue
            new_linear1.weight.data.view((512 * 7 * 7 + 2, 512, 7 * self.scale,
                                          7 * self.scale))[-2, :, row_idx * 7:(row_idx + 1) * 7, col_idx * 7:(col_idx + 1) * 7] = 1.0
            new_linear1.weight.data.view((512 * 7 * 7 + 2, 512, 7 * self.scale,
                                          7 * self.scale))[-1, :, row_idx * 7:(row_idx + 1) * 7, col_idx * 7:(col_idx + 1) * 7] = -1.0

        new_linear2 = torch.nn.Linear(
            512 * 7 * 7 + 2, 4096, bias=True)
        new_linear2.weight.data.fill_(0.0)
        new_linear2.bias.data = self.base_model.classifier[0].bias
        new_linear2.weight.data[:, :-2].view((4096, 512, 7, 7))[
            :] = self.base_model.classifier[0].weight.view((4096, 512, 7, 7))
        new_linear2.weight.data[:, -2] = 1.0
        new_linear2.weight.data[:, -1] = 1.0

        self.base_model.classifier_mod = torch.nn.Sequential(new_linear1, new_linear2, self.base_model.classifier[1], self.base_model.classifier[
                                                             2], self.base_model.classifier[3], self.base_model.classifier[4], self.base_model.classifier[5], self.base_model.classifier[6])

        self.base_model.feature_list = torch.nn.ModuleList([self.base_model.features, copy.deepcopy(
            self.base_model.features), copy.deepcopy(self.base_model.features), copy.deepcopy(self.base_model.features)])

        del self.base_model.features
        del self.base_model.classifier

        self.conv_act_ids = [[1, 4, 7, 9, 12, 14, 17, 19],
                             [2, 5, 7, 10, 12, 15, 17, 20]]

        self.single_spatial = False
        self.single_head = False

        self.start_conv_layer_idx = None
        self.use_softmax = False

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None, loc_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        if conv_layer_idx is not None:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, loc_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.base_model.feature_list[0])):
                y = self.base_model.feature_list[0][fidx](y)
                if (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                    conv_acts = y
                    return None, conv_acts
        else:
            combined = torch.zeros(
                (x.shape[0], 512, 7 * self.scale, 7 * self.scale)).to(x.device)
            for head_idx in range(self.scale * self.scale):
                y_coord, x_coord, h, w = get_augmentation_range(
                    x.shape, self.scale, head_idx)
                # print(y_coord, x_coord, h, w)
                y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
                for fidx in range(start_idx, len(self.base_model.feature_list[0])):
                    y = self.base_model.feature_list[head_idx][fidx](y)
                yc, xc, hc, wc = get_augmentation_range(
                    combined.shape, self.scale, head_idx)
                # print(yc, xc, hc, wc)
                combined[:, :, yc:yc + hc, xc:xc + wc] = y

            # print(combined.tolist())
            # import sys
            # sys.exit(0)
            outs = self.base_model.classifier_mod(
                combined.reshape(combined.shape[0], -1))
            # y = combined.reshape(combined.shape[0], -1)
            # y1 = self.base_model.classifier_mod[0](y)
            # y2 = self.base_model.classifier_mod[1](y1)
            # print(y1.shape, y2.shape)
            # print(y1.tolist(), y2.tolist())
            # sys.exit(0)
            # for idx in range(len(self.base_model.classifier_mod)):
            # print(y)
            # y = self.base_model.classifier_mod[idx](y)

            if self.use_softmax:
                outs = torch.nn.functional.softmax(outs, dim=1)

            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)


class ImagenetContainerVGG11CommonSpatial(torch.nn.Module):

    def __init__(self, scale=3, output_head_idx=None, transform_batchnorm=True):
        super(ImagenetContainerVGG11CommonSpatial, self).__init__()
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        self.base_model = torchvision.models.vgg11(pretrained=True)

        self.conv_act_ids = [[1, 4, 7, 9, 12, 14, 17, 19],
                             [2, 5, 7, 10, 12, 15, 17, 20]]

        self.single_spatial = True
        self.single_head = False

        self.start_conv_layer_idx = None
        self.use_softmax = False

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        for fidx in range(start_idx, len(self.base_model.features)):
            y = self.base_model.features[fidx](y)
            if (conv_layer_idx is not None) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                conv_acts = y

        ya, xa, ha, wa = get_augmentation_range(
            y.shape, self.scale, self.output_head_idx)

        outs = self.base_model.classifier(
            y[:, :, ya:ya + ha, xa:xa + wa].reshape(y.shape[0], -1))

        if self.use_softmax:
            outs = torch.nn.functional.softmax(outs, dim=1)

        if conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        for idx, layer in enumerate(self.base_model.features.children()):
            if isinstance(layer, torch.nn.ReLU):
                self.base_model.features[idx].inplace = value
        for idx, layer in enumerate(self.base_model.classifier.children()):
            if isinstance(layer, torch.nn.ReLU):
                self.base_model.classifier[idx].inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def set_lower_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(1, 11):
            layer = self.base_model.features[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.features[idx].rule = rule()
                    else:
                        self.base_model.features[idx].rule = rule(rule_param)

    def set_input_rule(self, rule, rule_param):
        if rule_param is None:
            self.base_model.features[0].rule = rule()
        else:
            self.base_model.features[0].rule = rule(rule_param)

    def set_middle_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(11, 21):
            layer = self.base_model.features[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.features[idx].rule = rule()
                    else:
                        self.base_model.features[idx].rule = rule(rule_param)

    def set_upper_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(0, 7):
            layer = self.base_model.classifier[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.classifier[idx].rule = rule()
                    else:
                        self.base_model.classifier[idx].rule = rule(rule_param)


class ImagenetContainerVGG11BNPointingGame(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True):
        super(ImagenetContainerVGG11BNPointingGame, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert self.output_head_idx is None or (
            self.output_head_idx >= 0 and self.output_head_idx < self.num_heads)

        self.base_model = torchvision.models.vgg11_bn(pretrained=True)

        self.conv_act_ids = [[2, 6, 10, 13, 17, 20, 24, 27]]

        self.single_spatial = True
        self.single_head = True

        self.start_conv_layer_idx = None
        self.avgpool2d = torch.nn.AvgPool2d(kernel_size=8)
        self.use_softmax = False

        self.conv1 = torch.nn.Conv2d(512, 4096, kernel_size=7, padding=0)
        self.conv2 = torch.nn.Conv2d(4096, 4096, kernel_size=1, padding=0)
        self.conv3 = torch.nn.Conv2d(4096, 1000, kernel_size=1, padding=0)

        self.conv1.weight.data = self.base_model.classifier[0].weight.reshape(
            (4096, 512, 7, 7))
        self.conv1.bias.data = self.base_model.classifier[0].bias
        self.conv2.weight.data = self.base_model.classifier[3].weight.reshape(
            (4096, 4096, 1, 1))
        self.conv2.bias.data = self.base_model.classifier[3].bias
        self.conv3.weight.data = self.base_model.classifier[6].weight.reshape(
            (1000, 4096, 1, 1))
        self.conv3.bias.data = self.base_model.classifier[6].bias

    def randomize(self, layer_idx=-1, only_zplus_layers=False, independent=False):
        conv_layers = [0, 4, 8, 11, 15, 18, 22, 25][::-1]
        if layer_idx == -1:
            return
        if independent:
            if layer_idx == 0:
                torch.nn.init.normal_(self.conv3.weight, 0, 0.01)
                torch.nn.init.constant_(self.conv3.bias, 0)
            if layer_idx == 1:
                torch.nn.init.normal_(self.conv2.weight, 0, 0.01)
                torch.nn.init.constant_(self.conv2.bias, 0)
            if layer_idx == 2:
                torch.nn.init.normal_(self.conv1.weight, 0, 0.01)
                torch.nn.init.constant_(self.conv1.bias, 0)
            for idx, layer_pos in enumerate(conv_layers):
                if idx + 3 == layer_idx:
                    torch.nn.init.kaiming_normal_(
                        self.base_model.features[layer_pos].weight, mode="fan_out", nonlinearity="relu")
                    if self.base_model.features[layer_pos].bias is not None:
                        torch.nn.init.constant_(
                            self.base_model.features[layer_pos].bias, 0)
                    torch.nn.init.constant_(
                        self.base_model.features[layer_pos + 1].weight, 1)
                    torch.nn.init.constant_(
                        self.base_model.features[layer_pos + 1].bias, 0)
        else:
            if only_zplus_layers:
                conv_layers = conv_layers[:1]
            else:
                if layer_idx >= 0:
                    torch.nn.init.normal_(self.conv3.weight, 0, 0.01)
                    torch.nn.init.constant_(self.conv3.bias, 0)
                if layer_idx >= 1:
                    torch.nn.init.normal_(self.conv2.weight, 0, 0.01)
                    torch.nn.init.constant_(self.conv2.bias, 0)
                if layer_idx >= 2:
                    torch.nn.init.normal_(self.conv1.weight, 0, 0.01)
                    torch.nn.init.constant_(self.conv1.bias, 0)
            for idx, layer_pos in enumerate(conv_layers):
                if idx + 2 >= layer_idx:
                    break
                torch.nn.init.kaiming_normal_(
                    self.base_model.features[layer_pos].weight, mode="fan_out", nonlinearity="relu")
                if self.base_model.features[layer_pos].bias is not None:
                    torch.nn.init.constant_(
                        self.base_model.features[layer_pos].bias, 0)
                torch.nn.init.constant_(
                    self.base_model.features[layer_pos + 1].weight, 1)
                torch.nn.init.constant_(
                    self.base_model.features[layer_pos + 1].bias, 0)

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        for fidx in range(start_idx, len(self.base_model.features)):
            y = self.base_model.features[fidx](y)
            if (conv_layer_idx is not None) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                conv_acts = y

        outs = self.conv1(y)
        outs = self.base_model.classifier[1](outs)
        outs = self.base_model.classifier[2](outs)
        outs = self.conv2(outs)
        outs = self.base_model.classifier[4](outs)
        outs = self.base_model.classifier[5](outs)
        classifier_outputs = self.conv3(outs)

        outs = self.avgpool2d(
            classifier_outputs).squeeze(2).squeeze(2)

        if self.use_softmax:
            outs = torch.nn.functional.softmax(outs, dim=1)

        if conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        # return self.__call__(x, *kwargs)
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        for idx, layer in enumerate(self.base_model.features.children()):
            if isinstance(layer, torch.nn.ReLU):
                self.base_model.features[idx].inplace = value
        for idx, layer in enumerate(self.base_model.classifier.children()):
            if isinstance(layer, torch.nn.ReLU):
                self.base_model.classifier[idx].inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def set_lower_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(1, 11):
            layer = self.base_model.features[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.features[idx].rule = rule()
                    else:
                        self.base_model.features[idx].rule = rule(rule_param)

    def set_input_rule(self, rule, rule_param):
        if rule_param is None:
            self.base_model.features[0].rule = rule()
        else:
            self.base_model.features[0].rule = rule(rule_param)

    def set_middle_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(11, 21):
            layer = self.base_model.features[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.features[idx].rule = rule()
                    else:
                        self.base_model.features[idx].rule = rule(rule_param)

    def set_upper_rule(self, rule, rule_param):
        if rule_param is None:
            self.avgpool2d.rule = rule()
            self.conv1.rule = rule()
            self.conv2.rule = rule()
            self.conv3.rule = rule()
        else:
            self.avgpool2d.rule = rule(rule_param)
            self.conv1.rule = rule(rule_param)
            self.conv2.rule = rule(rule_param)
            self.conv3.rule = rule(rule_param)


class ImagenetContainerVGG11BNDisconnected(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True):
        super(ImagenetContainerVGG11BNDisconnected, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        self.base_model = torchvision.models.vgg11_bn(pretrained=True)

        self.conv_act_ids = [[2, 6, 10, 13, 17, 20, 24, 27]]

        self.single_spatial = False
        self.single_head = False

        self.start_conv_layer_idx = None
        self.use_softmax = False

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None, loc_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        if conv_layer_idx is not None:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, loc_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.base_model.features)):
                y = self.base_model.features[fidx](y)
                if (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                    conv_acts = y
                    return None, conv_acts
        else:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, self.output_head_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.base_model.features)):
                y = self.base_model.features[fidx](y)

            outs = self.base_model.classifier(y.reshape(y.shape[0], -1))

            if self.use_softmax:
                outs = torch.nn.functional.softmax(outs, dim=1)

            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        for idx, layer in enumerate(self.base_model.features.children()):
            if isinstance(layer, torch.nn.ReLU):
                self.base_model.features[idx].inplace = value
        for idx, layer in enumerate(self.base_model.classifier.children()):
            if isinstance(layer, torch.nn.ReLU):
                self.base_model.classifier[idx].inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def set_lower_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(1, 11):
            layer = self.base_model.features[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.features[idx].rule = rule()
                    else:
                        self.base_model.features[idx].rule = rule(rule_param)

    def set_input_rule(self, rule, rule_param):
        if rule_param is None:
            self.base_model.features[0].rule = rule()
        else:
            self.base_model.features[0].rule = rule(rule_param)

    def set_middle_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(11, 21):
            layer = self.base_model.features[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.features[idx].rule = rule()
                    else:
                        self.base_model.features[idx].rule = rule(rule_param)

    def set_upper_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(0, 7):
            layer = self.base_model.classifier[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.classifier[idx].rule = rule()
                    else:
                        self.base_model.classifier[idx].rule = rule(rule_param)


class ImagenetContainerVGG11BNCommonSpatial(torch.nn.Module):

    def __init__(self, scale=3, output_head_idx=None, transform_batchnorm=True):
        super(ImagenetContainerVGG11BNCommonSpatial, self).__init__()
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        self.base_model = torchvision.models.vgg11_bn(pretrained=True)

        self.conv_act_ids = [[2, 6, 10, 13, 17, 20, 24, 27]]

        self.single_spatial = True
        self.single_head = False

        self.start_conv_layer_idx = None
        self.use_softmax = False

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        for fidx in range(start_idx, len(self.base_model.features)):
            y = self.base_model.features[fidx](y)
            if (conv_layer_idx is not None) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                conv_acts = y

        ya, xa, ha, wa = get_augmentation_range(
            y.shape, self.scale, self.output_head_idx)

        outs = self.base_model.classifier(
            y[:, :, ya:ya + ha, xa:xa + wa].reshape(y.shape[0], -1))

        if self.use_softmax:
            outs = torch.nn.functional.softmax(outs, dim=1)

        if conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        for idx, layer in enumerate(self.base_model.features.children()):
            if isinstance(layer, torch.nn.ReLU):
                self.base_model.features[idx].inplace = value
        for idx, layer in enumerate(self.base_model.classifier.children()):
            if isinstance(layer, torch.nn.ReLU):
                self.base_model.classifier[idx].inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def set_lower_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(1, 11):
            layer = self.base_model.features[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.features[idx].rule = rule()
                    else:
                        self.base_model.features[idx].rule = rule(rule_param)

    def set_input_rule(self, rule, rule_param):
        if rule_param is None:
            self.base_model.features[0].rule = rule()
        else:
            self.base_model.features[0].rule = rule(rule_param)

    def set_middle_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(11, 21):
            layer = self.base_model.features[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.features[idx].rule = rule()
                    else:
                        self.base_model.features[idx].rule = rule(rule_param)

    def set_upper_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(0, 7):
            layer = self.base_model.classifier[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.classifier[idx].rule = rule()
                    else:
                        self.base_model.classifier[idx].rule = rule(rule_param)


class ImagenetContainerVGG11NoClassifierKernelPointingGame(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True):
        super(ImagenetContainerVGG11NoClassifierKernelPointingGame, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert self.output_head_idx is None or (
            self.output_head_idx >= 0 and self.output_head_idx < self.num_heads)

        self.base_model = torchvision.models.vgg11(pretrained=True)

        self.conv_act_ids = [[1, 4, 7, 9, 12, 14, 17, 19],
                             [2, 5, 7, 10, 12, 15, 17, 20]]

        self.single_spatial = True
        self.single_head = True
        self.use_softmax = False

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        for fidx in range(start_idx, len(self.base_model.features)):
            y = self.base_model.features[fidx](y)
            if (conv_layer_idx is not None) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                conv_acts = y

        y = self.base_model.avgpool(y)
        outs = self.base_model.classifier(y.flatten(start_dim=1))

        if self.use_softmax:
            outs = torch.nn.functional.softmax(outs, dim=1)

        if conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        for idx, layer in enumerate(self.base_model.features.children()):
            if isinstance(layer, torch.nn.ReLU):
                self.base_model.features[idx].inplace = value
        for idx, layer in enumerate(self.base_model.classifier.children()):
            if isinstance(layer, torch.nn.ReLU):
                self.base_model.classifier[idx].inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def set_lower_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(1, 11):
            layer = self.base_model.features[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.features[idx].rule = rule()
                    else:
                        self.base_model.features[idx].rule = rule(rule_param)

    def set_input_rule(self, rule, rule_param):
        if rule_param is None:
            self.base_model.features[0].rule = rule()
        else:
            self.base_model.features[0].rule = rule(rule_param)

    def set_middle_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(11, 21):
            layer = self.base_model.features[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.features[idx].rule = rule()
                    else:
                        self.base_model.features[idx].rule = rule(rule_param)
        if rule_param is None:
            self.base_model.avgpool.rule = rule()
        else:
            self.base_model.avgpool.rule = rule(rule_param)

    def set_upper_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(0, 7):
            layer = self.base_model.classifier[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.classifier[idx].rule = rule()
                    else:
                        self.base_model.classifier[idx].rule = rule(rule_param)


####################################################################################################

class ImagenetContainerResnet18PointingGame(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True, batchnorm_to_conv=False, every_layer=False):
        super(ImagenetContainerResnet18PointingGame, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert self.output_head_idx is None or (
            self.output_head_idx >= 0 and self.output_head_idx < self.num_heads)

        # self.base_model = torchvision.models.resnet18(pretrained=True)
        if batchnorm_to_conv:
            self._load_batchnorm_to_conv_model()
            self.conv_act_ids = [[2, 4, 5, 6, 7]]
        elif transform_batchnorm:
            self._load_model()
            self.conv_act_ids = [[1, 3, 4, 5, 6]]  # [[2, 4, 5, 6, 7]]
        else:
            self.base_model = torchvision.models.resnet18(pretrained=True)
            self.conv_act_ids = [[2, 4, 5, 6, 7]]

        self.single_spatial = True
        self.single_head = True

        self.all_layers = list(self.base_model.named_children())

        self.start_conv_layer_idx = None

    def randomize(self, layer_idx=-1, only_zplus_layers=False, independent=False):
        if layer_idx == -1:
            return
        if independent:
            if layer_idx == 0:
                torch.nn.init.normal_(self.base_model.fc.weight, 0, 0.01)
                torch.nn.init.constant_(self.base_model.fc.bias, 0)
            if layer_idx == 1:
                for m in self.base_model.layer4.modules():
                    if isinstance(m, torch.nn.Conv2d):
                        torch.nn.init.kaiming_normal_(
                            m.weight, mode="fan_out", nonlinearity="relu")
                    elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                        torch.nn.init.constant_(m.weight, 1)
                        torch.nn.init.constant_(m.bias, 0)
            if layer_idx == 2:
                for m in self.base_model.layer3.modules():
                    if isinstance(m, torch.nn.Conv2d):
                        torch.nn.init.kaiming_normal_(
                            m.weight, mode="fan_out", nonlinearity="relu")
                    elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                        torch.nn.init.constant_(m.weight, 1)
                        torch.nn.init.constant_(m.bias, 0)
            if layer_idx == 3:
                for m in self.base_model.layer2.modules():
                    if isinstance(m, torch.nn.Conv2d):
                        torch.nn.init.kaiming_normal_(
                            m.weight, mode="fan_out", nonlinearity="relu")
                    elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                        torch.nn.init.constant_(m.weight, 1)
                        torch.nn.init.constant_(m.bias, 0)
            if layer_idx == 4:
                for m in self.base_model.layer1.modules():
                    if isinstance(m, torch.nn.Conv2d):
                        torch.nn.init.kaiming_normal_(
                            m.weight, mode="fan_out", nonlinearity="relu")
                    elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                        torch.nn.init.constant_(m.weight, 1)
                        torch.nn.init.constant_(m.bias, 0)
            if layer_idx == 5:
                torch.nn.init.kaiming_normal_(
                    self.base_model.conv1.weight, mode="fan_out", nonlinearity="relu")
                torch.nn.init.constant_(self.base_model.bn1.weight, 1)
                torch.nn.init.constant_(self.base_model.bn1.bias, 0)
        else:
            if layer_idx >= 0 and not only_zplus_layers:
                torch.nn.init.normal_(self.base_model.fc.weight, 0, 0.01)
                torch.nn.init.constant_(self.base_model.fc.bias, 0)
            if layer_idx >= 1:
                for m in self.base_model.layer4.modules():
                    if isinstance(m, torch.nn.Conv2d):
                        torch.nn.init.kaiming_normal_(
                            m.weight, mode="fan_out", nonlinearity="relu")
                    elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                        torch.nn.init.constant_(m.weight, 1)
                        torch.nn.init.constant_(m.bias, 0)
            if layer_idx >= 2:
                for m in self.base_model.layer3.modules():
                    if isinstance(m, torch.nn.Conv2d):
                        torch.nn.init.kaiming_normal_(
                            m.weight, mode="fan_out", nonlinearity="relu")
                    elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                        torch.nn.init.constant_(m.weight, 1)
                        torch.nn.init.constant_(m.bias, 0)
            if layer_idx >= 3:
                for m in self.base_model.layer2.modules():
                    if isinstance(m, torch.nn.Conv2d):
                        torch.nn.init.kaiming_normal_(
                            m.weight, mode="fan_out", nonlinearity="relu")
                    elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                        torch.nn.init.constant_(m.weight, 1)
                        torch.nn.init.constant_(m.bias, 0)
            if layer_idx >= 4:
                for m in self.base_model.layer1.modules():
                    if isinstance(m, torch.nn.Conv2d):
                        torch.nn.init.kaiming_normal_(
                            m.weight, mode="fan_out", nonlinearity="relu")
                    elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                        torch.nn.init.constant_(m.weight, 1)
                        torch.nn.init.constant_(m.bias, 0)
            if layer_idx >= 5 and not only_zplus_layers:
                torch.nn.init.kaiming_normal_(
                    self.base_model.conv1.weight, mode="fan_out", nonlinearity="relu")
                torch.nn.init.constant_(self.base_model.bn1.weight, 1)
                torch.nn.init.constant_(self.base_model.bn1.bias, 0)

    def _load_model(self):
        self.base_model = resnet_for_lrp.resnet18(pretrained=False)
        state_dict = torch.hub.load_state_dict_from_url(
            resnet_for_lrp.model_urls['resnet18'])
        # print(state_dict.keys())
        self._update_state_dict(state_dict)
        self.base_model.load_state_dict(state_dict)

    def _load_batchnorm_to_conv_model(self):
        self.base_model = resnet_for_lrp_conv.resnet18(pretrained=False)
        # print(self.base_model)
        state_dict = torch.hub.load_state_dict_from_url(
            resnet_for_lrp_conv.model_urls['resnet18'])
        # print(state_dict.keys())
        self._update_state_dict_bn_to_conv(state_dict)
        self.base_model.load_state_dict(state_dict)

    def _update_state_dict(self, state_dict):
        self._update_state_dict_single(state_dict, "bn1", "conv1")
        for layer in range(4):
            block_idx = 0
            end_of_block = False
            base_layer_name = "layer" + str(layer + 1)
            while not end_of_block:
                conv_bn_idx = 0
                base_block_name = base_layer_name + "." + str(block_idx)
                while True:
                    conv_layer_name = base_block_name + \
                        ".conv" + str(conv_bn_idx + 1)
                    bn_layer_name = base_block_name + \
                        ".bn" + str(conv_bn_idx + 1)
                    if conv_layer_name + ".weight" not in state_dict:
                        if conv_bn_idx == 0:
                            end_of_block = True
                        break
                    self._update_state_dict_single(
                        state_dict, bn_layer_name, conv_layer_name)
                    conv_bn_idx += 1
                downsample_conv_name = base_block_name + ".downsample.0"
                downsample_bn_name = base_block_name + ".downsample.1"
                if downsample_conv_name + ".weight" in state_dict:
                    self._update_state_dict_single(
                        state_dict, downsample_bn_name, downsample_conv_name)
                block_idx += 1

    def _update_state_dict_bn_to_conv(self, state_dict):
        self._update_state_dict_single_conv(state_dict, "bn1")
        for layer in range(4):
            block_idx = 0
            end_of_block = False
            base_layer_name = "layer" + str(layer + 1)
            while not end_of_block:
                conv_bn_idx = 0
                base_block_name = base_layer_name + "." + str(block_idx)
                while True:
                    conv_layer_name = base_block_name + \
                        ".conv" + str(conv_bn_idx + 1)
                    bn_layer_name = base_block_name + \
                        ".bn" + str(conv_bn_idx + 1)
                    if conv_layer_name + ".weight" not in state_dict:
                        if conv_bn_idx == 0:
                            end_of_block = True
                        break
                    self._update_state_dict_single_conv(
                        state_dict, bn_layer_name)
                    conv_bn_idx += 1
                downsample_conv_name = base_block_name + ".downsample.0"
                downsample_bn_name = base_block_name + ".downsample.1"
                if downsample_conv_name + ".weight" in state_dict:
                    self._update_state_dict_single_conv(
                        state_dict, downsample_bn_name)
                block_idx += 1

    def _update_state_dict_single(self, state_dict, name1, name2):
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"], state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"])
        for idx in range(len(conv_weight)):
            state_dict[name2 + ".weight"].data[idx] *= conv_weight[idx]
        state_dict[name2 + ".bias"] = conv_bias
        del state_dict[name1 + ".weight"]
        del state_dict[name1 + ".bias"]
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]

    def _update_state_dict_single_conv(self, state_dict, name1):
        # print(name1)
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"], state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"])
        conv_weight_matrix = torch.zeros(
            (len(conv_weight), len(conv_weight), 1, 1))
        for idx in range(len(conv_weight)):
            conv_weight_matrix.data[idx, idx] = conv_weight[idx]
        state_dict[name1 + ".weight"] = conv_weight_matrix
        state_dict[name1 + ".bias"] = conv_bias
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]

    def _get_equivalent_conv_weights(self, weight, bias, running_mean, running_var):
        num_features = len(weight)
        eps = 1e-5
        conv_weight = torch.zeros(num_features)
        conv_bias = torch.zeros(num_features)
        conv_weight.data = weight / torch.sqrt(running_var + eps)
        conv_bias.data = ((-1.0 * running_mean * weight) /
                          torch.sqrt(running_var + eps)) + bias
        return conv_weight, conv_bias

    def _remove_bias(self):
        self.base_model.conv1.bias = None
        for layer in range(4):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    conv_layer.bias = None
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    downsample_conv.bias = None
        self.base_model.fc.bias = None

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        # print(x.get_device(), self.all_layers[0][1].weight.get_device())
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        for fidx in range(start_idx, len(self.all_layers) - 1):
            y = self.all_layers[fidx][1](y)
            if (conv_layer_idx is not None) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                conv_acts = y

        outs = self.all_layers[-1][1](y.reshape(y.shape[0], -1))

        if conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        self.base_model.relu.inplace = value
        for layer in range(4):
            block = eval("self.base_model.layer" + str(layer + 1))
            for block_idx in range(len(block)):
                block[block_idx].relu.inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def _assign_rule(self, layer, rule, rule_param):
        if rule_param is None:
            layer.rule = rule()
        else:
            layer.rule = rule(rule_param)

    def set_lower_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.maxpool, rule, rule_param)
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_middle_rule(self, rule, rule_param):
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 3)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_upper_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.fc, rule, rule_param)

    def set_input_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.conv1, rule, rule_param)


class ImagenetContainerResnet18Disconnected(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True, batchnorm_to_conv=False):
        super(ImagenetContainerResnet18Disconnected, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        # self.base_model = torchvision.models.resnet18(pretrained=True)
        if batchnorm_to_conv:
            self._load_batchnorm_to_conv_model()
            self.conv_act_ids = [[2, 4, 5, 6, 7]]
        elif transform_batchnorm:
            self._load_model()
            self.conv_act_ids = [[1, 3, 4, 5, 6]]  # [[2, 4, 5, 6, 7]]
        else:
            self.base_model = torchvision.models.resnet18(pretrained=True)
            self.conv_act_ids = [[2, 4, 5, 6, 7]]

        self.single_spatial = False
        self.single_head = False

        self.all_layers = list(self.base_model.named_children())

        self.start_conv_layer_idx = None

    def _load_model(self):
        self.base_model = resnet_for_lrp.resnet18(pretrained=False)
        state_dict = torch.hub.load_state_dict_from_url(
            resnet_for_lrp.model_urls['resnet18'])
        # print(state_dict.keys())
        self._update_state_dict(state_dict)
        self.base_model.load_state_dict(state_dict)

    def _load_batchnorm_to_conv_model(self):
        self.base_model = resnet_for_lrp_conv.resnet18(pretrained=False)
        # print(self.base_model)
        state_dict = torch.hub.load_state_dict_from_url(
            resnet_for_lrp_conv.model_urls['resnet18'])
        # print(state_dict.keys())
        self._update_state_dict_bn_to_conv(state_dict)
        self.base_model.load_state_dict(state_dict)

    def _update_state_dict(self, state_dict):
        self._update_state_dict_single(state_dict, "bn1", "conv1")
        for layer in range(4):
            block_idx = 0
            end_of_block = False
            base_layer_name = "layer" + str(layer + 1)
            while not end_of_block:
                conv_bn_idx = 0
                base_block_name = base_layer_name + "." + str(block_idx)
                while True:
                    conv_layer_name = base_block_name + \
                        ".conv" + str(conv_bn_idx + 1)
                    bn_layer_name = base_block_name + \
                        ".bn" + str(conv_bn_idx + 1)
                    if conv_layer_name + ".weight" not in state_dict:
                        if conv_bn_idx == 0:
                            end_of_block = True
                        break
                    self._update_state_dict_single(
                        state_dict, bn_layer_name, conv_layer_name)
                    conv_bn_idx += 1
                downsample_conv_name = base_block_name + ".downsample.0"
                downsample_bn_name = base_block_name + ".downsample.1"
                if downsample_conv_name + ".weight" in state_dict:
                    self._update_state_dict_single(
                        state_dict, downsample_bn_name, downsample_conv_name)
                block_idx += 1

    def _update_state_dict_bn_to_conv(self, state_dict):
        self._update_state_dict_single_conv(state_dict, "bn1")
        for layer in range(4):
            block_idx = 0
            end_of_block = False
            base_layer_name = "layer" + str(layer + 1)
            while not end_of_block:
                conv_bn_idx = 0
                base_block_name = base_layer_name + "." + str(block_idx)
                while True:
                    conv_layer_name = base_block_name + \
                        ".conv" + str(conv_bn_idx + 1)
                    bn_layer_name = base_block_name + \
                        ".bn" + str(conv_bn_idx + 1)
                    if conv_layer_name + ".weight" not in state_dict:
                        if conv_bn_idx == 0:
                            end_of_block = True
                        break
                    self._update_state_dict_single_conv(
                        state_dict, bn_layer_name)
                    conv_bn_idx += 1
                downsample_conv_name = base_block_name + ".downsample.0"
                downsample_bn_name = base_block_name + ".downsample.1"
                if downsample_conv_name + ".weight" in state_dict:
                    self._update_state_dict_single_conv(
                        state_dict, downsample_bn_name)
                block_idx += 1

    def _update_state_dict_single(self, state_dict, name1, name2):
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"], state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"])
        for idx in range(len(conv_weight)):
            state_dict[name2 + ".weight"].data[idx] *= conv_weight[idx]
        state_dict[name2 + ".bias"] = conv_bias
        del state_dict[name1 + ".weight"]
        del state_dict[name1 + ".bias"]
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]

    def _update_state_dict_single_conv(self, state_dict, name1):
        # print(name1)
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"], state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"])
        conv_weight_matrix = torch.zeros(
            (len(conv_weight), len(conv_weight), 1, 1))
        for idx in range(len(conv_weight)):
            conv_weight_matrix.data[idx, idx] = conv_weight[idx]
        state_dict[name1 + ".weight"] = conv_weight_matrix
        state_dict[name1 + ".bias"] = conv_bias
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]

    def _get_equivalent_conv_weights(self, weight, bias, running_mean, running_var):
        num_features = len(weight)
        eps = 1e-5
        conv_weight = torch.zeros(num_features)
        conv_bias = torch.zeros(num_features)
        conv_weight.data = weight / torch.sqrt(running_var + eps)
        conv_bias.data = ((-1.0 * running_mean * weight) /
                          torch.sqrt(running_var + eps)) + bias
        return conv_weight, conv_bias

    def _remove_bias(self):
        self.base_model.conv1.bias = None
        for layer in range(4):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    conv_layer.bias = None
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    downsample_conv.bias = None
        self.base_model.fc.bias = None

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None, loc_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        if conv_layer_idx is not None:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, loc_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.all_layers) - 1):
                y = self.all_layers[fidx][1](y)
                if (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                    conv_acts = y
                    return None, conv_acts
        else:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, self.output_head_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.all_layers) - 1):
                y = self.all_layers[fidx][1](y)

            outs = self.base_model.fc(y.reshape(y.shape[0], -1))

        return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        self.base_model.relu.inplace = value
        for layer in range(4):
            block = eval("self.base_model.layer" + str(layer + 1))
            for block_idx in range(len(block)):
                block[block_idx].relu.inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def _assign_rule(self, layer, rule, rule_param):
        if rule_param is None:
            layer.rule = rule()
        else:
            layer.rule = rule(rule_param)

    def set_lower_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.maxpool, rule, rule_param)
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_middle_rule(self, rule, rule_param):
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 3)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_upper_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.fc, rule, rule_param)

    def set_input_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.conv1, rule, rule_param)


class ImagenetContainerResnet18Disconnected2(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True, batchnorm_to_conv=False):
        super(ImagenetContainerResnet18Disconnected2, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        if batchnorm_to_conv:
            raise NotImplementedError
        elif transform_batchnorm:
            raise NotImplementedError
        else:
            self.base_model = torchvision.models.resnet18(pretrained=True)
            self.conv_act_ids = [[2, 4, 5, 6, 7]]

        self.base_model.fc_mod = torch.nn.Linear(
            512 * self.scale * self.scale, 1000, bias=True)
        self.base_model.fc_mod.weight.data.fill_(0.0)
        self.base_model.fc_mod.bias.data = self.base_model.fc.bias
        row_idx = output_head_idx // self.scale
        col_idx = output_head_idx % self.scale
        self.base_model.fc_mod.weight.data.view((1000, 512, self.scale, self.scale))[
            :, :, row_idx:row_idx + 1, col_idx:col_idx + 1] = self.base_model.fc.weight.view((1000, 512, 1, 1))
        self.base_model.features = torch.nn.Sequential(self.base_model.conv1, self.base_model.bn1, self.base_model.relu, self.base_model.maxpool,
                                                       self.base_model.layer1, self.base_model.layer2, self.base_model.layer3, self.base_model.layer4, self.base_model.avgpool)

        self.base_model.feature_list = torch.nn.ModuleList([copy.deepcopy(
            self.base_model.features) for _ in range(self.scale * self.scale)])

        del self.base_model.features
        del self.base_model.fc
        del self.base_model.conv1
        del self.base_model.bn1
        del self.base_model.relu
        del self.base_model.maxpool
        del self.base_model.layer1
        del self.base_model.layer2
        del self.base_model.layer3
        del self.base_model.layer4
        del self.base_model.avgpool

        self.single_spatial = False
        self.single_head = False

        self.start_conv_layer_idx = None

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None, loc_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        if conv_layer_idx is not None:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, loc_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.base_model.feature_list[0])):
                y = self.base_model.feature_list[0][fidx](y)
                if (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                    conv_acts = y
                    return None, conv_acts
        else:
            combined = torch.zeros(
                (x.shape[0], 512, self.scale, self.scale)).to(x.device)
            for head_idx in range(self.scale * self.scale):
                y_coord, x_coord, h, w = get_augmentation_range(
                    x.shape, self.scale, head_idx)
                y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
                for fidx in range(start_idx, len(self.base_model.feature_list[0])):
                    y = self.base_model.feature_list[head_idx][fidx](y)
                yc, xc, hc, wc = get_augmentation_range(
                    combined.shape, self.scale, head_idx)
                combined[:, :, yc:yc + hc, xc:xc + wc] = y

            outs = self.base_model.fc_mod(
                combined.reshape(combined.shape[0], -1))

        return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)


class ImagenetContainerResnet18Disconnected4(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True, batchnorm_to_conv=False):
        super(ImagenetContainerResnet18Disconnected4, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        if batchnorm_to_conv:
            raise NotImplementedError
        elif transform_batchnorm:
            raise NotImplementedError
        else:
            self.base_model = torchvision.models.resnet18(pretrained=True)
            self.conv_act_ids = [[2, 4, 5, 6, 7]]

        new_linear1 = torch.nn.Linear(
            512 * self.scale * self.scale, 512 + 2, bias=False)
        new_linear1.weight.data.fill_(0.0)
        hrow_idx = output_head_idx // self.scale
        hcol_idx = output_head_idx % self.scale
        temp_w = new_linear1.weight.data[:-2].reshape((512, 512, self.scale, self.scale))[
            :, :, hrow_idx:hrow_idx + 1, hcol_idx:hcol_idx + 1].reshape((512, 512))
        for idx in range(temp_w.shape[0]):
            temp_w[idx, idx] = 1.0
        new_linear1.weight.data[:-2].reshape((512, 512, self.scale, self.scale))[
            :, :, hrow_idx:hrow_idx + 1, hcol_idx:hcol_idx + 1] = temp_w.reshape((512, 512, 1, 1))

        for head_idx in range(self.scale * self.scale):
            row_idx = head_idx // self.scale
            col_idx = head_idx % self.scale
            if (row_idx == hrow_idx) and (col_idx == hcol_idx):
                continue
            new_linear1.weight.data.view(
                (512 + 2, 512, self.scale, self.scale))[-2, :, row_idx:row_idx + 1, col_idx:col_idx + 1] = 1.0
            new_linear1.weight.data.view(
                (512 + 2, 512, self.scale, self.scale))[-1, :, row_idx:row_idx + 1, col_idx:col_idx + 1] = -1.0

        new_linear2 = torch.nn.Linear(512 + 2, 1000, bias=True)
        new_linear2.weight.data.fill_(0.0)
        new_linear2.bias.data = self.base_model.fc.bias
        new_linear2.weight.data[:, :-2].view((1000, 512, 1, 1))[
            :] = self.base_model.fc.weight.view((1000, 512, 1, 1))
        new_linear2.weight.data[:, -2] = 1.0
        new_linear2.weight.data[:, -1] = 1.0

        self.base_model.fc_mod = torch.nn.Sequential(new_linear1, new_linear2)

        self.base_model.features = torch.nn.Sequential(self.base_model.conv1, self.base_model.bn1, self.base_model.relu, self.base_model.maxpool,
                                                       self.base_model.layer1, self.base_model.layer2, self.base_model.layer3, self.base_model.layer4, self.base_model.avgpool)

        self.base_model.feature_list = torch.nn.ModuleList([copy.deepcopy(
            self.base_model.features) for _ in range(self.scale * self.scale)])

        del self.base_model.features
        del self.base_model.fc
        del self.base_model.conv1
        del self.base_model.bn1
        del self.base_model.relu
        del self.base_model.maxpool
        del self.base_model.layer1
        del self.base_model.layer2
        del self.base_model.layer3
        del self.base_model.layer4
        del self.base_model.avgpool

        self.single_spatial = False
        self.single_head = False

        self.start_conv_layer_idx = None

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None, loc_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        if conv_layer_idx is not None:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, loc_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.base_model.feature_list[0])):
                y = self.base_model.feature_list[0][fidx](y)
                if (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                    conv_acts = y
                    return None, conv_acts
        else:
            combined = torch.zeros(
                (x.shape[0], 512, self.scale, self.scale)).to(x.device)
            for head_idx in range(self.scale * self.scale):
                y_coord, x_coord, h, w = get_augmentation_range(
                    x.shape, self.scale, head_idx)
                y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
                for fidx in range(start_idx, len(self.base_model.feature_list[0])):
                    y = self.base_model.feature_list[head_idx][fidx](y)
                yc, xc, hc, wc = get_augmentation_range(
                    combined.shape, self.scale, head_idx)
                combined[:, :, yc:yc + hc, xc:xc + wc] = y

            outs = self.base_model.fc_mod(
                combined.reshape(combined.shape[0], -1))

        return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)


class ImagenetContainerResnet18CommonSpatial(torch.nn.Module):

    def __init__(self, scale=3, output_head_idx=None, transform_batchnorm=True, batchnorm_to_conv=False):
        super(ImagenetContainerResnet18CommonSpatial, self).__init__()
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        # self.base_model = torchvision.models.resnet18(pretrained=True)
        if batchnorm_to_conv:
            self._load_batchnorm_to_conv_model()
            self.conv_act_ids = [[2, 4, 5, 6, 7]]
        elif transform_batchnorm:
            self._load_model()
            self.conv_act_ids = [[1, 3, 4, 5, 6]]  # [[2, 4, 5, 6, 7]]
        else:
            self.base_model = torchvision.models.resnet18(pretrained=True)
            self.conv_act_ids = [[2, 4, 5, 6, 7]]

        self.pooled_dims = (7, 7)
        self.single_spatial = True
        self.single_head = False

        self.all_layers = list(self.base_model.named_children())

        self.start_conv_layer_idx = None

    def _load_model(self):
        self.base_model = resnet_for_lrp.resnet18(pretrained=False)
        state_dict = torch.hub.load_state_dict_from_url(
            resnet_for_lrp.model_urls['resnet18'])
        # print(state_dict.keys())
        self._update_state_dict(state_dict)
        self.base_model.load_state_dict(state_dict)

    def _load_batchnorm_to_conv_model(self):
        self.base_model = resnet_for_lrp_conv.resnet18(pretrained=False)
        # print(self.base_model)
        state_dict = torch.hub.load_state_dict_from_url(
            resnet_for_lrp_conv.model_urls['resnet18'])
        # print(state_dict.keys())
        self._update_state_dict_bn_to_conv(state_dict)
        self.base_model.load_state_dict(state_dict)

    def _update_state_dict(self, state_dict):
        self._update_state_dict_single(state_dict, "bn1", "conv1")
        for layer in range(4):
            block_idx = 0
            end_of_block = False
            base_layer_name = "layer" + str(layer + 1)
            while not end_of_block:
                conv_bn_idx = 0
                base_block_name = base_layer_name + "." + str(block_idx)
                while True:
                    conv_layer_name = base_block_name + \
                        ".conv" + str(conv_bn_idx + 1)
                    bn_layer_name = base_block_name + \
                        ".bn" + str(conv_bn_idx + 1)
                    if conv_layer_name + ".weight" not in state_dict:
                        if conv_bn_idx == 0:
                            end_of_block = True
                        break
                    self._update_state_dict_single(
                        state_dict, bn_layer_name, conv_layer_name)
                    conv_bn_idx += 1
                downsample_conv_name = base_block_name + ".downsample.0"
                downsample_bn_name = base_block_name + ".downsample.1"
                if downsample_conv_name + ".weight" in state_dict:
                    self._update_state_dict_single(
                        state_dict, downsample_bn_name, downsample_conv_name)
                block_idx += 1

    def _update_state_dict_bn_to_conv(self, state_dict):
        self._update_state_dict_single_conv(state_dict, "bn1")
        for layer in range(4):
            block_idx = 0
            end_of_block = False
            base_layer_name = "layer" + str(layer + 1)
            while not end_of_block:
                conv_bn_idx = 0
                base_block_name = base_layer_name + "." + str(block_idx)
                while True:
                    conv_layer_name = base_block_name + \
                        ".conv" + str(conv_bn_idx + 1)
                    bn_layer_name = base_block_name + \
                        ".bn" + str(conv_bn_idx + 1)
                    if conv_layer_name + ".weight" not in state_dict:
                        if conv_bn_idx == 0:
                            end_of_block = True
                        break
                    self._update_state_dict_single_conv(
                        state_dict, bn_layer_name)
                    conv_bn_idx += 1
                downsample_conv_name = base_block_name + ".downsample.0"
                downsample_bn_name = base_block_name + ".downsample.1"
                if downsample_conv_name + ".weight" in state_dict:
                    self._update_state_dict_single_conv(
                        state_dict, downsample_bn_name)
                block_idx += 1

    def _update_state_dict_single(self, state_dict, name1, name2):
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"], state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"])
        for idx in range(len(conv_weight)):
            state_dict[name2 + ".weight"].data[idx] *= conv_weight[idx]
        state_dict[name2 + ".bias"] = conv_bias
        del state_dict[name1 + ".weight"]
        del state_dict[name1 + ".bias"]
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]

    def _update_state_dict_single_conv(self, state_dict, name1):
        # print(name1)
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"], state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"])
        conv_weight_matrix = torch.zeros(
            (len(conv_weight), len(conv_weight), 1, 1))
        for idx in range(len(conv_weight)):
            conv_weight_matrix.data[idx, idx] = conv_weight[idx]
        state_dict[name1 + ".weight"] = conv_weight_matrix
        state_dict[name1 + ".bias"] = conv_bias
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]

    def _get_equivalent_conv_weights(self, weight, bias, running_mean, running_var):
        num_features = len(weight)
        eps = 1e-5
        conv_weight = torch.zeros(num_features)
        conv_bias = torch.zeros(num_features)
        conv_weight.data = weight / torch.sqrt(running_var + eps)
        conv_bias.data = ((-1.0 * running_mean * weight) /
                          torch.sqrt(running_var + eps)) + bias
        return conv_weight, conv_bias

    def _remove_bias(self):
        self.base_model.conv1.bias = None
        for layer in range(4):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    conv_layer.bias = None
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    downsample_conv.bias = None
        self.base_model.fc.bias = None

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        for fidx in range(start_idx, len(self.all_layers) - 2):
            y = self.all_layers[fidx][1](y)
            if (conv_layer_idx is not None) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                conv_acts = y

        ya, xa, ha, wa = get_augmentation_range(
            y.shape, self.scale, self.output_head_idx)
        y = self.all_layers[-2][1](y[:, :, ya:ya + ha, xa:xa + wa])
        outs = self.base_model.fc(y.reshape(y.shape[0], -1))

        if conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        self.base_model.relu.inplace = value
        for layer in range(4):
            block = eval("self.base_model.layer" + str(layer + 1))
            for block_idx in range(len(block)):
                block[block_idx].relu.inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def _assign_rule(self, layer, rule, rule_param):
        if rule_param is None:
            layer.rule = rule()
        else:
            layer.rule = rule(rule_param)

    def set_lower_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.maxpool, rule, rule_param)
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_middle_rule(self, rule, rule_param):
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 3)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_upper_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.fc, rule, rule_param)

    def set_input_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.conv1, rule, rule_param)


###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################


class ImagenetContainerVGG19PointingGame(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True, every_layer=True):
        super(ImagenetContainerVGG19PointingGame, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert self.output_head_idx is None or (
            self.output_head_idx >= 0 and self.output_head_idx < self.num_heads)

        self.base_model = torchvision.models.vgg19(pretrained=True)

        if every_layer:
            self.conv_act_ids = [[1, 3, 6, 8, 11, 13,
                                  15, 17, 20, 22, 24, 26, 29, 31, 33, 35]]
        else:
            self.conv_act_ids = [[1, None, None, None, 20, None, None, 35]]

        self.single_spatial = True
        self.single_head = True

        self.start_conv_layer_idx = None
        self.avgpool2d = torch.nn.AvgPool2d(kernel_size=8)
        self.use_softmax = False

        self.conv1 = torch.nn.Conv2d(512, 4096, kernel_size=7, padding=0)
        self.conv2 = torch.nn.Conv2d(4096, 4096, kernel_size=1, padding=0)
        self.conv3 = torch.nn.Conv2d(4096, 1000, kernel_size=1, padding=0)

        self.conv1.weight.data = self.base_model.classifier[0].weight.reshape(
            (4096, 512, 7, 7))
        self.conv1.bias.data = self.base_model.classifier[0].bias
        self.conv2.weight.data = self.base_model.classifier[3].weight.reshape(
            (4096, 4096, 1, 1))
        self.conv2.bias.data = self.base_model.classifier[3].bias
        self.conv3.weight.data = self.base_model.classifier[6].weight.reshape(
            (1000, 4096, 1, 1))
        self.conv3.bias.data = self.base_model.classifier[6].bias

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        for fidx in range(start_idx, len(self.base_model.features)):
            y = self.base_model.features[fidx](y)
            if (conv_layer_idx is not None) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                conv_acts = y

        outs = self.conv1(y)
        outs = self.base_model.classifier[1](outs)
        outs = self.base_model.classifier[2](outs)
        outs = self.conv2(outs)
        outs = self.base_model.classifier[4](outs)
        outs = self.base_model.classifier[5](outs)
        classifier_outputs = self.conv3(outs)
        if classifier_outputs.shape[2] == 1:
            outs = classifier_outputs.squeeze(2).squeeze(2)
        else:
            outs = self.avgpool2d(
                classifier_outputs).squeeze(2).squeeze(2)

        if self.use_softmax:
            outs = torch.nn.functional.softmax(outs, dim=1)

        if conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        # return self.__call__(x, *kwargs)
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        for idx, layer in enumerate(self.base_model.features.children()):
            if isinstance(layer, torch.nn.ReLU):
                self.base_model.features[idx].inplace = value
        for idx, layer in enumerate(self.base_model.classifier.children()):
            if isinstance(layer, torch.nn.ReLU):
                self.base_model.classifier[idx].inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def set_lower_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(1, 19):
            layer = self.base_model.features[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.features[idx].rule = rule()
                    else:
                        self.base_model.features[idx].rule = rule(rule_param)

    def set_input_rule(self, rule, rule_param):
        if rule_param is None:
            self.base_model.features[0].rule = rule()
        else:
            self.base_model.features[0].rule = rule(rule_param)

    def set_middle_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(19, 37):
            layer = self.base_model.features[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.features[idx].rule = rule()
                    else:
                        self.base_model.features[idx].rule = rule(rule_param)

    def set_upper_rule(self, rule, rule_param):
        if rule_param is None:
            self.avgpool2d.rule = rule()
            self.conv1.rule = rule()
            self.conv2.rule = rule()
            self.conv3.rule = rule()
        else:
            self.avgpool2d.rule = rule(rule_param)
            self.conv1.rule = rule(rule_param)
            self.conv2.rule = rule(rule_param)
            self.conv3.rule = rule(rule_param)


class ImagenetContainerVGG19Disconnected(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True):
        super(ImagenetContainerVGG19Disconnected, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        self.base_model = torchvision.models.vgg19(pretrained=True)

        self.conv_act_ids = [[1, None, None, None, 20, None, None, 35]]

        self.single_spatial = False
        self.single_head = False

        self.start_conv_layer_idx = None
        self.use_softmax = False

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None, loc_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        if conv_layer_idx is not None:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, loc_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.base_model.features)):
                y = self.base_model.features[fidx](y)
                if (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                    conv_acts = y
                    return None, conv_acts
        else:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, self.output_head_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.base_model.features)):
                y = self.base_model.features[fidx](y)

            outs = self.base_model.classifier(y.reshape(y.shape[0], -1))

            if self.use_softmax:
                outs = torch.nn.functional.softmax(outs, dim=1)

            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        for idx, layer in enumerate(self.base_model.features.children()):
            if isinstance(layer, torch.nn.ReLU):
                self.base_model.features[idx].inplace = value
        for idx, layer in enumerate(self.base_model.classifier.children()):
            if isinstance(layer, torch.nn.ReLU):
                self.base_model.classifier[idx].inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def set_lower_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(1, 19):
            layer = self.base_model.features[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.features[idx].rule = rule()
                    else:
                        self.base_model.features[idx].rule = rule(rule_param)

    def set_input_rule(self, rule, rule_param):
        if rule_param is None:
            self.base_model.features[0].rule = rule()
        else:
            self.base_model.features[0].rule = rule(rule_param)

    def set_middle_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(19, 37):
            layer = self.base_model.features[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.features[idx].rule = rule()
                    else:
                        self.base_model.features[idx].rule = rule(rule_param)

    def set_upper_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(0, 7):
            layer = self.base_model.classifier[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.classifier[idx].rule = rule()
                    else:
                        self.base_model.classifier[idx].rule = rule(rule_param)

class ImagenetContainerVGG19Disconnected4(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True):
        super(ImagenetContainerVGG19Disconnected4, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        self.base_model = torchvision.models.vgg19(pretrained=True)

        new_linear1 = torch.nn.Linear(
            512 * 7 * 7 * self.scale * self.scale, 512 * 7 * 7 + 2, bias=False)
        new_linear1.weight.data.fill_(0.0)
        hrow_idx = output_head_idx // self.scale
        hcol_idx = output_head_idx % self.scale
        temp_w = new_linear1.weight.data[:-2].reshape((512 * 7 * 7, 512, 7 * self.scale, 7 * self.scale))[:, :, hrow_idx * 7:(
            hrow_idx + 1) * 7, hcol_idx * 7:(hcol_idx + 1) * 7].reshape((512 * 7 * 7, 512 * 7 * 7))
        # print(temp_w)
        # print(temp_w.shape)
        for idx in range(temp_w.shape[0]):
            temp_w[idx, idx] = 1.0
        # print(temp_w)
        new_linear1.weight.data[:-2].reshape((512 * 7 * 7, 512, 7 * self.scale, 7 * self.scale))[:, :, hrow_idx * 7:(
            hrow_idx + 1) * 7, hcol_idx * 7:(hcol_idx + 1) * 7] = temp_w.reshape((512 * 7 * 7, 512, 7, 7))
        # print(torch.where(new_linear1.weight == 1),
        #   torch.where(new_linear1.weight == 1)[0].shape)
        # import sys
        # sys.exit(0)

        for head_idx in range(self.scale * self.scale):
            row_idx = head_idx // self.scale
            col_idx = head_idx % self.scale
            if (row_idx == hrow_idx) and (col_idx == hcol_idx):
                continue
            new_linear1.weight.data.view((512 * 7 * 7 + 2, 512, 7 * self.scale,
                                          7 * self.scale))[-2, :, row_idx * 7:(row_idx + 1) * 7, col_idx * 7:(col_idx + 1) * 7] = 1.0
            new_linear1.weight.data.view((512 * 7 * 7 + 2, 512, 7 * self.scale,
                                          7 * self.scale))[-1, :, row_idx * 7:(row_idx + 1) * 7, col_idx * 7:(col_idx + 1) * 7] = -1.0

        new_linear2 = torch.nn.Linear(
            512 * 7 * 7 + 2, 4096, bias=True)
        new_linear2.weight.data.fill_(0.0)
        new_linear2.bias.data = self.base_model.classifier[0].bias
        new_linear2.weight.data[:, :-2].view((4096, 512, 7, 7))[
            :] = self.base_model.classifier[0].weight.view((4096, 512, 7, 7))
        new_linear2.weight.data[:, -2] = 1.0
        new_linear2.weight.data[:, -1] = 1.0

        self.base_model.classifier_mod = torch.nn.Sequential(new_linear1, new_linear2, self.base_model.classifier[1], self.base_model.classifier[
                                                             2], self.base_model.classifier[3], self.base_model.classifier[4], self.base_model.classifier[5], self.base_model.classifier[6])

        self.base_model.feature_list = torch.nn.ModuleList([self.base_model.features, copy.deepcopy(
            self.base_model.features), copy.deepcopy(self.base_model.features), copy.deepcopy(self.base_model.features)])

        del self.base_model.features
        del self.base_model.classifier

        self.conv_act_ids = [[1, None, None, None, 20, None, None, 35]]

        self.single_spatial = False
        self.single_head = False

        self.start_conv_layer_idx = None
        self.use_softmax = False

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None, loc_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        if conv_layer_idx is not None:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, loc_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.base_model.feature_list[0])):
                y = self.base_model.feature_list[0][fidx](y)
                if (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                    conv_acts = y
                    return None, conv_acts
        else:
            combined = torch.zeros(
                (x.shape[0], 512, 7 * self.scale, 7 * self.scale)).to(x.device)
            for head_idx in range(self.scale * self.scale):
                y_coord, x_coord, h, w = get_augmentation_range(
                    x.shape, self.scale, head_idx)
                # print(y_coord, x_coord, h, w)
                y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
                for fidx in range(start_idx, len(self.base_model.feature_list[0])):
                    y = self.base_model.feature_list[head_idx][fidx](y)
                yc, xc, hc, wc = get_augmentation_range(
                    combined.shape, self.scale, head_idx)
                # print(yc, xc, hc, wc)
                combined[:, :, yc:yc + hc, xc:xc + wc] = y

            # print(combined.tolist())
            # import sys
            # sys.exit(0)
            outs = self.base_model.classifier_mod(
                combined.reshape(combined.shape[0], -1))
            # y = combined.reshape(combined.shape[0], -1)
            # y1 = self.base_model.classifier_mod[0](y)
            # y2 = self.base_model.classifier_mod[1](y1)
            # print(y1.shape, y2.shape)
            # print(y1.tolist(), y2.tolist())
            # sys.exit(0)
            # for idx in range(len(self.base_model.classifier_mod)):
            # print(y)
            # y = self.base_model.classifier_mod[idx](y)

            if self.use_softmax:
                outs = torch.nn.functional.softmax(outs, dim=1)

            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)


class ImagenetContainerVGG19CommonSpatial(torch.nn.Module):

    def __init__(self, scale=3, output_head_idx=None, transform_batchnorm=True):
        super(ImagenetContainerVGG19CommonSpatial, self).__init__()
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        self.base_model = torchvision.models.vgg19(pretrained=True)

        self.conv_act_ids = [[1, None, None, None, 20, None, None, 35]]

        self.single_spatial = True
        self.single_head = False

        self.start_conv_layer_idx = None
        self.use_softmax = False

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        for fidx in range(start_idx, len(self.base_model.features)):
            y = self.base_model.features[fidx](y)
            if (conv_layer_idx is not None) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                conv_acts = y

        ya, xa, ha, wa = get_augmentation_range(
            y.shape, self.scale, self.output_head_idx)

        outs = self.base_model.classifier(
            y[:, :, ya:ya + ha, xa:xa + wa].reshape(y.shape[0], -1))

        if self.use_softmax:
            outs = torch.nn.functional.softmax(outs, dim=1)

        if conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        for idx, layer in enumerate(self.base_model.features.children()):
            if isinstance(layer, torch.nn.ReLU):
                self.base_model.features[idx].inplace = value
        for idx, layer in enumerate(self.base_model.classifier.children()):
            if isinstance(layer, torch.nn.ReLU):
                self.base_model.classifier[idx].inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def set_lower_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(1, 19):
            layer = self.base_model.features[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.features[idx].rule = rule()
                    else:
                        self.base_model.features[idx].rule = rule(rule_param)

    def set_input_rule(self, rule, rule_param):
        if rule_param is None:
            self.base_model.features[0].rule = rule()
        else:
            self.base_model.features[0].rule = rule(rule_param)

    def set_middle_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(19, 37):
            layer = self.base_model.features[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.features[idx].rule = rule()
                    else:
                        self.base_model.features[idx].rule = rule(rule_param)

    def set_upper_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(0, 7):
            layer = self.base_model.classifier[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.classifier[idx].rule = rule()
                    else:
                        self.base_model.classifier[idx].rule = rule(rule_param)


class ImagenetContainerVGG19BNPointingGame(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True):
        super(ImagenetContainerVGG19BNPointingGame, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert self.output_head_idx is None or (
            self.output_head_idx >= 0 and self.output_head_idx < self.num_heads)

        self.base_model = torchvision.models.vgg19_bn(pretrained=True)

        self.conv_act_ids = [[2, None, None, None, 29, None, None, 51]]

        self.single_spatial = True
        self.single_head = True

        self.start_conv_layer_idx = None
        self.avgpool2d = torch.nn.AvgPool2d(kernel_size=8)
        self.use_softmax = False

        self.conv1 = torch.nn.Conv2d(512, 4096, kernel_size=7, padding=0)
        self.conv2 = torch.nn.Conv2d(4096, 4096, kernel_size=1, padding=0)
        self.conv3 = torch.nn.Conv2d(4096, 1000, kernel_size=1, padding=0)

        self.conv1.weight.data = self.base_model.classifier[0].weight.reshape(
            (4096, 512, 7, 7))
        self.conv1.bias.data = self.base_model.classifier[0].bias
        self.conv2.weight.data = self.base_model.classifier[3].weight.reshape(
            (4096, 4096, 1, 1))
        self.conv2.bias.data = self.base_model.classifier[3].bias
        self.conv3.weight.data = self.base_model.classifier[6].weight.reshape(
            (1000, 4096, 1, 1))
        self.conv3.bias.data = self.base_model.classifier[6].bias

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        for fidx in range(start_idx, len(self.base_model.features)):
            y = self.base_model.features[fidx](y)
            if (conv_layer_idx is not None) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                conv_acts = y

        outs = self.conv1(y)
        outs = self.base_model.classifier[1](outs)
        outs = self.base_model.classifier[2](outs)
        outs = self.conv2(outs)
        outs = self.base_model.classifier[4](outs)
        outs = self.base_model.classifier[5](outs)
        classifier_outputs = self.conv3(outs)

        outs = self.avgpool2d(
            classifier_outputs).squeeze(2).squeeze(2)

        if self.use_softmax:
            outs = torch.nn.functional.softmax(outs, dim=1)

        if conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        # return self.__call__(x, *kwargs)
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)


class ImagenetContainerVGG19BNDisconnected(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True):
        super(ImagenetContainerVGG19BNDisconnected, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        self.base_model = torchvision.models.vgg19_bn(pretrained=True)

        self.conv_act_ids = [[2, None, None, None, 29, None, None, 51]]

        self.single_spatial = False
        self.single_head = False

        self.start_conv_layer_idx = None
        self.use_softmax = False

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None, loc_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        if conv_layer_idx is not None:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, loc_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.base_model.features)):
                y = self.base_model.features[fidx](y)
                if (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                    conv_acts = y
                    return None, conv_acts
        else:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, self.output_head_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.base_model.features)):
                y = self.base_model.features[fidx](y)

            outs = self.base_model.classifier(y.reshape(y.shape[0], -1))

            if self.use_softmax:
                outs = torch.nn.functional.softmax(outs, dim=1)

            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)


class ImagenetContainerVGG19BNCommonSpatial(torch.nn.Module):

    def __init__(self, scale=3, output_head_idx=None, transform_batchnorm=True):
        super(ImagenetContainerVGG19BNCommonSpatial, self).__init__()
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        self.base_model = torchvision.models.vgg19_bn(pretrained=True)

        self.conv_act_ids = [[2, None, None, None, 29, None, None, 51]]

        self.single_spatial = True
        self.single_head = False

        self.start_conv_layer_idx = None
        self.use_softmax = False

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        for fidx in range(start_idx, len(self.base_model.features)):
            y = self.base_model.features[fidx](y)
            if (conv_layer_idx is not None) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                conv_acts = y

        ya, xa, ha, wa = get_augmentation_range(
            y.shape, self.scale, self.output_head_idx)

        outs = self.base_model.classifier(
            y[:, :, ya:ya + ha, xa:xa + wa].reshape(y.shape[0], -1))

        if self.use_softmax:
            outs = torch.nn.functional.softmax(outs, dim=1)

        if conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)


class ImagenetContainerVGG19NoClassifierKernelPointingGame(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True):
        super(ImagenetContainerVGG19NoClassifierKernelPointingGame, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert self.output_head_idx is None or (
            self.output_head_idx >= 0 and self.output_head_idx < self.num_heads)

        self.base_model = torchvision.models.vgg19(pretrained=True)

        self.conv_act_ids = [[1, None, None, None, 20, None, None, 35]]

        self.single_spatial = True
        self.single_head = True
        self.use_softmax = False

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        for fidx in range(start_idx, len(self.base_model.features)):
            y = self.base_model.features[fidx](y)
            if (conv_layer_idx is not None) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                conv_acts = y

        y = self.base_model.avgpool(y)
        outs = self.base_model.classifier(y.flatten(start_dim=1))

        if self.use_softmax:
            outs = torch.nn.functional.softmax(outs, dim=1)

        if conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        for idx, layer in enumerate(self.base_model.features.children()):
            if isinstance(layer, torch.nn.ReLU):
                self.base_model.features[idx].inplace = value
        for idx, layer in enumerate(self.base_model.classifier.children()):
            if isinstance(layer, torch.nn.ReLU):
                self.base_model.classifier[idx].inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def set_lower_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(1, 19):
            layer = self.base_model.features[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.features[idx].rule = rule()
                    else:
                        self.base_model.features[idx].rule = rule(rule_param)

    def set_input_rule(self, rule, rule_param):
        if rule_param is None:
            self.base_model.features[0].rule = rule()
        else:
            self.base_model.features[0].rule = rule(rule_param)

    def set_middle_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(19, 37):
            layer = self.base_model.features[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.features[idx].rule = rule()
                    else:
                        self.base_model.features[idx].rule = rule(rule_param)
        if rule_param is None:
            self.base_model.avgpool.rule = rule()
        else:
            self.base_model.avgpool.rule = rule(rule_param)

    def set_upper_rule(self, rule, rule_param):
        layer_types = [torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.Linear]
        for idx in range(0, 7):
            layer = self.base_model.classifier[idx]
            for ltype in layer_types:
                if isinstance(layer, ltype):
                    if rule_param is None:
                        self.base_model.classifier[idx].rule = rule()
                    else:
                        self.base_model.classifier[idx].rule = rule(rule_param)


class ImagenetContainerResnet152PointingGame(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True, batchnorm_to_conv=False, every_layer=False):
        super(ImagenetContainerResnet152PointingGame, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert self.output_head_idx is None or (
            self.output_head_idx >= 0 and self.output_head_idx < self.num_heads)

        if batchnorm_to_conv:
            self._load_batchnorm_to_conv_model()
            self.conv_act_ids = [[2, 4, 5, 6, 7]]
        elif transform_batchnorm:
            self._load_model()
            self.conv_act_ids = [[1, 3, 4, 5, 6]]  # [[2, 4, 5, 6, 7]]
        else:
            self.base_model = torchvision.models.resnet152(pretrained=True)
            self.conv_act_ids = [[2, 4, 5, 6, 7]]

        self.single_spatial = True
        self.single_head = True

        self.all_layers = list(self.base_model.named_children())

        self.start_conv_layer_idx = None

    def _load_model(self):
        self.base_model = resnet_for_lrp.resnet152(pretrained=False)
        state_dict = torch.hub.load_state_dict_from_url(
            resnet_for_lrp.model_urls['resnet152'])
        # print(state_dict.keys())
        self._update_state_dict(state_dict)
        self.base_model.load_state_dict(state_dict)

    def _load_batchnorm_to_conv_model(self):
        self.base_model = resnet_for_lrp_conv.resnet152(pretrained=False)
        # print(self.base_model)
        state_dict = torch.hub.load_state_dict_from_url(
            resnet_for_lrp_conv.model_urls['resnet152'])
        # print(state_dict.keys())
        self._update_state_dict_bn_to_conv(state_dict)
        self.base_model.load_state_dict(state_dict)

    def _update_state_dict(self, state_dict):
        self._update_state_dict_single(state_dict, "bn1", "conv1")
        for layer in range(4):
            block_idx = 0
            end_of_block = False
            base_layer_name = "layer" + str(layer + 1)
            while not end_of_block:
                conv_bn_idx = 0
                base_block_name = base_layer_name + "." + str(block_idx)
                while True:
                    conv_layer_name = base_block_name + \
                        ".conv" + str(conv_bn_idx + 1)
                    bn_layer_name = base_block_name + \
                        ".bn" + str(conv_bn_idx + 1)
                    if conv_layer_name + ".weight" not in state_dict:
                        if conv_bn_idx == 0:
                            end_of_block = True
                        break
                    self._update_state_dict_single(
                        state_dict, bn_layer_name, conv_layer_name)
                    conv_bn_idx += 1
                downsample_conv_name = base_block_name + ".downsample.0"
                downsample_bn_name = base_block_name + ".downsample.1"
                if downsample_conv_name + ".weight" in state_dict:
                    self._update_state_dict_single(
                        state_dict, downsample_bn_name, downsample_conv_name)
                block_idx += 1

    def _update_state_dict_bn_to_conv(self, state_dict):
        self._update_state_dict_single_conv(state_dict, "bn1")
        for layer in range(4):
            block_idx = 0
            end_of_block = False
            base_layer_name = "layer" + str(layer + 1)
            while not end_of_block:
                conv_bn_idx = 0
                base_block_name = base_layer_name + "." + str(block_idx)
                while True:
                    conv_layer_name = base_block_name + \
                        ".conv" + str(conv_bn_idx + 1)
                    bn_layer_name = base_block_name + \
                        ".bn" + str(conv_bn_idx + 1)
                    if conv_layer_name + ".weight" not in state_dict:
                        if conv_bn_idx == 0:
                            end_of_block = True
                        break
                    self._update_state_dict_single_conv(
                        state_dict, bn_layer_name)
                    conv_bn_idx += 1
                downsample_conv_name = base_block_name + ".downsample.0"
                downsample_bn_name = base_block_name + ".downsample.1"
                if downsample_conv_name + ".weight" in state_dict:
                    self._update_state_dict_single_conv(
                        state_dict, downsample_bn_name)
                block_idx += 1

    def _update_state_dict_single(self, state_dict, name1, name2):
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"], state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"])
        for idx in range(len(conv_weight)):
            state_dict[name2 + ".weight"].data[idx] *= conv_weight[idx]
        state_dict[name2 + ".bias"] = conv_bias
        del state_dict[name1 + ".weight"]
        del state_dict[name1 + ".bias"]
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]

    def _update_state_dict_single_conv(self, state_dict, name1):
        # print(name1)
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"], state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"])
        conv_weight_matrix = torch.zeros(
            (len(conv_weight), len(conv_weight), 1, 1))
        for idx in range(len(conv_weight)):
            conv_weight_matrix.data[idx, idx] = conv_weight[idx]
        state_dict[name1 + ".weight"] = conv_weight_matrix
        state_dict[name1 + ".bias"] = conv_bias
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]

    def _get_equivalent_conv_weights(self, weight, bias, running_mean, running_var):
        num_features = len(weight)
        eps = 1e-5
        conv_weight = torch.zeros(num_features)
        conv_bias = torch.zeros(num_features)
        conv_weight.data = weight / torch.sqrt(running_var + eps)
        conv_bias.data = ((-1.0 * running_mean * weight) /
                          torch.sqrt(running_var + eps)) + bias
        return conv_weight, conv_bias

    def _remove_bias(self):
        self.base_model.conv1.bias = None
        for layer in range(4):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    conv_layer.bias = None
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    downsample_conv.bias = None
        self.base_model.fc.bias = None

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        # print(x.get_device(), self.all_layers[0][1].weight.get_device())
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        for fidx in range(start_idx, len(self.all_layers) - 1):
            y = self.all_layers[fidx][1](y)
            if (conv_layer_idx is not None) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                conv_acts = y

        outs = self.all_layers[-1][1](y.reshape(y.shape[0], -1))

        if conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        self.base_model.relu.inplace = value
        for layer in range(4):
            block = eval("self.base_model.layer" + str(layer + 1))
            for block_idx in range(len(block)):
                block[block_idx].relu.inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def _assign_rule(self, layer, rule, rule_param):
        if rule_param is None:
            layer.rule = rule()
        else:
            layer.rule = rule(rule_param)

    def set_lower_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.maxpool, rule, rule_param)
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_middle_rule(self, rule, rule_param):
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 3)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_upper_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.fc, rule, rule_param)

    def set_input_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.conv1, rule, rule_param)


class ImagenetContainerResnet152Disconnected(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True, batchnorm_to_conv=False):
        super(ImagenetContainerResnet152Disconnected, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        if batchnorm_to_conv:
            self._load_batchnorm_to_conv_model()
            self.conv_act_ids = [[2, 4, 5, 6, 7]]
        elif transform_batchnorm:
            self._load_model()
            self.conv_act_ids = [[1, 3, 4, 5, 6]]  # [[2, 4, 5, 6, 7]]
        else:
            self.base_model = torchvision.models.resnet152(pretrained=True)
            self.conv_act_ids = [[2, 4, 5, 6, 7]]

        self.single_spatial = False
        self.single_head = False

        self.all_layers = list(self.base_model.named_children())

        self.start_conv_layer_idx = None

    def _load_model(self):
        self.base_model = resnet_for_lrp.resnet152(pretrained=False)
        state_dict = torch.hub.load_state_dict_from_url(
            resnet_for_lrp.model_urls['resnet152'])
        # print(state_dict.keys())
        self._update_state_dict(state_dict)
        self.base_model.load_state_dict(state_dict)

    def _load_batchnorm_to_conv_model(self):
        self.base_model = resnet_for_lrp_conv.resnet152(pretrained=False)
        # print(self.base_model)
        state_dict = torch.hub.load_state_dict_from_url(
            resnet_for_lrp_conv.model_urls['resnet152'])
        # print(state_dict.keys())
        self._update_state_dict_bn_to_conv(state_dict)
        self.base_model.load_state_dict(state_dict)

    def _update_state_dict(self, state_dict):
        self._update_state_dict_single(state_dict, "bn1", "conv1")
        for layer in range(4):
            block_idx = 0
            end_of_block = False
            base_layer_name = "layer" + str(layer + 1)
            while not end_of_block:
                conv_bn_idx = 0
                base_block_name = base_layer_name + "." + str(block_idx)
                while True:
                    conv_layer_name = base_block_name + \
                        ".conv" + str(conv_bn_idx + 1)
                    bn_layer_name = base_block_name + \
                        ".bn" + str(conv_bn_idx + 1)
                    if conv_layer_name + ".weight" not in state_dict:
                        if conv_bn_idx == 0:
                            end_of_block = True
                        break
                    self._update_state_dict_single(
                        state_dict, bn_layer_name, conv_layer_name)
                    conv_bn_idx += 1
                downsample_conv_name = base_block_name + ".downsample.0"
                downsample_bn_name = base_block_name + ".downsample.1"
                if downsample_conv_name + ".weight" in state_dict:
                    self._update_state_dict_single(
                        state_dict, downsample_bn_name, downsample_conv_name)
                block_idx += 1

    def _update_state_dict_bn_to_conv(self, state_dict):
        self._update_state_dict_single_conv(state_dict, "bn1")
        for layer in range(4):
            block_idx = 0
            end_of_block = False
            base_layer_name = "layer" + str(layer + 1)
            while not end_of_block:
                conv_bn_idx = 0
                base_block_name = base_layer_name + "." + str(block_idx)
                while True:
                    conv_layer_name = base_block_name + \
                        ".conv" + str(conv_bn_idx + 1)
                    bn_layer_name = base_block_name + \
                        ".bn" + str(conv_bn_idx + 1)
                    if conv_layer_name + ".weight" not in state_dict:
                        if conv_bn_idx == 0:
                            end_of_block = True
                        break
                    self._update_state_dict_single_conv(
                        state_dict, bn_layer_name)
                    conv_bn_idx += 1
                downsample_conv_name = base_block_name + ".downsample.0"
                downsample_bn_name = base_block_name + ".downsample.1"
                if downsample_conv_name + ".weight" in state_dict:
                    self._update_state_dict_single_conv(
                        state_dict, downsample_bn_name)
                block_idx += 1

    def _update_state_dict_single(self, state_dict, name1, name2):
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"], state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"])
        for idx in range(len(conv_weight)):
            state_dict[name2 + ".weight"].data[idx] *= conv_weight[idx]
        state_dict[name2 + ".bias"] = conv_bias
        del state_dict[name1 + ".weight"]
        del state_dict[name1 + ".bias"]
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]

    def _update_state_dict_single_conv(self, state_dict, name1):
        # print(name1)
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"], state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"])
        conv_weight_matrix = torch.zeros(
            (len(conv_weight), len(conv_weight), 1, 1))
        for idx in range(len(conv_weight)):
            conv_weight_matrix.data[idx, idx] = conv_weight[idx]
        state_dict[name1 + ".weight"] = conv_weight_matrix
        state_dict[name1 + ".bias"] = conv_bias
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]

    def _get_equivalent_conv_weights(self, weight, bias, running_mean, running_var):
        num_features = len(weight)
        eps = 1e-5
        conv_weight = torch.zeros(num_features)
        conv_bias = torch.zeros(num_features)
        conv_weight.data = weight / torch.sqrt(running_var + eps)
        conv_bias.data = ((-1.0 * running_mean * weight) /
                          torch.sqrt(running_var + eps)) + bias
        return conv_weight, conv_bias

    def _remove_bias(self):
        self.base_model.conv1.bias = None
        for layer in range(4):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    conv_layer.bias = None
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    downsample_conv.bias = None
        self.base_model.fc.bias = None

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None, loc_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        if conv_layer_idx is not None:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, loc_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.all_layers) - 1):
                y = self.all_layers[fidx][1](y)
                if (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                    conv_acts = y
                    return None, conv_acts
        else:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, self.output_head_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.all_layers) - 1):
                y = self.all_layers[fidx][1](y)

            outs = self.base_model.fc(y.reshape(y.shape[0], -1))

        return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        self.base_model.relu.inplace = value
        for layer in range(4):
            block = eval("self.base_model.layer" + str(layer + 1))
            for block_idx in range(len(block)):
                block[block_idx].relu.inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def _assign_rule(self, layer, rule, rule_param):
        if rule_param is None:
            layer.rule = rule()
        else:
            layer.rule = rule(rule_param)

    def set_lower_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.maxpool, rule, rule_param)
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_middle_rule(self, rule, rule_param):
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 3)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_upper_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.fc, rule, rule_param)

    def set_input_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.conv1, rule, rule_param)


class ImagenetContainerResnet152CommonSpatial(torch.nn.Module):

    def __init__(self, scale=3, output_head_idx=None, transform_batchnorm=True, batchnorm_to_conv=False):
        super(ImagenetContainerResnet152CommonSpatial, self).__init__()
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        if batchnorm_to_conv:
            self._load_batchnorm_to_conv_model()
            self.conv_act_ids = [[2, 4, 5, 6, 7]]
        elif transform_batchnorm:
            self._load_model()
            self.conv_act_ids = [[1, 3, 4, 5, 6]]  # [[2, 4, 5, 6, 7]]
        else:
            self.base_model = torchvision.models.resnet152(pretrained=True)
            self.conv_act_ids = [[2, 4, 5, 6, 7]]

        self.single_spatial = True
        self.single_head = False

        self.all_layers = list(self.base_model.named_children())

        self.start_conv_layer_idx = None

    def _load_model(self):
        self.base_model = resnet_for_lrp.resnet152(pretrained=False)
        state_dict = torch.hub.load_state_dict_from_url(
            resnet_for_lrp.model_urls['resnet152'])
        # print(state_dict.keys())
        self._update_state_dict(state_dict)
        self.base_model.load_state_dict(state_dict)

    def _load_batchnorm_to_conv_model(self):
        self.base_model = resnet_for_lrp_conv.resnet152(pretrained=False)
        # print(self.base_model)
        state_dict = torch.hub.load_state_dict_from_url(
            resnet_for_lrp_conv.model_urls['resnet152'])
        # print(state_dict.keys())
        self._update_state_dict_bn_to_conv(state_dict)
        self.base_model.load_state_dict(state_dict)

    def _update_state_dict(self, state_dict):
        self._update_state_dict_single(state_dict, "bn1", "conv1")
        for layer in range(4):
            block_idx = 0
            end_of_block = False
            base_layer_name = "layer" + str(layer + 1)
            while not end_of_block:
                conv_bn_idx = 0
                base_block_name = base_layer_name + "." + str(block_idx)
                while True:
                    conv_layer_name = base_block_name + \
                        ".conv" + str(conv_bn_idx + 1)
                    bn_layer_name = base_block_name + \
                        ".bn" + str(conv_bn_idx + 1)
                    if conv_layer_name + ".weight" not in state_dict:
                        if conv_bn_idx == 0:
                            end_of_block = True
                        break
                    self._update_state_dict_single(
                        state_dict, bn_layer_name, conv_layer_name)
                    conv_bn_idx += 1
                downsample_conv_name = base_block_name + ".downsample.0"
                downsample_bn_name = base_block_name + ".downsample.1"
                if downsample_conv_name + ".weight" in state_dict:
                    self._update_state_dict_single(
                        state_dict, downsample_bn_name, downsample_conv_name)
                block_idx += 1

    def _update_state_dict_bn_to_conv(self, state_dict):
        self._update_state_dict_single_conv(state_dict, "bn1")
        for layer in range(4):
            block_idx = 0
            end_of_block = False
            base_layer_name = "layer" + str(layer + 1)
            while not end_of_block:
                conv_bn_idx = 0
                base_block_name = base_layer_name + "." + str(block_idx)
                while True:
                    conv_layer_name = base_block_name + \
                        ".conv" + str(conv_bn_idx + 1)
                    bn_layer_name = base_block_name + \
                        ".bn" + str(conv_bn_idx + 1)
                    if conv_layer_name + ".weight" not in state_dict:
                        if conv_bn_idx == 0:
                            end_of_block = True
                        break
                    self._update_state_dict_single_conv(
                        state_dict, bn_layer_name)
                    conv_bn_idx += 1
                downsample_conv_name = base_block_name + ".downsample.0"
                downsample_bn_name = base_block_name + ".downsample.1"
                if downsample_conv_name + ".weight" in state_dict:
                    self._update_state_dict_single_conv(
                        state_dict, downsample_bn_name)
                block_idx += 1

    def _update_state_dict_single(self, state_dict, name1, name2):
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"], state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"])
        for idx in range(len(conv_weight)):
            state_dict[name2 + ".weight"].data[idx] *= conv_weight[idx]
        state_dict[name2 + ".bias"] = conv_bias
        del state_dict[name1 + ".weight"]
        del state_dict[name1 + ".bias"]
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]

    def _update_state_dict_single_conv(self, state_dict, name1):
        # print(name1)
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"], state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"])
        conv_weight_matrix = torch.zeros(
            (len(conv_weight), len(conv_weight), 1, 1))
        for idx in range(len(conv_weight)):
            conv_weight_matrix.data[idx, idx] = conv_weight[idx]
        state_dict[name1 + ".weight"] = conv_weight_matrix
        state_dict[name1 + ".bias"] = conv_bias
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]

    def _get_equivalent_conv_weights(self, weight, bias, running_mean, running_var):
        num_features = len(weight)
        eps = 1e-5
        conv_weight = torch.zeros(num_features)
        conv_bias = torch.zeros(num_features)
        conv_weight.data = weight / torch.sqrt(running_var + eps)
        conv_bias.data = ((-1.0 * running_mean * weight) /
                          torch.sqrt(running_var + eps)) + bias
        return conv_weight, conv_bias

    def _remove_bias(self):
        self.base_model.conv1.bias = None
        for layer in range(4):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    conv_layer.bias = None
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    downsample_conv.bias = None
        self.base_model.fc.bias = None

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        for fidx in range(start_idx, len(self.all_layers) - 2):
            y = self.all_layers[fidx][1](y)
            if (conv_layer_idx is not None) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                conv_acts = y

        ya, xa, ha, wa = get_augmentation_range(
            y.shape, self.scale, self.output_head_idx)
        y = self.all_layers[-2][1](y[:, :, ya:ya + ha, xa:xa + wa])
        outs = self.base_model.fc(y.reshape(y.shape[0], -1))

        if conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        self.base_model.relu.inplace = value
        for layer in range(4):
            block = eval("self.base_model.layer" + str(layer + 1))
            for block_idx in range(len(block)):
                block[block_idx].relu.inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def _assign_rule(self, layer, rule, rule_param):
        if rule_param is None:
            layer.rule = rule()
        else:
            layer.rule = rule(rule_param)

    def set_lower_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.maxpool, rule, rule_param)
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_middle_rule(self, rule, rule_param):
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 3)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_upper_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.fc, rule, rule_param)

    def set_input_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.conv1, rule, rule_param)


class ImagenetContainerGooglenetPointingGame(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True, every_layer=False):
        super(ImagenetContainerGooglenetPointingGame, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert self.output_head_idx is None or (
            self.output_head_idx >= 0 and self.output_head_idx < self.num_heads)

        self.conv_layer_names = ["conv1", "conv2", "conv3"]
        self.inception_layer_names = ["inception3a", "inception3b", "inception4a", "inception4b",
                                      "inception4c", "inception4d", "inception4e", "inception5a", "inception5b"]
        self.aux_layer_names = ["aux1", "aux2"]

        if transform_batchnorm:
            self._load_model()
        else:
            self.base_model = torchvision.models.googlenet(pretrained=True)

        if every_layer:
            self.conv_act_ids = [[0,2,3,5,6,8,9,10,11,12,14,15]]
        else:
            self.conv_act_ids = [[0, None, None, None, 6, None, None, 15]]
        

        self.single_spatial = True
        self.single_head = True

        self.all_layers = list(self.base_model.named_children())

        self.start_conv_layer_idx = None

    def _load_model(self):
        self.base_model = googlenet_for_lrp.googlenet(pretrained=False)
        state_dict = torch.hub.load_state_dict_from_url(
            googlenet_for_lrp.model_urls['googlenet'])
        # print(state_dict.keys())
        self._update_state_dict(state_dict)
        self.base_model.load_state_dict(state_dict)

    def _update_state_dict(self, state_dict):
        for conv_layer_name in self.conv_layer_names:
            conv_name = conv_layer_name + ".conv"
            bn_name = conv_layer_name + ".bn"
            self._update_state_dict_single(state_dict, bn_name, conv_name)
        for inception_layer_name in self.inception_layer_names:
            for branch_idx in range(4):
                base_branch_name = inception_layer_name + \
                    ".branch" + str(branch_idx + 1)
                if branch_idx == 0:
                    conv_name = base_branch_name + ".conv"
                    bn_name = base_branch_name + ".bn"
                    self._update_state_dict_single(
                        state_dict, bn_name, conv_name)
                    continue
                for lidx in range(2):
                    conv_name = base_branch_name + \
                        "." + str(lidx) + ".conv"
                    bn_name = base_branch_name + "." + str(lidx) + ".bn"
                    if conv_name + ".weight" not in state_dict:
                        continue
                    self._update_state_dict_single(
                        state_dict, bn_name, conv_name)
        for aux_layer_name in self.aux_layer_names:
            conv_name = aux_layer_name + ".conv.conv"
            bn_name = aux_layer_name + ".conv.bn"
            self._update_state_dict_single(state_dict, bn_name, conv_name)

    def _update_state_dict_single(self, state_dict, name1, name2):
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"], state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"])
        for idx in range(len(conv_weight)):
            state_dict[name2 + ".weight"].data[idx] *= conv_weight[idx]
        state_dict[name2 + ".bias"] = conv_bias
        del state_dict[name1 + ".weight"]
        del state_dict[name1 + ".bias"]
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]
        del state_dict[name1 + ".num_batches_tracked"]

    def _get_equivalent_conv_weights(self, weight, bias, running_mean, running_var):
        num_features = len(weight)
        eps = 0.001
        conv_weight = torch.zeros(num_features)
        conv_bias = torch.zeros(num_features)
        conv_weight.data = weight / torch.sqrt(running_var + eps)
        conv_bias.data = ((-1.0 * running_mean * weight) /
                          torch.sqrt(running_var + eps)) + bias
        return conv_weight, conv_bias

    def _remove_bias(self):
        for conv_layer_name in self.conv_layer_names:
            conv_layer = eval("self.base_model." + conv_layer_name + ".conv")
            conv_layer.bias = None
        for inception_layer_name in self.inception_layer_names:
            base_layer_name = "self.base_model." + inception_layer_name
            for branch_idx in range(4):
                base_branch_name = base_layer_name + \
                    ".branch" + str(branch_idx + 1)
                if branch_idx == 0:
                    conv_layer = eval(base_branch_name + ".conv")
                    conv_layer.bias = None
                elif branch_idx == 3:
                    conv_layer = eval(base_branch_name + "[1].conv")
                    conv_layer.bias = None
                else:
                    for lidx in range(2):
                        conv_layer = eval(
                            base_branch_name + "[" + str(lidx) + "].conv")
                        conv_layer.bias = None
        for aux_layer_name in self.aux_layer_names:
            base_layer_name = "self.base_model." + aux_layer_name
            conv_layer = eval(base_layer_name + ".conv.conv")
            conv_layer.bias = None
            fc1_layer = eval(base_layer_name + ".fc1")
            fc1_layer.bias = None
            fc2_layer = eval(base_layer_name + ".fc2")
            fc2_layer.bias = None
        self.base_model.fc.bias = None

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        # print(x.get_device(), self.all_layers[0][1].weight.get_device())
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        for fidx in range(start_idx, 16):
            y = self.all_layers[fidx][1](y)
            if (conv_layer_idx is not None) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                conv_acts = y

        y = self.base_model.avgpool(y)
        outs = self.base_model.fc(y.reshape(y.shape[0], -1))

        if conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def disable_inplace(self):
        pass

    def _assign_rule(self, layer, rule, rule_param):
        if rule_param is None:
            layer.rule = rule()
        else:
            layer.rule = rule(rule_param)

    def set_lower_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.maxpool1, rule, rule_param)
        self._assign_rule(self.base_model.conv2.conv, rule, rule_param)
        self._assign_rule(self.base_model.conv3.conv, rule, rule_param)
        self._assign_rule(self.base_model.maxpool2, rule, rule_param)
        self._assign_rule(self.base_model.maxpool3, rule, rule_param)
        for inception_layer_name in self.inception_layer_names[:2]:
            base_layer_name = "self.base_model." + inception_layer_name
            block = eval(base_layer_name)
            self._assign_rule(block.branch1.conv, rule, rule_param)
            self._assign_rule(block.branch2[0].conv, rule, rule_param)
            self._assign_rule(block.branch2[1].conv, rule, rule_param)
            self._assign_rule(block.branch3[0].conv, rule, rule_param)
            self._assign_rule(block.branch3[1].conv, rule, rule_param)
            self._assign_rule(block.branch4[0], rule, rule_param)
            self._assign_rule(block.branch4[1].conv, rule, rule_param)

    def set_middle_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.maxpool4, rule, rule_param)
        self._assign_rule(self.base_model.avgpool, rule, rule_param)
        for inception_layer_name in self.inception_layer_names[2:]:
            base_layer_name = "self.base_model." + inception_layer_name
            block = eval(base_layer_name)
            self._assign_rule(block.branch1.conv, rule, rule_param)
            self._assign_rule(block.branch2[0].conv, rule, rule_param)
            self._assign_rule(block.branch2[1].conv, rule, rule_param)
            self._assign_rule(block.branch3[0].conv, rule, rule_param)
            self._assign_rule(block.branch3[1].conv, rule, rule_param)
            self._assign_rule(block.branch4[0], rule, rule_param)
            self._assign_rule(block.branch4[1].conv, rule, rule_param)

    def set_upper_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.fc, rule, rule_param)

    def set_input_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.conv1.conv, rule, rule_param)


class ImagenetContainerGooglenetDisconnected(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True):
        super(ImagenetContainerGooglenetDisconnected, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        self.conv_layer_names = ["conv1", "conv2", "conv3"]
        self.inception_layer_names = ["inception3a", "inception3b", "inception4a", "inception4b",
                                      "inception4c", "inception4d", "inception4e", "inception5a", "inception5b"]
        self.aux_layer_names = ["aux1", "aux2"]

        if transform_batchnorm:
            self._load_model()
        else:
            self.base_model = torchvision.models.googlenet(pretrained=True)

        self.conv_act_ids = [[0, None, None, None, 6, None, None, 15]]

        self.single_spatial = False
        self.single_head = False

        self.all_layers = list(self.base_model.named_children())

        self.start_conv_layer_idx = None

    def _load_model(self):
        self.base_model = googlenet_for_lrp.googlenet(pretrained=False)
        state_dict = torch.hub.load_state_dict_from_url(
            googlenet_for_lrp.model_urls['googlenet'])
        # print(state_dict.keys())
        self._update_state_dict(state_dict)
        self.base_model.load_state_dict(state_dict)

    def _update_state_dict(self, state_dict):
        for conv_layer_name in self.conv_layer_names:
            conv_name = conv_layer_name + ".conv"
            bn_name = conv_layer_name + ".bn"
            self._update_state_dict_single(state_dict, bn_name, conv_name)
        for inception_layer_name in self.inception_layer_names:
            for branch_idx in range(4):
                base_branch_name = inception_layer_name + \
                    ".branch" + str(branch_idx + 1)
                if branch_idx == 0:
                    conv_name = base_branch_name + ".conv"
                    bn_name = base_branch_name + ".bn"
                    self._update_state_dict_single(
                        state_dict, bn_name, conv_name)
                    continue
                for lidx in range(2):
                    conv_name = base_branch_name + \
                        "." + str(lidx) + ".conv"
                    bn_name = base_branch_name + "." + str(lidx) + ".bn"
                    if conv_name + ".weight" not in state_dict:
                        continue
                    self._update_state_dict_single(
                        state_dict, bn_name, conv_name)
        for aux_layer_name in self.aux_layer_names:
            conv_name = aux_layer_name + ".conv.conv"
            bn_name = aux_layer_name + ".conv.bn"
            self._update_state_dict_single(state_dict, bn_name, conv_name)

    def _update_state_dict_single(self, state_dict, name1, name2):
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"], state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"])
        for idx in range(len(conv_weight)):
            state_dict[name2 + ".weight"].data[idx] *= conv_weight[idx]
        state_dict[name2 + ".bias"] = conv_bias
        del state_dict[name1 + ".weight"]
        del state_dict[name1 + ".bias"]
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]
        del state_dict[name1 + ".num_batches_tracked"]

    def _get_equivalent_conv_weights(self, weight, bias, running_mean, running_var):
        num_features = len(weight)
        eps = 0.001
        conv_weight = torch.zeros(num_features)
        conv_bias = torch.zeros(num_features)
        conv_weight.data = weight / torch.sqrt(running_var + eps)
        conv_bias.data = ((-1.0 * running_mean * weight) /
                          torch.sqrt(running_var + eps)) + bias
        return conv_weight, conv_bias

    def _remove_bias(self):
        for conv_layer_name in self.conv_layer_names:
            conv_layer = eval("self.base_model." + conv_layer_name + ".conv")
            conv_layer.bias = None
        for inception_layer_name in self.inception_layer_names:
            base_layer_name = "self.base_model." + inception_layer_name
            for branch_idx in range(4):
                base_branch_name = base_layer_name + \
                    ".branch" + str(branch_idx + 1)
                if branch_idx == 0:
                    conv_layer = eval(base_branch_name + ".conv")
                    conv_layer.bias = None
                elif branch_idx == 3:
                    conv_layer = eval(base_branch_name + "[1].conv")
                    conv_layer.bias = None
                else:
                    for lidx in range(2):
                        conv_layer = eval(
                            base_branch_name + "[" + str(lidx) + "].conv")
                        conv_layer.bias = None
        for aux_layer_name in self.aux_layer_names:
            base_layer_name = "self.base_model." + aux_layer_name
            conv_layer = eval(base_layer_name + ".conv.conv")
            conv_layer.bias = None
            fc1_layer = eval(base_layer_name + ".fc1")
            fc1_layer.bias = None
            fc2_layer = eval(base_layer_name + ".fc2")
            fc2_layer.bias = None
        self.base_model.fc.bias = None

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None, loc_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        if conv_layer_idx is not None:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, loc_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, 16):
                y = self.all_layers[fidx][1](y)
                if (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                    conv_acts = y
                    return None, conv_acts
        else:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, self.output_head_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, 16):
                y = self.all_layers[fidx][1](y)
            y = self.base_model.avgpool(y)

            outs = self.base_model.fc(y.reshape(y.shape[0], -1))

        return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def disable_inplace(self):
        pass

    def _assign_rule(self, layer, rule, rule_param):
        if rule_param is None:
            layer.rule = rule()
        else:
            layer.rule = rule(rule_param)

    def set_lower_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.maxpool1, rule, rule_param)
        self._assign_rule(self.base_model.conv2.conv, rule, rule_param)
        self._assign_rule(self.base_model.conv3.conv, rule, rule_param)
        self._assign_rule(self.base_model.maxpool2, rule, rule_param)
        self._assign_rule(self.base_model.maxpool3, rule, rule_param)
        for inception_layer_name in self.inception_layer_names[:2]:
            base_layer_name = "self.base_model." + inception_layer_name
            block = eval(base_layer_name)
            self._assign_rule(block.branch1.conv, rule, rule_param)
            self._assign_rule(block.branch2[0].conv, rule, rule_param)
            self._assign_rule(block.branch2[1].conv, rule, rule_param)
            self._assign_rule(block.branch3[0].conv, rule, rule_param)
            self._assign_rule(block.branch3[1].conv, rule, rule_param)
            self._assign_rule(block.branch4[0], rule, rule_param)
            self._assign_rule(block.branch4[1].conv, rule, rule_param)

    def set_middle_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.maxpool4, rule, rule_param)
        self._assign_rule(self.base_model.avgpool, rule, rule_param)
        for inception_layer_name in self.inception_layer_names[2:]:
            base_layer_name = "self.base_model." + inception_layer_name
            block = eval(base_layer_name)
            self._assign_rule(block.branch1.conv, rule, rule_param)
            self._assign_rule(block.branch2[0].conv, rule, rule_param)
            self._assign_rule(block.branch2[1].conv, rule, rule_param)
            self._assign_rule(block.branch3[0].conv, rule, rule_param)
            self._assign_rule(block.branch3[1].conv, rule, rule_param)
            self._assign_rule(block.branch4[0], rule, rule_param)
            self._assign_rule(block.branch4[1].conv, rule, rule_param)

    def set_upper_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.fc, rule, rule_param)

    def set_input_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.conv1.conv, rule, rule_param)


class ImagenetContainerGooglenetCommonSpatial(torch.nn.Module):

    def __init__(self, scale=3, output_head_idx=None, transform_batchnorm=True):
        super(ImagenetContainerGooglenetCommonSpatial, self).__init__()
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        self.conv_layer_names = ["conv1", "conv2", "conv3"]
        self.inception_layer_names = ["inception3a", "inception3b", "inception4a", "inception4b",
                                      "inception4c", "inception4d", "inception4e", "inception5a", "inception5b"]
        self.aux_layer_names = ["aux1", "aux2"]

        if transform_batchnorm:
            self._load_model()
        else:
            self.base_model = torchvision.models.googlenet(pretrained=True)

        self.conv_act_ids = [[0, None, None, None, 6, None, None, 15]]

        self.single_spatial = True
        self.single_head = False

        self.all_layers = list(self.base_model.named_children())

        self.start_conv_layer_idx = None

    def _load_model(self):
        self.base_model = googlenet_for_lrp.googlenet(pretrained=False)
        state_dict = torch.hub.load_state_dict_from_url(
            googlenet_for_lrp.model_urls['googlenet'])
        # print(state_dict.keys())
        self._update_state_dict(state_dict)
        self.base_model.load_state_dict(state_dict)

    def _update_state_dict(self, state_dict):
        for conv_layer_name in self.conv_layer_names:
            conv_name = conv_layer_name + ".conv"
            bn_name = conv_layer_name + ".bn"
            self._update_state_dict_single(state_dict, bn_name, conv_name)
        for inception_layer_name in self.inception_layer_names:
            for branch_idx in range(4):
                base_branch_name = inception_layer_name + \
                    ".branch" + str(branch_idx + 1)
                if branch_idx == 0:
                    conv_name = base_branch_name + ".conv"
                    bn_name = base_branch_name + ".bn"
                    self._update_state_dict_single(
                        state_dict, bn_name, conv_name)
                    continue
                for lidx in range(2):
                    conv_name = base_branch_name + \
                        "." + str(lidx) + ".conv"
                    bn_name = base_branch_name + "." + str(lidx) + ".bn"
                    if conv_name + ".weight" not in state_dict:
                        continue
                    self._update_state_dict_single(
                        state_dict, bn_name, conv_name)
        for aux_layer_name in self.aux_layer_names:
            conv_name = aux_layer_name + ".conv.conv"
            bn_name = aux_layer_name + ".conv.bn"
            self._update_state_dict_single(state_dict, bn_name, conv_name)

    def _update_state_dict_single(self, state_dict, name1, name2):
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"], state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"])
        for idx in range(len(conv_weight)):
            state_dict[name2 + ".weight"].data[idx] *= conv_weight[idx]
        state_dict[name2 + ".bias"] = conv_bias
        del state_dict[name1 + ".weight"]
        del state_dict[name1 + ".bias"]
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]
        del state_dict[name1 + ".num_batches_tracked"]

    def _get_equivalent_conv_weights(self, weight, bias, running_mean, running_var):
        num_features = len(weight)
        eps = 0.001
        conv_weight = torch.zeros(num_features)
        conv_bias = torch.zeros(num_features)
        conv_weight.data = weight / torch.sqrt(running_var + eps)
        conv_bias.data = ((-1.0 * running_mean * weight) /
                          torch.sqrt(running_var + eps)) + bias
        return conv_weight, conv_bias

    def _remove_bias(self):
        for conv_layer_name in self.conv_layer_names:
            conv_layer = eval("self.base_model." + conv_layer_name + ".conv")
            conv_layer.bias = None
        for inception_layer_name in self.inception_layer_names:
            base_layer_name = "self.base_model." + inception_layer_name
            for branch_idx in range(4):
                base_branch_name = base_layer_name + \
                    ".branch" + str(branch_idx + 1)
                if branch_idx == 0:
                    conv_layer = eval(base_branch_name + ".conv")
                    conv_layer.bias = None
                elif branch_idx == 3:
                    conv_layer = eval(base_branch_name + "[1].conv")
                    conv_layer.bias = None
                else:
                    for lidx in range(2):
                        conv_layer = eval(
                            base_branch_name + "[" + str(lidx) + "].conv")
                        conv_layer.bias = None
        for aux_layer_name in self.aux_layer_names:
            base_layer_name = "self.base_model." + aux_layer_name
            conv_layer = eval(base_layer_name + ".conv.conv")
            conv_layer.bias = None
            fc1_layer = eval(base_layer_name + ".fc1")
            fc1_layer.bias = None
            fc2_layer = eval(base_layer_name + ".fc2")
            fc2_layer.bias = None
        self.base_model.fc.bias = None

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        for fidx in range(start_idx, 16):
            y = self.all_layers[fidx][1](y)
            if (conv_layer_idx is not None) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                conv_acts = y

        ya, xa, ha, wa = get_augmentation_range(
            y.shape, self.scale, self.output_head_idx)
        y = self.base_model.avgpool(y[:, :, ya:ya + ha, xa:xa + wa])

        outs = self.base_model.fc(y.reshape(y.shape[0], -1))

        if conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def disable_inplace(self):
        pass

    def _assign_rule(self, layer, rule, rule_param):
        if rule_param is None:
            layer.rule = rule()
        else:
            layer.rule = rule(rule_param)

    def set_lower_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.maxpool1, rule, rule_param)
        self._assign_rule(self.base_model.conv2.conv, rule, rule_param)
        self._assign_rule(self.base_model.conv3.conv, rule, rule_param)
        self._assign_rule(self.base_model.maxpool2, rule, rule_param)
        self._assign_rule(self.base_model.maxpool3, rule, rule_param)
        for inception_layer_name in self.inception_layer_names[:2]:
            base_layer_name = "self.base_model." + inception_layer_name
            block = eval(base_layer_name)
            self._assign_rule(block.branch1.conv, rule, rule_param)
            self._assign_rule(block.branch2[0].conv, rule, rule_param)
            self._assign_rule(block.branch2[1].conv, rule, rule_param)
            self._assign_rule(block.branch3[0].conv, rule, rule_param)
            self._assign_rule(block.branch3[1].conv, rule, rule_param)
            self._assign_rule(block.branch4[0], rule, rule_param)
            self._assign_rule(block.branch4[1].conv, rule, rule_param)

    def set_middle_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.maxpool4, rule, rule_param)
        self._assign_rule(self.base_model.avgpool, rule, rule_param)
        for inception_layer_name in self.inception_layer_names[2:]:
            base_layer_name = "self.base_model." + inception_layer_name
            block = eval(base_layer_name)
            self._assign_rule(block.branch1.conv, rule, rule_param)
            self._assign_rule(block.branch2[0].conv, rule, rule_param)
            self._assign_rule(block.branch2[1].conv, rule, rule_param)
            self._assign_rule(block.branch3[0].conv, rule, rule_param)
            self._assign_rule(block.branch3[1].conv, rule, rule_param)
            self._assign_rule(block.branch4[0], rule, rule_param)
            self._assign_rule(block.branch4[1].conv, rule, rule_param)

    def set_upper_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.fc, rule, rule_param)

    def set_input_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.conv1.conv, rule, rule_param)


class ImagenetContainerResnextPointingGame(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True, every_layer=False):
        super(ImagenetContainerResnextPointingGame, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert self.output_head_idx is None or (
            self.output_head_idx >= 0 and self.output_head_idx < self.num_heads)

        if transform_batchnorm:
            raise NotImplementedError
            self._load_model()
            self.conv_act_ids = [[1, None, None, None, 4, None, None, 6]]
        else:
            self.base_model = torchvision.models.resnext50_32x4d(
                pretrained=True)
            if every_layer:
                self.conv_act_ids = [[2, 4, 5, 6, 7]]
            else:
                self.conv_act_ids = [[2, None, None, None, 5, None, None, 7]]

        self.single_spatial = True
        self.single_head = True

        self.all_layers = list(self.base_model.named_children())

        self.start_conv_layer_idx = None

    def _load_model(self):
        self.base_model = resnet_for_lrp.resnext50_32x4d(pretrained=False)
        state_dict = torch.hub.load_state_dict_from_url(
            resnet_for_lrp.model_urls['resnext50_32x4d'])
        # print(state_dict.keys())
        self._update_state_dict(state_dict)
        self.base_model.load_state_dict(state_dict)

    def _update_state_dict(self, state_dict):
        self._update_state_dict_single(state_dict, "bn1", "conv1")
        for layer in range(4):
            block_idx = 0
            end_of_block = False
            base_layer_name = "layer" + str(layer + 1)
            while not end_of_block:
                conv_bn_idx = 0
                base_block_name = base_layer_name + "." + str(block_idx)
                while True:
                    conv_layer_name = base_block_name + \
                        ".conv" + str(conv_bn_idx + 1)
                    bn_layer_name = base_block_name + \
                        ".bn" + str(conv_bn_idx + 1)
                    if conv_layer_name + ".weight" not in state_dict:
                        if conv_bn_idx == 0:
                            end_of_block = True
                        break
                    self._update_state_dict_single(
                        state_dict, bn_layer_name, conv_layer_name)
                    conv_bn_idx += 1
                downsample_conv_name = base_block_name + ".downsample.0"
                downsample_bn_name = base_block_name + ".downsample.1"
                if downsample_conv_name + ".weight" in state_dict:
                    self._update_state_dict_single(
                        state_dict, downsample_bn_name, downsample_conv_name)
                block_idx += 1

    def _update_state_dict_single(self, state_dict, name1, name2):
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"], state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"])
        for idx in range(len(conv_weight)):
            state_dict[name2 + ".weight"].data[idx] *= conv_weight[idx]
        state_dict[name2 + ".bias"] = conv_bias
        del state_dict[name1 + ".weight"]
        del state_dict[name1 + ".bias"]
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]
        del state_dict[name1 + ".num_batches_tracked"]

    def _get_equivalent_conv_weights(self, weight, bias, running_mean, running_var):
        num_features = len(weight)
        eps = 1e-5
        conv_weight = torch.zeros(num_features)
        conv_bias = torch.zeros(num_features)
        conv_weight.data = weight / torch.sqrt(running_var + eps)
        conv_bias.data = ((-1.0 * running_mean * weight) /
                          torch.sqrt(running_var + eps)) + bias
        return conv_weight, conv_bias

    def _remove_bias(self):
        self.base_model.conv1.bias = None
        for layer in range(4):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    conv_layer.bias = None
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    downsample_conv.bias = None
        self.base_model.fc.bias = None

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        # print(x.get_device(), self.all_layers[0][1].weight.get_device())
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        for fidx in range(start_idx, len(self.all_layers) - 1):
            y = self.all_layers[fidx][1](y)
            if (conv_layer_idx is not None) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                conv_acts = y

        outs = self.all_layers[-1][1](y.reshape(y.shape[0], -1))

        if conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        self.base_model.relu.inplace = value
        for layer in range(4):
            block = eval("self.base_model.layer" + str(layer + 1))
            for block_idx in range(len(block)):
                block[block_idx].relu.inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def _assign_rule(self, layer, rule, rule_param):
        if rule_param is None:
            layer.rule = rule()
        else:
            layer.rule = rule(rule_param)

    def set_lower_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.maxpool, rule, rule_param)
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_middle_rule(self, rule, rule_param):
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 3)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_upper_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.fc, rule, rule_param)

    def set_input_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.conv1, rule, rule_param)


class ImagenetContainerResnextDisconnected(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True):
        super(ImagenetContainerResnextDisconnected, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        if transform_batchnorm:
            self._load_model()
            self.conv_act_ids = [[1, None, None, None, 4, None, None, 6]]
        else:
            self.base_model = torchvision.models.resnext50_32x4d(
                pretrained=True)
            self.conv_act_ids = [[2, None, None, None, 5, None, None, 7]]

        self.single_spatial = False
        self.single_head = False

        self.all_layers = list(self.base_model.named_children())

        self.start_conv_layer_idx = None

    def _load_model(self):
        self.base_model = resnet_for_lrp.resnext50_32x4d(pretrained=False)
        state_dict = torch.hub.load_state_dict_from_url(
            resnet_for_lrp.model_urls['resnext50_32x4d'])
        # print(state_dict.keys())
        self._update_state_dict(state_dict)
        self.base_model.load_state_dict(state_dict)

    def _update_state_dict(self, state_dict):
        self._update_state_dict_single(state_dict, "bn1", "conv1")
        for layer in range(4):
            block_idx = 0
            end_of_block = False
            base_layer_name = "layer" + str(layer + 1)
            while not end_of_block:
                conv_bn_idx = 0
                base_block_name = base_layer_name + "." + str(block_idx)
                while True:
                    conv_layer_name = base_block_name + \
                        ".conv" + str(conv_bn_idx + 1)
                    bn_layer_name = base_block_name + \
                        ".bn" + str(conv_bn_idx + 1)
                    if conv_layer_name + ".weight" not in state_dict:
                        if conv_bn_idx == 0:
                            end_of_block = True
                        break
                    self._update_state_dict_single(
                        state_dict, bn_layer_name, conv_layer_name)
                    conv_bn_idx += 1
                downsample_conv_name = base_block_name + ".downsample.0"
                downsample_bn_name = base_block_name + ".downsample.1"
                if downsample_conv_name + ".weight" in state_dict:
                    self._update_state_dict_single(
                        state_dict, downsample_bn_name, downsample_conv_name)
                block_idx += 1

    def _update_state_dict_single(self, state_dict, name1, name2):
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"], state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"])
        for idx in range(len(conv_weight)):
            state_dict[name2 + ".weight"].data[idx] *= conv_weight[idx]
        state_dict[name2 + ".bias"] = conv_bias
        del state_dict[name1 + ".weight"]
        del state_dict[name1 + ".bias"]
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]
        del state_dict[name1 + ".num_batches_tracked"]

    def _get_equivalent_conv_weights(self, weight, bias, running_mean, running_var):
        num_features = len(weight)
        eps = 1e-5
        conv_weight = torch.zeros(num_features)
        conv_bias = torch.zeros(num_features)
        conv_weight.data = weight / torch.sqrt(running_var + eps)
        conv_bias.data = ((-1.0 * running_mean * weight) /
                          torch.sqrt(running_var + eps)) + bias
        return conv_weight, conv_bias

    def _remove_bias(self):
        self.base_model.conv1.bias = None
        for layer in range(4):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    conv_layer.bias = None
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    downsample_conv.bias = None
        self.base_model.fc.bias = None

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None, loc_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        if conv_layer_idx is not None:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, loc_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.all_layers) - 1):
                y = self.all_layers[fidx][1](y)
                if (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                    conv_acts = y
                    return None, conv_acts
        else:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, self.output_head_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.all_layers) - 1):
                y = self.all_layers[fidx][1](y)

            outs = self.base_model.fc(y.reshape(y.shape[0], -1))

        return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        self.base_model.relu.inplace = value
        for layer in range(4):
            block = eval("self.base_model.layer" + str(layer + 1))
            for block_idx in range(len(block)):
                block[block_idx].relu.inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def _assign_rule(self, layer, rule, rule_param):
        if rule_param is None:
            layer.rule = rule()
        else:
            layer.rule = rule(rule_param)

    def set_lower_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.maxpool, rule, rule_param)
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_middle_rule(self, rule, rule_param):
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 3)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_upper_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.fc, rule, rule_param)

    def set_input_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.conv1, rule, rule_param)


class ImagenetContainerResnextCommonSpatial(torch.nn.Module):

    def __init__(self, scale=3, output_head_idx=None, transform_batchnorm=True):
        super(ImagenetContainerResnextCommonSpatial, self).__init__()
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        if transform_batchnorm:
            self._load_model()
            self.conv_act_ids = [[1, None, None, None, 4, None, None, 6]]
        else:
            self.base_model = torchvision.models.resnext50_32x4d(
                pretrained=True)
            self.conv_act_ids = [[2, None, None, None, 5, None, None, 7]]

        self.single_spatial = True
        self.single_head = False

        self.all_layers = list(self.base_model.named_children())

        self.start_conv_layer_idx = None

    def _load_model(self):
        self.base_model = resnet_for_lrp.resnext50_32x4d(pretrained=False)
        state_dict = torch.hub.load_state_dict_from_url(
            resnet_for_lrp.model_urls['resnext50_32x4d'])
        # print(state_dict.keys())
        self._update_state_dict(state_dict)
        self.base_model.load_state_dict(state_dict)

    def _update_state_dict(self, state_dict):
        self._update_state_dict_single(state_dict, "bn1", "conv1")
        for layer in range(4):
            block_idx = 0
            end_of_block = False
            base_layer_name = "layer" + str(layer + 1)
            while not end_of_block:
                conv_bn_idx = 0
                base_block_name = base_layer_name + "." + str(block_idx)
                while True:
                    conv_layer_name = base_block_name + \
                        ".conv" + str(conv_bn_idx + 1)
                    bn_layer_name = base_block_name + \
                        ".bn" + str(conv_bn_idx + 1)
                    if conv_layer_name + ".weight" not in state_dict:
                        if conv_bn_idx == 0:
                            end_of_block = True
                        break
                    self._update_state_dict_single(
                        state_dict, bn_layer_name, conv_layer_name)
                    conv_bn_idx += 1
                downsample_conv_name = base_block_name + ".downsample.0"
                downsample_bn_name = base_block_name + ".downsample.1"
                if downsample_conv_name + ".weight" in state_dict:
                    self._update_state_dict_single(
                        state_dict, downsample_bn_name, downsample_conv_name)
                block_idx += 1

    def _update_state_dict_single(self, state_dict, name1, name2):
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"], state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"])
        for idx in range(len(conv_weight)):
            state_dict[name2 + ".weight"].data[idx] *= conv_weight[idx]
        state_dict[name2 + ".bias"] = conv_bias
        del state_dict[name1 + ".weight"]
        del state_dict[name1 + ".bias"]
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]
        del state_dict[name1 + ".num_batches_tracked"]

    def _get_equivalent_conv_weights(self, weight, bias, running_mean, running_var):
        num_features = len(weight)
        eps = 1e-5
        conv_weight = torch.zeros(num_features)
        conv_bias = torch.zeros(num_features)
        conv_weight.data = weight / torch.sqrt(running_var + eps)
        conv_bias.data = ((-1.0 * running_mean * weight) /
                          torch.sqrt(running_var + eps)) + bias
        return conv_weight, conv_bias

    def _remove_bias(self):
        self.base_model.conv1.bias = None
        for layer in range(4):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    conv_layer.bias = None
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    downsample_conv.bias = None
        self.base_model.fc.bias = None

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        for fidx in range(start_idx, len(self.all_layers) - 2):
            y = self.all_layers[fidx][1](y)
            if (conv_layer_idx is not None) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                conv_acts = y

        ya, xa, ha, wa = get_augmentation_range(
            y.shape, self.scale, self.output_head_idx)
        y = self.all_layers[-2][1](y[:, :, ya:ya + ha, xa:xa + wa])
        outs = self.base_model.fc(y.reshape(y.shape[0], -1))

        if conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        self.base_model.relu.inplace = value
        for layer in range(4):
            block = eval("self.base_model.layer" + str(layer + 1))
            for block_idx in range(len(block)):
                block[block_idx].relu.inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def _assign_rule(self, layer, rule, rule_param):
        if rule_param is None:
            layer.rule = rule()
        else:
            layer.rule = rule(rule_param)

    def set_lower_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.maxpool, rule, rule_param)
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_middle_rule(self, rule, rule_param):
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 3)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_upper_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.fc, rule, rule_param)

    def set_input_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.conv1, rule, rule_param)


class ImagenetContainerWideResnetPointingGame(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True, every_layer=False):
        super(ImagenetContainerWideResnetPointingGame, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert self.output_head_idx is None or (
            self.output_head_idx >= 0 and self.output_head_idx < self.num_heads)

        if transform_batchnorm:
            raise NotImplementedError
            self._load_model()
            self.conv_act_ids = [[1, None, None, None, 4, None, None, 6]]
        else:
            self.base_model = torchvision.models.wide_resnet50_2(
                pretrained=True)
            if every_layer:
                self.conv_act_ids = [[2,4,5,6,7]]
            else:
                self.conv_act_ids = [[2, None, None, None, 5, None, None, 7]]

        self.single_spatial = True
        self.single_head = True

        self.all_layers = list(self.base_model.named_children())

        self.start_conv_layer_idx = None

    def _load_model(self):
        self.base_model = resnet_for_lrp.wide_resnet50_2(pretrained=False)
        state_dict = torch.hub.load_state_dict_from_url(
            resnet_for_lrp.model_urls['wide_resnet50_2'])
        # print(state_dict.keys())
        self._update_state_dict(state_dict)
        self.base_model.load_state_dict(state_dict)

    def _update_state_dict(self, state_dict):
        self._update_state_dict_single(state_dict, "bn1", "conv1")
        for layer in range(4):
            block_idx = 0
            end_of_block = False
            base_layer_name = "layer" + str(layer + 1)
            while not end_of_block:
                conv_bn_idx = 0
                base_block_name = base_layer_name + "." + str(block_idx)
                while True:
                    conv_layer_name = base_block_name + \
                        ".conv" + str(conv_bn_idx + 1)
                    bn_layer_name = base_block_name + \
                        ".bn" + str(conv_bn_idx + 1)
                    if conv_layer_name + ".weight" not in state_dict:
                        if conv_bn_idx == 0:
                            end_of_block = True
                        break
                    self._update_state_dict_single(
                        state_dict, bn_layer_name, conv_layer_name)
                    conv_bn_idx += 1
                downsample_conv_name = base_block_name + ".downsample.0"
                downsample_bn_name = base_block_name + ".downsample.1"
                if downsample_conv_name + ".weight" in state_dict:
                    self._update_state_dict_single(
                        state_dict, downsample_bn_name, downsample_conv_name)
                block_idx += 1

    def _update_state_dict_single(self, state_dict, name1, name2):
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"], state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"])
        for idx in range(len(conv_weight)):
            state_dict[name2 + ".weight"].data[idx] *= conv_weight[idx]
        state_dict[name2 + ".bias"] = conv_bias
        del state_dict[name1 + ".weight"]
        del state_dict[name1 + ".bias"]
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]
        del state_dict[name1 + ".num_batches_tracked"]

    def _get_equivalent_conv_weights(self, weight, bias, running_mean, running_var):
        num_features = len(weight)
        eps = 1e-5
        conv_weight = torch.zeros(num_features)
        conv_bias = torch.zeros(num_features)
        conv_weight.data = weight / torch.sqrt(running_var + eps)
        conv_bias.data = ((-1.0 * running_mean * weight) /
                          torch.sqrt(running_var + eps)) + bias
        return conv_weight, conv_bias

    def _remove_bias(self):
        self.base_model.conv1.bias = None
        for layer in range(4):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    conv_layer.bias = None
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    downsample_conv.bias = None
        self.base_model.fc.bias = None

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        # print(x.get_device(), self.all_layers[0][1].weight.get_device())
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        for fidx in range(start_idx, len(self.all_layers) - 1):
            y = self.all_layers[fidx][1](y)
            if (conv_layer_idx is not None) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                conv_acts = y

        outs = self.all_layers[-1][1](y.reshape(y.shape[0], -1))

        if conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        self.base_model.relu.inplace = value
        for layer in range(4):
            block = eval("self.base_model.layer" + str(layer + 1))
            for block_idx in range(len(block)):
                block[block_idx].relu.inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def _assign_rule(self, layer, rule, rule_param):
        if rule_param is None:
            layer.rule = rule()
        else:
            layer.rule = rule(rule_param)

    def set_lower_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.maxpool, rule, rule_param)
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_middle_rule(self, rule, rule_param):
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 3)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_upper_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.fc, rule, rule_param)

    def set_input_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.conv1, rule, rule_param)


class ImagenetContainerWideResnetDisconnected(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True):
        super(ImagenetContainerWideResnetDisconnected, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        if transform_batchnorm:
            self._load_model()
            self.conv_act_ids = [[1, None, None, None, 4, None, None, 6]]
        else:
            self.base_model = torchvision.models.wide_resnet50_2(
                pretrained=True)
            self.conv_act_ids = [[2, None, None, None, 5, None, None, 7]]

        self.single_spatial = False
        self.single_head = False

        self.all_layers = list(self.base_model.named_children())

        self.start_conv_layer_idx = None

    def _load_model(self):
        self.base_model = resnet_for_lrp.wide_resnet50_2(pretrained=False)
        state_dict = torch.hub.load_state_dict_from_url(
            resnet_for_lrp.model_urls['wide_resnet50_2'])
        # print(state_dict.keys())
        self._update_state_dict(state_dict)
        self.base_model.load_state_dict(state_dict)

    def _update_state_dict(self, state_dict):
        self._update_state_dict_single(state_dict, "bn1", "conv1")
        for layer in range(4):
            block_idx = 0
            end_of_block = False
            base_layer_name = "layer" + str(layer + 1)
            while not end_of_block:
                conv_bn_idx = 0
                base_block_name = base_layer_name + "." + str(block_idx)
                while True:
                    conv_layer_name = base_block_name + \
                        ".conv" + str(conv_bn_idx + 1)
                    bn_layer_name = base_block_name + \
                        ".bn" + str(conv_bn_idx + 1)
                    if conv_layer_name + ".weight" not in state_dict:
                        if conv_bn_idx == 0:
                            end_of_block = True
                        break
                    self._update_state_dict_single(
                        state_dict, bn_layer_name, conv_layer_name)
                    conv_bn_idx += 1
                downsample_conv_name = base_block_name + ".downsample.0"
                downsample_bn_name = base_block_name + ".downsample.1"
                if downsample_conv_name + ".weight" in state_dict:
                    self._update_state_dict_single(
                        state_dict, downsample_bn_name, downsample_conv_name)
                block_idx += 1

    def _update_state_dict_single(self, state_dict, name1, name2):
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"], state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"])
        for idx in range(len(conv_weight)):
            state_dict[name2 + ".weight"].data[idx] *= conv_weight[idx]
        state_dict[name2 + ".bias"] = conv_bias
        del state_dict[name1 + ".weight"]
        del state_dict[name1 + ".bias"]
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]
        del state_dict[name1 + ".num_batches_tracked"]

    def _get_equivalent_conv_weights(self, weight, bias, running_mean, running_var):
        num_features = len(weight)
        eps = 1e-5
        conv_weight = torch.zeros(num_features)
        conv_bias = torch.zeros(num_features)
        conv_weight.data = weight / torch.sqrt(running_var + eps)
        conv_bias.data = ((-1.0 * running_mean * weight) /
                          torch.sqrt(running_var + eps)) + bias
        return conv_weight, conv_bias

    def _remove_bias(self):
        self.base_model.conv1.bias = None
        for layer in range(4):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    conv_layer.bias = None
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    downsample_conv.bias = None
        self.base_model.fc.bias = None

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None, loc_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        if conv_layer_idx is not None:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, loc_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.all_layers) - 1):
                y = self.all_layers[fidx][1](y)
                if (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                    conv_acts = y
                    return None, conv_acts
        else:
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, self.output_head_idx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.all_layers) - 1):
                y = self.all_layers[fidx][1](y)

            outs = self.base_model.fc(y.reshape(y.shape[0], -1))

        return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        self.base_model.relu.inplace = value
        for layer in range(4):
            block = eval("self.base_model.layer" + str(layer + 1))
            for block_idx in range(len(block)):
                block[block_idx].relu.inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def _assign_rule(self, layer, rule, rule_param):
        if rule_param is None:
            layer.rule = rule()
        else:
            layer.rule = rule(rule_param)

    def set_lower_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.maxpool, rule, rule_param)
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_middle_rule(self, rule, rule_param):
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 3)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_upper_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.fc, rule, rule_param)

    def set_input_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.conv1, rule, rule_param)


class ImagenetContainerWideResnetCommonSpatial(torch.nn.Module):

    def __init__(self, scale=3, output_head_idx=None, transform_batchnorm=True):
        super(ImagenetContainerWideResnetCommonSpatial, self).__init__()
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert (self.output_head_idx >=
                0 and self.output_head_idx < self.num_heads)

        if transform_batchnorm:
            self._load_model()
            self.conv_act_ids = [[1, None, None, None, 4, None, None, 6]]
        else:
            self.base_model = torchvision.models.wide_resnet50_2(
                pretrained=True)
            self.conv_act_ids = [[2, None, None, None, 5, None, None, 7]]

        self.single_spatial = True
        self.single_head = False

        self.all_layers = list(self.base_model.named_children())

        self.start_conv_layer_idx = None

    def _load_model(self):
        self.base_model = resnet_for_lrp.wide_resnet50_2(pretrained=False)
        state_dict = torch.hub.load_state_dict_from_url(
            resnet_for_lrp.model_urls['wide_resnet50_2'])
        # print(state_dict.keys())
        self._update_state_dict(state_dict)
        self.base_model.load_state_dict(state_dict)

    def _update_state_dict(self, state_dict):
        self._update_state_dict_single(state_dict, "bn1", "conv1")
        for layer in range(4):
            block_idx = 0
            end_of_block = False
            base_layer_name = "layer" + str(layer + 1)
            while not end_of_block:
                conv_bn_idx = 0
                base_block_name = base_layer_name + "." + str(block_idx)
                while True:
                    conv_layer_name = base_block_name + \
                        ".conv" + str(conv_bn_idx + 1)
                    bn_layer_name = base_block_name + \
                        ".bn" + str(conv_bn_idx + 1)
                    if conv_layer_name + ".weight" not in state_dict:
                        if conv_bn_idx == 0:
                            end_of_block = True
                        break
                    self._update_state_dict_single(
                        state_dict, bn_layer_name, conv_layer_name)
                    conv_bn_idx += 1
                downsample_conv_name = base_block_name + ".downsample.0"
                downsample_bn_name = base_block_name + ".downsample.1"
                if downsample_conv_name + ".weight" in state_dict:
                    self._update_state_dict_single(
                        state_dict, downsample_bn_name, downsample_conv_name)
                block_idx += 1

    def _update_state_dict_single(self, state_dict, name1, name2):
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"], state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"])
        for idx in range(len(conv_weight)):
            state_dict[name2 + ".weight"].data[idx] *= conv_weight[idx]
        state_dict[name2 + ".bias"] = conv_bias
        del state_dict[name1 + ".weight"]
        del state_dict[name1 + ".bias"]
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]
        del state_dict[name1 + ".num_batches_tracked"]

    def _get_equivalent_conv_weights(self, weight, bias, running_mean, running_var):
        num_features = len(weight)
        eps = 1e-5
        conv_weight = torch.zeros(num_features)
        conv_bias = torch.zeros(num_features)
        conv_weight.data = weight / torch.sqrt(running_var + eps)
        conv_bias.data = ((-1.0 * running_mean * weight) /
                          torch.sqrt(running_var + eps)) + bias
        return conv_weight, conv_bias

    def _remove_bias(self):
        self.base_model.conv1.bias = None
        for layer in range(4):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    conv_layer.bias = None
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    downsample_conv.bias = None
        self.base_model.fc.bias = None

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        for fidx in range(start_idx, len(self.all_layers) - 2):
            y = self.all_layers[fidx][1](y)
            if (conv_layer_idx is not None) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                conv_acts = y

        ya, xa, ha, wa = get_augmentation_range(
            y.shape, self.scale, self.output_head_idx)
        y = self.all_layers[-2][1](y[:, :, ya:ya + ha, xa:xa + wa])
        outs = self.base_model.fc(y.reshape(y.shape[0], -1))

        if conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        self.base_model.relu.inplace = value
        for layer in range(4):
            block = eval("self.base_model.layer" + str(layer + 1))
            for block_idx in range(len(block)):
                block[block_idx].relu.inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def _assign_rule(self, layer, rule, rule_param):
        if rule_param is None:
            layer.rule = rule()
        else:
            layer.rule = rule(rule_param)

    def set_lower_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.maxpool, rule, rule_param)
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 1)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_middle_rule(self, rule, rule_param):
        for layer in range(2):
            base_layer_name = "self.base_model.layer" + str(layer + 3)
            block = eval(base_layer_name)
            for block_idx in range(len(block)):
                base_block_name = base_layer_name + "[" + str(block_idx) + "]"
                self._assign_rule(
                    block[block_idx].addition_module, rule, rule_param)
                conv_idx = 0
                while True:
                    if not hasattr(block[block_idx], "conv" + str(conv_idx + 1)):
                        break
                    conv_layer = eval(base_block_name +
                                      ".conv" + str(conv_idx + 1))
                    self._assign_rule(conv_layer, rule, rule_param)
                    conv_idx += 1
                if hasattr(block[block_idx], "downsample") and block[block_idx].downsample is not None:
                    downsample_conv = eval(base_block_name + ".downsample[0]")
                    self._assign_rule(downsample_conv, rule, rule_param)

    def set_upper_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.fc, rule, rule_param)

    def set_input_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.conv1, rule, rule_param)


class ImagenetContainerResnet101PointingGame(torch.nn.Module):
    def __init__(self, scale=3, activation=None, output_head_idx=None, transform_batchnorm=True, batchnorm_to_conv=False, every_layer=False):
        super(ImagenetContainerResnet101PointingGame, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx
        self.num_heads = self.scale * self.scale
        assert self.output_head_idx is None or (0 <= self.output_head_idx < self.num_heads)

        if batchnorm_to_conv:
            self._load_batchnorm_to_conv_model()
            self.conv_act_ids = [[2, 4, 5, 6, 7]]  # Can adjust if needed
        elif transform_batchnorm:
            self._load_model()
            self.conv_act_ids = [[1, 3, 4, 5, 6]]
        else:
            self.base_model = torchvision.models.resnet101(pretrained=True)
            self.conv_act_ids = [[2, 4, 5, 6, 7]]

        self.single_spatial = True
        self.single_head = True
        self.all_layers = list(self.base_model.named_children())
        self.start_conv_layer_idx = None

    def _load_model(self):
        self.base_model = resnet_for_lrp.resnet101(pretrained=False)
        state_dict = torch.hub.load_state_dict_from_url(resnet_for_lrp.model_urls['resnet101'])
        self._update_state_dict(state_dict)
        self.base_model.load_state_dict(state_dict)

    def _load_batchnorm_to_conv_model(self):
        self.base_model = resnet_for_lrp_conv.resnet101(pretrained=False)
        state_dict = torch.hub.load_state_dict_from_url(resnet_for_lrp_conv.model_urls['resnet101'])
        self._update_state_dict_bn_to_conv(state_dict)
        self.base_model.load_state_dict(state_dict)

    # --- The following utility methods remain unchanged ---
    def _update_state_dict(self, state_dict):
        self._update_state_dict_single(state_dict, "bn1", "conv1")
        for layer in range(4):
            block_idx = 0
            end_of_block = False
            base_layer_name = "layer" + str(layer + 1)
            while not end_of_block:
                conv_bn_idx = 0
                base_block_name = base_layer_name + "." + str(block_idx)
                while True:
                    conv_layer_name = base_block_name + ".conv" + str(conv_bn_idx + 1)
                    bn_layer_name = base_block_name + ".bn" + str(conv_bn_idx + 1)
                    if conv_layer_name + ".weight" not in state_dict:
                        if conv_bn_idx == 0:
                            end_of_block = True
                        break
                    self._update_state_dict_single(state_dict, bn_layer_name, conv_layer_name)
                    conv_bn_idx += 1
                downsample_conv_name = base_block_name + ".downsample.0"
                downsample_bn_name = base_block_name + ".downsample.1"
                if downsample_conv_name + ".weight" in state_dict:
                    self._update_state_dict_single(state_dict, downsample_bn_name, downsample_conv_name)
                block_idx += 1

    def _update_state_dict_bn_to_conv(self, state_dict):
        self._update_state_dict_single_conv(state_dict, "bn1")
        for layer in range(4):
            block_idx = 0
            end_of_block = False
            base_layer_name = "layer" + str(layer + 1)
            while not end_of_block:
                conv_bn_idx = 0
                base_block_name = base_layer_name + "." + str(block_idx)
                while True:
                    conv_layer_name = base_block_name + ".conv" + str(conv_bn_idx + 1)
                    bn_layer_name = base_block_name + ".bn" + str(conv_bn_idx + 1)
                    if conv_layer_name + ".weight" not in state_dict:
                        if conv_bn_idx == 0:
                            end_of_block = True
                        break
                    self._update_state_dict_single_conv(state_dict, bn_layer_name)
                    conv_bn_idx += 1
                downsample_conv_name = base_block_name + ".downsample.0"
                downsample_bn_name = base_block_name + ".downsample.1"
                if downsample_conv_name + ".weight" in state_dict:
                    self._update_state_dict_single_conv(state_dict, downsample_bn_name)
                block_idx += 1

    def _update_state_dict_single(self, state_dict, name1, name2):
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"],
            state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"]
        )
        for idx in range(len(conv_weight)):
            state_dict[name2 + ".weight"].data[idx] *= conv_weight[idx]
        state_dict[name2 + ".bias"] = conv_bias
        del state_dict[name1 + ".weight"]
        del state_dict[name1 + ".bias"]
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]

    def _update_state_dict_single_conv(self, state_dict, name1):
        conv_weight, conv_bias = self._get_equivalent_conv_weights(
            state_dict[name1 + ".weight"], state_dict[name1 + ".bias"],
            state_dict[name1 + ".running_mean"], state_dict[name1 + ".running_var"]
        )
        conv_weight_matrix = torch.zeros((len(conv_weight), len(conv_weight), 1, 1))
        for idx in range(len(conv_weight)):
            conv_weight_matrix.data[idx, idx] = conv_weight[idx]
        state_dict[name1 + ".weight"] = conv_weight_matrix
        state_dict[name1 + ".bias"] = conv_bias
        del state_dict[name1 + ".running_mean"]
        del state_dict[name1 + ".running_var"]

    def _get_equivalent_conv_weights(self, weight, bias, running_mean, running_var):
        eps = 1e-5
        conv_weight = weight / torch.sqrt(running_var + eps)
        conv_bias = (-running_mean * weight / torch.sqrt(running_var + eps)) + bias
        return conv_weight, conv_bias

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        start_idx = 0 if start_conv_layer_idx is None else self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        for fidx in range(start_idx, len(self.all_layers) - 1):
            y = self.all_layers[fidx][1](y)
            if conv_layer_idx is not None and fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]:
                conv_acts = y
        outs = self.all_layers[-1][1](y.reshape(y.shape[0], -1))
        return (outs, conv_acts) if conv_layer_idx is not None else outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)

    def _set_inplace_ReLU(self, value):
        assert value in [True, False]
        self.base_model.relu.inplace = value
        for layer in range(4):
            block = eval("self.base_model.layer" + str(layer + 1))
            for block_idx in range(len(block)):
                block[block_idx].relu.inplace = value

    def disable_inplace(self):
        self._set_inplace_ReLU(False)

    def _assign_rule(self, layer, rule, rule_param):
        layer.rule = rule() if rule_param is None else rule(rule_param)

    def set_lower_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.maxpool, rule, rule_param)
        for layer in range(2):
            for block in getattr(self.base_model, f"layer{layer+1}"):
                self._assign_rule(block.addition_module, rule, rule_param)
                for conv_layer in [l for name, l in block.named_children() if 'conv' in name]:
                    self._assign_rule(conv_layer, rule, rule_param)
                if hasattr(block, "downsample") and block.downsample:
                    self._assign_rule(block.downsample[0], rule, rule_param)

    def set_middle_rule(self, rule, rule_param):
        for layer in range(2, 4):
            for block in getattr(self.base_model, f"layer{layer+1}"):
                self._assign_rule(block.addition_module, rule, rule_param)
                for conv_layer in [l for name, l in block.named_children() if 'conv' in name]:
                    self._assign_rule(conv_layer, rule, rule_param)
                if hasattr(block, "downsample") and block.downsample:
                    self._assign_rule(block.downsample[0], rule, rule_param)

    def set_upper_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.fc, rule, rule_param)

    def set_input_rule(self, rule, rule_param):
        self._assign_rule(self.base_model.conv1, rule, rule_param)


class CIFAR10ContainerVGG11GridPG(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None):
        super(CIFAR10ContainerVGG11GridPG, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert self.output_head_idx is None or (
            self.output_head_idx >= 0 and self.output_head_idx < self.num_heads)

        self.base_model_path = "/BS/srao/work/EB/models/vgg11_lc.tar"
        self.base_model = vgg_model.vgg11_lessconv()
        self.base_model.features = torch.nn.DataParallel(
            self.base_model.features)
        self.base_model.load_state_dict(
            torch.load(self.base_model_path)["state_dict"])
        self.base_model.features = self.base_model.features.module
        # self.avgpool2d = torch.nn.AvgPool2d(kernel_size=15)

        self.conv_act_ids = [[1, 4, 7, 9, 12, 14], [2, 5, 7, 10, 12, 15]]

        self.conv1 = torch.nn.Conv2d(512, 512, kernel_size=2, padding=0)
        self.conv2 = torch.nn.Conv2d(512, 512, kernel_size=1, padding=0)
        self.conv3 = torch.nn.Conv2d(512, 10, kernel_size=1, padding=0)

        self.conv1.weight.data = self.base_model.classifier[1].weight.reshape(
            (512, 512, 2, 2))
        self.conv1.bias.data = self.base_model.classifier[1].bias
        self.conv2.weight.data = self.base_model.classifier[4].weight.reshape(
            (512, 512, 1, 1))
        self.conv2.bias.data = self.base_model.classifier[4].bias
        self.conv3.weight.data = self.base_model.classifier[6].weight.reshape(
            (10, 512, 1, 1))
        self.conv3.bias.data = self.base_model.classifier[6].bias

        kernel_size = 2 * self.scale - 2 + 1
        self.avgpool2d = torch.nn.AvgPool2d(kernel_size=kernel_size)


        self.pooled_dims = (2, 2)
        # self.start_conv_layer_idx = None
        # self.after_maxpool = None
        self.single_spatial = True
        self.single_head = True
        # self.layer_block_size = 8
        # self.layer_start_idx = 192

        self.start_conv_layer_idx = None
        print("Here")

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        print("H2")
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        for fidx in range(start_idx, len(self.base_model.features)):
            y = self.base_model.features[fidx](y)
            if (conv_layer_idx is not None) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                conv_acts = y

        outs = self.base_model.classifier[0](y)
        outs = self.conv1(outs)
        outs = self.base_model.classifier[2](outs)
        outs = self.base_model.classifier[3](outs)
        outs = self.conv2(outs)
        outs = self.base_model.classifier[5](outs)
        classifier_outputs = self.conv3(outs)

        print(classifier_outputs.shape)
        outs = self.avgpool2d(
            classifier_outputs)
        print(outs.shape)

        outs = self.avgpool2d(
            classifier_outputs).squeeze(2).squeeze(2)
        print(outs.shape)

        # classifier_grid_dims = y.shape[2], y.shape[3]
        # kernel_traversal_dims = classifier_grid_dims[0] - self.pooled_dims[0] + \
        #     1, classifier_grid_dims[1] - self.pooled_dims[1] + 1

        # classifier_outputs = torch.zeros(
        #     (x.shape[0], 10, kernel_traversal_dims[0], kernel_traversal_dims[1])).cuda(x.get_device())
        # for ridx in range(kernel_traversal_dims[0]):
        #     for cidx in range(kernel_traversal_dims[1]):
        #         classifier_outputs[:, :, ridx, cidx] = self.base_model.classifier(
        #             y[:, :, ridx:ridx + self.pooled_dims[0], cidx:cidx + self.pooled_dims[1]].reshape(y.shape[0], -1))
        # avgpool2d = torch.nn.AvgPool2d(kernel_size=kernel_traversal_dims[0])
        # outs = avgpool2d(
        #     classifier_outputs).squeeze(2).squeeze(2)

        if conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        # return self.__call__(x, *kwargs)
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)


class CIFAR10ContainerVGG11DiFull(torch.nn.Module):

    def __init__(self, scale=3, activation=None, output_head_idx=None):
        super(CIFAR10ContainerVGG11DiFull, self).__init__()
        self.activation = activation
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert self.output_head_idx is None or (
            self.output_head_idx >= 0 and self.output_head_idx < self.num_heads)

        self.base_model_path = "/BS/srao/work/EB/models/vgg11_lc.tar"
        self.base_model = vgg_model.vgg11_lessconv()
        self.base_model.features = torch.nn.DataParallel(
            self.base_model.features)
        self.base_model.load_state_dict(
            torch.load(self.base_model_path)["state_dict"])
        self.base_model.features = self.base_model.features.module

        self.conv_act_ids = [[1, 4, 7, 9, 12, 14], [2, 5, 7, 10, 12, 15]]

        self.pooled_dims = (2, 2)
        # self.start_conv_layer_idx = None
        # self.after_maxpool = None
        self.single_spatial = False
        self.single_head = False
        # print("Here")
        # self.layer_block_size = 8
        # self.final_layer_idx = 2 + self.layer_block_size * scale * scale
        # self.num_fc_layers = 3

        self.classification_heads = []
        for hidx in range(self.num_heads):
            classification_head = copy.deepcopy(self.base_model.classifier)
            classification_head[1] = torch.nn.Linear(in_features=(
                2048 * self.num_heads), out_features=512, bias=True)
            classification_head[1].weight.data.fill_(0.0)
            classification_head[1].bias.data = self.base_model.classifier[1].bias
            row_idx = hidx // self.scale
            col_idx = hidx % self.scale
            classification_head[1].weight.data.view((512, 512, self.scale * 2, self.scale * 2))[:, :, row_idx * 2:(
                row_idx + 1) * 2, col_idx * 2:(col_idx + 1) * 2] = self.base_model.classifier[1].weight.view((512, 512, 2, 2))
            self.classification_heads.append(classification_head)
        self.classification_heads = torch.nn.ModuleList(
            self.classification_heads)

        self.start_conv_layer_idx = None

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None, loc_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        combined = torch.zeros(
            (x.shape[0], 512, 2 * self.scale, 2 * self.scale)).cuda(x.get_device())
        for lidx in range(self.num_heads):
            y_coord, x_coord, h, w = get_augmentation_range(
                x.shape, self.scale, lidx)
            y = x[:, :, y_coord:y_coord + h, x_coord:x_coord + w]
            for fidx in range(start_idx, len(self.base_model.features)):
                y = self.base_model.features[fidx](y)
                if (conv_layer_idx is not None) and (lidx == loc_idx) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                    conv_acts = y
            yc_comb, xc_comb, h_comb, w_comb = get_augmentation_range(
                combined.shape, self.scale, lidx)
            # print(combined.shape, y.shape)
            combined[:, :, yc_comb:yc_comb + h_comb,
                     xc_comb:xc_comb + w_comb] = y

        outs = []
        for head_idx in range(self.num_heads):
            outs.append(self.classification_heads[head_idx](
                combined.reshape(combined.shape[0], -1)))

        if self.output_head_idx is not None:
            if conv_layer_idx is not None:
                return outs[self.output_head_idx], conv_acts
            else:
                return outs[self.output_head_idx]
        elif conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        # return self.__call__(x, *kwargs)
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)


class CIFAR10ContainerVGG11DiPart(torch.nn.Module):

    def __init__(self, scale=3, output_head_idx=None):
        super(CIFAR10ContainerVGG11DiPart, self).__init__()
        self.scale = scale
        self.output_head_idx = output_head_idx

        self.num_heads = self.scale * self.scale
        assert self.output_head_idx is None or (
            self.output_head_idx >= 0 and self.output_head_idx < self.num_heads)

        self.base_model_path = "/BS/srao/work/EB/models/vgg11_lc.tar"
        self.base_model = vgg_model.vgg11_lessconv()
        self.base_model.features = torch.nn.DataParallel(
            self.base_model.features)
        self.base_model.load_state_dict(
            torch.load(self.base_model_path)["state_dict"])
        self.base_model.features = self.base_model.features.module
        # self.avgpool2d = torch.nn.AvgPool2d(kernel_size=7)

        self.conv_act_ids = [[1, 4, 7, 9, 12, 14], [2, 5, 7, 10, 12, 15]]

        self.pooled_dims = (2, 2)
        # self.start_conv_layer_idx = None
        # self.after_maxpool = None
        self.single_spatial = True
        self.single_head = False

        self.classification_heads = []
        for hidx in range(self.num_heads):
            classification_head = copy.deepcopy(self.base_model.classifier)
            classification_head[1] = torch.nn.Linear(in_features=(
                2048 * self.num_heads), out_features=512, bias=True)
            classification_head[1].weight.data.fill_(0.0)
            classification_head[1].bias.data = self.base_model.classifier[1].bias
            row_idx = hidx // self.scale
            col_idx = hidx % self.scale
            classification_head[1].weight.data.view((512, 512, self.scale * 2, self.scale * 2))[:, :, row_idx * 2:(
                row_idx + 1) * 2, col_idx * 2:(col_idx + 1) * 2] = self.base_model.classifier[1].weight.view((512, 512, 2, 2))
            self.classification_heads.append(classification_head)
        self.classification_heads = torch.nn.ModuleList(
            self.classification_heads)

        self.start_conv_layer_idx = None

    def forward(self, x, conv_layer_idx=None, after_maxpool=False, start_conv_layer_idx=None):
        if self.start_conv_layer_idx is not None:
            start_conv_layer_idx = self.start_conv_layer_idx
        if start_conv_layer_idx is None:
            start_idx = 0
        else:
            start_idx = self.conv_act_ids[after_maxpool][start_conv_layer_idx] + 1
        y = x
        # print(x.shape,conv_layer_idx,after_maxpool,start_conv_layer_idx)
        for fidx in range(start_idx, len(self.base_model.features)):
            y = self.base_model.features[fidx](y)
            # print(fidx, self.base_model.features[fidx], y.shape)
            if (conv_layer_idx is not None) and (fidx == self.conv_act_ids[after_maxpool][conv_layer_idx]):
                conv_acts = y

        # print(y.shape)

        outs = []
        for head_idx in range(self.num_heads):
            outs.append(self.classification_heads[head_idx](
                y.reshape(y.shape[0], -1)))

        if self.output_head_idx is not None:
            if conv_layer_idx is not None:
                return outs[self.output_head_idx], conv_acts
            else:
                return outs[self.output_head_idx]
        elif conv_layer_idx is not None:
            return outs, conv_acts
        else:
            return outs

    def __getitem__(self, idx):
        return list(self.children())[idx]

    def predict(self, x, *kwargs):
        self.eval()
        # return self.__call__(x, *kwargs)
        return torch.nn.functional.softmax(self.__call__(x, *kwargs), dim=1)
