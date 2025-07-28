from .configs import attributor_configs
import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, GuidedBackprop, InputXGradient, Saliency, LayerAttribution
from skimage.transform import resize
from tqdm import tqdm
import os
from .utils import limit_n_images
import sys


def get_attributor(model_setting, attributor_name, config_name):
    """
    Maps the names and configurations to attributors for a specific model and setting

    :param model_setting: Evaluation setting containing the model to evaluate on
    :type model_setting: GridContainerBase
    :param attributor_name: Name of the attribution method
    :type attributor_name: str
    :param config_name: Name of the attribution method configuration
    :type config_name: str
    :return: Attributor object
    :rtype: AttributorBase
    """
    attributor_map = {
        "Grad": Grad,
        "GB": GB,
        "IntGrad": IntGrad,
        "IxG": IxG,
        'Cam': Cam,
        "GradCam": GradCam,
        "GradCamPlusPlus": GradCamPlusPlus,
        "AblationCam": AblationCam,
        "ScoreCam": ScoreCam,
        "LayerCam": LayerCam,
        "Occlusion": Occlusion,
        "RISE": RISE,
    }
    return attributor_map[attributor_name](model_setting, **attributor_configs[attributor_name][config_name])


class AttributorContainer:
    """
    Container to evaluate an attribution method on a model on a specific classification head at a specific layer
    """

    def __init__(self, model_setting, base_exp, base_config):
        """
        Constructor
        :param model_setting: Setting object containing the model to evaluate on
        :type model_setting: GridContainerBase
        :param base_exp: Attribution method to evaluate on
        :type base_exp: str
        :param base_config: Attribution configuration to evaluate on
        :type base_config: str
        """
        self.model_setting = model_setting
        self.base_attributor = get_attributor(
            self.model_setting, base_exp, base_config)

    def attribute(self, img, target, output_head_idx=0, conv_layer_idx=0, **kwargs):
        """
        Runs the attribution method on a batch of images on the model in the prescribed setting on a specific layer, and if applicable, for a specific classification head
        :param img: Images to obtain attributions for
        :type img: torch.Tensor of the shape (B, C, H, W), where B is the batch size, C is the number of channels, H is the image height, and W is the image width
        :param target: Output logits from which to obtain attributions
        :type target: torch.Tensor of the shape (B, 1)
        :param output_head_idx: Classification head to evaluate on, defaults to 0. Required for the DiFull and DiPart settings. Classification heads for a N x N grid are counted zero-indexed row-wise from the top-left corner to the bottom-right corner.
        :type output_head_idx: int, optional
        :param conv_layer_idx: Layer to evaluate on, defaults to 0. A value of zero is equivalent to evaluating at the input.
        :type conv_layer_idx: int, optional
        :return: Attributions
        :rtype: torch.Tensor of the shape (B, 1, 1, H, W)
        """
        if isinstance(conv_layer_idx, str):
            conv_layer_idx = self.model_setting.model.layer_map[conv_layer_idx]
        assert conv_layer_idx >= 0 and conv_layer_idx < len(
            self.model_setting.model.conv_layer_ids)
        model_name = self.model_setting.model.__class__.__name__
        is_vit = 'ViT' in model_name
        is_input_layer = conv_layer_idx == 0

        # Get appropriate input to attribute over
        if is_input_layer and is_vit:
            inputs = img.requires_grad_(True)
        else:
            inputs = self.model_setting.get_intermediate_activations(img, end_layer=conv_layer_idx)


        if self.base_attributor.use_original_img:
            attrs = self.base_attributor.attribute(
                inputs, target=target, original_img=img,
                additional_forward_args=(output_head_idx, conv_layer_idx),
                **self.base_attributor.configs, **kwargs
            )
        else:
            attrs = self.base_attributor.attribute(
                inputs, target=target,
                additional_forward_args=(output_head_idx, conv_layer_idx),
                **self.base_attributor.configs, **kwargs
            )

        # Postprocessing

        if is_vit and is_input_layer:
            # Return full-resolution 224x224 pixel heatmap
            attrs = attrs.abs().sum(dim=1, keepdim=True).float()
        elif is_vit and attrs.ndim == 3:
            # Remove CLS token
            patch_attrs = attrs[:, 1:, :]  # [B, 196, 1]
            # Squeeze out the last dim to get scalar scores per patch
            patch_scores = patch_attrs.norm(p=1, dim=2)  # [1, 196] 
            # Dynamically reshape to spatial map
            B, num_patches = patch_scores.shape
            H = W = int(num_patches ** 0.5)
            assert H * W == num_patches, f"Expected square number of patches, got {num_patches}"

            attrs = patch_scores.view(B, 1, H, W).float()  # [B, 1, 14, 14]
        else:
            attrs = attrs.sum(dim=1, keepdim=True).float()
        return attrs.detach()

    def attribute_selection(self, img, target, output_head_idx=0, conv_layer_idx=0, **kwargs):
        """
        Runs the attribution method on a batch of images for multiple targets on the model in the prescribed setting on a specific layer, and if applicable, for a specific classification head

        :param img: Images to obtain attributions for
        :type img: torch.Tensor of the shape (B, C, H, W), where B is the batch size, C is the number of channels, H is the image height, and W is the image width
        :param target: Output logits from which to obtain attributions
        :type target: torch.Tensor of the shape (B, K), where K is the number of targets per image.
        :param output_head_idx: Classification head to evaluate on, defaults to 0. Required for the DiFull and DiPart settings. Classification heads for a N x N grid are counted zero-indexed row-wise from the top-left corner to the bottom-right corner.
        :type output_head_idx: int, optional
        :param conv_layer_idx: Layer to evaluate on, defaults to 0. A value of zero is equivalent to evaluating at the input.
        :type conv_layer_idx: int, optional
        :return: Attributions
        :rtype: torch.Tensor of the shape (B, K, 1, H, W)
        """
        out = []
        for tgt_idx in range(target.shape[1]):
            out.append(self.attribute(img, target=target[:, tgt_idx].tolist(),
                                      output_head_idx=output_head_idx, conv_layer_idx=conv_layer_idx, **kwargs).detach().cpu())

        out = torch.stack(out, dim=1)
        return out


class AttributorBase:
    """
    Base class for attribution methods
    """

    def __init__(self, model_setting, **configs):
        self.model_setting = model_setting
        self.configs = configs
        self.use_original_img = False


class IntGrad(AttributorBase, IntegratedGradients):
    """
    Integrated Gradient attributions
    Reference: https://arxiv.org/abs/1703.01365
    """

    def __init__(self, model_setting, n_steps=20, internal_batch_size=1):
        AttributorBase.__init__(self,
                                model_setting, n_steps=n_steps, internal_batch_size=internal_batch_size)
        IntegratedGradients.__init__(self, self.model_setting)


class GB(AttributorBase, GuidedBackprop):
    """
    Guided Backprop attributions
    Reference: https://arxiv.org/abs/1412.6806
    """

    def __init__(self, model_setting, apply_abs=True):
        AttributorBase.__init__(self, model_setting)
        GuidedBackprop.__init__(self, self.model_setting)
        self.abs = apply_abs

    def attribute(self, img, target, additional_forward_args=None, original_img=None, **kwargs):
        attrs = super(GB, self).attribute(
            img, target, additional_forward_args=additional_forward_args)
        if self.abs:
            attrs = torch.abs(attrs)
        return attrs


class IxG(AttributorBase, InputXGradient):
    """
    InputxGradient attributions
    Reference: https://arxiv.org/abs/1704.02685
    """

    def __init__(self, model_setting):
        AttributorBase.__init__(self, model_setting)
        InputXGradient.__init__(self, self.model_setting)


class Grad(AttributorBase, Saliency):
    """
    Gradient attributions
    Reference: https://arxiv.org/abs/1312.6034
    """

    def __init__(self, model_setting, apply_abs=True, **configs):
        AttributorBase.__init__(self, model_setting, **configs)
        Saliency.__init__(self, self.model_setting)


class RISE(AttributorBase, nn.Module):
    """
    RISE attributions
    Reference: https://arxiv.org/abs/1806.07421
    """

    def __init__(self, model_setting, mask_path, batch_size=2, n=6000, s=6, p1=0.1):
        nn.Module.__init__(self)
        AttributorBase.__init__(self, model_setting)
        self.path_tmplt = os.path.join(mask_path, "masks{}_{}.npy")
        self.batch_size = batch_size
        self.max_imgs_bs = 1
        self.N = n
        self.s = s
        self.p1 = p1
        self.masks = None

    def generate_masks(self, savepath="masks.npy", input_size=None):
        print("Generating masks for", input_size)
        p1, s = self.p1, self.s
        if not os.path.isdir(os.path.dirname(savepath)):
            os.makedirs(os.path.dirname(savepath))
        cell_size = np.ceil(np.array(input_size) / s)
        up_size = (s + 1) * cell_size
        grid = np.random.rand(self.N, s, s) < p1
        grid = grid.astype("float32")
        masks = np.empty((self.N, *input_size))
        for i in tqdm(range(self.N)):
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            masks[i, :, :] = resize((grid[i]), up_size, order=1, mode="reflect", anti_aliasing=False)[
                x:x + input_size[0], y:y + input_size[1]]

        masks = (masks.reshape)(*(-1, 1), *input_size)
        np.save(savepath, masks)

    def load_masks(self, filepath):
        if not os.path.exists(filepath):
            size = int(os.path.basename(filepath)[
                       len("masks") + len(str(self.N)) + len("_"):-len(".npy")])
            self.generate_masks(savepath=(filepath[:-4]),
                                input_size=(size, size))
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float()
        self.N = self.masks.shape[0]
        return self.masks

    @limit_n_images
    @torch.no_grad()
    def attribute(self, x, target, return_all=False, original_img=None, additional_forward_args=None):
        N = self.N
        is_vit = x.ndim == 3  # [B, T, D]
        if is_vit:
            H = W = int((x.shape[1]-1)**0.5)
            cls_token = x[:, :1, :]     # [B, 1, D]
            patch_tokens = x[:, 1:, :]  # [B, 196, D]
        else:
            _, _, H, W = x.size()
        if self.masks is None or self.masks.shape[(-1)] != H:
            self.masks = self.load_masks(
                self.path_tmplt.format(int(N), int(H)))
        self.masks = self.masks.to(x.device)

        if is_vit:
            B, T, D = x.shape
            if T == H * W + 1:
                cls_token = x[:, :1, :]       # [B, 1, D]
                patch_tokens = x[:, 1:, :]    # [B, H*W, D]
            elif T == H * W:
                cls_token = None
                patch_tokens = x              # [B, H*W, D]
            else:
                raise ValueError(f"ViT input token count {T} doesn't match mask size {H*W} (+CLS optional)")

            # Reshape masks from [N, H, W] → [N, H*W]
            masks = self.masks.view(N, -1)                # [N, T']
            masks = masks.unsqueeze(1).unsqueeze(-1)      # [N, 1, T', 1]
            patch_tokens = patch_tokens.unsqueeze(0)      # [1, B, T', D]
            # Apply masks
            masked_patches = patch_tokens * masks         # [N, B, T', D]

            # Reattach CLS token if needed
            if cls_token is not None:
                cls_token_exp = cls_token.expand(N, -1, -1, -1)  # [N, B, 1, D]
                masked_inputs = torch.cat([cls_token_exp, masked_patches], dim=2)  # [N, B, T, D]
            else:
                masked_inputs = masked_patches
            # Flatten [N, B, T, D] → [N * B, T, D]
            stack = masked_inputs.view(-1, masked_inputs.shape[2], D)  # [N * B, T, D]
        else:
            stack = torch.mul(self.masks, x.data)
        p = []
        for i in range(0, N, self.batch_size):
            if additional_forward_args is not None:
                p.append((self.model_setting.predict)(
                    stack[i:min(i + self.batch_size, N)], *additional_forward_args))
            else:
                p.append(self.model_setting.predict(
                    stack[i:min(i + self.batch_size, N)]))

        p = torch.cat(p)
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        sal = sal.view((CL, 1, H, W))
        sal = sal / N / self.p1
        if return_all:
            return sal
        return sal[int(target[0])][None]

    def attribute_selection(self, x, tgts, additional_forward_args=None):
        return self.attribute(x, tgts, return_all=True, additional_forward_args=additional_forward_args)[tgts]


class Occlusion(AttributorBase):
    def __init__(self, model_setting, stride=32, ks=32, batch_size=8, only_positive=False):
        super().__init__(model_setting)
        self.masks = None
        self.participated = None
        self.n_part = None
        self.max_imgs_bs = 1
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.ks = ks
        self.batch_size = batch_size
        self.only_positive = only_positive

    def make_masks(self, img):
        device = img.device
        stride = self.stride
        ks = self.ks
        total = (img.shape[-1] // stride[-1] + ks // stride[-1] - 1) * \
                (img.shape[-2] // stride[-2] + ks // stride[-2] - 1)
        strided_shape = (np.array(img.shape[-2:]) / np.array(stride)).astype(int) + ks // stride[-1] - 1
        off = 0 if ks % 2 == 1 else ks // stride[-1] - 1
        ks2 = (ks - 1) // 2 if ks % 2 == 1 else ks

        occlusion_masks = []
        for idx in range(total):
            mask = torch.ones(img.shape[-2:], device=device)
            wpos, hpos = np.unravel_index(idx, shape=strided_shape)
            x1 = max(0, (hpos + off) * stride[0] - ks2) if ks % 2 == 1 else max(0, (hpos - off) * stride[0])
            x2 = min(img.shape[-1], x1 + ks)
            y1 = max(0, (wpos + off) * stride[1] - ks2) if ks % 2 == 1 else max(0, (wpos - off) * stride[1])
            y2 = min(img.shape[-2], y1 + ks)
            mask[y1:y2, x1:x2] = 0
            occlusion_masks.append(mask)

        self.masks = torch.stack(occlusion_masks, dim=0)[:, None].to(device)

    @limit_n_images
    @torch.no_grad()
    def attribute(self, img, target, return_all=True, additional_forward_args=None, original_img=None):
        self.model_setting.zero_grad()
        device = img.device
        batch_size = self.batch_size
        is_vit = img.ndim == 3  # ViT: [B, T, D]
        B = img.shape[0]

        org_out = self.model_setting(img, *additional_forward_args) if additional_forward_args else self.model_setting(img)
        if isinstance(target, list):
            target = torch.tensor(target, device=img.device)
        else:
            target = target.to(img.device)

        if is_vit:
            cls_token = img[:, :1, :]
            patch_tokens = img[:, 1:, :]
            T = patch_tokens.shape[1]
            D = patch_tokens.shape[2]
            H = W = int(T ** 0.5)

            weights = torch.zeros(B, T, 1, device=device)
            for i in range(T):
                occluded = patch_tokens.clone()
                occluded[:, i, :] = 0
                modified_input = torch.cat([cls_token, occluded], dim=1)
                out = self.model_setting(modified_input, *additional_forward_args) if additional_forward_args else self.model_setting(modified_input)
                delta = (org_out[:, target].diag() - out[:, target].diag()) / org_out[:, target].diag()
                weights[:, i, 0] = delta

            sal = torch.relu(weights).view(B, 1, H, W)
            return sal if return_all else sal[:, int(target)][..., None]
        else:
            if self.masks is None or self.masks.shape[-1] != img.shape[-1]:
                self.make_masks(img)
                masks = self.masks
                participated = (1 - masks).abs()[:, 0]
                n_part = participated.view(masks.shape[0], -1).sum(1)[:, None, None, None]
                self.participated = participated[:, None]
                self.n_part = n_part

            masked_input = img * self.masks
            pert_out = torch.cat([
                self.model_setting(masked_input[i * batch_size:(i + 1) * batch_size], *additional_forward_args)
                if additional_forward_args else
                self.model_setting(masked_input[i * batch_size:(i + 1) * batch_size])
                for i in range((len(masked_input) + batch_size - 1) // batch_size)
            ], dim=0)

            diff = org_out - pert_out
            if self.only_positive:
                diff = diff.clamp(min=0)

            diff = diff[:, :, None, None]
            diff2 = diff[:, target.flatten(), :, :]
            influence = self.participated * diff2 / self.n_part
            return influence.sum(0, keepdim=True) if return_all else influence.sum(0, keepdim=True)[:, int(target)][..., None]

    def attribute_selection(self, img, targets):
        return self.attribute(img, targets, return_all=True).unsqueeze(2)

    
class Cam(AttributorBase):
    """
    Fine-Grained Class Activation Mapping (CAM) for CNNs.
    Reference: Zhang et al., "Exploring Fine-Grained Class Activation Mapping for Weakly Supervised Object Localization" (BMVC 2020).
    Paper: https://arxiv.org/abs/2007.09823

    Extended to support input-layer CAM by approximating the class activation from the first conv layer weights.
    Supports ResNet18, ResNet50_2, ResNet152, VGG19, and similar CNNs.
    """
    def __init__(self, model_setting):
        super(Cam, self).__init__(model_setting)
        self.use_original_img = False

    def attribute(self, feats, target, additional_forward_args=None, original_img=None, **kwargs):
        model = self.model_setting.model

        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, device=feats.device)
        if target.ndim == 0:
            target = target.unsqueeze(0)

        # Get classifier weights from final linear layer
        if hasattr(model, 'fc') and isinstance(model.fc, torch.nn.Linear):
            class_weights = model.fc.weight  # [num_classes, D]
        elif hasattr(model, 'classifier') and isinstance(model.classifier, torch.nn.Linear):
            class_weights = model.classifier.weight
        elif hasattr(model, 'use_classifier_kernel') and model.use_classifier_kernel:
            # If classifier kernel is enabled, we use the weights from the convolutional classifier layers
            # Access the convolutional layers that replace the original fully connected layers in VGG11
            class_weights = model.classifier_conv1.weight[:, :, 0, 0]  # [1000, 4096]  This is the final classifier convolutional kernel
        else:
            raise ValueError("Unsupported model architecture for CAM: no final linear layer found.")
        # Get first conv layer weights for input CAM projection
        conv1 = None

        # Case 1: For ResNet-style models with .conv1
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'conv1'):
            conv1 = model.base_model.conv1

        # ✅ Case 2: For VGG-style models (like vgg11)
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'features'):
            for layer in model.base_model.features:
                if isinstance(layer, torch.nn.Conv2d):
                    conv1 = layer
                    break

        if conv1 is None:
            raise ValueError("Could not find first convolutional layer for input CAM support.")

        
        cams = []
        for b in range(feats.size(0)):
            x = feats[b:b+1]  # [1, C, H, W] (can be input image or features)
            C = x.shape[1]
            weights = class_weights[target[b]]  # [D] if using conv features
            
            if weights.shape[0] != C:
                # raise ValueError(
                #     f"[CAM ERROR] Feature map has {C} channels, but classifier expects {weights.shape[0]} features. "
                #     "You must match the classifier weights to the conv layer output. "
                #     "Either set `conv_layer_idx='Final'` and use matching weights, or fix the classifier setup."
                # )
                # Project classifier weights through first conv layer to input space
                W_conv = conv1.weight  # [F, 3, k, k]
                F = W_conv.shape[0]

                # Truncate or interpolate weights if mismatched
                W_class_proj = weights[:F] if weights.shape[0] >= F else torch.nn.functional.interpolate(
                    weights.view(1, -1, 1), size=F, mode='linear', align_corners=False).view(-1)

                W_eff = torch.sum(W_class_proj.view(-1, 1, 1, 1) * W_conv, dim=0)  # [3, k, k]
                W_eff = W_eff.unsqueeze(0)  # [1, 3, k, k]

                cam_map = torch.nn.functional.conv2d(x, W_eff, padding=conv1.padding)  # [1, 1, H, W]
            else:
                cam_map = torch.nn.functional.conv2d(x, weights.view(1, C, 1, 1))  # [1, 1, H, W]

            cam_map = torch.nn.functional.relu(cam_map)

            if original_img is not None:
                target_size = original_img.shape[2:]
                cam_map = torch.nn.functional.interpolate(cam_map, size=target_size, mode='bilinear', align_corners=False)

            cams.append(cam_map)

        return torch.cat(cams, dim=0)  # [B, 1, H, W]


class GradCamBase(AttributorBase):
    """
    Base class for Grad-CAM like methods
    """

    def __init__(self, model_setting, pool_grads=True, only_positive_grads=False, use_higher_order_grads=False):
        super(GradCamBase, self).__init__(model_setting)
        self.pool_grads = pool_grads
        self.only_positive_grads = only_positive_grads
        self.use_higher_order_grads = use_higher_order_grads
        self.use_original_img = True

    def attribute(self, img, target, additional_forward_args=None, **kwargs):
        img.requires_grad = True
        outs = self.model_setting(
            img, *additional_forward_args) if additional_forward_args is not None else self.model_setting(img)
        grads = torch.autograd.grad(outs[:, target].diag(), img,
                                    grad_outputs=(torch.ones_like(outs[:, target].diag())))[0]
        
        if grads.ndim == 3: #ViT
            mean_dim = 1
            sum_dim = 2
        elif grads.ndim == 4: # CNN
            mean_dim = (2, 3)
            sum_dim = 1
        with torch.no_grad():
            if self.only_positive_grads:
                grads = torch.nn.functional.relu(grads)
            if self.pool_grads:
                if self.use_higher_order_grads:
                    weights = self._get_higher_order_grads(
                        img, grads, outs[:, target].diag())
                    prods = weights * img
                else:
                    prods = torch.mean(grads, dim=mean_dim,
                                       keepdim=True) * img
            else:
                prods = grads * img
                
            attrs = torch.nn.functional.relu(
                torch.sum(prods, axis=sum_dim, keepdim=True))
        return attrs.detach()

    def _get_higher_order_grads(self, conv_acts, grads, logits, orig_input_shape=(224, 224)):
        if grads.ndim == 3: # ViT
            sum_dim = 2
        elif grads.ndim == 4: # CNN
            sum_dim = (2, 3)
        alpha_num = torch.pow(grads, 2)
        alpha_den = 2 * torch.pow(grads, 2) + torch.pow(grads, 3) * conv_acts.sum(dim=sum_dim, keepdim=True)
        alpha_den = torch.where(alpha_den != 0.0, alpha_den, torch.ones_like(alpha_den))
        alpha = alpha_num / alpha_den

        if grads.ndim==3: # ViT
            prod = torch.exp(logits).reshape(-1, 1, 1 ).expand_as(grads) * grads
        elif grads.ndim == 4: # CNN
            prod = torch.exp(logits).reshape(-1, 1, 1, 1).expand_as(grads) * grads
        weights = (torch.nn.functional.relu(prod)* alpha).sum(dim=sum_dim, keepdim=True)
        return weights


class GradCam(GradCamBase):
    """
    Grad-CAM attributions
    Reference: https://arxiv.org/abs/1610.02391
    """

    def __init__(self, model_setting):
        super(GradCam, self).__init__(model_setting=model_setting, pool_grads=True,
                                      only_positive_grads=False,
                                      use_higher_order_grads=False)


class GradCamPlusPlus(GradCamBase):
    """
    Grad-CAM++ attributions
    Reference: https://arxiv.org/abs/1710.11063
    """

    def __init__(self, model_setting):
        super(GradCamPlusPlus, self).__init__(model_setting=model_setting, pool_grads=True,
                                              only_positive_grads=False,
                                              use_higher_order_grads=True)


class LayerCam(GradCamBase):
    """
    Layer-CAM attributions
    Reference: https://ieeexplore.ieee.org/document/9462463
    """

    def __init__(self, model_setting):
        super(LayerCam, self).__init__(model_setting=model_setting, pool_grads=False,
                                       only_positive_grads=True,
                                       use_higher_order_grads=False)


class AblationCam(AttributorBase):
    """
    Ablation-CAM attributions
    Reference: https://ieeexplore.ieee.org/document/9093360
    """

    def __init__(self, model_setting):
        super(AblationCam, self).__init__(model_setting)

    def attribute(self, img, target, additional_forward_args=None, original_img=None, **kwargs):
        with torch.no_grad():
            is_vit = img.dim() == 3  # [B, Tokens, D]
            B = img.shape[0]

            # Forward pass on original
            outs = self.model_setting(
                img, *additional_forward_args) if additional_forward_args is not None else self.model_setting(img)

            # Init weight tensor
            if is_vit:
                weights = torch.zeros(B, img.shape[1], 1).to(img.device)  # [B, Tokens, 1]
            else:
                weights = torch.zeros(B, img.shape[1], 1, 1).to(img.device)  # [B, C, 1, 1]

            for act_idx in range(img.shape[1]):
                acts_temp = torch.clone(img).detach()
                if is_vit:
                    acts_temp[:, act_idx, :] = 0
                else:
                    acts_temp[:, act_idx, :, :] = 0

                act_outs = self.model_setting(
                    acts_temp, *additional_forward_args) if additional_forward_args is not None else self.model_setting(acts_temp)

                original_preds = outs[:, target].diag()
                act_preds = act_outs[:, target].diag()
                delta = (original_preds - act_preds) / original_preds

                if is_vit:
                    weights[:, act_idx, 0] = delta
                else:
                    weights[:, act_idx, 0, 0] = delta

            # Compute attributions
            prods = weights * img
            if is_vit:
                attrs = torch.relu(prods.sum(dim=2, keepdim=True))  # [B, 1, D]
            else:
                attrs = torch.relu(prods.sum(dim=1, keepdim=True))  # [B, 1, H, W]

            return attrs.detach()


class ScoreCam(AttributorBase):
    """
    Score-CAM attributions
    Reference: https://arxiv.org/abs/1910.01279
    """

    def __init__(self, model_setting):
        super(ScoreCam, self).__init__(model_setting)
        self.use_original_img = True

    def attribute(self, img, target, original_img, additional_forward_args=None, **kwargs):
        with torch.no_grad():
            weights = torch.zeros((
                img.shape[0], img.shape[1], 1, 1))
            weights = weights.to(img.device)
            for act_idx in range(img.shape[1]):
                upsampled_acts = LayerAttribution.interpolate(img[:, act_idx, :, :].unsqueeze(1),
                                                              (original_img.shape[2], original_img.shape[3]), interpolate_mode="bilinear").detach()
                min_acts = torch.min(upsampled_acts.view(-1, original_img.shape[2] * original_img.shape[3]),
                                     dim=1)[0].reshape(-1, 1, 1, 1)
                max_acts = torch.max(upsampled_acts.view(-1, original_img.shape[2] * original_img.shape[3]),
                                     dim=1)[0].reshape(-1, 1, 1, 1)
                normalized_acts = (upsampled_acts - min_acts) / \
                    (max_acts - min_acts)
                mod_imgs = original_img * normalized_acts
                mod_outs = self.model_setting(mod_imgs).detach()
                weights[:, act_idx, 0, 0] = torch.nn.functional.softmax(mod_outs, dim=1)[:,
                                                                                         target].diag()

            prods = weights * img
            attrs = torch.nn.functional.relu(
                torch.sum(prods, axis=1, keepdim=True))
        return attrs.detach()
