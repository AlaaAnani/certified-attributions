import argparse
import os
import pickle
import random
import sys
from pprint import pprint

import numpy as np
import torch
import torchvision
from torch.utils.data import Subset
from torchvision import transforms
from tqdm import tqdm
import yaml

# Project-specific modules
from attribution_certification.attribution import attributors
from attribution_certification.attribution.lrp_utils import *
from attribution_certification.models import models, settings
from attribution_certification.certifier.utils import process_attributions
from utils import DotDict
# Set a random seed for reproducibility
SEED = 73
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Certifier:
    def __init__(self):
        """
        Initialize Certifier: parse config, load dataset, initialize model, and run selected mode.
        """
        self.parse_args()
        self.test_loader = self.init_dataset()
        self.model = self.init_model()
        if self.config.mode == 'cache_samples':
            self.sample_with_noise_chunks(self.model, self.test_loader)
        elif self.config.mode == 'certify':
            self.smoothed_inference(self.model, self.test_loader)
            
    def parse_args(self):
        """
        Parse command-line arguments and merge them with YAML config.
        """

        parser = argparse.ArgumentParser(
            description="Runs an attribution method using a specified configuration at a specified layer.")
        parser.add_argument(
            '--mode',
            type=str,
            choices=['cache_samples', 'certify'],
            required=True,
            help='Mode of operation: "cache_samples" to generate and save noisy samples, "certify" to compute certified attributions'
        )
        parser.add_argument('--exp', default='LRP')
        parser.add_argument('--layer', default='Input', choices=['Input', 'Middle', 'Final'])
        parser.add_argument('--num_images', default=100, type=int)
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--sigma', default=0.15, type=float)
        parser.add_argument('--dataset_path', default="data/imagenet", type=str)
        parser.add_argument(
            '--Ks',
            nargs='+',
            type=int,
            default=[50, 25, 5],
            help='Sparsify the continuous attributions by setting top K%% of attribution pixels to 1, else 0.'
        )
        parser.add_argument('--n', default=100, type=int)
        parser.add_argument('--model', default="resnet18", type=str)
        parser.add_argument('--cuda', action='store_true', default=False)
        parser.add_argument('--seed', type=int, default=73)
        parser.add_argument('--save_dir', type=str, default='outputs')

        # LRP 
        parser.add_argument('--composite', type=str, default='EpsilonPlusBox',
                            choices=["None", "EpsilonGammaBox", "EpsilonPlus", "EpsilonAlpha2Beta1", "EpsilonPlusFlat", "EpsilonAlpha2Beta1Flat", "ExcitationBackprop", "EpsilonPlusBox", "Epsilon025PlusBox"])
        parser.add_argument('--only_corners', action='store_true', default=False)
        parser.add_argument('--head_idx', type=int, default=0)
        parser.add_argument('--ignore_non_corners', action='store_true', default=False)
        parser.add_argument('--use_box_stabilizer', action='store_true', default=False)
        parser.add_argument('--num_conv_epsilon', type=int, default=0)
        parser.add_argument('--gamma', type=float, default=None)

        args = parser.parse_args() # the args preceed the .yaml config in priority
        with open('configs/imagenet/imagenet.yaml') as f:
            config = DotDict(yaml.safe_load(f))   
        for arg_name, arg_value in vars(args).items():
            if arg_name in config:
                config[arg_name] = arg_value
           
        if args.exp == 'LRP':
            config['batch_size'] = 1
            
        self.config = config
        pprint(self.config)

    def init_dataset(self):
        """
        Load and prepare the test dataset and indices, return DataLoader.
        """
        test_data_dict = torch.load(os.path.join(self.config.dataset_path, 'test.pt'))

        test_data = torch.utils.data.TensorDataset(
            test_data_dict["data"], test_data_dict["labels"])
        
        indices_path = self.config.grids_indices_path if 'grid' in self.config.dataset_path else self.config.images_indices_path
        if not os.path.exists(indices_path):
            indices = random.sample(range(0, len(test_data_dict['data']+1)), len(test_data_dict['data']))
            pickle.dump(indices, open(indices_path, 'wb'))
        else:
            indices = pickle.load(open(indices_path, 'rb'))
        if 'grid' in self.config.dataset_path or 'certified' in self.config.dataset_path: indices = list(range(self.config.num_images))
        
        subset = Subset(test_data, indices[:self.config.num_images])
        
        test_loader = torch.utils.data.DataLoader(
           subset, batch_size=1, shuffle=False,)
        
        scale = test_data_dict["scale"]
        grid_size = scale * scale
        
        self.scale = scale
        self.grid_size = grid_size
        return test_loader
        
    def init_model(self):
        """
        Initialize the model and attribution method based on the config.
        """
        if not settings.eval_only_corners(self.config.setting):
            self.head_list = [0]
        else:
            self.head_list = [0, self.grid_size - 1]
        if self.config.exp != 'LRP':
            model = models.get_model(self.config.model)()
            model_setting = settings.get_setting(self.config.setting)(model=model, scale=self.scale)
            if self.config.cuda:
                model_setting.cuda()
            model_setting.eval()
            self.model_setting = model_setting

            if self.config.exp == 'Occlusion' and self.config.layer == 'Final':
                key = 'Occ5_2'
            elif self.config.exp == 'Occlusion' and self.config.layer == 'Input':
                key = 'Occ16_8'
            else:
                key = self.config.config
            attributor = attributors.AttributorContainer(
                model_setting=model_setting, base_exp=self.config.exp, base_config=key)
        else:
            self.config.batch_size == 1
            model, attributor = lrp_model_setup(self.config, self.scale)
            layer_map = {'vgg11': {"Input": -1, "Middle": 4, "Final": 7},
                        'resnet18': {"Input":-1, "Middle":2, "Final":4},
                        'resnet50_2': {'Input': -1, 'Middle':2, 'Final':4},
                        'resnet101': {'Input': -1, 'Middle':2, 'Final':4},
                        'resnet152': {'Input': -1, 'Middle':2, 'Final':4},
                        'vgg19': {"Input": -1, "Middle": 8, "Final": 15},
                        "vit_b_16": { "Input": -1, "Middle": 11,  "Final": 11}} # last encoder block
            self.layer_idx = layer_map[self.config.model][self.config.layer]
            self.lrp_model = model
            self.lrp_attributor = attributor
            return None
        return attributor
    
    def smoothed_inference_noisy_batch(self, attributor, test_X, test_y, batch_size, sigma):
        noisy_X = []
        if self.config.normalized:
            imagenet_inv_normalize_transform = torchvision.transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            )
            test_X = imagenet_inv_normalize_transform(test_X)
        imagenet_normalized_transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        if 'grid' in self.config.dataset_path and 'vit' in self.config.model:
            test_X = transforms.functional.resize(test_X, [224, 224])
        for i in range(batch_size):
            noisy_img = test_X.cpu() + sigma * torch.randn_like(test_X.cpu()) # (B, C, H, W)
            noisy_img = imagenet_normalized_transform(noisy_img)
            noisy_X.append(noisy_img)
        noisy_X = torch.cat(noisy_X, dim=0)
        test_y = test_y.repeat(noisy_X.shape[0], 1)
        if self.config.cuda:
            noisy_X = noisy_X.cuda().requires_grad_(True)
            test_y = test_y.cuda()
        batch_attributions = []
        for head_pos_idx in self.head_list:
            if self.config.exp != 'LRP':
                if self.model_setting.single_head:
                    batch_attributions.append(
                        attributor.attribute_selection(img=noisy_X, target=test_y, conv_layer_idx=self.config.layer).sum(dim=2, keepdim=True))
                else:
                    batch_attributions.append(attributor.attribute_selection(img=noisy_X, target=test_y[:, head_pos_idx].reshape(
                        -1, 1), output_head_idx=head_pos_idx, conv_layer_idx=self.config.layer).sum(dim=2, keepdim=True))
            else:
                batch_attributions.append(self.get_relevance(test_X=noisy_X, test_y=test_y, 
                                                             model=self.lrp_model, scale=self.scale, head_pos_idx=0, 
                                                             attributor=self.lrp_attributor, args=self.config).unsqueeze(0))

        attributions = torch.cat(batch_attributions, dim=0).detach().cpu() 
        return attributions # (B, K, 1, H, W)
    
    def sample(self, attributor, test_X, test_y, n, sigma, show_progress_bar=False):
        """
        Sample noisy attributions `n` times for a single image and return the aggregated tensor.

        Parameters:
            - test_X: Tensor of shape (1, C, H, W) — input image
            - test_y: Tensor of shape (1, num_classes) or (1,) — label or target vector
            - n: Number of noisy samples to draw
            - sigma: Standard deviation of Gaussian noise
            - show_progress_bar: Whether to show tqdm progress

        Returns:
            - sampled_attributions: Tensor of shape (n, B, 1, H, W), where:
                - n: number of noisy samples
                - B: number of output heads (always 1 here)
                - 1: single attribution channel
                - H, W: spatial dimensions (scaled if needed)
        """
        out = []
        BS = self.config.batch_size
        remaining = n
        with tqdm(total=n, disable= not show_progress_bar, desc="sampling") as pbar:
            while remaining > 0:
                cnt = min(remaining, BS)
                attributions = self.smoothed_inference_noisy_batch(attributor, test_X, test_y, cnt, sigma) # (BS, nxn, channels, W, H)
                out.append(attributions)
                remaining -=cnt
                pbar.update(cnt)
        sampled_attributions = torch.cat(out, dim=0)
        return sampled_attributions
    

    def smoothed_inference(self, attributor, test_loader):
        """
        Run full certified attribution pipeline on the dataset and save results.

        This function performs the following:
        - Computes a raw attribution (single clean input)
            Shape: (1, 1, 1, H, W)
        - Samples noisy inputs and computes attributions on them
            Shape: (1, n, 1, H, W)
        - Applies smoothing, sparsification, and certification over each image
        - Generates an overlayed certified map where each pixel has a discrete importance degree label
            Shape: (1, H*scale, W*scale)
        - Aggregates and saves results:
            - raw_attributions: (num_images, 1, 1, H, W)
            - certified_attributions: (num_images, H*scale, W*scale)
        """
        idx = 0
        raw_attributions = []
        certified_attributions = []

        for (test_X, test_y) in tqdm(test_loader, desc="certifying dataset"):
            # Raw attribution
            batch_attributions = self.sample(attributor, test_X, test_y, n=1, sigma=0).unsqueeze(1)  # (BS, nxn, channels=1, W, H)
            batch_attributions = batch_attributions[:, :, :1, :, :, :]
            
            # Sample attributions on noisy input
            batch_noisy_samples = self.sample(attributor, test_X, test_y, 
                                        self.config.n, 
                                        self.config.sigma).unsqueeze(0)
            batch_noisy_samples = batch_noisy_samples[:, :, :1, :, :, :]
            
            img_dims, scale, n, tau = test_X.shape[2:], self.scale, self.config.n, self.config.tau
            Ks = sorted(self.config.Ks, reverse=True) # sort the Ks in ascneding importance (starting from higher K)
            batch_attributions, scale_factor = process_attributions(batch_attributions, 
                                steps_dict=
                                { 
                                'interpolate': {'img_dims': tuple(img_dims), 'scale': scale},
                                'normalize':  {'clip_percentile': 99.5, 'scale_factor': None},
                                'positive': {},
                                })
            noisy_samples = process_attributions(batch_noisy_samples, 
                                steps_dict=
                                {
                                'interpolate': {'img_dims': tuple(img_dims), 'scale': scale},
                                'normalize':   {'clip_percentile': 99.5, 'scale_factor': scale_factor},
                                'positive': {},
                                },
                                return_scale=False)
            overlayed = torch.zeros((1, self.scale*img_dims[0], self.scale*img_dims[1]))

            for deg_idx, K in enumerate(Ks):
                sparsified = process_attributions(noisy_samples, 
                        steps_dict={'sparsify': {'spars_method': 'all_percentile', 'spars_param': [K], 'head_idx': 0}}, 
                        return_scale=False)
                certified = process_attributions(sparsified, 
                        steps_dict={'certify': {'n':n, 'n0':10, 'tau':tau, 'alpha': 0.001, 'head_idx':0}}, 
                        return_scale=False)
                overlayed[certified == 1] = deg_idx + 1

            raw_attributions.append(batch_attributions)
            certified_attributions.append(overlayed)
            idx +=1
            if idx == self.config.num_images: break
        raw_attributions = torch.cat(raw_attributions, dim=0).detach().cpu()
        certified_attributions = torch.cat(certified_attributions, dim=0).detach().cpu()    
        
        # save certified attributions
        dir_type = 'grid' if 'grid' in self.config.dataset_path else 'images'
        ext = f'{dir_type}_{self.config.sigma}_{self.config.layer}_{self.config.exp}_{self.config.model}_{self.config.Ks}_{self.config.n}_{self.config.tau}'
        save_path = os.path.join(self.config.save_dir, 'certified', f'{ext}_certified.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print('Saving certified attribution at', save_path)
        torch.save(certified_attributions, save_path)
        
    def sample_with_noise(self, attributor, test_loader):
        """
        Sample raw and noisy attributions and save them as tensors.

        - For each image:
            - Raw attribution: Shape (1, 1, 1, H, W)
            - Noisy attribution samples: Shape (1, n, 1, H, W)
        - Saves entire dataset as:
            - raw_attributions: (num_images, 1, 1, H, W)
            - raw_samples: (num_images, 1, n, 1, H, W)
        """
        dir_type = 'grid' if 'grid' in self.config.dataset_path else 'images'
        self.config.save_dir = f'{self.config.save_dir}/noisy_samples/{dir_type}/sigma_{self.config.sigma}/{self.config.layer}'
        os.makedirs(self.config.save_dir, exist_ok=True)
        print(self.config.save_dir, self.config.exp)
        for attr_type in ['raw', 'noisy_samples']:
            full_save_dir = os.path.join(self.config.save_dir, self.config.exp,
                                        f'{attr_type}_{self.config.model}.pt')
            if os.path.exists(full_save_dir):
                x = torch.load(full_save_dir)
                num_samples = x.shape[0]
                print(f'Number of already cached {attr_type} =', num_samples)
                if attr_type == 'raw':
                    num_cached_images = num_samples
                    raw_attributions = [x[i].unsqueeze(0) for i in range(x.size(0))]
                if attr_type == 'noisy_samples':
                    raw_samples = [x[i].unsqueeze(0) for i in range(x.size(0))]
            
            else:
                num_cached_images = 0
                if attr_type == 'raw':
                    raw_attributions = []
                if attr_type == 'noisy_samples':
                    raw_samples = []
        idx = 0
        if self.config.num_images <= num_cached_images:
            print(f'You already have {self.config.num_images}/{num_cached_images}', ' cahced images.')
            return
        self.config.num_images = self.config.num_images - num_cached_images
        print('Will sample', self.config.num_images, ' more images.')
        
        for (test_X, test_y) in tqdm(test_loader, desc="dataset loader"):
            # text_X (BS, c, scale*im_width, scale*im_height)
            if idx < num_cached_images:
                idx +=1
                continue
            # raw attribution
            batch_attributions = self.sample(attributor, test_X, test_y, n=1, sigma=0) # (BS, nxn, channels=1, W, H)
            # sample attributions on noisy input
            batch_samples = self.sample(attributor, test_X, test_y, 
                                        self.config.n, 
                                        self.config.sigma, 
                                        ).unsqueeze(0)
            raw_samples.append(batch_samples)
            raw_attributions.append(batch_attributions)
            idx +=1
            if idx == self.config.num_images + num_cached_images: break


        raw_attributions = torch.cat(raw_attributions, dim=0).detach().cpu()
        raw_samples = torch.cat(raw_samples, dim=0).detach().cpu()
        
        
        for x, y in [('raw',   raw_attributions), 
                     ('noisy_samples',    raw_samples)]:
            full_save_dir = os.path.join(self.config.save_dir, self.config.exp,
                                        f'{x}_{self.config.model}.pt')
            os.makedirs(os.path.dirname(full_save_dir), exist_ok=True)
            print("Saving attributions at", full_save_dir)
            torch.save(y, full_save_dir)

    def sample_with_noise_chunks(self, attributor, test_loader):
        """
        Sample raw and noisy attributions in chunks for efficiency.

        - Same structure as `sample_with_noise`, but saves in batches.
        - File tensors:
            - raw chunk: (chunk_size, 1, 1, H, W)
            - noisy_samples chunk: (chunk_size, 1, n, 1, H, W)
        - Useful when dataset is large and memory is constrained (This is what we use in all our experiments).
        """
        dir_type = 'grid' if 'grid' in self.config.dataset_path else 'images'
        subd = 'noisy_samples_chunks'
        if 'certified' in self.config.dataset_path:
            part = '_'.join(self.config.dataset_path.split('_')[-3:])
            subd = part +'_'+ subd
        self.config.save_dir = f'{self.config.save_dir}/{subd}/{dir_type}/sigma_{self.config.sigma}/{self.config.layer}'
        os.makedirs(self.config.save_dir, exist_ok=True)
        print(self.config.save_dir, self.config.exp)
        
        # Initialize counters and lists
        num_cached_images = 0
        idx = 0
        chunk_size = 5  # Adjust chunk size as needed
        raw_attributions_chunk = []
        raw_samples_chunk = []
        
        for attr_type in ['raw', 'noisy_samples']:
            base_save_dir = os.path.join(self.config.save_dir, self.config.exp)
            os.makedirs(base_save_dir, exist_ok=True)
            
            # Check already cached tensors
            existing_files = sorted(
                [f for f in os.listdir(base_save_dir) if f.endswith(f"_{attr_type}_{self.config.model}.pt")]
            )
            
            if existing_files:
                print(f"Found cached {attr_type} tensors:")
                for file_name in existing_files:
                    full_save_dir = os.path.join(base_save_dir, file_name)
                    tensor = torch.load(full_save_dir)
                    print(f"Loaded cached tensor from {file_name}, shape: {tensor.shape}")
                    if attr_type == 'noisy_samples':
                        num_cached_images += tensor.size(0)
            else:
                print(f"No cached {attr_type} tensors found.")

        # Adjust the number of images to sample
        if self.config.num_images <= num_cached_images:
            print(f"You already have {self.config.num_images}/{num_cached_images} cached images.")
            return
        
        self.config.num_images = self.config.num_images - num_cached_images
        print('Will sample', self.config.num_images, 'more images.')
        
        # Process and save tensors incrementally
        for (test_X, test_y) in tqdm(test_loader, desc="dataset loader"):
            if idx < num_cached_images:
                idx += 1
                continue
            
            # Raw attribution
            batch_attributions = self.sample(attributor, test_X, test_y, n=1, sigma=0).unsqueeze(1)  # (BS, nxn, channels=1, W, H)
            batch_attributions = batch_attributions[:, :, :1, :, :, :]
            
            # Sample attributions on noisy input
            batch_samples = self.sample(attributor, test_X, test_y, 
                                        self.config.n, 
                                        self.config.sigma).unsqueeze(0)
            batch_samples = batch_samples[:, :, :1, :, :, :]
            
            # Append to chunks
            raw_attributions_chunk.append(batch_attributions.detach().cpu())
            raw_samples_chunk.append(batch_samples.detach().cpu())
            
            idx += 1
            
            # Save when chunk size is reached
            if len(raw_attributions_chunk) == chunk_size:
                # Save raw attributions
                file_name = f"{str((idx // chunk_size)).zfill(3)}_raw_{self.config.model}.pt"
                full_save_dir = os.path.join(self.config.save_dir, self.config.exp, file_name)
                torch.save(torch.cat(raw_attributions_chunk, dim=0), full_save_dir)
                print(f"Saved raw attributions chunk at {full_save_dir}")
                raw_attributions_chunk = []

                # Save noisy samples
                file_name = f"{str((idx // chunk_size)).zfill(3)}_noisy_samples_{self.config.model}.pt"
                full_save_dir = os.path.join(self.config.save_dir, self.config.exp, file_name)
                torch.save(torch.cat(raw_samples_chunk, dim=0), full_save_dir)
                print(f"Saved noisy samples chunk at {full_save_dir}")
                raw_samples_chunk = []

            # Stop if required number of images is reached
            if idx == self.config.num_images + num_cached_images:
                break
        
        # Save remaining tensors if any
        if raw_attributions_chunk:
            file_name = f"{str((idx // chunk_size)).zfill(3)}_raw_{self.config.model}.pt"
            full_save_dir = os.path.join(self.config.save_dir, self.config.exp, file_name)
            torch.save(torch.cat(raw_attributions_chunk, dim=0), full_save_dir)
            print(f"Saved final raw attributions chunk at {full_save_dir}")

        if raw_samples_chunk:
            file_name = f"{str((idx // chunk_size)).zfill(3)}_noisy_samples_{self.config.model}.pt"
            full_save_dir = os.path.join(self.config.save_dir, self.config.exp, file_name)
            torch.save(torch.cat(raw_samples_chunk, dim=0), full_save_dir)
            print(f"Saved final noisy samples chunk at {full_save_dir}")
            
    def get_relevance(self, test_X, test_y, model, scale, head_pos_idx, attributor, args):  
        """
        Compute LRP relevance for a single input using the specified model and head index.

        Input:
            - test_X: (B, C, H, W)
            - test_y: (B, n*n) or (B,)

        Returns:
            - relevance: (B, 1, H, W)
        """   
        if args.cuda:
            test_X = test_X.cuda().requires_grad_(True)
            test_y = test_y.cuda()
        inp = get_intermediate_activations(model, test_X, self.layer_idx, scale)
        if self.layer_idx != -1:
            model.start_conv_layer_idx = self.layer_idx
        out = model(inp)
        out = out[:, test_y[:, head_pos_idx]].item()
        with attributor:
            target = torch.eye(1000, device=test_y.device)[[test_y[:, head_pos_idx]]] * out
            if args.cuda:
                target = target.cuda()
            _, relevance = attributor(inp, target)
        nm_relevance = relevance.sum(dim=1, keepdim=True).detach().cpu()
        model.start_conv_layer_idx = None
        return nm_relevance
    
if __name__ == "__main__":
    Certifier()
