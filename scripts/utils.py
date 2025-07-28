import torch
from attribution_certification.evaluation import utils
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import visualization as viz
import ast
from twosample import binom_test
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import sys
import os
from torch.utils.data import Dataset, DataLoader
import glob

color_map = {
    'Grad': '#25517c',        # Dark indigo purple
    'GB': '#51242c',          # Dark blue
    'IxG': '#2a9d8f',         # Deep green
    'IntGrad': '#91be6d',     # Forest green
    'LRP': '#B40C1C',

    'RISE': '#289bd0',
    'Occlusion': '#BC930D',
    
    'Cam': '#F5DF31',
    'GradCam': '#f8961e',     # Dark orange
    'GradCam++': '#9D1B9D',   # Rich burnt orange
    'AblationCam': '#6F6F3A', # Saddle brown
    'LayerCam': '#552061',    # Dark brown
    #'Abstain ($\oslash$)': 'lightgray',
}

def to_numpy_recursive(x):
    if isinstance(x, list):
        return np.array([to_numpy_recursive(i) for i in x])
    else:
        return np.array(x)
def get_plot_dict(data_dict, xai_methods,
                  models=['resnet18'],
                  exclude_dict={},
                  sigmas=[ 0.15,  0.25, 0.33], 
                  ns=[100], 
                  Ks=[50], 
                  taus=[0.75],
                  layers=['Input', 'Final'],
                  data_type='images',
                  ):
    better_names = {'GradCamPlusPlus': 'GradCam++'}
    plot_dict = {}
    for model in models:
        if model not in plot_dict:
            plot_dict[model] = {}
        for layer in layers:
            if layer not in plot_dict[model]:
                plot_dict[model][layer] = {}
            for xai_method in xai_methods:
                if xai_method in better_names:
                    xai_method_ = better_names[xai_method]
                else: xai_method_ = xai_method
                if model in exclude_dict and xai_method in exclude_dict[model]: continue
                plot_dict[model][layer][xai_method_] = {'certified_1': [], 'certified_0': [], 'abstain': [], 'certified_pg': [], 'uncertified_pg': []}
                for sigma in sigmas:
                    for K in Ks:
                        for n in ns:
                            for tau in taus:
                                d = data_dict[model][data_type][f'sigma_{sigma}'][layer][xai_method][K][n][tau]
                                if data_type == 'images':
                                    num_pixels = d['num_pixels']
                                    certified_1_pct = (d['certified_1'] / num_pixels) *100
                                    certified_0_pct = (d['certified_0'] / num_pixels) *100
                                    abstain_pct = (d['abstain'] / num_pixels) * 100

                                    plot_dict[model][layer][xai_method_]['certified_0'].append(certified_0_pct)
                                    plot_dict[model][layer][xai_method_]['certified_1'].append(certified_1_pct)
                                    plot_dict[model][layer][xai_method_]['abstain'].append(abstain_pct)
                                if data_type == 'grid':
                                    c_pg = d['certified_localization_scores'].cpu().numpy()
                                    uc_pg = d['raw_localization_scores'].cpu().numpy()
                                    plot_dict[model][layer][xai_method_]['certified_pg'].append(c_pg)
                                    plot_dict[model][layer][xai_method_]['uncertified_pg'].append(uc_pg)
    return plot_dict

def fast_load(tensor_folder, model='vgg11', num_images=100, im_indices=[], num_workers=10):
    batch_size = 5  # fixed; adjust if needed

    # Compute which global batch files we need
    if len(im_indices) > 0:
        global_batch_ids = sorted(set(i // batch_size for i in im_indices))
        dataset = ChunkedTensorDataset(tensor_folder, model, num_images, im_indices=im_indices)
    else:
        dataset = ChunkedTensorDataset(tensor_folder, model, num_images)

    # Map global_batch_id -> list of global image indices needed from that batch
    index_lookup = {}
    for i in im_indices:
        batch_id = i // batch_size
        idx_within = i % batch_size
        index_lookup.setdefault(batch_id, []).append(idx_within)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    noisy_out, raw_out = [], []

    with SuppressOutput():
        for batch_i, (raw, noisy) in enumerate(data_loader):
            # raw/noisy shape: [1, 5, C, H, W]
            raw = raw[0]      # shape: [5, C, H, W]
            noisy = noisy[0]  # shape: [5, C, H, W]

            # What is the corresponding global batch ID for this one?
            if len(im_indices) > 0:
                global_batch_id = list([i // batch_size for i in im_indices])[batch_i]
                if global_batch_id in index_lookup:
                    for local_idx in index_lookup[global_batch_id]:
                        raw_out.append(raw[local_idx])
                        noisy_out.append(noisy[local_idx])
            else:
                raw_out.append(raw)
                noisy_out.append(noisy)

    if len(im_indices) > 0:
        return torch.stack(noisy_out), torch.stack(raw_out)
    else:
        return torch.cat(noisy_out, dim=0), torch.cat(raw_out, dim=0)




class ChunkedTensorDataset(Dataset):
    def __init__(self, folder_path, model='vgg11', num_images=100, im_indices=[]):
        self.folder_path = folder_path
        self.noisy_files = np.array(sorted(glob.glob(os.path.join(folder_path, f"*_noisy_samples_{model}.pt"))))
        self.raw_files = np.array(sorted(glob.glob(os.path.join(folder_path, f"*_raw_{model}.pt"))))
        if len(im_indices) > 0:
            batch_indices = np.array(list([int(i // 5) for i in im_indices]))
            self.noisy_files = self.noisy_files[batch_indices]
            self.raw_files = self.raw_files[batch_indices]
        if not isinstance(self.noisy_files, list):
            self.noisy_files = list(self.noisy_files)
            self.raw_files = list(self.raw_files)
    def __len__(self):
        return len(self.noisy_files)
    
    def __getitem__(self, idx):
        raw_path, noisy_path = self.raw_files[idx], self.noisy_files[idx]
        with open(raw_path, 'rb') as raw_file:
            raw = torch.load(raw_file)
        with open(noisy_path, 'rb') as noisy_file:
            noisy = torch.load(noisy_file)
        if len(raw.shape) == 5:
            raw = raw.unsqueeze(1)
        return raw.clone(), noisy.clone() 

class SuppressOutput:
    def __enter__(self):
        self._stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')  # Redirect stderr to null

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stderr.close()
        sys.stderr = self._stderr  # Restore stderr
    
class MulticolorPatch:
    def __init__(self, cmap, ncolors=100):
        self.ncolors = ncolors
        
        if isinstance(cmap, str):
            self.cmap = plt.get_cmap(cmap)
        else:    
            self.cmap = cmap

# Define a handler for both `MulticolorPatch` and single color patches
class MulticolorPatchHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        if isinstance(orig_handle, MulticolorPatch):
            # Multicolor gradient entry
            n = orig_handle.ncolors
            width, height = handlebox.width, handlebox.height
            patches = []
            for i, color in enumerate(orig_handle.cmap(i / n) for i in range(n)):
                patches.append(
                    plt.Rectangle(
                        (width / n * i - handlebox.xdescent, -handlebox.ydescent),
                        width / n,
                        height,
                        facecolor=color,
                        edgecolor='none'
                    )
                )
            patch = PatchCollection(patches, match_original=True)
            handlebox.add_artist(patch)
            return patch
        else:
            # Single color entry
            patch = mpatches.Rectangle(
                (0, 0), handlebox.width, handlebox.height,
                facecolor=orig_handle, edgecolor='black', linewidth=0.5
            )
            handlebox.add_artist(patch)
            return patch
        
def fast_certify(samples_flattened, n0, n, tau=0.75, alpha=0.001, stats=None, non_ignore_idx=None):
    assert samples_flattened.shape[0] == n
    if not isinstance(samples_flattened, torch.Tensor):
        samples_flattened = torch.from_numpy(samples_flattened)
    if non_ignore_idx is None:
        non_ignore_idx = [True]*samples_flattened.shape[1]
    modes, _ = torch.mode(samples_flattened[:n0], 0)
    counts_at_modes = (samples_flattened[n0:] == modes.unsqueeze(0)).sum(0)
    pvals_ = binom_test(np.array(counts_at_modes), np.array([n-n0]*len(samples_flattened[0])), np.array([tau]*len(samples_flattened[0])), alt='greater')
    abstain = pvals_ > alpha/len(samples_flattened[0])
    modes = modes.cpu().numpy()    
    modes[np.array(abstain)] = -1
    if stats is not None:
        d = {}
        d['fluctuations'] = samples_flattened[:, abstain & non_ignore_idx].cpu().numpy()
        return modes, d
    return torch.from_numpy(modes)

def process_attributions(attributions, steps_dict={}, return_scale=True):
    # noisy_samples (num_images, num_samples, n, n, w, h)
    # raw (num_images, n, n, w, h)
    
    if len(steps_dict.keys()) == 0:
        return attributions
    assert all([step in ['smooth', 'interpolate', 'positive', 'normalize', 'sparsify', 'certify'] for step in steps_dict.keys()])
    
    for step, d in steps_dict.items():
        if step == 'smooth':
            attributions = attributions.mean(1)
        elif step == 'interpolate':
            num_images, num_samples, grid_size, c, w, h = attributions.shape
            attributions = attributions.reshape(num_images*num_samples, grid_size, c, w, h)
            grid_img_dims = tuple([d['scale'] * dim for dim in d['img_dims']])
            attributions = utils.interpolate_attributions(
                    attributions, grid_img_dims)
            attributions = attributions.reshape(num_images, num_samples, grid_size, c, d['img_dims'][0]*d['scale'], d['img_dims'][1]*d['scale'])
        elif step == 'positive':
            attributions[attributions < 0] = - attributions[attributions < 0]
            attributions = utils.get_positive_attributions(
                attributions)
        elif step == 'normalize':
            if d['scale_factor'] == None:
                scale_factor = np.percentile(attributions, d['clip_percentile'])
            else: scale_factor = d['scale_factor']
            if scale_factor != 0:
                norm_attributions = viz._normalize_scale(attributions, scale_factor)
            else:
                norm_attributions = attributions
            attributions = norm_attributions
        elif step == 'sparsify':
            assert d['spars_method'] in ['all_percentile', 'positive_percentile', 'threshold']
            spars_method = d['spars_method']
            spars_param = d['spars_param']
            num_images, num_samples, grid_size, c, w, h = attributions.shape
            attributions = attributions[:, :, d['head_idx'], :, :, :]
            attributions = attributions.reshape(num_images, num_samples, -1)
            sparsified_attributions = []
            if 'percentile' in d['spars_method']:
                if isinstance(spars_param, str):
                    spars_param = ast.literal_eval(spars_param)
                percentiles = np.array([100-k for k in list(spars_param)])
                for idx in range(num_images):
                    samples = attributions[idx] # (num_samples, -1)
                    
                    if spars_method == 'positive_percentile':
                        
                        for i in range(num_samples):
                            sample_i = samples[i]
                            positive_idx = sample_i > 1e-5
                            pos_sample_values = sample_i[positive_idx]
                            thresholds = np.percentile(sample_i.cpu().numpy(), percentiles)
                            thresholds_tensor = torch.tensor(thresholds, dtype=samples.dtype, device=samples.device)
                            categories = torch.bucketize(pos_sample_values, thresholds_tensor)
                            sample_i[positive_idx] = categories.float()
                            sample_i[positive_idx == 0] = 0
                            sparsified_attributions.append(sample_i.unsqueeze(0))
                    
                    elif spars_method == 'all_percentile':
                        
                        thresholds = np.percentile(samples, percentiles, axis=1)
                        for i in range(num_samples):
                            categories = torch.bucketize(samples[i].clone().detach(), torch.tensor(list(thresholds))[:, i],)
                            sparsified_attributions.append(categories.unsqueeze(0))
                sparsified_attributions = torch.cat(sparsified_attributions, dim=0).reshape(num_images, num_samples, 1, c, w, h)
            elif spars_method == 'threshold':
                sparsified_attributions = torch.bucketize(attributions, torch.tensor(list(d['spars_param'])))
            attributions = sparsified_attributions
        elif step == 'certify':
            certified_attributions = []
            num_images = attributions.shape[0]
            attributions_ = attributions[:, :d['n'], d['head_idx'], :, :, :].reshape(num_images, d['n'], -1)
            assert d['n'] == attributions_.shape[1]
            for idx in range(num_images):
                certified_attr = fast_certify(attributions_[idx], d['n0'], d['n'], tau=d['tau'], alpha=d['alpha'], stats=None, non_ignore_idx=None)
                certified_attributions.append(certified_attr.unsqueeze(0))
            attributions = torch.cat(certified_attributions, dim=0).reshape(num_images, attributions.shape[-2], attributions.shape[-1])

    if return_scale:
        return attributions, scale_factor
    return attributions


from matplotlib.colors import LinearSegmentedColormap

def get_certified_rgb(certified, cmap, degrees, abstain_color=None, certified_1_color=None, certified_0_color=(0, 0, 0, 1),):
    # either use enforced colors or a colormap
    certified_ = certified.copy()
    certified_ = np.clip(certified_, 1, len(degrees)) -1
    certified_ = certified_/len(degrees)
    rgb_certified = cmap(certified_)
    
    if certified_1_color is not None:
        rgb_certified[certified > 0] = certified_1_color
    if len(degrees) == 1:
        if abstain_color is not None:
            rgb_certified[certified == -1] = abstain_color
        else: rgb_certified[certified == -1] = (0.65, 0.65, 0.65, 1)
    rgb_certified[certified == 0] = certified_0_color
    return rgb_certified

def draw_segments(segments, ax):
    x_scale = 1
    rectangle_width_scale = 0.61  
    x_shift = 0.19
    # Draw segments as a bar below the certified image
    for segment in segments:
        ax.add_patch(mpatches.Rectangle(
            (segment[0]*x_scale*rectangle_width_scale+x_shift, 0.15),  # Adjust 'y' position as necessary
            (segment[1] - segment[0])*x_scale*rectangle_width_scale,  # Width of the rectangle
            0.3,  # Height of the rectangle
            facecolor=segment[2], 
            #edgecolor='black', 
            linewidth=0.1
        ))

    # Calculate the boundaries of all rectangles for the border
    min_x = min(segment[0] for segment in segments) * x_scale * rectangle_width_scale + x_shift
    max_x = max(segment[1] for segment in segments) * x_scale * rectangle_width_scale + x_shift
    border_height = 0.3  # Set the height for the border rectangle

    # Draw a single black border rectangle around all segments
    ax.add_patch(mpatches.Rectangle(
        (min_x, 0.15),  # Starting position for the border rectangle
        max_x - min_x,  # Width calculated from min and max x values
        border_height,   # Height of the border rectangle
        facecolor='none',  # No fill color
        edgecolor='black',  # Black border
        linewidth=0.4  # Adjust border thickness as needed
    ))
    ax.set_ylim(0, 0.5)  # Set y-axis limit to allow full visibility of rectangles
    ax.set_xlim(0, x_scale)  # Set y-axis limit to allow full visibility of rectangles

def generate_bottom_title(certified, degrees, colors, abstain_color, bg_color=(1, 1, 1, 1)):
    certified_labels = [i for i in range(len(degrees)+1)]
    text = ''
    total_count = np.prod(certified.shape) 
    non_negative_count = (certified != -1).sum() 
    # Initialize a variable to hold the total percentage calculated
    total_percent = 0
    percentages_dict = {}
    for l in certified_labels:
        if l != -1:  # Calculate only for valid labels
            label_count = (certified == l).sum()  # Count occurrences of the current label
            label_i_certified_pr = (label_count / total_count) * 100  # Calculate percentage
            label_i_certified_pr = "{:.1f}".format(label_i_certified_pr)
            if l == 0:
                text += f'0: {label_i_certified_pr}\% \n'
            else:
                text += f'top {100-degrees[l-1][0]}\%: {label_i_certified_pr}\% \n'
            total_percent += float(label_i_certified_pr)  # Accumulate total percentage
            percentages_dict[l] = float(label_i_certified_pr)
    # Calculate the percentage for the \oslash label
    oslash_percentage = 100 - total_percent
    oslash_label = r'$\oslash$'
    percentages_dict[-1] = float(oslash_percentage)

    # Only add \oslash if there are any -1 entries
    if total_count > 0 and non_negative_count < total_count:  # Only if there are entries with -1
        text += f'{oslash_label}: {oslash_percentage:.1f}\% \n'

    text = text[:-2]  # Remove the last newline if needed

    # Create segments
    start = 0
    segments = []
    
    for label, percentage in percentages_dict.items():
        end = start + percentage
        if label == -1:
            color = abstain_color
        elif label == 0:
            color = bg_color
        else:
            color = colors[label-1]
        segments.append((start/100, end/100, color))
        start = end
    return text, segments

def generate_certified_legend(spars_dict, cmap, abstain_color, bg_color=(0, 0, 0, 1)):
    degrees = spars_dict['degrees']
    certified_legend_colors = cmap([(i+1)/len(degrees) for i in range(len(degrees))]).tolist()
    legend_txt = []
    for degree in degrees:
        if spars_dict['spars_method'] == 'all_percentile':
            legend_txt.append(f'top {100-degree[0]}\%')
        elif spars_dict['spars_method'] == 'positive_percentile':
            legend_txt.append(f'top {100-degree[0]}\% (of +ve)')
        elif spars_dict['spars_method'] == 'threshold':
            legend_txt.append(f'$> {degree[0]}$')

    if len(degrees) == 1:
        certified_legend_colors.append(bg_color)
        legend_txt.append('0')
        certified_legend_colors.append(abstain_color)
        legend_txt.append(r'$\oslash$')

    return certified_legend_colors, legend_txt