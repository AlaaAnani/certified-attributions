"""
Code by Sukrut Rao https://github.com/sukrutrao/Attribution-Evaluation
"""
import torch
import torchvision
import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
from attribution_certification import utils


def get_augmented_shape(shape, scale):
    """
    Given an input shape and a scale factor, compute the new shape for augmented grid images.

    Args:
        shape (tuple or list): Original shape of the data tensor, typically (N, C, H, W).
        scale (int): Scale factor to determine grid size.

    Returns:
        list: Augmented shape with height and width scaled accordingly.
    """
    augmented_shape = list(shape)
    augmented_shape[2] *= scale
    augmented_shape[3] *= scale
    return augmented_shape


def create_grid(data, labels, selected_indices, num_classes, args):
    """
    Creates a dataset of grid images composed of sub-images from different classes. Each grid image contains
    `scale x scale` patches, each from a distinct class, and ensures that images are not repeated within a grid.

    Args:
        data (torch.Tensor): Input image data.
        labels (torch.Tensor): Corresponding class labels for the input data.
        selected_indices (list or np.ndarray): Indices of high-confidence classified images.
        num_classes (int): Total number of classes in the dataset.
        args (argparse.Namespace): Arguments containing configuration such as scale, num_test_images, etc.

    Returns:
        dict: A dictionary containing the augmented dataset, labels, and associated metadata.
    """
    assert args.scale >= 1
    assert args.scale * args.scale <= num_classes
    augmented_shape = get_augmented_shape(data.shape, args.scale)
    augmented_data = torch.zeros([args.num_test_images] + augmented_shape[1:])
    augmented_labels = torch.zeros(
        (args.num_test_images, args.scale * args.scale))
    class_data = []
    class_selected_indices = []
    class_data_idxs = []
    selected_indices = torch.tensor(selected_indices)
    for cidx in range(num_classes):
        permutation = torch.randperm(
            data[np.where(np.array(labels.squeeze().cpu()) == cidx)].shape[0])
        class_data.append(torch.tensor(
            data[np.where(np.array(labels.squeeze().cpu()) == cidx)])[permutation])
        class_selected_indices.append(torch.tensor(
            selected_indices[np.where(np.array(labels.squeeze().cpu()) == cidx)])[permutation])
        class_data_idxs.append(0)
    final_selected_indices = torch.zeros(args.num_test_images, args.scale*args.scale)
    for img_idx in tqdm(range(args.num_test_images)):
        while True:
            exit_flag = 1
            class_perm = torch.randperm(num_classes)[
                :(args.scale * args.scale)]
            for class_idx in class_perm:
                if class_data_idxs[class_idx] >= len(class_data[class_idx]):
                    exit_flag = 0
            if exit_flag:
                break
        for idx, class_idx in enumerate(class_perm):
            y, x, h, w = utils.get_augmentation_range(
                augmented_data.shape, args.scale, idx)
            augmented_data[img_idx, :, y:y + h, x:x +
                           w] = torch.clone(class_data[class_idx][class_data_idxs[class_idx]])
            final_selected_indices[img_idx,
                                   idx] = class_selected_indices[class_idx][class_data_idxs[class_idx]]
            class_data_idxs[class_idx] += 1
        augmented_labels[img_idx] = torch.clone(class_perm)
    return {'data': augmented_data, 'labels': augmented_labels.to(torch.long), 'input_dims': list(data.shape)[1:], 'num_classes': num_classes, 'scale': args.scale, 'final_selected_indices': final_selected_indices.to(torch.long)}


def create_grid_repclasscorners(data, labels, selected_indices, num_classes, args):
    """
    Creates a dataset of grid images where one class is repeated at both the top-left and bottom-right corners.
    This can be used to study spatial redundancy or class-specific attribution in specific regions.

    Args:
        data (torch.Tensor): Input image data.
        labels (torch.Tensor): Corresponding class labels for the input data.
        selected_indices (list or np.ndarray): Indices of high-confidence classified images.
        num_classes (int): Total number of classes in the dataset.
        args (argparse.Namespace): Arguments containing configuration such as scale, num_test_images, etc.

    Returns:
        dict: A dictionary containing the augmented dataset, labels, and associated metadata.
    """
    assert args.scale >= 1
    assert args.scale * args.scale <= num_classes
    augmented_shape = get_augmented_shape(data.shape, args.scale)
    augmented_data = torch.zeros([args.num_test_images] + augmented_shape[1:])
    augmented_labels = torch.zeros(
        (args.num_test_images, args.scale * args.scale))
    class_data = []
    class_selected_indices = []
    class_data_idxs = []
    selected_indices = torch.tensor(selected_indices)
    for cidx in range(num_classes):
        permutation = torch.randperm(
            data[np.where(np.array(labels.cpu()) == cidx)].shape[0])
        class_data.append(torch.tensor(
            data[np.where(np.array(labels.cpu()) == cidx)])[permutation])
        class_selected_indices.append(torch.tensor(
            selected_indices[np.where(np.array(labels.cpu()) == cidx)])[permutation])
        class_data_idxs.append(0)
    final_selected_indices = torch.zeros(args.num_test_images, args.scale*args.scale)
    for img_idx in tqdm(range(args.num_test_images)):
        while True:
            exit_flag = 1
            class_perm = torch.randperm(num_classes)[
                :(args.scale * args.scale)-1]
            class_perm = torch.hstack((class_perm, torch.tensor(class_perm[0])))
            for class_idx in class_perm:
                if class_data_idxs[class_idx] >= len(class_data[class_idx]):
                    exit_flag = 0
            if class_data_idxs[class_perm[0]] >= len(class_data[class_perm[0]])-1:
                exit_flag = 0
            if exit_flag:
                break
        for idx, class_idx in enumerate(class_perm):
            y, x, h, w = utils.get_augmentation_range(
                augmented_data.shape, args.scale, idx)
            augmented_data[img_idx, :, y:y + h, x:x +
                           w] = torch.clone(class_data[class_idx][class_data_idxs[class_idx]])
            final_selected_indices[img_idx,
                                   idx] = class_selected_indices[class_idx][class_data_idxs[class_idx]]
            class_data_idxs[class_idx] += 1
        augmented_labels[img_idx] = torch.clone(class_perm)
    return {'data': augmented_data, 'labels': augmented_labels.to(torch.long), 'input_dims': list(data.shape)[1:], 'num_classes': num_classes, 'scale': args.scale, 'final_selected_indices': final_selected_indices.to(torch.long)}


def filter_high_conf_data(data_dict, args):
    """
    Filters images from the dataset that are confidently classified by all specified models.
    Only retains images that each model classifies correctly with a softmax confidence above a given threshold.

    Args:
        data_dict (dict): Dictionary containing the dataset with keys "data" and "labels".
        args (argparse.Namespace): Arguments including model list, batch size, confidence threshold, and CUDA flag.

    Returns:
        tuple: Filtered data tensor, corresponding labels, selected indices, and number of classes.
    """
    dataset = torch.utils.data.TensorDataset(
        data_dict["data"], data_dict["labels"][:, 0])
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False)
    all_selected_indices = []
    for model_name in args.models:
        model_selected_indices = []
        model = torchvision.models.__dict__[model_name](pretrained=True)
        if args.cuda:
            model.cuda()
        model.eval()

        for batch_idx, (test_X, test_y) in enumerate(tqdm(data_loader)):
            if args.cuda:
                test_X = test_X.cuda()
                test_y = test_y.cuda()
            outs = model(test_X)
            softmaxes = torch.nn.functional.softmax(outs, dim=1)
            preds = torch.argmax(softmaxes, dim=1)
            pred_confs = softmaxes[:, test_y].diag()
            above_confs = torch.where((pred_confs >= args.conf_threshold)
                                      & (preds == test_y))[0]
            model_selected_indices.append(args.batch_size*batch_idx + above_confs)
        model_selected_indices = torch.cat(model_selected_indices).cpu().numpy()
        all_selected_indices.append(model_selected_indices)
    selected_indices_intersection = all_selected_indices[0]
    for idx in range(1, len(all_selected_indices)):
        selected_indices_intersection = np.intersect1d(
            selected_indices_intersection, all_selected_indices[idx])

    return data_dict["data"][selected_indices_intersection], data_dict["labels"][selected_indices_intersection], selected_indices_intersection, data_dict["num_classes"]


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_single_dict = torch.load(os.path.join(args.dataset_path, "test.pt"))

    filtered_data, filtered_labels, selected_indices, num_classes = filter_high_conf_data(
        dataset_single_dict, args)
    if args.repeat_classes_corners:
        test_data_augmented = create_grid_repclasscorners(
            filtered_data, filtered_labels, selected_indices, num_classes, args)
    else:
        test_data_augmented = create_grid(
            filtered_data, filtered_labels, selected_indices, num_classes, args)
    os.makedirs(args.save_path, exist_ok=False)
    print("Saving grid dataset at", os.path.join(args.save_path, 'test.pt'))
    torch.save(test_data_augmented, os.path.join(args.save_path, 'test.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates a dataset consisting of grids of images filtered by a minimum confidence threshold on a set of models.")
    parser.add_argument('--dataset_path', type=str, default="data/imagenet",
                        help="Path to dataset from which grids are to be generated.")
    parser.add_argument('--scale', type=int, default=2,
                        help="Dimension n for generating n x n grids.")
    parser.add_argument('--save_path', type=str, default='data/imagenet_grid_2x2',
                        help="Path to save generated grid dataset.")
    parser.add_argument('--seed', type=int, default=73, help="Random seed.")
    parser.add_argument('--num_test_images', type=int, default=500,
                        help="Number of grid images to generate.")
    parser.add_argument('--conf_threshold', type=float, default=0.99,
                        help="Confidence threshold for selecting images to place in the grid. Only images from the dataset that were classified with at least this confidence by all models will be used for the grid dataset.")
    parser.add_argument('--models', nargs='+', type=str, default=['vgg11', 'resnet18'],
                        help="Names of all Torchvision models that should classify all grid cells in the generated grid dataset with a confidence at least that as the confidence threshold, in a space-separated format.")
    parser.add_argument('--repeat_classes_corners', action='store_true', default=False,
                        help="Flag to enable repeating the same class at the top-left and bottom-right corners of the generated grids.")
    parser.add_argument('--batch_size', type=int, default=2,
                        help="Batch size for forward passes through the models when filtering the dataset by class confidence.")
    parser.add_argument('--cuda', action='store_true', 
                        help="Flag to enable computation on the GPU.")
    args = parser.parse_args()
    main(args)
