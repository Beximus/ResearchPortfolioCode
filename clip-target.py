# adapted from imagenet notebook
import argparse
import numpy as np
import torch
import clip
from tqdm import tqdm
from PIL import Image
import sys
import gzip
import html
import os
from functools import lru_cache
import os
import glob
import skimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import OrderedDict
import torch

import ftfy
import regex as re
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# images in skimage to use and their textual descriptions
description_pairs = [
    ["chelsea", "a facial photo of a tabby cat"],
    ["rocket", "a rocket standing on a launchpad"],
    ["astronaut", "a portrait of an astronaut with the American flag"],
    ["motorcycle_right", "a red motorcycle standing in a garage"],
    ["coffee", "a cup of coffee on a saucer"],
    ["camera", "a person looking at a camera on a tripod"],
    ["horse", "a black-and-white silhouette of a horse",],
    ["page", "a page of text about segmentation"],
]

def fetch_images(preprocess, image_files):
    images = []

    for filename in image_files:
        image = preprocess(Image.open(filename).convert("RGB"))
        images.append(image)

    return images

def do_image_features(model, images, image_mean, image_std):
    # image_input = torch.tensor(np.stack(images)).cuda()
    image_input = torch.tensor(np.stack(images)).cpu()
    image_input -= image_mean[:, None, None]
    image_input /= image_std[:, None, None]

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()

    return image_features

def do_text_features(model, texts):
    # text_input = clip.tokenize(texts).cuda()
    text_input = clip.tokenize(texts).cpu()

    with torch.no_grad():
        text_features = model.encode_text(text_input).float()

    return text_features

def calc_self_image_similarity(image_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy() @ image_features.cpu().numpy().T

def calc_target_similarity(target_features, image_features):
    target_features /= target_features.norm(dim=-1, keepdim=True)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return target_features.cpu().numpy() @ image_features.cpu().numpy().T

def show_target_similarity(outfile, similarity, images, target_image, target_text):
    x_count = len(images)

    fig = plt.figure(figsize=(12, 12), dpi=144)
    plt.imshow(similarity, vmin=0.1, vmax=0.9)
    # plt.colorbar()
    if target_text is not None:
        plt.yticks(range(1), [target_text], fontsize=18)
    else:
        plt.yticks([])
    plt.xticks([])
    if target_image is not None:
        # print("HERES THE TARGET IMAGE", target_image)
        plt.imshow(target_image.permute(1, 2, 0), extent=(-1.6, -0.6, 0 - 0.5, 0 + 0.5), origin="lower")        

    for i, image in enumerate(images):
        plt.imshow(image.permute(1, 2, 0), extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
      plt.gca().spines[side].set_visible(False)

    plt.xlim([-2, x_count - 0.5])
    plt.ylim([x_count + 0.5, -2])

    plt.title("Similarity images and image/text target", size=20)
    fig.savefig(outfile)

def main():
    parser = argparse.ArgumentParser(description="test CLIP")
    parser.add_argument('--input-glob', default='CLIP.png',
                        help="list of files to compare with target")
    parser.add_argument('--target-image', default=None,
                        help="use this image as the target")
    parser.add_argument('--target-text', default=None,
                        help="use this text string as the target")
    parser.add_argument('--outfile', default="target.png",
                        help="name of output file")
    parser.add_argument('--topk', type=int, default=10, help="filter glob to top K")
    args = parser.parse_args()

    if args.target_image is None and args.target_text is None:
        print("Program cannot run without target-image, target-text or both")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load("ViT-B/32")

    input_resolution = model.input_resolution.item()
    context_length = model.context_length.item()
    vocab_size = model.vocab_size.item()

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    preprocess = Compose([
        Resize(input_resolution, interpolation=Image.BICUBIC),
        CenterCrop(input_resolution),
        ToTensor()
    ])

    # image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    # image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cpu()
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cpu()

    image_files = list(sorted(glob.glob(args.input_glob)));

    images = fetch_images(preprocess, image_files);
    image_features = do_image_features(model, images, image_mean, image_std)

    target_features = None
    target_image = None
    target_text = None

    if args.target_image is not None:
        target_image_list = fetch_images(preprocess, [args.target_image])
        target_features = do_image_features(model, target_image_list, image_mean, image_std)
        target_image = target_image_list[0]

    if args.target_text is not None:
        target_text = args.target_text
        cur_target_features = do_text_features(model, [args.target_text])
        if target_features is None:
            target_features = cur_target_features
        else:
            target_features[0] = (target_features[0] + cur_target_features[0]) / 2

    target_similarity = calc_target_similarity(target_features, image_features)
    # this unsorted version is technically faster, etc.
    # temp = np.argpartition(-target_similarity[0], arg.topk)
    # topk_indices = temp[:args.topk]
    topk_indices = target_similarity[0].argsort()[-args.topk:]
    topk_indices = np.flip(topk_indices)
    filtered_images = [images[i] for i in topk_indices]
    filtered_similarity = np.asarray([target_similarity[0][topk_indices]])
    show_target_similarity(args.outfile, filtered_similarity, filtered_images, target_image, target_text)

if __name__ == '__main__':
    main()
