import numpy as np

def extract_patches(image, patch_sizes=[16, 32, 64], stride=16):
    patches = []
    h, w, _ = image.shape

    for p in patch_sizes:
        for i in range(0, h - p + 1, stride):
            for j in range(0, w - p + 1, stride):
                patch = image[i:i+p, j:j+p]
                patches.append(patch)

    return patches