import numpy as np
from helper import *

def movePatchOverImg(image, filter_size, apply_filter_to_patch):
    if filter_size %2 == 0:
        raise ValueError("Filter not odd")
    
    grey_image = np.dot(image[...,:3],[0.2989, 0.5870, 0.1140])
    save_image("grey.png",grey_image)
    
    h,w = grey_image.shape
    
    padding = filter_size // 2
    padded_image = np.pad(grey_image, pad_width=padding, mode='constant', constant_values=0)
    
    output_image = np.zeros_like(grey_image)
    
    for i in range(h):
        for j in range(w):
            patch = padded_image[i:i+filter_size,j:j+filter_size]
            output_image[i,j] = apply_filter_to_patch(patch)
    return output_image

def detect_horizontal_edge(image_patch):
    kernel = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])
    outputval = np.sum(kernel * image_patch)
    return outputval

def detect_vertical_edge(image_patch):
    kernel = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])
    outputval = np.sum(kernel * image_patch)
    return outputval

def detect_all_edges(image_patch):
    vert = detect_vertical_edge(image_patch)
    hori = detect_horizontal_edge(image_patch)
    outputval = np.sqrt(vert**2+hori**2)
    return outputval

def remove_noise(image_patch):
    outval = np.median(image_patch)
    return outval

def create_gaussian_kernel(size, sigma):
    output_kernel = np.fromfunction(lambda x, y: (1/ (2 * np.pi * sigma**2)) * np.exp(-(((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2))),(size, size))
    #normalizing to ensure sum of filter values to 1
    # print(output_kernel)
    return output_kernel / np.sum(output_kernel)

def gaussian_blur(image_patch):
    size = 25
    sigma = 1
    kernel = create_gaussian_kernel(size, sigma)
    outputval = np.sum(kernel * image_patch)
    return outputval

def unsharp_masking(image, scale):
    grey_image = np.dot(image[...,:3],[0.2989, 0.5870, 0.1140])
    # save_image("grey_image.png",grey_image)
    kernal_size = 25
    blurred_image = movePatchOverImg(image,kernal_size,gaussian_blur)
    # save_image("blurred_image.png",blurred_image)
    sub_image = grey_image-blurred_image
    # save_image("sub_image.png",sub_image)
    scaled_image = sub_image*scale
    # save_image("scaled_image.png",scaled_image)
    out_image = grey_image+scaled_image
    # save_image("out_image.png",out_image)
    return out_image

# TASK 1  
img=load_image("cutebird.png")
filter_size=3 #You may change this to any appropriate odd number
hori_edges = movePatchOverImg(img, filter_size, detect_horizontal_edge)
save_image("horizontal_edge.png",hori_edges)
filter_size=3 #You may change this to any appropriate odd number
vert_edges = movePatchOverImg(img, filter_size, detect_vertical_edge)
save_image("vertical_edge.png",vert_edges)
filter_size=3 #You may change this to any appropriate odd number
all_edges = movePatchOverImg(img, filter_size, detect_all_edges)
save_image("all_edge.png",all_edges)

# TASK 2
noisyimg=load_image("noisycutebird.png")
filter_size=3 #You may change this to any appropriate odd number
denoised = movePatchOverImg(noisyimg, filter_size, remove_noise)
save_image("denoised.png",denoised)

# # TASK 3
img=load_image("cutebird.png")
scale= 2#You may use any appropriate positive number (ideally between 1 and 3)
save_image("unsharpmask.png",unsharp_masking(img,scale))
