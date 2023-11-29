import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import random


def get_patch(ldct, ndct, patch_size=64):
    # patch_size = 64
    ## write your code
    # ldct는 저밀도 CT영상, ndct는 표준 복용량 CT
    # 대형 이미지의 작은 부분의 영역
    assert ldct.shape == ndct.shape
    patch_ldct = []
    patch_ndct = []
    for _ in range(20):
        i = random.randint(0, ldct.shape[1] - patch_size)
        j = random.randint(0, ldct.shape[2] - patch_size)
    
        ldct_patch = ldct[:, i:i + patch_size, j:j + patch_size]
        ndct_patch = ndct[:, i:i + patch_size, j:j + patch_size]
        patch_ldct.append(ldct_patch)
        patch_ndct.append(ndct_patch)
        
    return ldct_patch, ndct_patch

def augment(ldct, ndct, hflip=True, vflip=True, rot=True):
    
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    rot90 = rot * random.randint(0,4)
    
    # # write your code
    if hflip:
    ## write your code
    ## write your code
        ldct = np.flip(ldct, axis=2)  # 수평 뒤집기, 마지막 차원(너비) 변경
        ndct = np.flip(ndct, axis=2)
    if vflip:
    ## write your code
    ## write your code
        ldct = np.flip(ldct, axis=1)  # 수직 뒤집기, 세 번째 차원(높이) 변경
        ndct = np.flip(ndct, axis=1)
    if rot90:
        # ldct = np.rot90(ldct, rot90, [2, 3])  # 90도 회전, 높이와 너비 차원에서 회전
        # ndct = np.rot90(ndct, rot90, [2, 3])
        ldct = np.rot90(ldct, axes=(1,2))
        ndct = np.rot90(ndct, axes=(1,2))
    return ldct, ndct