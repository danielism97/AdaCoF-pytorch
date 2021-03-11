import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import pywt

def temporal_upsample(frames_lfr, ds):
    """
    Returns temporally nearest-neighbour upsampled version of the LFR test video,
    with number of frames equal to the reference HFR video.
    args:
    frames_lfr -- np array, of shape (num_frames,H,W)
    ds -- int, downsample ratio
    """

    num_frames, H, W = frames_lfr.shape

    return F.interpolate(torch.Tensor(frames_lfr[None,None,...]), size=(ds*num_frames, H, W), mode='nearest').numpy()[0,0]



def temporal_DWT(frames, filter, N):
    """
    Perform DWT along time axis. Return upsampled band coeffs of shape (N,num_frames,H,W)
    args:
    frames -- np.array of shape (num_frames,H,W)
    filter -- str, wavelet name
    N -- int, levels to decompose
    """

    # first flatten the images (num_frames, H, W) -> (HxW, num_frames)
    num_frames, H, W = frames.shape
    frames = frames.reshape((num_frames, H*W)).T # (HxW, num_frames)

    # perform 1D DWT along time axis at N levels and get HF coeffs (lo to hi, N to 1)
    HF_coeffs = pywt.wavedec(frames, 'haar', mode='symmetric', level=N, axis=1)[1:]

    # reshape and upsample each band coeff
    B = []
    for HF_band in HF_coeffs:
        HF_band = HF_band.T.reshape((HF_band.shape[1], H, W)) # (t',H,W)
        HF_band = F.interpolate(torch.Tensor(HF_band[None,None,...]), size=(num_frames, H, W), mode='nearest').numpy()[0,0] # (num_frames,H,W)
        B.append(HF_band)

    return np.array(B)



def pool_spatiotemporal(Dc, s, l):
    """
    Perform spatio-temporal pooling of the subband difference tensor.
    args:
    Dc -- np.array of shape (num_frames,H,W)
    s -- size of spatial window
    l -- length of temporal window
    """

    num_frames, H, W = Dc.shape

    if l > num_frames:
        l = 1

    # spatial pooling: take avg of 16x16 blocks, then take the max
    avg_filter = torch.ones(num_frames,1,s,s) / (s*s) # (num_frames,1,h,w)
    Dc_avg = F.conv2d(torch.Tensor(Dc[None,...]), avg_filter, stride=(s,s), groups=num_frames).numpy()[0] # (num_frames,h',w')
    Q_t = np.amax(Dc_avg, axis=(1,2)) # (num_frames,)

    # temporal pooling: take sum over temporal window, then take max over all sums
    sum_filter = torch.ones(1,1,l) / l # (1,1,l)
    Q = F.conv1d(torch.Tensor(Q_t[None,None,...]), sum_filter, stride=l).numpy()[0,0] # (num_frames/l, )
    Q = np.amax(Q)

    # convert to decibel units
    FRQM = 20 * np.log10(255/Q)

    return FRQM


def compute_frqm(frames_orig, frames_vfi, fps_h, fps_l):
    """
    args:
    frames_orig, frames_vfi -- luma samples, both of shape (t, H, W)
    fps -- int
    """

    # DWT
    N = int(np.ceil(np.log2(fps_h / fps_l)))
    B_ref = temporal_DWT(frames_orig, filter='haar', N=N) # (N,num_frames,H,W)
    B_test = temporal_DWT(frames_vfi, filter='haar', N=N) # (N,num_frames,H,W)

    # obtain HF subband difference
    D = np.abs(B_ref - B_test)

    # multiply each subband with corresponding weight from the paper
    for n in range(N, 0, -1):
        freq = fps_h / (2**n)
        weight = 0.01 if freq is 60 else 0.03 if freq is 30 else 0.14
        D[N-n] = D[N-n] * weight 
    
    # sum the weighted coeffs along subband axis
    Dc = np.sum(D, axis=0) # (num_frames,H,W)
    
    # spatio-temporal pooling
    FRQM = pool_spatiotemporal(Dc, s=16, l=int(fps_h/5))

    return FRQM