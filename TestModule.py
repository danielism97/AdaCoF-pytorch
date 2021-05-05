from PIL import Image
import torch
from torchvision import transforms
from math import log10
from torchvision.utils import save_image as imwrite
import os
from utility import to_variable
import numpy as np
# import pandas as pd
import pickle
import subprocess
import cv2
import math
from os.path import join, isdir, getsize
from datareader import read_frame_yuv2rgb
from pytorch_ssim import ssim as to_ssim

def to_psnr(rec, gt):
    mse = torch.nn.functional.mse_loss(rec, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    # ToTensor scales input images to [0.0, 1.0]
    intensity_max = 1.0
    
    psnr_list = [10.0 * math.log10(intensity_max / mse) for mse in mse_list]

    return psnr_list



class Middlebury_eval:
    def __init__(self, input_dir='./evaluation'):
        self.im_list = ['Backyard', 'Basketball', 'Dumptruck', 'Evergreen', 'Mequon', 'Schefflera', 'Teddy', 'Urban']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/input/' + item + '/frame10.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/input/' + item + '/frame11.png')).unsqueeze(0)))

    def Test(self, model, output_dir='./evaluation/output', output_name='frame10i11.png'):
        model.eval()
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))


class Middlebury_other:
    def __init__(self, input_dir, gt_dir):
        self.im_list = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame10.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame11.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(gt_dir + '/' + item + '/frame10i11.png')).unsqueeze(0)))

    def Test(self, model, output_dir, current_epoch, logfile=None, output_name='output.png'):
        model.eval()
        av_psnr = 0
        if logfile is not None:
            logfile.write('{:<7s}{:<3d}'.format('Epoch: ', current_epoch) + '\n')
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            gt = self.gt_list[idx]
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            av_psnr += psnr
            imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
            msg = '{:<15s}{:<20.16f}'.format(self.im_list[idx] + ': ', psnr) + '\n'
            print(msg, end='')
            if logfile is not None:
                logfile.write(msg)
        av_psnr /= len(self.im_list)
        msg = '{:<15s}{:<20.16f}'.format('Average: ', av_psnr) + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write(msg)


class Davis:
    def __init__(self, input_dir, gt_dir):
        self.im_list = ['bike-trial', 'boxing', 'burnout', 'choreography', 'demolition', 'dive-in', 'dolphins', 'e-bike', 'grass-chopper', 'hurdles', 'inflatable', 'juggle', 'kart-turn', 'kids-turning', 'lions', 'mbike-santa', 'monkeys', 'ocean-birds', 'pole-vault', 'running', 'selfie', 'skydive', 'speed-skating', 'swing-boy', 'tackle', 'turtle', 'varanus-tree', 'vietnam', 'wings-turn']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame10.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame11.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(gt_dir + '/' + item + '/frame10i11.png')).unsqueeze(0)))

    def Test(self, model, output_dir, logfile=None, output_name='output.png'):
        model.eval()
        av_psnr = 0
        if logfile is not None:
            logfile.write('{:<7s}{:<3d}'.format('Epoch: ', model.get_epoch()) + '\n')
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            gt = self.gt_list[idx]
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            av_psnr += psnr
            imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
            msg = '{:<15s}{:<20.16f}'.format(self.im_list[idx] + ': ', psnr) + '\n'
            print(msg, end='')
            if logfile is not None:
                logfile.write(msg)
        av_psnr /= len(self.im_list)
        msg = '{:<15s}{:<20.16f}'.format('Average: ', av_psnr) + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write(msg)


class ucf:
    def __init__(self, input_dir):
        self.im_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame0.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame2.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame1.png')).unsqueeze(0)))

    def Test(self, model, output_dir, logfile=None, output_name='output.png'):
        model.eval()
        av_psnr = 0
        if logfile is not None:
            logfile.write('{:<7s}{:<3d}'.format('Epoch: ', model.get_epoch()) + '\n')
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            gt = self.gt_list[idx]
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            av_psnr += psnr
            imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
            msg = '{:<15s}{:<20.16f}'.format(self.im_list[idx] + ': ', psnr) + '\n'
            print(msg, end='')
            if logfile is not None:
                logfile.write(msg)
        av_psnr /= len(self.im_list)
        msg = '{:<15s}{:<20.16f}'.format('Average: ', av_psnr) + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write(msg)


# class HomTex:
#     def __init__(self, db_dir, texture):
#         self.width = 256
#         self.height = 256
#         self.yuv_list = self._get_seq_list(db_dir, texture)
#         self.transform = transforms.Compose([transforms.ToTensor()])
#         self.bvi_texture_seqs = ['PaintingTilting', 'PaperStatic', 'BallUnderWater', 'BricksBushesStatic', 
#                                  'PlasmaFree', 'CalmingWater', 'LampLeaves', 'SmokeClear']

#     def Test(self, model, epoch, output_dir):
#         vmaf_dir = '/mnt/storage/home/mt20523/vmaf'
#         adacof_dir = '/mnt/storage/home/mt20523/AdaCoF-pytorch'
#         model.eval()

#         totensor = transforms.ToTensor()

#         vmaf_dict = dict()
#         psnr_dict = dict()
#         ssim_dict = dict()

#         for i, yuv_path in enumerate(self.yuv_list, 1):
#             print('processing the {}/120 sequence'.format(i))
#             oup_psnr, oup_ssim = [], []

#             # get seq name
#             seq_name = yuv_path.split('/')[-1].split('_')[:-1]
#             seq_name = seq_name[0] if len(seq_name) == 1 else seq_name[0]+'_'+seq_name[1]
#             fps = 25
#             for s in self.bvi_texture_seqs:
#                 if s in seq_name:
#                     fps = 60

#             # load sequence
#             stream = open(yuv_path, 'r')
#             file_size = os.path.getsize(yuv_path)
#             n_frames = file_size // (self.width*self.height*3 // 2)

#             # oup_frames = []
#             # add the first frame
#             # oup_frames.append(cv2.cvtColor(read_frame_yuv2rgb(stream, self.width, self.height, 0, 8), cv2.COLOR_RGB2BGR))
#             for t in range(0, n_frames-2, 2):
#                 with torch.no_grad():
#                 # read 3 frames (rgb)
#                     img1 = totensor(read_frame_yuv2rgb(stream, self.width, self.height, t, 8))[None,...].cuda() # 1x3xHxW
#                     img2 = totensor(read_frame_yuv2rgb(stream, self.width, self.height, t+1, 8))[None,...].cuda()
#                     img3 = totensor(read_frame_yuv2rgb(stream, self.width, self.height, t+2, 8))[None,...].cuda()

#                 # predict
#                     oup = model(img1, img3)
#                 # Calculate average PSNR
#                 oup_psnr.extend(to_psnr(oup, img2))
#                 # Calculate average SSIM
#                 oup_ssim.extend(to_ssim(oup, img2, size_average=False).cpu().numpy())

#                 # oup_rgb = np.moveaxis(oup[0].cpu().clamp(0.0, 1.0).numpy()*255.0, 0, -1).astype(np.uint8)

#                 # oup_frames.append(cv2.cvtColor(oup_rgb, cv2.COLOR_RGB2BGR)) #256x256x3
#                 # oup_frames.append(cv2.cvtColor(read_frame_yuv2rgb(stream, self.width, self.height, t+2, 8), cv2.COLOR_RGB2BGR))

#                 torch.cuda.empty_cache()
#             # add the last frame (index: 249)
#             # oup_frames.append(cv2.cvtColor(read_frame_yuv2rgb(stream, self.width, self.height, n_frames-1, 8), cv2.COLOR_RGB2BGR))
#             stream.close()

#             # # build interpolated video (mp4)
#             # out_mp4 = cv2.VideoWriter('tmp.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (256,256))
#             # for i in range(len(oup_frames)):
#             #     out_mp4.write(oup_frames[i])
#             # out_mp4.release()

#             # # convert mp4 to yuv
#             # os.system('ffmpeg -i tmp.mp4 tmp.yuv')

#             # # compute vmaf
#             # os.chdir(vmaf_dir)
#             # cmd = 'PYTHONPATH=python /content/vmaf/python/vmaf/script/run_vmaf.py yuv420p 256 256 {} {}/tmp.yuv --out-fmt text'.format(yuv_path, adacof_dir)
#             # result = subprocess.check_output(cmd, shell=True)
#             # os.chdir(adacof_dir)
#             # os.system('rm tmp.yuv tmp.mp4')

#             # oup_vmaf = float(str(result).split('Aggregate',1)[1].split('VMAF_score:')[1].split('\\')[0])

#             print('successfully test {} images in {}'.format(len(oup_psnr), seq_name))
#             # print('oup_PSNR:{0:.2f}, oup_SSIM:{1:.4f}, oup_VMAF:{1:.4f}'.format(sum(oup_psnr)/len(oup_psnr), sum(oup_ssim)/len(oup_ssim), oup_vmaf))
#             psnr_dict[seq_name] = sum(oup_psnr)/len(oup_psnr)
#             ssim_dict[seq_name] = sum(oup_ssim)/len(oup_ssim)
#             # vmaf_dict[seq_name] = oup_vmaf

#         with open(output_dir+'/adacof_psnr_epoch{}.pkl'.format(str(epoch).zfill(2)), 'wb+') as f:
#             pickle.dump(psnr_dict, f)
#         with open(output_dir+'/adacof_ssim_epoch{}.pkl'.format(str(epoch).zfill(2)), 'wb+') as f:
#             pickle.dump(ssim_dict, f)
#         # with open(output_dir+'/adacof_vmaf_epoch{}.pkl'.format(str(epoch).zfill(2)), 'wb+') as f:
#         #     pickle.dump(vmaf_dict, f)
    
#     def _get_seq_list(self, db_dir, texture):
#         # read annotations
#         csv_path = join(db_dir, 'Annotations.csv')
#         df = pd.read_csv(csv_path, sep=';', nrows=120)
#         seq_labels = dict()
#         for index, row in df.iterrows():
#             seq_name = row['Sequence name'].split('_')[:-1]
#             seq_name = seq_name[0] if len(seq_name) == 1 else seq_name[0]+'_'+seq_name[1]
#             if not seq_name.endswith('downsampled'):
#                 seq_name = seq_name.replace("_", "-", 1)
#             seq_labels[seq_name] = row['Dynamics (static or dynamic)'] + '-' + row['Structure (continuous, discrete)']
        
#         static_list = [join(db_dir, f+'_256x256.yuv') for f in seq_labels.keys() if seq_labels[f].startswith('static')]
#         dyndis_list = [join(db_dir, f+'_256x256.yuv') for f in seq_labels.keys() if seq_labels[f]=='dynamic-discrete']
#         dyncon_list = [join(db_dir, f+'_256x256.yuv') for f in seq_labels.keys() if seq_labels[f]=='dynamic-continuous']
#         if texture == 'mixed':
#             return static_list + dyncon_list + dyndis_list
#         elif texture == 'dyndis':
#             return dyndis_list
#         elif texture == 'dyncon':
#             return dyncon_list
#         elif texture == 'static':
#             return static_list
#         else:
#             print('wrong texture name')
#             return