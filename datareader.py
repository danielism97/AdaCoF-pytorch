import numpy as np
from os import listdir
from PIL import Image
from os.path import join, isdir, getsize
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import cv2

def cointoss(p):
    return random.random() < p

def read_frame_yuv2rgb(stream, width, height, iFrame, bit_depth):
    if bit_depth == 8:
        stream.seek(iFrame*1.5*width*height)
        Y = np.fromfile(stream, dtype=np.uint8, count=width*height).reshape((height, width))
        
        # read chroma samples and upsample since original is 4:2:0 sampling
        U = np.fromfile(stream, dtype=np.uint8, count=(width//2)*(height//2)).\
                                reshape((height//2, width//2))
        V = np.fromfile(stream, dtype=np.uint8, count=(width//2)*(height//2)).\
                                reshape((height//2, width//2))

    else:
        stream.seek(iFrame*3*width*height)
        Y = np.fromfile(stream, dtype=np.uint16, count=width*height).reshape((height, width))
                
        U = np.fromfile(stream, dtype=np.uint16, count=(width//2)*(height//2)).\
                                reshape((height//2, width//2))
        V = np.fromfile(stream, dtype=np.uint16, count=(width//2)*(height//2)).\
                                reshape((height//2, width//2))

    
    yuv = np.empty((height*3//2, width), dtype=np.uint8)
    yuv[0:height,:] = Y

    yuv[height:height+height//4,:] = U.reshape(-1, width)
    yuv[height+height//4:,:] = V.reshape(-1, width)

    #convert to rgb
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    # bgr = cv2.resize(bgr, (int(width/4),int(height/4)), interpolation = cv2.INTER_AREA)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) # (270,480,3)

    return rgb


class DBreader_Vimeo90k(Dataset):
    def __init__(self, db_dir, random_crop=None, resize=None, augment_s=True, augment_t=True):
        db_dir += '/sequences'
        self.random_crop = random_crop
        self.augment_s = augment_s
        self.augment_t = augment_t

        transform_list = []
        if resize is not None:
            transform_list += [transforms.Resize(resize)]

        transform_list += [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)

        self.folder_list = [(db_dir + '/' + f) for f in listdir(db_dir) if isdir(join(db_dir, f))]
        self.triplet_list = []
        for folder in self.folder_list:
            self.triplet_list += [(folder + '/' + f) for f in listdir(folder) if isdir(join(folder, f))]

        self.triplet_list = np.array(self.triplet_list)
        self.file_len = len(self.triplet_list)

    def __getitem__(self, index):
        rawFrame0 = Image.open(self.triplet_list[index] + "/im1.png")
        rawFrame1 = Image.open(self.triplet_list[index] + "/im2.png")
        rawFrame2 = Image.open(self.triplet_list[index] + "/im3.png")

        if self.random_crop is not None:
            i, j, h, w = transforms.RandomCrop.get_params(rawFrame1, output_size=self.random_crop)
            rawFrame0 = TF.crop(rawFrame0, i, j, h, w)
            rawFrame1 = TF.crop(rawFrame1, i, j, h, w)
            rawFrame2 = TF.crop(rawFrame2, i, j, h, w)

        if self.augment_s:
            if cointoss(0.5):
                rawFrame0 = TF.hflip(rawFrame0)
                rawFrame1 = TF.hflip(rawFrame1)
                rawFrame2 = TF.hflip(rawFrame2)
            if cointoss(0.5):
                rawFrame0 = TF.vflip(rawFrame0)
                rawFrame1 = TF.vflip(rawFrame1)
                rawFrame2 = TF.vflip(rawFrame2)

        frame0 = self.transform(rawFrame0)
        frame1 = self.transform(rawFrame1)
        frame2 = self.transform(rawFrame2)

        if self.augment_t:
            if cointoss(0.5):
                return frame2, frame1, frame0
            else:
                return frame0, frame1, frame2
        else:
            return frame0, frame1, frame2

    def __len__(self):
        return self.file_len


class DBreader_BVItexture(Dataset):
    def __init__(self, db_dir, texture='mixed', random_crop=None, resize=None, augment_s=True, augment_t=True):
        self.height = 1080
        self.width = 1920
        self.random_crop = random_crop
        self.augment_s = augment_s
        self.augment_t = augment_t

        transform_list = []
        if resize is not None:
            transform_list += [transforms.Resize(resize)]

        transform_list += [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)

        self.yuv_list = self._get_seq_list(db_dir, texture)
        self.file_len = len(self.yuv_list)

    def __getitem__(self, index):
        # first randomly sample a triplet
        stream = open(self.yuv_list[index], 'r')
        file_size = getsize(self.yuv_list[index])
        num_frames = file_size // (self.width*self.height*3 // 2)
        frame_idx = random.randint(1, num_frames-2)

        rawFrame0 = read_frame_yuv2rgb(stream, self.width, self.height, frame_idx-1, 8)
        rawFrame1 = read_frame_yuv2rgb(stream, self.width, self.height, frame_idx, 8)
        rawFrame2 = read_frame_yuv2rgb(stream, self.width, self.height, frame_idx+1, 8)
        stream.close()

        if self.random_crop is not None:
            i, j, h, w = transforms.RandomCrop.get_params(rawFrame1, output_size=self.random_crop)
            rawFrame0 = TF.crop(rawFrame0, i, j, h, w)
            rawFrame1 = TF.crop(rawFrame1, i, j, h, w)
            rawFrame2 = TF.crop(rawFrame2, i, j, h, w)

        if self.augment_s:
            if cointoss(0.5):
                rawFrame0 = TF.hflip(rawFrame0)
                rawFrame1 = TF.hflip(rawFrame1)
                rawFrame2 = TF.hflip(rawFrame2)
            if cointoss(0.5):
                rawFrame0 = TF.vflip(rawFrame0)
                rawFrame1 = TF.vflip(rawFrame1)
                rawFrame2 = TF.vflip(rawFrame2)

        frame0 = self.transform(rawFrame0)
        frame1 = self.transform(rawFrame1)
        frame2 = self.transform(rawFrame2)

        if self.augment_t:
            if cointoss(0.5):
                return frame2, frame1, frame0
            else:
                return frame0, frame1, frame2
        else:
            return frame0, frame1, frame2

    def __len__(self):
        return self.file_len

    def _get_seq_list(self, db_dir, texture):
        dyndis_list = [join(db_dir,'DynamicDiscrete',f) for f in listdir(join(db_dir,'DynamicDiscrete')) if f.endswith('.yuv')]
        dyncon_list = [join(db_dir,'DynamicContinuous',f) for f in listdir(join(db_dir,'DynamicContinuous')) if f.endswith('.yuv')]
        static_list = [join(db_dir,'Static',f) for f in listdir(join(db_dir,'Static')) if f.endswith('.yuv')]
        if texture == 'mixed':
            return static_list + dyncon_list + dyndis_list
        elif texture == 'dyndis':
            return dyndis_list
        elif texture == 'dyncon':
            return dyncon_list
        elif texture == 'static':
            return static_list
        else:
            print('wrong texture name')
            return


class DBreader_SynTex(Dataset):
    def __init__(self, db_dir, texture='mixed', random_crop=None, resize=None, augment_s=True, augment_t=True):
        self.height = 1080
        self.width = 1920
        self.random_crop = random_crop
        self.augment_s = augment_s
        self.augment_t = augment_t

        transform_list = []
        if resize is not None:
            transform_list += [transforms.Resize(resize)]

        transform_list += [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)

        self.yuv_list = self._get_seq_list(db_dir, texture)
        self.file_len = len(self.yuv_list)

    def __getitem__(self, index):
        # first randomly sample a triplet
        stream = open(self.yuv_list[index], 'r')
        file_size = getsize(self.yuv_list[index])
        num_frames = file_size // (self.width*self.height*3 // 2)
        frame_idx = random.randint(1, num_frames-2)

        rawFrame0 = read_frame_yuv2rgb(stream, self.width, self.height, frame_idx-1, 8)
        rawFrame1 = read_frame_yuv2rgb(stream, self.width, self.height, frame_idx, 8)
        rawFrame2 = read_frame_yuv2rgb(stream, self.width, self.height, frame_idx+1, 8)
        stream.close()

        if self.random_crop is not None:
            i, j, h, w = transforms.RandomCrop.get_params(rawFrame1, output_size=self.random_crop)
            rawFrame0 = TF.crop(rawFrame0, i, j, h, w)
            rawFrame1 = TF.crop(rawFrame1, i, j, h, w)
            rawFrame2 = TF.crop(rawFrame2, i, j, h, w)

        if self.augment_s:
            if cointoss(0.5):
                rawFrame0 = TF.hflip(rawFrame0)
                rawFrame1 = TF.hflip(rawFrame1)
                rawFrame2 = TF.hflip(rawFrame2)
            if cointoss(0.5):
                rawFrame0 = TF.vflip(rawFrame0)
                rawFrame1 = TF.vflip(rawFrame1)
                rawFrame2 = TF.vflip(rawFrame2)

        frame0 = self.transform(rawFrame0)
        frame1 = self.transform(rawFrame1)
        frame2 = self.transform(rawFrame2)

        if self.augment_t:
            if cointoss(0.5):
                return frame2, frame1, frame0
            else:
                return frame0, frame1, frame2
        else:
            return frame0, frame1, frame2

    def __len__(self):
        return self.file_len

    def _get_seq_list(self, db_dir, texture):
        dyndis_list = [join(db_dir,'DynamicDiscrete',f) for f in listdir(join(db_dir,'DynamicDiscrete')) if f.endswith('.yuv')]
        dyncon_list = [join(db_dir,'DynamicContinuous',f) for f in listdir(join(db_dir,'DynamicContinuous')) if f.endswith('.yuv')]
        static_list = [join(db_dir,'Static',f) for f in listdir(join(db_dir,'Static')) if f.endswith('.yuv')]
        if texture == 'mixed':
            return static_list + dyncon_list + dyndis_list
        elif texture == 'dyndis':
            return dyndis_list
        elif texture == 'dyncon':
            return dyncon_list
        elif texture == 'static':
            return static_list
        else:
            print('wrong texture name')
            return


class DBreader_DynTex(Dataset):
    def __init__(self, db_dir, texture='mixed', random_crop=None, resize=None, augment_s=True, augment_t=True):
        self.random_crop = random_crop
        self.augment_s = augment_s
        self.augment_t = augment_t

        transform_list = []
        if resize is not None:
            transform_list += [transforms.Resize(resize)]

        transform_list += [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)

        seq_list = self._get_seq_list(texture)
        self.avi_list = [join(db_dir, seq+'.avi') for seq in seq_list]

        self.file_len = len(self.avi_list)

    def __getitem__(self, index):
        # first randomly sample a triplet
        cap = cv2.VideoCapture(self.avi_list[index])
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = random.randint(1, num_frames-2)

        # read 3 frames
        cap.set(1,frame_idx-1)
        _, rawFrame0 = cap.read()
        cap.set(1,frame_idx)
        _, rawFrame1 = cap.read()
        cap.set(1,frame_idx+1)
        _, rawFrame2 = cap.read()


        if self.random_crop is not None:
            i, j, h, w = transforms.RandomCrop.get_params(rawFrame1, output_size=self.random_crop)
            rawFrame0 = TF.crop(rawFrame0, i, j, h, w)
            rawFrame1 = TF.crop(rawFrame1, i, j, h, w)
            rawFrame2 = TF.crop(rawFrame2, i, j, h, w)

        if self.augment_s:
            if cointoss(0.5):
                rawFrame0 = TF.hflip(rawFrame0)
                rawFrame1 = TF.hflip(rawFrame1)
                rawFrame2 = TF.hflip(rawFrame2)
            if cointoss(0.5):
                rawFrame0 = TF.vflip(rawFrame0)
                rawFrame1 = TF.vflip(rawFrame1)
                rawFrame2 = TF.vflip(rawFrame2)

        frame0 = self.transform(rawFrame0)
        frame1 = self.transform(rawFrame1)
        frame2 = self.transform(rawFrame2)

        if self.augment_t:
            if cointoss(0.5):
                return frame2, frame1, frame0
            else:
                return frame0, frame1, frame2
        else:
            return frame0, frame1, frame2

    def __len__(self):
        return self.file_len

    def _get_seq_list(self, texture):
        dyndis_list = ['644a310',
                        '6ame200',
                        '64ac510',
                        '649ed10',
                        '646c310',
                        '644c320',
                        '6489b10',
                        '645ab10',
                        '649ib10',
                        '645b210',
                        '646c410',
                        '646d210',
                        '645a550',
                        '649i710',
                        '649f210',
                        '649cb20',
                        '644ca10',
                        '64acb10',
                        '64aa910',
                        '64ad810',
                        '649h710',
                        '64ac110',
                        '649aa20',
                        '644a910',
                        '646ab10',
                        '6486110',
                        '648db10',
                        '648dc10',
                        '6amg900',
                        '6441610',
                        '646c210',
                        '6484i10',
                        '648a810',
                        '6amg800',
                        '644a810',
                        '64ad910',
                        '64aad10',
                        '6485610',
                        '645e310',
                        '645b110',
                        '646c110',
                        '6485210',
                        '649cb10',
                        '6481h20',
                        '64ac310',
                        '6450210',
                        '645a210',
                        '64ac620',
                        '64ab410',
                        '64aaa10',
                        '64ab510',
                        '64ad410',
                        '645b710',
                        '64ccf10',
                        '649i610',
                        '64aca10',
                        '6484j10',
                        '6488510',
                        '644b210',
                        '6486910',
                        '6441210',
                        '644b410',
                        '6488610',
                        '644b310',
                        '6486810',
                        '648a910',
                        '644b510',
                        '6489710',
                        '644a210',
                        '6481e10',
                        '64ccd10',
                        '64ad710',
                        '649i410',
                        '64ac220',
                        '64aac10',
                        '64ac410',
                        '64ac610',
                        '649g610',
                        '649a610',
                        '649i510',
                        '645d110',
                        '6450410',
                        '54pb410',
                        '56ua110',
                        '6488710',
                        '644b710',
                        '64ab810',
                        '644a410',
                        '644a710',
                        '64ce310',
                        '64ce210',
                        '6482a10',
                        '64acd10',
                        '64ccg10',
                        '646c220',
                        '649aa10',
                        '649a710',
                        '64cd110',
                        '645d210',
                        '644c410',
                        '645a610',
                        '648e410',
                        '646d310',
                        '645a710',
                        '644cc10',
                        '6481f10',
                        '6487510',
                        '649i720',
                        '648bg10',
                        '6484320',
                        '644b810',
                        '64ab710',
                        '6488810',
                        '648a920',
                        '64ce110',
                        '64ac710',
                        '645b610',
                        '644cb10',
                        '644ca20',
                        '6450610',
                        '6485710',
                        '648e310',
                        '6amha00',
                        '64ad210',
                        '6485510',
                        '64ad110',
                        '646a910',
                        '645a820',
                        '6484c10',
                        '6484d10',
                        '54pb110',
                        '6481a10',
                        '6484a10',
                        '6481c10',
                        '648d710',
                        '6481d10']
        dyncon_list = ['6481j10',
                        '6485410',
                        '571b310',
                        '647b610',
                        '644c610',
                        '649h320',
                        '6483130',
                        '649ba20',
                        '64caf10',
                        '55fc110',
                        '64ba110',
                        '648e810',
                        '64cac10',
                        '64ca510',
                        '6483a10',
                        '56ub210',
                        '64ba510',
                        '571c110',
                        '645a930',
                        '6483110',
                        '645c310',
                        '64ca910',
                        '649ch10',
                        '64cad10',
                        '649ci10',
                        '646a610',
                        '647b810',
                        '64adl10',
                        '55fc410',
                        '64cbg10',
                        '64cbe10',
                        '6ammi00',
                        '64cb810',
                        '649h920',
                        '6484110',
                        '648e910',
                        '54ab110',
                        '64bac10',
                        '64ca410',
                        '64cbc10',
                        '649h310',
                        '64ca210',
                        '6489310',
                        '644c110',
                        '6483410',
                        '64bad10',
                        '64cbk10',
                        '64bab10',
                        '64ca810',
                        '6482420',
                        '64ba810',
                        '64cb110',
                        '6481k10',
                        '6482c10',
                        '645b410',
                        '6483710',
                        '6485310',
                        '64ba310',
                        '646a520',
                        '64ba610',
                        '6ammb00',
                        '64cb210',
                        '64ba210',
                        '6450810',
                        '649h910',
                        '6484510',
                        '648ea10',
                        '64ca310',
                        '6ammg00',
                        '55fc310',
                        '649dc10',
                        '64aa210',
                        '648f210',
                        '646a210',
                        '571b110',
                        '6481n10',
                        '6482410',
                        '649f410',
                        '647c310',
                        '6amg500',
                        '6486510',
                        '64cb410',
                        '6481h10',
                        '64caa10',
                        '55fb210',
                        '64cbj10',
                        '647b710',
                        '64ca220',
                        '6482b10',
                        '649hb10',
                        '646c610',
                        '649cl20',
                        '6483310',
                        '64cbi10',
                        '649f320',
                        '64bae10',
                        '6484910',
                        '64ca110',
                        '647b320',
                        '648f310',
                        '6482e10',
                        '64cae10',
                        '647b310',
                        '64baa10',
                        '64cbb10',
                        '54pg210',
                        '6481l10',
                        '57db110',
                        '648de10',
                        '646a110',
                        '647b620',
                        '649di10',
                        '6481m10',
                        '6481o10',
                        '6482d10',
                        '64adk10',
                        '649ha10',
                        '64cab10',
                        '648di10',
                        '6483120',
                        '649f910',
                        '647ba10',
                        '648d510',
                        '73v192u',
                        '6484610',
                        '647b330',
                        '648e510',
                        '64ba710',
                        '6482710',
                        '6ammj00',
                        '6489110',
                        '644ba10',
                        '649ge10',
                        '648f110',
                        '648eb10',
                        '57da110',
                        '648dg10',
                        '64cb310',
                        '6482610',
                        '648ec10',
                        '6482810',
                        '571e110',
                        '649f810',
                        '571f110',
                        '649g110',
                        '64ba410',
                        '6483210',
                        '55fc510',
                        '55fa210',
                        '6484g10',
                        '649cm10',
                        '6487410',
                        '6485910',
                        '6482210',
                        '6483510',
                        '6487310',
                        '648d610',
                        '64add10',
                        '6486410',
                        '6484e10',
                        '6484410',
                        '6485810',
                        '6484310',
                        '64ade10',
                        '6484h10',
                        '6489510',
                        '6483610',
                        '649cf20',
                        '64adh10',
                        '6488410',
                        '6484810']
        static_list = ['648b320',
                        '54pe110',
                        '54pe310',
                        '648a410',
                        '647c510',
                        '54pe210',
                        '648a610',
                        '648aa10',
                        '648b610',
                        '648a110',
                        '647c730',
                        '648ab10',
                        '645c510',
                        '649i110',
                        '6489f10',
                        '6486e10',
                        '6441510',
                        '6441410',
                        '648bd10',
                        '55ra110',
                        '6487910',
                        '648b820',
                        '648b910',
                        '6487810',
                        '648b810',
                        '647c610',
                        '645c610',
                        '645c620',
                        '6486d10',
                        '648a210',
                        '648bc10',
                        '648be10',
                        '54ac110',
                        '6487a10']
        if texture == 'mixed':
            return static_list + dyncon_list + dyndis_list
        elif texture == 'dyndis':
            return dyndis_list
        elif texture == 'dyncon':
            return dyncon_list
        elif texture == 'static':
            return static_list
        else:
            print('wrong texture name')
            return


class Sampler(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.len_datasets = np.array([len(dataset) for dataset in self.datasets])
        self.p_datasets = self.len_datasets / np.sum(self.len_datasets)

    def __getitem__(self, index):
        # first randomly sample a dataset
        if index < self.len_datasets[0]:
            return self.datasets[0].__getitem__(index)
        elif index < np.sum(self.len_datasets[:1]):
            return self.datasets[1].__getitem__(index-self.len_datasets[0])
        else:
            return self.datasets[2].__getitem__(index-int(np.sum(self.len_datasets[:1])))
        

    def __len__(self):
        return int(np.sum(self.len_datasets))