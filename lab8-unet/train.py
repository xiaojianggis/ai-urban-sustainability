
## This is the script used to train the model
## by Xiaojiang Li, Penn, Dec 9, 2024

from torchvision.models import resnet50
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from PIL import Image
import torch
# import matplotlib.pyplot as plt
import time
import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from torch import nn
from skimage import io, transform
from skimage.transform import rotate, AffineTransform, warp
import rasterio as rio
from torchvision.transforms import Resize, CenterCrop, Normalize
import torchvision.transforms as transforms
from scipy import ndimage


class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        #parameters: in_channels, out_channels, kernel_size, padding
        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)
        
        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)
        
    # will be call when create instance
    def __call__(self, x): 
        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        
        return upconv1
    
    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )
        
        return contract
    
    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )
        return expand


class CustomizedDataset(Dataset):
    def __init__(self, img_dir, msk_dir, pytorch=True, transforms=None):
        super().__init__()
        
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(f, img_dir, msk_dir) for f in img_dir.iterdir() if not f.is_dir()]
        self.pytorch = pytorch
        self.transforms = transforms

        
    def combine_files(self, r_file: Path, img_dir, msk_dir):
        files = {'image': r_file, 
                 'mask': msk_dir/r_file.name.replace('naip_tiles', 'lu_masks')} #.replace('naip_tiles', 'lu_masks') is not necessary, just in case you have different name
        
        return files
        
    def __len__(self):
        return len(self.files)
    

    def open_as_array(self, idx, invert=False, include_nir=False, augment=False):
        img_pil = Image.open(self.files[idx]['image'])
        # augment the image
        if augment:
            img_pil = self.transforms(img_pil)
            
        raw_rgb = np.asarray(img_pil).astype(float)
#         src_raw_rgb = rio.open(self.files[idx]['image'])
#         raw_rgb = src_raw_rgb.read()
#         src_raw_rgb.close()
        
        raw_rgb = transform.resize(raw_rgb, (512, 512, 3))
        if invert:
            raw_rgb = raw_rgb.transpose((2,0,1))
        
        # normalize
        # return (raw_rgb / np.iinfo(raw_rgb.dtype).max)
        return (raw_rgb / 255.0)
        
        
    def open_mask(self, idx, add_dims=False, augment=False):
        mask_pil = Image.open(self.files[idx]['mask'])
        # augment the image
        if augment:
            mask_pil = self.transforms(mask_pil)

        raw_mask = np.array(mask_pil).astype(float)
#         print('The filename is:=======', self.files[idx]['mask'])
#         src_raw_mask = rio.open(self.files[idx]['mask'])
#         raw_mask = src_raw_mask.read()
#         src_raw_mask.close()
        
        raw_mask = transform.resize(raw_mask, (512, 512))
        raw_mask = np.where(raw_mask == 5, 1, 0)  #in the land use map of PHiladelphis, the tree is 1, 5 is for building
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask
        
        
    def __getitem__(self, idx):

        img_pil = Image.open(self.files[idx]['image'])
        mask_pil = Image.open(self.files[idx]['mask'])
        # mask_y = np.array(mask_pil).astype(float)

        ## just the colorization of the raw image, not need to change the mask
        if self.transforms:
            img_pil = self.transforms(img_pil)
            mask_pil = self.transforms(mask_pil)

        image_x = np.asarray(img_pil).astype(float)
        mask_y = np.array(mask_pil).astype(float)

        image_x = transform.resize(image_x, (512, 512, 3))
        mask_y = transform.resize(mask_y, (512, 512))

        image_x = image_x.transpose((2,0,1))/255.0
        mask_y = np.where(mask_y == 5, 1, 0)  #in the land use map of PHiladelphis, the tree is 1, 5 is for building

        # # 90 degree rotation
        # if np.random.rand()<0.5:
        #     angle = np.random.randint(4) * 90
        #     image_x = ndimage.rotate(image_x, angle,reshape=True)
        #     mask_y = ndimage.rotate(mask_y, angle, reshape=True)

        # # vertical flip
        # if np.random.rand()<0.5:
        #     image_x = np.flip(image_x, 0)
        #     mask_y = np.flip(mask_y, 0)
        
        # # horizonal flip
        # if np.random.rand()<0.5:
        #     image_x = np.flip(image_x, 1)
        #     mask_y = np.flip(mask_y, 1)

        ## add scale in future


        # x = torch.tensor(self.open_as_array(idx, \
        #                                     invert=self.pytorch, \
        #                                     include_nir=True, \
        #                                     augment=True), \
        #                                     dtype=torch.float32)
        # y = torch.tensor(self.open_mask(idx, add_dims=False, augment=True), \
        #                                 dtype=torch.torch.int64)
        
        # return x, y

        return torch.tensor(image_x, dtype=torch.float32), torch.tensor(mask_y, dtype=torch.int64)
    
    
    def open_as_pil(self, idx):
        arr = 256*self.open_as_array(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
    
    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())
        
        return s  
        

def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    # scaler = torch.cuda.amp.GradScaler()
    scaler = torch.amp.GradScaler('cuda')
    start = time.time()
    model.cuda()
    
    train_loss, valid_loss = [], []
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            step = 0

            # iterate over data
            for x, y in dataloader:
                #x = x.cuda()
                #y = y.cuda()
                
                x, y = x.to(device), y.to(device)
                
                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)

                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    # loss.backward()
                    # optimizer.step()
                    # # scheduler.step()
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    optimizer.step()
                    scaler.update()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())
                        
                # stats - whatever is the phase
                acc = acc_fn(outputs, y)
                
                running_acc  += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size 
                
                if step % 100 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc, torch.cuda.memory_allocated()/1024/1024))
                    # print(torch.cuda.memory_summary())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)
            
            # clear_output(wait=True)
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)
            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            print('-' * 10)

            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)
    

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    return train_loss, valid_loss    


def batch_to_img(xb, idx):
    img = np.array(xb[idx,0:3])
    return img.transpose((1,2,0))


def predb_to_mask(predb, idx):
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()



if __name__ == '__main__':
    # this is used to augment the image
    transform_aug = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize((300, 300)),
        # transforms.CenterCrop((100, 100)),
        # transforms.RandomCrop((80, 80)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-90, 90)),
        transforms.RandomVerticalFlip(p=0.5)
        # transforms.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = UNET(3,2).to(device)

    base_path = Path('../data/dataset/trainning')
    # data = CustomizedDataset(base_path/'imgs', 
    #                     base_path/'labels', 
    #                     transforms=transform_aug)

    data = CustomizedDataset(base_path/'imgs', 
                        base_path/'labels',
                        transforms=transform_aug)

    data_size = data.__len__()
    training_size = int(data_size*0.8)
    testing_size = data_size - training_size

    # train_ds, valid_ds = torch.utils.data.random_split(data, (2615, 872))
    train_ds, valid_ds = torch.utils.data.random_split(data, (training_size, testing_size))
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=4, shuffle=True)

    print('train_dl is:', len(train_dl))
    print('valid_dl is:', len(valid_dl))

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(unet.parameters(), lr=0.01)
    train_loss, valid_loss = train(unet, train_dl, valid_dl, loss_fn, opt, acc_metric, epochs=50)

    torch.save(unet.state_dict(), 'unet_build_model100epc_aug.pth')
