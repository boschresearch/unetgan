
from PIL import Image
import os
import pickle
import torch
import random
import torchvision

from torchvision.datasets.vision import VisionDataset
from torchvision import datasets, transforms, utils
import numpy as np
from matplotlib import pyplot as plt

class CocoAnimals(VisionDataset):
    def __init__(self, root=None, batch_size = 80, classes = None, transform=None, return_all=False , test_mode = False, imsize=128):
        self.num_classes = len(classes)
        if root==None:
            root = os.path.join(os.environ["SSD"],"animals")

        self.root = root
        self.return_all = return_all
        self.names = classes
        self.nclasses = len(classes)
        print(self.names)
        self.mask_trans =transforms.Compose(
                [ transforms.Resize(imsize),
                    transforms.CenterCrop(imsize),
                    transforms.ToTensor(),
                ])
        self.fixed_transform = transforms.Compose(
                [ transforms.Resize(imsize),
                    transforms.CenterCrop(imsize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

        self.image_files = [os.listdir(os.path.join(self.root, n )) for n in self.names]
        self.lenghts = [len(f) for f in self.image_files]

        print("images per class ", self.lenghts)
        self.transform = transform
        self.totensor = transforms.ToTensor()
        self.batch_size = batch_size

        self.k = int(batch_size/self.num_classes)
        self.length = sum([len(folder) for folder in self.image_files])

        all_files = [ [os.path.join(self.root, n , f ) for f in files] for n, files in zip(self.names,self.image_files)]
        self.all_files = []
        self.all_labels = []
        for i, f in enumerate(all_files):
            self.all_files += f
            self.all_labels += [i]*len(f)

        with open( os.path.join(root, "merged_bbdict_v2.p"), "rb") as h:
            self.bbox = pickle.load(h)


        with open(os.path.join(root,"coco_ann_dict.p"),"rb") as h:
            self.mask_coco = pickle.load(h)

        self.fixed_files = []
        self.fixed_impaths = []
        self.fixed_labels = []

        for _ in range(batch_size):
            id = np.random.randint(self.nclasses)
            file = random.choice(self.image_files[id])
            image_path = os.path.join(self.root, self.names[id], file)
            self.fixed_files.append(file)
            self.fixed_impaths.append(image_path)
            self.fixed_labels.append(id)
        self.fixed_labels = torch.tensor(self.fixed_labels).long().cuda()


    def __len__(self):
        # Here, we need to return the number of samples in this dataset.
        return self.length

    def fixed_batch(self, return_labels = False):
        if return_labels == True:
            images = torch.stack([self.random_batch(0,0,file=fi,image_path=im)[0].cuda() for fi,im in zip(self.fixed_files,self.fixed_impaths) ])
            labels = self.fixed_labels
            return images, labels
        else:
            return torch.stack([self.random_batch(0,0,file=fi,image_path=im)[0].cuda() for fi,im in zip(self.fixed_files,self.fixed_impaths) ])

    def single_batch(self):
        w = [self.__getitem__(np.random.randint(self.length)) for _ in range(self.batch_size)  ]
        x =  torch.stack([e[0].cuda() for e in w])
        y = torch.stack([e[1].long().cuda() for e in w])
        return x, y

    def random_batch(self,index, id=0, file=None, image_path=None):
        # this function adds some data augmentation by cropping and resizing the
        # images with the help of the bounding boxes. In particular we make sure
        # that the animal is still in the frame if we crop randomly and resize.

        if image_path==None:
            id = np.random.randint(self.nclasses)
            file = random.choice(self.image_files[id])
            image_path = os.path.join(self.root, self.names[id], file)

            #image_name = self.random.choice(self.image_files[random.choice(self.names)]["image_files"])
            fixed = False
            usebbx = True
        else:
            fixed = True
            usebbx = False

        img = Image.open(  image_path  ).convert('RGB')

        w,h = img.size
        im_id = file.strip(".jpg")
        if usebbx:
            if im_id.isdigit():
                origin = "coco"
                bbox = {"bbox":self.mask_coco[file]["bbox"], "label": self.mask_coco[file]["category_id"]}
            else:
                origin = "oi"
                bbox = self.bbox[file.strip(".jpg")]

            if len(bbox["bbox"])>0:
                usebbx = True
            else:
                usebbx = False

        if usebbx:
            if isinstance(bbox["bbox"][0], list):
                idx = random.choice(np.arange(len(bbox["bbox"])))
                bbox = bbox["bbox"][idx]  # choose a random bbox from the list
            else:
                bbox = bbox["bbox"]

        if usebbx:
            if origin == "coco":
                a = bbox[0]
                b = bbox[1]
                c = bbox[2]
                d = bbox[3]
            else:
                a = float(bbox[0])*w
                b = float(bbox[1])*w
                c = float(bbox[2])*h
                d = float(bbox[3])*h

                a, b, c, d = a, c, b-a, d-c

            eps = 0.0001
            longer_side  = max(h,d)
            r_max = min(float(longer_side)/(d+eps), float(longer_side)/(c+eps))
            r_min = 1.5

            if r_max > r_min and w > 200 and h > 200 and c*d > 30*30:
                r = 1 + np.random.rand()*(r_max-1)
                d_new = r*d
                c_new = r*c

                longer_side = min ( max(c_new  ,d_new ) , h, w)

                d_new = max(longer_side, 150)
                c_new = max(longer_side, 150)

                a_new = max(0, a - 0.5*(c_new - c) )
                b_new = max(0, b - 0.5*(d_new - d) )

                if c_new+a_new > w:
                    a_new = max(0,a_new - ((c_new+a_new)-w))
                if d_new+b_new>h:
                    b_new = max(0,b_new - ((d_new+b_new)-h))

                c_new = c_new + a_new
                d_new = d_new + b_new

                img  =  img.crop((a_new,b_new,c_new,d_new))

        idx = image_path

        if not fixed:
            img = self.transform(img)
        elif fixed:
            img = self.fixed_transform(img)

        label = torch.LongTensor([id])

        bbox = self.bbox[file.strip(".jpg")]
        out = (img, label , idx)

        return out

    def exact_batch(self,index):

        image_path = self.all_files[index]
        img = Image.open(  image_path ).convert('RGB')
        img = self.transform(img)
        id = self.all_labels[index]
        label = torch.LongTensor([id])
        return img, label , image_path

    def __getitem__(self,index):

        if self.return_all:
            return self.exact_batch(index)
        else:
            return self.random_batch(index)


class FFHQ(VisionDataset):

    def __init__(self, root, transform, batch_size = 60, test_mode = False, return_all = False, imsize=256):

        self.root = root
        self.transform = transform
        self.return_all = return_all

        print("root:",self.root)
        all_folders = os.listdir(self.root)

        self.length = sum([len(os.listdir(os.path.join(self.root,folder))) for folder in all_folders]) # = 70000
        self.fixed_transform = transforms.Compose(
                [ transforms.Resize(imsize),
                    transforms.CenterCrop(imsize),
                    #transforms.RandomHorizontalFlip(),
                    #transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

        self.fixed_indices = []

        for _ in range(batch_size):
            id = np.random.randint(self.length)
            self.fixed_indices.append(id)

    def __len__(self):
        return self.length


    def fixed_batch(self, random = False):
        if random == False:
            return torch.stack([self.random_batch(idx, True)[0].cuda() for idx in self.fixed_indices])
        else:
            return torch.stack([self.random_batch(np.random.randint(self.length), True)[0].cuda() for _ in range(len(self.fixed_indices))])

    def random_batch(self,index, fixed=False):

        folder = str(int(np.floor(index/1000)*1000)).zfill(5)
        file = str(index).zfill(5) + ".png"
        image_path = os.path.join(self.root, folder , file )
        img = Image.open( image_path).convert('RGB')
        if fixed:
            img = self.fixed_transform(img)
        else:
            img = self.transform(img)


        return img, torch.zeros(1).long(), image_path

    def __getitem__(self,index):

        if self.return_all:
            return self.exact_batch(index)
        else:
            return self.random_batch(index)


class Celeba(VisionDataset):

    def __init__(self, root, transform, batch_size = 60, test_mode = False, return_all = False, imsize=128):

        self.root = root
        self.transform = transform
        self.return_all = return_all
        all_files = os.listdir(self.root)
        self.length = len(all_files)
        self.fixed_transform = transforms.Compose(
                [ transforms.Resize(imsize),
                    transforms.CenterCrop(imsize),
                    #transforms.RandomHorizontalFlip(),
                    #transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

        self.fixed_indices = []

        for _ in range(batch_size):
            id = np.random.randint(self.length)
            self.fixed_indices.append(id)

    def __len__(self):
        return self.length


    def fixed_batch(self):
        return torch.stack([self.random_batch(idx, True)[0].cuda() for idx in self.fixed_indices])


    def random_batch(self,index, fixed=False):

        file = str(index+1).zfill(6) + '.png'
        image_path = os.path.join(self.root, file )
        img = Image.open( image_path).convert('RGB')
        if fixed:
            img = self.fixed_transform(img)
        else:
            img = self.transform(img)

        return img, torch.zeros(1).long(), image_path

    def __getitem__(self,index):

        if self.return_all:
            return self.exact_batch(index)
        else:
            return self.random_batch(index)
