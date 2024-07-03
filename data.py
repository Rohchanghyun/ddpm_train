from torchvision import transforms
from torch.utils.data import dataset, dataloader
from torchvision.datasets.folder import default_loader
from utils.RandomSampler import RandomSampler
from opt import opt
import os
import re

class Data():
    def __init__(self):
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=3),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=3),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.vis_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor()
        ])

        self.trainset = Market1501(self.train_transform, 'train', opt.data_path)
        self.testset = Market1501(self.test_transform, 'test', opt.data_path)
        self.queryset = Market1501(self.test_transform, 'query', opt.data_path)

        # 데이터셋이 올바르게 생성되었는지 확인
        assert len(self.trainset) > 0, "Trainset is empty"
        assert len(self.testset) > 0, "Testset is empty"
        assert len(self.queryset) > 0, "Queryset is empty"

        self.train_loader = dataloader.DataLoader(self.trainset,
                                                  sampler=RandomSampler(self.trainset, batch_id=opt.batchid,
                                                                        batch_image=opt.batchimage),
                                                  batch_size=opt.batchid * opt.batchimage, num_workers=8,
                                                  pin_memory=True)
        self.test_loader = dataloader.DataLoader(self.testset, batch_size=opt.batchtest, num_workers=8, pin_memory=True)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=opt.batchtest, num_workers=8,
                                                  pin_memory=True)

        if opt.mode == 'rank':
            self.query_image = self.test_transform(default_loader(opt.query_image))
            self.query_id = self.get_id(opt.query_image)
            
    def get_id(self,image_path):
        return Market1501.id(image_path)

class Market1501(dataset.Dataset):
    def __init__(self, transform, dtype, data_path):
        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path

        if dtype == 'train':
            self.data_path += '/bounding_box_train'
        elif dtype == 'test':
            self.data_path += '/bounding_box_test'
        else:
            self.data_path += '/query'

        # 데이터셋 경로가 올바른지 확인
        if not os.path.isdir(self.data_path):
            raise ValueError(f"Dataset directory {self.data_path} does not exist.")

        self.imgs = [path for path in self.list_pictures(self.data_path) if self.id(path) != -1]

        # 이미지 리스트가 비어있지 않은지 확인
        if not self.imgs:
            raise ValueError(f"No images found in {self.data_path}.")

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self._id2label[self.id(path)]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        return int(file_path.split('/')[-1].split('_')[1][1])

    @property
    def ids(self):
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        return sorted(set(self.ids))

    @property
    def cameras(self):
        return [self.camera(path) for path in self.imgs]

    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), f'Dataset directory does not exist: {directory}'

        return sorted([os.path.join(root, f)
                       for root, _, files in os.walk(directory) for f in files
                       if re.match(r'([\w]+\.(?:' + ext + '))', f)])
