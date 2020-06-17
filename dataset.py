import torch
import torch.utils.data as data
import cv2

import pandas as pd
from torchvision.transforms import transforms

means = [0.5950, 0.5952, 0.5956]
stds = [0.2949, 0.2949, 0.2948]

class CTData():
    """This class provides that CT images to Dataloader
    """

    def __init__(self, type_data='train', transform=None, weights=None):
        """Initialize the dataset
        Args:
            type_data : whether to load the train or val data, or test data. Choose from 'train','test','val'
            transform : which transforms to apply
            weights (Tensor) : Give wieghted loss to postive class eg. `weights=torch.tensor([2.223, 22.22])` 
                Dimension should be equal to no. of classes.
        """
        self.data_frame = pd.read_csv('./data/{}.csv'.format(type_data))
        # self.transforms = transforms
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(means,stds)
        ])

        # Train without weights in baseline model
        # self.weights = weights

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.data_frame)

    def __getitem__(self, index):
        """Returns `(images,labels)` pair"""

        image_raw = cv2.imread("./data/images/" + self.data_frame['path'][index])
        image_raw = cv2.resize(image_raw,(224,224),interpolation=cv2.INTER_AREA)

        img_tensor = self.transforms(image_raw)
        label = self.data_frame['label'][index]
        if label == 1:
            label = torch.FloatTensor([1])
        else:
            label = torch.FloatTensor([0])

        return img_tensor, label




        
