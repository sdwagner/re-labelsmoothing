import os
import csv
from PIL import Image
from torch.utils.data import Dataset

"""
Dataset for CUB_200_2011
Used for loading training and validation of the CUB_200_2011 dataset.
This requires the zip file to be unpacked in the chosen directory.
"""
class CUB_200_2011 (Dataset):
    """
    CUB_200_2011 Dataset
    
    Args:
    directory (string): Contains the directory in which the zip has been unpacked.
    split (string, optional): Contains the prefered split, either train or val. 
        The default is train, any other then the above values leads to an error.
    transforms (callable, optional): Contains a transforms that given a PIL Image, returns
        a transformed image. This is used for data augmentation.
    """
    def __init__(self, directory, split = "train", transforms = None):
        
        self.data = []
        self.targets = []
        self.transforms = transforms
        
        # Loading the training or validation data
        if (split == "train" or split == "val"):
            
            split = "0" if (split == "val") else "1"
            
            #Loading text files containing images paths and labels
            images_txt = os.path.join(directory, "CUB_200_2011", "images.txt")
            train_test_split_txt = os.path.join(directory, "CUB_200_2011", "train_test_split.txt")
            image_class_labels_txt = os.path.join(directory, "CUB_200_2011", "image_class_labels.txt")
            
            #Adding only correct images and labels
            with open(images_txt, "r") as paths, open(train_test_split_txt, "r") as train_test_split, open(image_class_labels_txt, "r") as classes:
                
                p = csv.reader(paths, delimiter=' ')
                sp = csv.reader(train_test_split, delimiter=' ')
                c = csv.reader(classes, delimiter=' ')
                
                for path, t, label in zip(p, sp, c):
                    if (t[1] == split):
                        self.data.append(os.path.join(directory, "CUB_200_2011", "images", path[1]))
                        self.targets.append(int(label[1]) - 1)
        
        # Error for any other split
        else:
            raise ValueError(f"The split can only be train, test or val, but {split} was given.")

    """
    Returns the total amount of images.
    """ 
    def __len__(self):
        return(len(self.data))
    
    """
    Returns the image and label at the given index. If a transform was previously given the image is transformed b it.

    Args:
    idx (int): Index of the image.
    """
    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert("RGB")
            
        if (self.transforms is not None):
            return self.transforms(img), self.targets[idx]
        return img, self.targets[idx]