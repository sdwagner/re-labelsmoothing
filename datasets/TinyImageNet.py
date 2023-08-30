import os
import csv
from PIL import Image
from torch.utils.data import Dataset

"""
Dataset for TinyImageNet
Used for loading training, validation and testing of the TinyImageNet dataset.
This requires the zip file to be unpacked in the chosen directory, which can be found here: 
"""
class TinyImageNet (Dataset):
    """
    TinyImageNet Dataset
    
    Args:
    directory (string): Contains the directory in which the zip has been unpacked.
    split (string, optional): Contains the prefered split, either train, test or val. 
        The default is train, any other then the above values leads to an error.
    transforms (callable, optional): Contains a transforms that given a PIL Image, returns
        a transformed image. This is used for data augmentation.
    """
    def __init__(self, directory, split = "train", transforms = None):
        
        self.data = []
        self.targets = []
        self.transforms = transforms
        
        # Loading the classes from wnids.txt for training and validation data
        wnids = os.path.join(directory, "tiny-imagenet-200", "wnids.txt")
        with open(wnids, "r") as file:
            classes = [line.strip()  for line in file]
        
        # Loading the training data
        if (split == "train"):
            
            train = os.path.join(directory, "tiny-imagenet-200", "train")
            
            # Foreach class folder
            for cur_class, img_class in enumerate(classes):
                directory = os.path.join(train, img_class, "images")
                images = os.listdir(directory)
                
                # Foreach image in the class folder
                for image in images:
                    if image.endswith(".JPEG"):
                        path = os.path.join(directory, image)
                        self.data.append(path)
                        self.targets.append(cur_class)
            
        # Loading the testing data (without any labels)
        elif (split == "test"):
            test = os.path.join(directory, "tiny-imagenet-200", "test", "images")
            
            # Foreach image in the folder
            for i in range(0, 10000):
                path = os.path.join(test, f"test_{i}.JPEG")
                self.data.append(path)
                self.targets.append(-1)
                
        # Loading the validation data 
        elif (split == "val"):
            
            # Loading validation annotaions
            val_annotations = os.path.join(directory, "tiny-imagenet-200", "val", "val_annotations.txt")
            val = os.path.join(directory, "tiny-imagenet-200", "val", "images")
            
            with open(val_annotations, "r") as file:
                reader = csv.reader(file, delimiter = "\t")

                # Foreach image entry in the validation annotaions
                for row in reader:
                    
                    path = os.path.join(val, row[0])
                    self.data.append(path)
                                    
                    cur_class = row[1]
                    
                    # Finding the corrosponding class
                    for i in range(len(classes)):
                        if (classes[i] == cur_class):
                            cur_class = i
                                    
                    self.targets.append(cur_class)

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