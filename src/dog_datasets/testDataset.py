from typing import List, Tuple, Dict, Any
import torch
from typing import Any, Callable, List, Optional, Union, Tuple
from PIL import Image
from torch.utils.data import Dataset
import os
from tqdm import tqdm
# this is the class for dog-breed-recognition
class TestDataSet(Dataset):
    def __init__(self,img_dir:str,
        transform: Optional[Callable] = None
    ) -> None:
        self.img_dir=img_dir     
        self.img_indexes=os.listdir(self.img_dir)
        self.transform=transform


    def __getitem__(self, index: int) -> Any:
        img_index=self.img_indexes[index]
        img = Image.open(
            os.path.join(
                self.img_dir,
                img_index,
            )
        )
        if self.transform is not None:
            img = self.transform(img)  
        return img_index,img


    def __len__(self) -> int:
        return len(self.img_indexes)


    def collate(self, batch):
        indexes=[item[0].strip('.jpg') for item in batch]
        imgs=torch.stack(
            [item[1] for item in batch]
        )
        return indexes,imgs