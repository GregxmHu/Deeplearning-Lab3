from typing import List, Tuple, Dict, Any
import torch
from typing import Any, Callable, List, Optional, Union, Tuple
from PIL import Image
from torch.utils.data import Dataset
import os
from tqdm import tqdm
# this is the class for dog-breed-recognition
class DogBreed(Dataset):
    def __init__(self,label_file:str,img_dir:str,
        transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
    ) -> None:
        skip=True
        self.classes=['affenpinscher',
 'afghan_hound',
 'african_hunting_dog',
 'airedale',
 'american_staffordshire_terrier',
 'appenzeller',
 'australian_terrier',
 'basenji',
 'basset',
 'beagle',
 'bedlington_terrier',
 'bernese_mountain_dog',
 'black-and-tan_coonhound',
 'blenheim_spaniel',
 'bloodhound',
 'bluetick',
 'border_collie',
 'border_terrier',
 'borzoi',
 'boston_bull',
 'bouvier_des_flandres',
 'boxer',
 'brabancon_griffon',
 'briard',
 'brittany_spaniel',
 'bull_mastiff',
 'cairn',
 'cardigan',
 'chesapeake_bay_retriever',
 'chihuahua',
 'chow',
 'clumber',
 'cocker_spaniel',
 'collie',
 'curly-coated_retriever',
 'dandie_dinmont',
 'dhole',
 'dingo',
 'doberman',
 'english_foxhound',
 'english_setter',
 'english_springer',
 'entlebucher',
 'eskimo_dog',
 'flat-coated_retriever',
 'french_bulldog',
 'german_shepherd',
 'german_short-haired_pointer',
 'giant_schnauzer',
 'golden_retriever',
 'gordon_setter',
 'great_dane',
 'great_pyrenees',
 'greater_swiss_mountain_dog',
 'groenendael',
 'ibizan_hound',
 'irish_setter',
 'irish_terrier',
 'irish_water_spaniel',
 'irish_wolfhound',
 'italian_greyhound',
 'japanese_spaniel',
 'keeshond',
 'kelpie',
 'kerry_blue_terrier',
 'komondor',
 'kuvasz',
 'labrador_retriever',
 'lakeland_terrier',
 'leonberg',
 'lhasa',
 'malamute',
 'malinois',
 'maltese_dog',
 'mexican_hairless',
 'miniature_pinscher',
 'miniature_poodle',
 'miniature_schnauzer',
 'newfoundland',
 'norfolk_terrier',
 'norwegian_elkhound',
 'norwich_terrier',
 'old_english_sheepdog',
 'otterhound',
 'papillon',
 'pekinese',
 'pembroke',
 'pomeranian',
 'pug',
 'redbone',
 'rhodesian_ridgeback',
 'rottweiler',
 'saint_bernard',
 'saluki',
 'samoyed',
 'schipperke',
 'scotch_terrier',
 'scottish_deerhound',
 'sealyham_terrier',
 'shetland_sheepdog',
 'shih-tzu',
 'siberian_husky',
 'silky_terrier',
 'soft-coated_wheaten_terrier',
 'staffordshire_bullterrier',
 'standard_poodle',
 'standard_schnauzer',
 'sussex_spaniel',
 'tibetan_mastiff',
 'tibetan_terrier',
 'toy_poodle',
 'toy_terrier',
 'vizsla',
 'walker_hound',
 'weimaraner',
 'welsh_springer_spaniel',
 'west_highland_white_terrier',
 'whippet',
 'wire-haired_fox_terrier',
 'yorkshire_terrier']
        self.label_map={}
        with open(label_file,'r') as f:
            for item in tqdm(f,desc="creating img-label mapping..."):
                if skip:
                    skip=False
                    continue
                img_index,img_label=item.strip('\n').split(',')
                #if img_label not in self.classes:
                #    self.classes.append(img_label)
                self.label_map[img_index]=self.classes.index(img_label)
        #print(len(self.classes))
        self.img_dir=img_dir
        self.img_indexes=os.listdir(self.img_dir)
        self.img_indexes=[item.strip('.jpg') for item in self.img_indexes]
        self.transform=transform
        self.label_transform=label_transform


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_index=self.img_indexes[index]
        img_label=self.label_map[img_index]
        img = Image.open(
            os.path.join(
                self.img_dir,
                "{}.jpg".format(img_index)
            )
        )
        if self.transform is not None:
            img = self.transform(img)
        if self.label_transform is not None:
            img_label = self.label_transform(img_label)

        
        return img,img_label


    def __len__(self) -> int:
        return len(self.img_indexes)


    def collate(self, batch):
        imgs=torch.stack(
            [item[0] for item in batch]
        )
        labels=torch.stack(
            [item[1] for item in batch]
        )
        return imgs,labels
