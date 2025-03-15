from datasets import load_dataset
from hydra.utils import instantiate
from torch.utils.data import DataLoader, SubsetRandomSampler
import os
import omegaconf

def build_data_loader(dataset_path, preprocess, batch_size, split: str = "train", debug: bool = os.getenv("DEBUG", "0") == "1"):
    preprocess = instantiate(preprocess) if isinstance(preprocess, omegaconf.DictConfig) else preprocess
    dataset = load_dataset(dataset_path, split=split)
    def transform(input):
        return {'image': [preprocess(image) for image in input['image']]}
    dataset.set_transform(transform)
    loader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=not debug, 
        sampler=SubsetRandomSampler(range(batch_size * 5)) if debug else None,
    )
    return loader   