from torch.utils.data import DataLoader
from nf_trainer.dataloaders.base import BaseDataLoader

__all__ = ['NodefluxDataLoader']

class NodefluxDataLoader(BaseDataLoader):
    def __init__(self, dataset, batch_size, shuffle, pin_memory, num_workers):
        super().__init__(dataset, batch_size, shuffle, pin_memory, num_workers)

        self.dataloader = DataLoader(
                dataset = dataset,
                batch_size = batch_size,
                shuffle = shuffle,
                pin_memory = pin_memory,
                num_workers = num_workers)



