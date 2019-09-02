from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, num_workers, pin_memory, collate_fn=default_collate):
        
        self.shuffle = shuffle
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'pin_memory': pin_memory,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(**self.init_kwargs)
        