from torch.utils.data import Dataset


import numpy as np

class simple_np_ds(Dataset):
    """
    Dataset encapsulating simple numpy dataset
    """


    def __init__(self, path, val_split, test_split, shuffle=True, transform=None, reduced_dataset_size=None, seed=42):
        
        
        self.f=np.load(path)
        data_l = self.f["data"]
        labels_l=self.f["labels"]
        
        self.data = data_l.astype(np.float32)
        self.labels = labels_l.astype(np.int64)
        
        assert self.data.shape[0] == self.labels.shape[0]
        
        self.transform=transform
        
        self.reduced_size = reduced_dataset_size
        
        #save prng state
        rstate=np.random.get_state()
        if seed is not None:
            np.random.seed(seed)

        indices = np.arange(len(self.labels))

        if self.reduced_size is not None:
            print("Reduced size: {}".format(self.reduced_size))
            assert len(indices)>=self.reduced_size
            indices = np.random.choice(self.labels.shape[0], reduced_dataset_size)

        #shuffle index array
        if shuffle:
            np.random.shuffle(indices)
        
        #restore the prng state
        if seed is not None:
            np.random.set_state(rstate)

        n_val = int(len(indices) * val_split)
        n_test = int(len(indices) * test_split)
        self.train_indices = indices[:-n_val-n_test]
        self.val_indices = indices[-n_test-n_val:-n_test]
        self.test_indices = indices[-n_test:]
        
        
    def __getitem__(self,index):
        if self.transform is None:
            return self.data[index,:],  self.labels[index]
        else:
            return self.transform(self.data[index,:]),  self.labels[index]



    def __len__(self):
        if self.reduced_size is None:
            return self.labels.shape[0]
        else:
            return self.reduced_size


    def __del__(self):
        self.f.close()
