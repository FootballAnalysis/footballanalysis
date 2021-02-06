# two aligned dataset
from data.aligned_dataset import AlignedDataset
import random

class TwoAlignedDataset:
    def initialize(self, opt):
        assert opt.isTrain == True       
        # set different phases (folders of image)
        opt1 = opt
        opt1.phase = opt.phase1
        opt1.dataset_model = 'aligned'        
        self.dataset1 = AlignedDataset()
        self.dataset1.initialize(opt1)

        opt2 = opt
        opt2.phase = opt.phase2
        opt2.dataset_model = 'aligned'
        self.dataset2 = AlignedDataset()
        self.dataset2.initialize(opt2)

    def __getitem__(self, index):        
        # make crop and flip same in two datasets
        w = self.dataset1.opt.loadSize
        h = self.dataset1.opt.loadSize
        w_offset = random.randint(0, max(0, w - self.dataset1.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.dataset1.opt.fineSize - 1))
        is_flip = random.random() < 0.5        
        item1 = self.dataset1.get_item(index, w_offset, h_offset, is_flip)
        item2 = self.dataset2.get_item(index, w_offset, h_offset, is_flip)
        #item1 = self.dataset1[index]
        #item2 = self.dataset2[index]        
        return {'dataset1_input':item1, 'dataset2_input':item2}
       
       
    def __len__(self):
        assert(len(self.dataset1) == len(self.dataset2))
        return len(self.dataset1)
    
    def name(self):
        return 'TwoAlignedDataset'

