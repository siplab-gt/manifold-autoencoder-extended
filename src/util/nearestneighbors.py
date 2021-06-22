import torch
from torchvision.transforms import ToPILImage

class neighbors():

    def __init__(self, k):
        self.neighbors = None
        self.similarity = None
        self.target = None
        self.index = None
        self.k = k
        self.toimg = ToPILImage(mode='RGB')

    def update(self, images, simi, target,indices):
        #if self.neighbors is not None:
        #    print("Current neigh {}, simi {}".format(self.neighbors.shape, self.similarity.shape))
        #print("Images {}, simi {}".format(images.shape, simi.shape))
        image_all = images
        simi_all = simi
        target_all = target
        indices_all = indices
        if self.neighbors is not None:
            image_all = torch.cat((image_all, self.neighbors),  dim=0)
            simi_all = torch.cat((simi_all, self.similarity))
            target_all = torch.cat((target_all, self.target))
            indices_all = torch.cat((indices_all, self.index))

        _, sort_idx = torch.sort(simi_all, descending=True)


        self.neighbors = image_all[sort_idx][:self.k]
        self.similarity = simi_all[sort_idx][:self.k]
        self.target = target_all[sort_idx][:self.k]
        self.index = indices_all[sort_idx][:self.k]

    def get_neighbors(self):
        return self.neighbors

    def save(self, dir, img, tar, denormalize):
        img_ = self.toimg(denormalize(img.cpu().squeeze()))
        img_.save('{}/target_{}.png'.format(dir, tar))
        for i in range(self.neighbors.size(0)):
            ne_ = self.toimg(denormalize(self.neighbors[i].cpu().squeeze()))
            ne_.save('{}/neighbor{}_{}.png'.format(dir,i, self.target[i].item()))
