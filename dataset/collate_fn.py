from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import transforms

from dataset.transforms import RandomMixup, RandomCutmix


class MixupCutmixCollateFn(nn.Module):
    def __init__(self, num_classes):
        super(MixupCutmixCollateFn, self).__init__()
        self.mixupcutmix = transforms.RandomChoice([
            RandomMixup(num_classes=num_classes, p=1.0, alpha=0.2),
            RandomCutmix(num_classes=num_classes, p=1.0, alpha=1.0)
        ])

    def __call__(self, batch):
        return self.mixupcutmix(default_collate(batch))
