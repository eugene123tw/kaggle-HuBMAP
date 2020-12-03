from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import rasterio

import pathlib

identity = rasterio.Affine(1, 0, 0, 0, 1, 0)


class HubDataset(Dataset):

    def __init__(self, root_dir, transform,
                 window=256, overlap=32, threshold=100):
        self.path = pathlib.Path(root_dir)
        self.overlap = overlap
        self.window = window
        self.transform = transform
        self.csv = pd.read_csv((self.path / 'train.csv').as_posix(),
                               index_col=[0])
        self.threshold = threshold

        self.x, self.y = [], []
        self.build_slices()
        self.len = len(self.x)
        self.as_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),
        ])

    def build_slices(self):
        self.masks = []
        self.files = []
        self.slices = []
        for i, filename in enumerate(self.csv.index.values):
            filepath = (self.path / 'train' / (filename + '.tiff')).as_posix()
            self.files.append(filepath)

            print('Transform', filename)
            with rasterio.open(filepath, transform=identity) as dataset:
                self.masks.append(rle_decode(self.csv.loc[filename, 'encoding'], dataset.shape))
                slices = make_grid(dataset.shape, window=self.window, min_overlap=self.overlap)

                for slc in tqdm(slices):
                    x1, x2, y1, y2 = slc
                    if self.masks[-1][x1:x2, y1:y2].sum() > self.threshold or np.random.randint(100) > 120:
                        self.slices.append([i, x1, x2, y1, y2])

                        image = dataset.read([1, 2, 3],
                                             window=Window.from_slices((x1, x2), (y1, y2)))

                        #                         if image.std().mean() < 10:
                        #                             continue

                        # print(image.std().mean(), self.masks[-1][x1:x2,y1:y2].sum())
                        image = np.moveaxis(image, 0, -1)
                        self.x.append(image)
                        self.y.append(self.masks[-1][x1:x2, y1:y2])

    # get data operation
    def __getitem__(self, index):
        image, mask = self.x[index], self.y[index]
        augments = self.transform(image=image, mask=mask)
        return self.as_tensor(augments['image']), augments['mask'][None]

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
