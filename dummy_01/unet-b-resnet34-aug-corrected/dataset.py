from ..common import *
from ..hubmap_v2 import *

#############################################################################

'''
train-0
valid-0
test-all
'''


def make_image_id(mode):
    train_image_id = {
        0: '2f6ecfcdf',
        1: 'aaa6a05cc',
        2: 'cb2d976f4',
        3: '0486052bb',
        4: 'e79de561c',
        5: '095bf7a1f',
        6: '54f2eec69',
        7: '1e2425f28',
    }
    test_image_id = {
        0: 'b9a3865fc',
        1: 'b2dc8411c',
        2: '26dc41664',
        3: 'c68fe75ea',
        4: 'afa5e8098',
    }
    if 'pseudo-all' == mode:
        test_id = [test_image_id[i] for i in [0, 1, 2, 3, 4]]
        return test_id

    if 'test-all' == mode:
        test_id = [test_image_id[i] for i in [0, 1, 2, 3, 4]]  # list(test_image_id.values()) #
        return test_id

    if 'train-all' == mode:
        train_id = [train_image_id[i] for i in [0, 1, 2, 3, 4, 5, 6, 7]]  # list(test_image_id.values()) #
        return train_id

    if 'valid' in mode or 'train' in mode:
        fold = int(mode[-1])
        valid = [fold, ]
        train = list({0, 1, 2, 3, 4, 5, 6, 7} - {fold, })
        valid_id = [train_image_id[i] for i in valid]
        train_id = [train_image_id[i] for i in train]

        if 'valid' in mode: return valid_id
        if 'train' in mode: return train_id


class HuDataset(Dataset):
    # /0.25_320_192_train
    def __init__(self, image_id, image_dir, augment=None):
        self.augment = augment
        self.image_id = image_id
        self.image_dir = image_dir

        tile_id = []
        for i in range(len(image_dir)):
            for id in image_id[i]:
                df = pd.read_csv(data_dir + '/etc/tile/%s/%s.csv' % (self.image_dir[i], id))
                tile_id += ('%s/%s/' % (self.image_dir[i], id) + df.tile_id).tolist()

        self.tile_id = tile_id
        self.len = len(self.tile_id)

    def __len__(self):
        return self.len

    def __str__(self):
        string = ''
        string += '\tlen  = %d\n' % len(self)
        string += '\timage_dir = %s\n' % self.image_dir
        string += '\timage_id  = %s\n' % str(self.image_id)
        string += '\t          = %d\n' % sum(len(i) for i in self.image_id)
        return string

    def __getitem__(self, index):
        id = self.tile_id[index]
        image = cv2.imread(data_dir + '/etc/tile/%s.png' % (id), cv2.IMREAD_COLOR)
        mask = cv2.imread(data_dir + '/etc/tile/%s.mask.png' % (id), cv2.IMREAD_GRAYSCALE)
        # print(data_dir + '/tile/%s/%s.png'%(self.image_dir,id))

        image = image.astype(np.float32) / 255
        mask = mask.astype(np.float32) / 255
        r = {
            'index': index,
            'tile_id': id,
            'mask': mask,
            'image': image,
        }
        if self.augment is not None: r = self.augment(r)
        return r


def null_collate(batch):
    batch_size = len(batch)
    index = []
    mask = []
    image = []
    for r in batch:
        index.append(r['index'])
        mask.append(r['mask'])
        image.append(r['image'])

    image = np.stack(image)
    image = image[..., ::-1]
    image = image.transpose(0, 3, 1, 2)
    image = np.ascontiguousarray(image)

    mask = np.stack(mask)
    mask = np.ascontiguousarray(mask)

    # ---
    image = torch.from_numpy(image).contiguous().float()
    mask = torch.from_numpy(mask).contiguous().unsqueeze(1)
    mask = (mask > 0.5).float()

    return {
        'index': index,
        'mask': mask,
        'image': image,
    }


## augmentation ######################################################################
# flip
def do_random_flip_transpose(image, mask):
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    if np.random.rand() > 0.5:
        image = image.transpose(1, 0, 2)
        mask = mask.transpose(1, 0)

    image = np.ascontiguousarray(image)
    mask = np.ascontiguousarray(mask)
    return image, mask


# geometric
def do_random_crop(image, mask, size):
    height, width = image.shape[:2]
    x = np.random.choice(width - size)
    y = np.random.choice(height - size)
    image = image[y:y + size, x:x + size]
    mask = mask[y:y + size, x:x + size]
    return image, mask


def do_random_scale_crop(image, mask, size, mag):
    height, width = image.shape[:2]

    s = 1 + np.random.uniform(-1, 1) * mag
    s = int(s * size)

    x = np.random.choice(width - s)
    y = np.random.choice(height - s)
    image = image[y:y + s, x:x + s]
    mask = mask[y:y + s, x:x + s]
    if s != size:
        image = cv2.resize(image, dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(size, size), interpolation=cv2.INTER_LINEAR)
    return image, mask


def do_random_rotate_crop(image, mask, size, mag=30):
    angle = 1 + np.random.uniform(-1, 1) * mag

    height, width = image.shape[:2]
    dst = np.array([
        [0, 0], [size, size], [size, 0], [0, size],
    ])

    c = np.cos(angle / 180 * 2 * PI)
    s = np.sin(angle / 180 * 2 * PI)
    src = (dst - size // 2) @ np.array([[c, -s], [s, c]]).T
    src[:, 0] -= src[:, 0].min()
    src[:, 1] -= src[:, 1].min()

    src[:, 0] = src[:, 0] + np.random.uniform(0, width - src[:, 0].max())
    src[:, 1] = src[:, 1] + np.random.uniform(0, height - src[:, 1].max())


    transform = cv2.getAffineTransform(src[:3].astype(np.float32), dst[:3].astype(np.float32))
    image = cv2.warpAffine(image, transform, (size, size), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpAffine(mask, transform, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask


# warp/elastic deform ...
# <todo>

# noise
def do_random_noise(image, mask, mag=0.1):
    height, width = image.shape[:2]
    noise = np.random.uniform(-1, 1, (height, width, 1)) * mag
    image = image + noise
    image = np.clip(image, 0, 1)
    return image, mask


# intensity
def do_random_contast(image, mask, mag=0.3):
    alpha = 1 + random.uniform(-1, 1) * mag
    image = image * alpha
    image = np.clip(image, 0, 1)
    return image, mask


def do_random_gain(image, mask, mag=0.3):
    alpha = 1 + random.uniform(-1, 1) * mag
    image = image ** alpha
    image = np.clip(image, 0, 1)
    return image, mask


def do_random_hsv(image, mask, mag=[0.15, 0.25, 0.25]):
    image = (image * 255).astype(np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0].astype(np.float32)  # hue
    s = hsv[:, :, 1].astype(np.float32)  # saturation
    v = hsv[:, :, 2].astype(np.float32)  # value
    h = (h * (1 + random.uniform(-1, 1) * mag[0])) % 180
    s = s * (1 + random.uniform(-1, 1) * mag[1])
    v = v * (1 + random.uniform(-1, 1) * mag[2])

    hsv[:, :, 0] = np.clip(h, 0, 180).astype(np.uint8)
    hsv[:, :, 1] = np.clip(s, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(v, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    image = image.astype(np.float32) / 255
    return image, mask


# shuffle block, etc
# <todo>


# post process ---
# https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv/42812226

# min_radius = 50
# min_area = 7853
#
#
def filter_small(mask, min_size):
    m = (mask * 255).astype(np.uint8)

    num_comp, comp, stat, centroid = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_comp == 1: return mask

    filtered = np.zeros(comp.shape, dtype=np.uint8)
    area = stat[:, -1]
    for i in range(1, num_comp):
        if area[i] >= min_size:
            filtered[comp == i] = 255
    return filtered


######################################################################################

def run_check_dataset():
    dataset = HuDataset(
        image_id=[
            make_image_id('valid-0'),
        ],
        image_dir=[
            '0.25_480_240_train',
        ]
    )
    print(dataset)

    for i in range(1000):
        i = np.random.choice(len(dataset))  # 98 #
        r = dataset[i]

        print(r['index'])
        print(r['tile_id'])
        print(r['image'].shape)
        print(r['mask'].shape)
        print('')

        filtered = filter_small(r['mask'], min_size=800 * 0.25)

        image_show_norm('image', r['image'], min=0, max=1)
        image_show_norm('mask', r['mask'], min=0, max=1)
        image_show('filtered', filtered)
        cv2.waitKey(0)
        # exit(0)


def run_check_augment():
    def augment(image, mask):
        # image, mask = do_random_crop(image, mask, size=320)
        # image, mask = do_random_scale_crop(image, mask, size=320, mag=0.1)
        # image, mask = do_random_rotate_crop(image, mask, size=320, mag=30 )
        # image, mask = do_random_contast(image, mask, mag=0.8 )
        image, mask = do_random_hsv(image, mask, mag=[0.1, 0.2, 0])
        image, mask = do_random_gain(image, mask, mag=0.8)
        # image, mask = do_random_noise(image, mask, mag=0.1)

        return image, mask

    dataset = HuDataset(
        image_id=[make_image_id('train-0')],
        image_dir=['0.25_480_240_train'],
    )  # '0.25_320_192_train'
    print(dataset)

    for i in range(1000):
        # for i in np.random.choice(len(dataset),100):
        r = dataset[i]
        image = r['image']
        mask = r['mask']

        print('%2d --------------------------- ' % (i))
        overlay = np.hstack([image, np.tile(mask.reshape(*image.shape[:2], 1), (1, 1, 3)), ])
        image_show_norm('overlay', overlay, min=0, max=1)
        cv2.waitKey(1)

        if 1:
            for i in range(100):
                image1, mask1 = augment(image.copy(), mask.copy())
                overlay1 = np.hstack([image1, np.tile(mask1.reshape(*image1.shape[:2], 1), (1, 1, 3)), ])
                image_show_norm('overlay1', overlay1, min=0, max=1)
                cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    # run_check_dataset()
    run_check_augment()
