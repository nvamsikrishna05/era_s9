import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms():
    """Gets instance of train and test transforms"""

    train_transform = A.Compose([
        A.HorizontalFlip(),
        A.ShiftScaleRotate(rotate_limit=10),
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=0.4468, mask_fill_value=None),
        A.Normalize(mean=(0.4915, 0.4823, .4468), std=(0.2470, 0.2435, 0.2616)),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Normalize(mean=(0.4915, 0.4823, .4468), std=(0.2470, 0.2435, 0.2616)),
        ToTensorV2()
    ])

    return train_transform, test_transform