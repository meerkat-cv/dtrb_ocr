import imgaug as ia
from imgaug import augmenters as iaa

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential([
    # execute 0 to 5 of the following (less important) augmenters per image
    # don't execute all of them, as that would often be way too strong
    sometimes(iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
        rotate=(-6, 6), # rotate by -45 to +45 degrees
        shear=(-10, 10), # shear by -16 to +16 degrees
        order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
        cval=(0, 255), # if mode is constant, use a cval between 0 and 255
        mode='constant' # use any of scikit-image's warping modes (see 2nd image from the top for examples)
    )),
    iaa.SomeOf((0, 5),
        [
            iaa.OneOf([
                iaa.GaussianBlur((0, 0.5)), # blur images with a sigma between 0 and 3.0
                iaa.AverageBlur(k=(2, 3)), # blur image using local means with kernel sizes between 2 and 7
                iaa.MedianBlur(k=(1, 3)), # blur image using local medians with kernel sizes between 2 and 7
            ]),
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.85, 1.15)), # sharpen images
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.001*255), per_channel=0.5), # add gaussian noise to images
            iaa.Dropout((0.0001, 0.001), per_channel=0.5), # randomly remove up to 10% of the pixels
            iaa.Invert(0.05, per_channel=True), # invert color channels
            iaa.Add((-4, 4), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
            iaa.AddToHueAndSaturation((-10, 10)), # change hue and saturation
            # either change the brightness of the whole image (sometimes
            # per channel) or change the brightness of subareas
            iaa.Multiply((0.85, 1.15), per_channel=0.5),
            iaa.ContrastNormalization((0.85, 1.25), per_channel=0.5), # improve or worsen the contrast
            iaa.Grayscale(alpha=(0.0, 1.0)),
            sometimes(iaa.ElasticTransformation(alpha=(0.8, 1.2), sigma=0.25)), # move pixels locally around (with random strengths)
        ],
        random_order=True
    )
])
