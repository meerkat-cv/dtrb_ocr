import imgaug as ia
from imgaug import augmenters as iaa

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# execute 0 to 5 of the following (less important) augmenters per image
# don't execute all of them, as that would often be way too strong
im_filters = iaa.SomeOf((0, 5),
    [
        iaa.OneOf([
            iaa.GaussianBlur((0, 1.5)), # blur images with a sigma between 0 and 3.0
            iaa.AverageBlur(k=(2, 3)), # blur image using local means with kernel sizes between 2 and 7
            iaa.MedianBlur(k=(3, 3)), # blur image using local medians with kernel sizes between 2 and 7
        ]),
        iaa.Sharpen(alpha=(0, 0.1), lightness=(0.75, 1.5)), # sharpen images
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
        iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
        iaa.Add((-20, 20)), # change brightness of images (by -10 to 10 of original value)
        iaa.OneOf([
            iaa.ContrastNormalization((0.75, 1.75), per_channel=0.5), # improve or worsen the contrast
            iaa.ContrastNormalization((0.5, 1.5)), # improve or worsen the contrast
        ]),
        iaa.Grayscale(alpha=(0.0, 1.0)),
    ],
    random_order=True
)

seq = sometimes(
    iaa.Sequential(
        [
            sometimes(iaa.Affine(
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.025, 0.025), "y": (-0.05, 0.025)}, # translate by -20 to +20 percent (per axis)
                #       rotate=(0, 0), # rotate by -45 to +45 degrees
            shear=(-6, 6), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode='edge' # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            #   sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
        im_filters
        ],
        random_order=True
    )
)
