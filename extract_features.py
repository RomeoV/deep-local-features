import argparse

import numpy as np

import imageio

import torch

from tqdm import tqdm

import scipy
import scipy.io
import scipy.misc

#from lib.model_test import D2Net
from externals.d2net.lib.utils import preprocess_image
#from lib.pyramid import process_multiscale

# added
from lib.feature_extractor import extraction_model as em
from lib import autoencoder, attention_model
#from externals.d2net.lib import localization, utils



# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Argument parsing
parser = argparse.ArgumentParser(description='Feature extraction script')

parser.add_argument(
    '--image_list_file', type=str, default='hpatches_sequences/image_list_hpatches_sequences.txt',
    help='path to a file containing a list of images to process'
)

parser.add_argument(
    '--preprocessing', type=str, default='caffe',
    help='image preprocessing (caffe or torch)'
)
parser.add_argument(
    '--model_file', type=str, default='models/d2_tf.pth',
    help='path to the full model'
)

parser.add_argument(
    '--max_edge', type=int, default=1600,
    help='maximum image size at network input'
)
parser.add_argument(
    '--max_sum_edges', type=int, default=2800,
    help='maximum sum of image sizes at network input'
)

parser.add_argument(
    '--output_extension', type=str, default='.our-model',
    help='extension for the output'
)
parser.add_argument(
    '--output_type', type=str, default='npz',
    help='output file type (npz or mat)'
)

parser.add_argument(
    '--multiscale', dest='multiscale', action='store_true',
    help='extract multiscale features'
)
parser.set_defaults(multiscale=False)

parser.add_argument(
    '--no-relu', dest='use_relu', action='store_false',
    help='remove ReLU after the dense feature extraction module'
)
parser.set_defaults(use_relu=True)

args = parser.parse_args()

print(args)

# # Creating CNN model
# model = D2Net(
#     model_file=args.model_file,
#     use_relu=args.use_relu,
#     use_cuda=use_cuda
# )
encoder = autoencoder.FeatureEncoder1.load_from_checkpoint('/home/witi/Downloads/autoenc.ckpt', load_tf_weights=False).eval()
attention = attention_model.MultiAttentionLayer.load_from_checkpoint('/home/witi/Downloads/attention.ckpt', feature_encoder=encoder).eval()

extraction_model = em.ExtractionModel(attention, max_features=None,  use_d2net_detection=False, num_upsampling=3, thresh=0.15)

# Process the file
with open(args.image_list_file, 'r') as f:
    lines = f.readlines()
for line in tqdm(lines, total=len(lines)):
    path = line.strip()

    image = imageio.imread(path)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
    resized_image = image
    if max(resized_image.shape) > args.max_edge:
        resized_image = scipy.misc.imresize(
            resized_image,
            args.max_edge / max(resized_image.shape)
        ).astype('float')
    if sum(resized_image.shape[: 2]) > args.max_sum_edges:
        resized_image = scipy.misc.imresize(
            resized_image,
            args.max_sum_edges / sum(resized_image.shape[: 2])
        ).astype('float')

    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    input_image = preprocess_image(
        resized_image,
        preprocessing=args.preprocessing
    )
    with torch.no_grad():
        #if args.multiscale:

        print("SHAPE", input_image.shape)

        print("SHAPE2", input_image[np.newaxis, :, :, :].astype(np.float32).shape)
        im = torch.tensor(input_image[np.newaxis, :, :, :].astype(np.float32))

        #keypoints, scores, descriptors, _ = extraction_model(im)
        keypoints, descriptors, scores, _ = extraction_model(im)

    print("KP", keypoints.shape)
    print("DESC",descriptors.shape)
    print("SCOR", scores.shape)

    # Input image coordinates
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    # i, j -> u, v
    #TODO Find out what the following line is for
    #keypoints = keypoints[:, [1, 0, 2]]

    if args.output_type == 'npz':
        with open(path + args.output_extension, 'wb') as output_file:
            np.savez(
                output_file,
                keypoints=keypoints,
                scores=scores,
                descriptors=descriptors
            )
    elif args.output_type == 'mat':
        with open(path + args.output_extension, 'wb') as output_file:
            scipy.io.savemat(
                output_file,
                {
                    'keypoints': keypoints,
                    'scores': scores,
                    'descriptors': descriptors
                }
            )
    else:
        raise ValueError('Unknown output type.')
