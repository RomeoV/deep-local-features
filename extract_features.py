import argparse
from threading import active_count

import numpy as np

import imageio

import torch

from tqdm import tqdm

import scipy
import scipy.io
import scipy.misc

# from lib.model_test import D2Net
from externals.d2net.lib.utils import preprocess_image
# from lib.pyramid import process_multiscale

# added
from lib.feature_extractor import extraction_model as em
from lib import autoencoder, attention_model
from lib import load_checkpoint
from lib.train_shared_fe64 import CorrespondenceEncoder
#from pqdm.threads import pqdm
import os
# from externals.d2net.lib import localization, utils


# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Argument parsing
parser = argparse.ArgumentParser(description='Feature extraction script')

parser.add_argument(
    '--encoder_model', type=str, default='FeatureEncoder64Up',
    help='encoder model name'
)
parser.add_argument(
    '--attention_model', type=str, default='auto',
    help='attention model name'
)
parser.add_argument(
    '--encoder_ckpt', type=str, default='correspondence_encoder_lr1e3',
    help='path to encoder checkpoint'
)
parser.add_argument(
    '--attention_ckpt', type=str, default='cfe64_multi_attention_model2_distinctiveness+_lossN64_lambda1',
    help='path to attention checkpoint'
)
parser.add_argument(
    '--output_extension', type=str, default='.our-model',
    help='extension for the output. Same name must be added to hpatches_sequences/HPatches-Sequences-Matching-Benchmark.ipynb'
)
parser.add_argument(
    '--d2net_localization', action='store_true', default=False, help='Help whether or not to extract feature as in the D2Net paper'
)
parser.add_argument(
    '--nogpu', action='store_true', default=False, help="Whether or not to use gpu"
)
parser.add_argument(
    '--scale', action='store_true', default=False, help="Preprocess and scale large images"
)
parser.add_argument(
    '--append', action='store_true', default=False, help="Append ones to descriptors"
)
parser.add_argument(
    '--load_from_folder', action='store_true', default=False, help='load checkpoints from ./checkpoints/'
)
parser.add_argument(
    '--smart_name', action='store_true', default=False, help='override args.output_extension with smart name and write to checkpoints/extensions.txt'
)
parser.add_argument(
    '--shared', action='store_true', default=False, help='encoder weights are contained in the attention weights'
)
parser.add_argument('--thresh', type=float, default=0.2,
                    help="Threshold for detection")
parser.add_argument('--replace_strides', action='store_true', default=False,
                    help="Whether or not to replace strides with dilated convolution")
parser.add_argument('--first_stride', type=int, default=1,
                    help="Replace the first stride in the network (this is not affected by replace strides)")
parser.add_argument('--nouse_nms', action='store_true',
                    default=False, help="Disable NMS")

parser.add_argument(
    '--image_list_file', type=str, default='hpatches_sequences/image_list_hpatches_sequences.txt',
    help='path to a file containing a list of images to process'
)

parser.add_argument(
    '--preprocessing', type=str, default='torch',
    help='image preprocessing (caffe or torch)')

parser.add_argument(
    '--max_edge', type=int, default=1600,
    help='maximum image size at network input'
)
parser.add_argument(
    '--max_sum_edges', type=int, default=2800,
    help='maximum sum of image sizes at network input'
)

parser.add_argument(
    '--output_type', type=str, default='npz',
    help='output file type (npz or mat)'
)

parser.add_argument("--sum_descriptors", action='store_true', default=False, help="Wether or not to sum descriptors of different scales")

args = parser.parse_args()

print(args)

if args.smart_name:
    name = "." + args.attention_ckpt + "_" + args.encoder_ckpt + "_"
    if args.encoder_model == "auto": name+= "CORMODEL"
    name += "_NOMAXF"
    if args.scale:
        name += "_SCALE"
    args.output_extension = name
    f = open("checkpoints/extensions.txt", "a")
    f.write(name + '\n')
    f.close()
    print('Extension: ' + name)

if args.replace_strides:
    # (True, True, True) # Default: (False, False, False)
    replace_stride_with_dilation = (True, True, True)
    num_upsampling_extraction = 2
    no_upsampling = True  # Set to true when replacing strides
else:
    replace_stride_with_dilation = (False, False, False)
    num_upsampling_extraction = 3
    no_upsampling = False

if args.first_stride == 1:
    num_upsampling_extraction -= 1

#num_upsampling_extraction += 1

if args.load_from_folder:
    encoder_ckpt = 'checkpoints/' + args.encoder_ckpt + '.ckpt'
else:
    encoder_ckpt = load_checkpoint.get_attention_ckpt(args.encoder_ckpt)

exec('EncoderModule = autoencoder.' + args.encoder_model)

if args.shared:
    encoder = EncoderModule.load_from_checkpoint(encoder_ckpt,
                                                 no_upsampling=no_upsampling,
                                                 replace_stride_with_dilation=replace_stride_with_dilation,
                                                 first_stride=args.first_stride,
                                                 load_tf_weights=False).eval()
else:
    encoder = EncoderModule(no_upsampling=no_upsampling,
                            replace_stride_with_dilation=replace_stride_with_dilation,
                            first_stride=args.first_stride,
                            load_tf_weights=False)

if args.load_from_folder:
    attention_ckpt = 'checkpoints/' + args.attention_ckpt + '.ckpt'
else:
    attention_ckpt = load_checkpoint.get_attention_ckpt(args.attention_ckpt)

if args.attention_model == 'auto':
    if 'multi_attention_model2' in args.attention_ckpt:
        args.attention_model = 'MultiAttentionLayer2'
    elif 'multi_attention_model' in args.attention_ckpt:
        args.attention_model = 'MultiAttentionLayer'
    elif 'attention_model' in args.attention_ckpt:
        args.attention_model = 'AttentionLayer2'
    else:
        raise Exception("Can't find matching attention model. Set --attention_model manually")
    print ("USING ATTENTION MODEL: " + args.attention_model)


exec('AttentionModule = attention_model.' + args.attention_model)
attention = AttentionModule.load_from_checkpoint(
    attention_ckpt, feature_encoder=encoder).eval()


extraction_model = em.ExtractionModel(
    attention,
    max_features=None,
    use_d2net_detection=args.d2net_localization,
    num_upsampling=num_upsampling_extraction,
    thresh=args.thresh,
    use_nms=not args.nouse_nms,
    sum_descriptors=args.sum_descriptors)

if not args.nogpu:
    extraction_model = extraction_model.cuda()

# Process the file
with open(args.image_list_file, 'r') as f:
    lines = f.readlines()

for l in lines:
    l = os.path.join(os.path.dirname(os.path.realpath(__file__)), l.strip())
    if not os.path.isfile(l.strip()):
        print(f"Failed to find {l.strip()}")
        raise Exception("File not found")


for line in tqdm(lines, total=len(lines)):
    path = line.strip()
    image = imageio.imread(path)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
    resized_image = image
    if args.scale:
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
        im = torch.tensor(
            input_image[np.newaxis, :, :, :].astype(np.float32))
        if not args.nogpu:
            im = im.cuda()

        # keypoints, scores, descriptors, _ = extraction_model(im)
        keypoints, descriptors, scores, _ = extraction_model(im)

    # # Input image coordinates
    if args.scale:
        keypoints[:, 0] *= fact_i
        keypoints[:, 1] *= fact_j
    # i, j -> u, v
    # TODO Find out what the following line is for
    if args.append:
        keypoints = torch.cat([
            keypoints,
            torch.ones([keypoints.size(0), 1]),  # * 1 / scale,
        ], dim=1)

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

    #del keypoints, descriptors, scores, input_image, resized_image, image, im
    #if not args.nogpu:
    #    torch.cuda.empty_cache()
