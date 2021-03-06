#from hashlib import scrypt
#from numpy.core.numeric import indices
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from externals.d2net.lib import localization, utils
from lib import autoencoder, attention_model
from PIL import Image
import matplotlib.pyplot as plt


class ExtractionModel(nn.Module):
    def __init__(self, attention_model, thresh=0.0, max_features=None, use_d2net_detection=False, num_upsampling=4, use_nms=True, sum_descriptors=False, mult_with_d2=False) -> None:
        super().__init__()
        self._extraction_model = attention_model.feature_encoder

        self.localization = localization.HandcraftedLocalizationModule()
        self.attent_detection = DetectionModule(attention_model)
        self.d2_net_detection = localization.HardDetectionModule()
        self._use_d2net_detection = use_d2net_detection
        self.extraction = lambda x: self._extraction_model(x)
        self._num_upsampling = num_upsampling

        self._thresh = thresh
        self._max_features = max_features
        self._use_nms = use_nms
        self._max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self._sum_descriptors = sum_descriptors
        self._mult_with_d2 = mult_with_d2

    def nms(self, x):
        lmax = self._max_pool(x)
        x[lmax != x] = 0
        return x

    def forward(self, images: torch.Tensor):
        n, _, h, w = images.size()

        dense_features = self.extraction(images)

        early = dense_features['early']
        middle = dense_features['middle']
        deep = dense_features['deep']
        n, _, h_map, w_map = early.size()

        detections = None
        displacements_input = None
        d2_net_map = []
        if self._use_d2net_detection or self._mult_with_d2:
            displacements_input = [early, middle, deep]
            detections = []
            for df in [early, middle, deep]:
                d, m = self.d2_net_detection(df)
                detections.append(d)
                d2_net_map.append(m.cpu())
        if not self._use_d2net_detection:
            detections = self.attent_detection(dense_features)
            # for i,d in enumerate(detections):
            #     detections[i][d < self._thresh] = 0
            if self._mult_with_d2:
                detections = [detections[i].cpu()*d2_net_map[i] for i in range(3)]
            displacements_input = detections
            if self._use_nms:
                detections = [self.nms(d) for d in detections]
        dense_features = [early.cpu(), middle.cpu(), deep.cpu()]
        detections = [d.cpu() for d in detections]

        fmap_pos = [torch.nonzero(d[0].cpu()).t() for d in detections]

        displacements = [self.localization(
            features)[0].cpu() for features in displacements_input]

        displacements_i = [displacements[i][
            0, fmap_pos[i][0, :], fmap_pos[i][1, :], fmap_pos[i][2, :]
        ] for i in range(3)]
        displacements_j = [displacements[i][
            1, fmap_pos[i][0, :], fmap_pos[i][1, :], fmap_pos[i][2, :]
        ] for i in range(3)]

        del displacements

        masks = [torch.min(
            torch.abs(displacements_i[i]) < 0.5,
            torch.abs(displacements_j[i]) < 0.5
        ) for i in range(3)]

        fmap_pos = [fmap_pos[i][:, masks[i]] for i in range(3)]

        valid_displacements = [torch.stack([
            displacements_i[i][masks[i]],
            displacements_j[i][masks[i]]
        ], dim=0) for i in range(3)]

        del masks, displacements_i, displacements_j

        fmap_keypoints = [fmap_pos[i][1:, :].float(
        ) + valid_displacements[i] for i in range(3)]

        raw_desriptors = []
        ids = []

        if not self._sum_descriptors:
            for i in range(3):
                try:
                    rd, _, _ids = utils.interpolate_dense_features(
                        fmap_keypoints[i], dense_features[i][0])
                    raw_desriptors.append(rd)
                    ids.append(_ids)
                except utils.EmptyTensorError:
                    raw_desriptors.append(None)
                    ids.append(None)
        else:
            score_inp = detections if not self._use_d2net_detection else d2_net_map
            for i in range(3):
                rd_use = None
                _ids_use = None
                for j in range(3):
                    try:
                        rd, _, _ids = utils.interpolate_dense_features(
                            fmap_keypoints[i], score_inp[i][0]*dense_features[i][0])
                        if rd_use is None:
                            rd_use = rd 
                        else:
                            rd_use += rd 
                        if i == j:
                            _ids_use = _ids
                    except utils.EmptyTensorError:
                        pass
                
                raw_desriptors.append(rd_use)
                ids.append(_ids_use)
        fmap_pos = [fmap_pos[i][:, ids[i]] for i in range(3)]
        fmap_keypoints = [fmap_keypoints[i][:, ids[i]] for i in range(3)]

        keypoints = [utils.upscale_positions(
            k, scaling_steps=self._num_upsampling) for k in fmap_keypoints]

        del fmap_keypoints
        # for i in range(3):
        #     keypoints[i][0, :] *= h / h_map * 0.25
        #     keypoints[i][1, :] *= w / w_map * 0.25

        descriptors = [F.normalize(rd, dim=0) for rd in raw_desriptors]

        score_extraction_input = detections if not self._use_d2net_detection else dense_features
        scores = [
            score_extraction_input[i][0, fmap_pos[i][0, :], fmap_pos[i][1, :], fmap_pos[i][2, :]] for i in range(3)
        ]

        del fmap_pos

        keypoints = [k.t() for k in keypoints]
        keypoints = torch.cat(keypoints)

        descriptors = [d.t() for d in descriptors]
        descriptors = torch.cat(descriptors)

        scores = torch.cat(scores)
        scores, idx = torch.sort(scores, descending=True)
        descriptors = descriptors[idx, :]
        keypoints = keypoints[idx, :]

        if self._thresh != 0 or self._max_features is not None:
            mask = scores >= self._thresh
            scores = scores[mask]
            descriptors = descriptors[mask, :]
            keypoints = keypoints[mask, :]

            if self._max_features is not None and scores.size()[0] > self._max_features:
                scores = scores[:self._max_features]
                keypoints = keypoints[:self._max_features, :]
                descriptors = descriptors[:self._max_features, :]

        keypoints = keypoints.detach()
        keypoints = keypoints[:, [1, 0]]  # To work with cv indexing

        if self._use_d2net_detection:
            d2_net_map = [d.detach().cpu().numpy()[0, :, :]
                          for d in d2_net_map]
            return keypoints, descriptors.detach(), scores.detach(
            ), d2_net_map
        else:
            return keypoints, descriptors.detach(), scores.detach(), [d.detach().numpy()[0, 0, :, :] for d in detections]


class DetectionModule(nn.Module):
    def __init__(self, attention_model) -> None:
        super().__init__()
        self._attention_model = attention_model

    def forward(self, encoded_features):
        scores = self._attention_model(encoded_features)
        early = scores['early']
        middle = scores['middle']
        deep = scores['deep']

        return early, middle, deep


if __name__ == "__main__":
    encoder = autoencoder.FeatureEncoder()
    attention = attention_model.MultiAttentionLayer(encoder)

    extraction_model = ExtractionModel(
        encoder, attention, use_d2net_detection=True)
    image = Image.open('/home/mark/Downloads/test.jpg')
    image_np = np.array(image)
    image = utils.preprocess_image(image_np)
    image = torch.as_tensor(image)
    image = torch.unsqueeze(image, 0)

    keypoints, descriptors, scores = extraction_model(image)

    print(scores)

    plt.imshow(image_np)
    plt.scatter(keypoints[:, 1], keypoints[:, 0])
    plt.show()

    # print(displacements[0])
