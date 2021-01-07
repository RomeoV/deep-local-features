import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from externals.d2net.lib import localization, utils
from lib import autoencoder, attention_model
from PIL import Image
import matplotlib.pyplot as plt


class ExtractionModel(nn.Module):
    def __init__(self, extraction_model, attention_model, thresh=0.0, max_features=None, use_d2net_detection=False) -> None:
        super().__init__()
        self._extraction_model = extraction_model

        self.localization = localization.HandcraftedLocalizationModule()
        self.detection = DetectionModule(
            attention_model, thresh, max_features) if not use_d2net_detection else localization.HardDetectionModule()
        self._use_d2net_detection = use_d2net_detection
        self.extraction = lambda x: self._extraction_model(x)

    def forward(self, images: torch.Tensor):
        n, _, h, w = images.size()

        dense_features = self.extraction(images)
        early = dense_features['early']
        middle = dense_features['middle']
        deep = dense_features['deep']
        n, _, h_map, w_map = early.size()

        if self._use_d2net_detection:
            dense_features = [early, middle, deep]
            detections = [self.detection(df) for df in dense_features]
            print(detections[0])
        else:
            detections = self.detection(dense_features)
        dense_features = [early, middle, deep]

        fmap_pos = [torch.nonzero(d[0].cpu()).t() for d in detections]

        displacements = [self.localization(features)[0].cpu() for features in [
            early, middle, deep]]

        displacements_i = [displacements[i][
            0, fmap_pos[i][0, :], fmap_pos[i][1, :], fmap_pos[i][2, :]
        ] for i in range(3)]
        displacements_j = [displacements[i][
            1, fmap_pos[i][0, :], fmap_pos[i][1, :], fmap_pos[i][2, :]
        ] for i in range(3)]

        masks = [torch.min(
            torch.abs(displacements_i[i]) < 0.5,
            torch.abs(displacements_j[i]) < 0.5
        ) for i in range(3)]

        fmap_pos = [fmap_pos[i][:, masks[i]] for i in range(3)]

        valid_displacements = [torch.stack([
            displacements_i[i][masks[i]],
            displacements_j[i][masks[i]]
        ], dim=0) for i in range(3)]

        fmap_keypoints = [fmap_pos[i][1:, :].float(
        ) + valid_displacements[i] for i in range(3)]

        raw_desriptors = []
        ids = []

        for i in range(3):
            try:
                rd, _, _ids = utils.interpolate_dense_features(
                    fmap_keypoints[i], dense_features[i][0])
                raw_desriptors.append(rd)
                ids.append(_ids)
            except utils.EmptyTensorError:
                raw_desriptors.append(None)
                ids.append(None)
        fmap_pos = [fmap_pos[i][:, ids[i]] for i in range(3)]
        fmap_keypoints = [fmap_keypoints[i][:, ids[i]] for i in range(3)]

        keypoints = [utils.upscale_positions(
            k, scaling_steps=4) for k in fmap_keypoints]
        # for i in range(3):
        #     keypoints[i][0, :] *= h / h_map * 0.25
        #     keypoints[i][1, :] *= w / w_map * 0.25

        descriptors = [F.normalize(rd, dim=0) for rd in raw_desriptors]

        score_extraction_input = detections if not self._use_d2net_detection else dense_features
        scores = [
            F.normalize(score_extraction_input[i][0, fmap_pos[i][0, :], fmap_pos[i][1, :], fmap_pos[i][2, :]], dim=0) for i in range(3)
        ]

        keypoints = [k.t() for k in keypoints]
        keypoints = torch.cat(keypoints)

        descriptors = [d.t() for d in descriptors]
        descriptors = torch.cat(descriptors)

        scores = torch.cat(scores)

        return keypoints.detach(), descriptors, scores


class DetectionModule(nn.Module):
    def __init__(self, attention_model, thresh=0.0, max_features=None) -> None:
        super().__init__()
        self._attention_model = attention_model
        self._thresh = thresh
        self._max_features = max_features

    def forward(self, encoded_features):
        scores = self._attention_model(encoded_features)
        early = scores['early']
        middle = scores['middle']
        deep = scores['deep']
        # TODO normalize
        if self._max_features is not None:
            all_scores = torch.cat([
                torch.flatten(early),
                torch.flatten(middle),
                torch.flatten(deep)
            ])
            if self._max_features >= all_scores.size()[0]:
                self._thresh = 0
            else:
                all_scores = torch.sort(all_scores, descending=True).values
                print(all_scores.size())
                self._thresh = all_scores[self._max_features]

        early[early < self._thresh] = 0
        middle[middle < self._thresh] = 0
        deep[deep < self._thresh] = 0

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
