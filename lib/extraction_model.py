import torch
from torch.serialization import load

from lib import autoencoder
from lib.feature_extractor import feature_extraction
from lib import attention_model


class FeatureExtractor:
    def __init__(self, feature_encoder, rfs, strides, paddings):
        self.feature_encoder = feature_encoder
        self.attention = attention_model.MultiAttentionLayer(feature_encoder)
        self.rfs = rfs
        self.strides = strides
        self.paddings = paddings

    def forward(self, x):
        h = None
        w = None
        if len(x.shape) == 4:
            _, _, h, w = x.shape
        else:
            _, h, w = x.shape

        y = self.feature_encoder(x)
        y_att = self.attention(y)
        early = y['early']
        middle = y['middle']
        deep = y['deep']

        early_att = y_att['early']
        middle_att = y_att['middle']
        deep_att = y_att['deep']

        early_centers, early_features, early_scores = feature_extraction.ExtractFeatures(
            early, early_att, self.rfs[0], self.strides[0], self.paddings[0], image_width=w, image_height=h, sort=True)

        middle_centers, middle_features, middle_scores = feature_extraction.ExtractFeatures(
            middle, middle_att, self.rfs[1], self.strides[1], self.paddings[1], image_width=w, image_height=h, sort=True)

        deep_centers, deep_features, deep_scores = feature_extraction.ExtractFeatures(
            deep, deep_att, self.rfs[2], self.strides[2], self.paddings[2], image_width=w, image_height=h, sort=True)

        centers = torch.cat(
            [early_centers, middle_centers, deep_centers], axis=0)
        features = torch.cat(
            [early_features, middle_features, deep_features], axis=0)
        scores = torch.cat([early_scores, middle_scores, deep_scores], axis=0)

        return centers, features, scores


def main_tests():
    encoder = autoencoder.FeatureEncoder1(load_tf_weights=True)
    extractor = FeatureExtractor(encoder, [43, 99, 355], [
                                 8, 8, 8], [17, 49, 145])
    image = torch.rand(size=(1, 3, 480, 640))

    centers, features, scores = extractor.forward(image)

    print(centers.shape, centers.numpy())
    print(features.shape, features.detach().numpy())
    print(scores.shape, scores.detach().numpy())


if __name__ == "__main__":
    main_tests()
