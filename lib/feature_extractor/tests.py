import torch

from lib.feature_extractor import feature_extraction


def main():
    feature_map = torch.normal(0, 1, size=[30, 40, 16])
    attention_scores = torch.normal(0, 1, size=[30, 40])

    centers, features, scores = feature_extraction.ExtractFeatures(
        feature_map, attention_scores, rf=99.0, stride=16, padding=49.0, thresh=0.1, sort=True, image_height=480, image_width=640)

    print(centers.shape, centers)
    print(features.shape, features)
    print(scores.shape, scores)


if __name__ == "__main__":
    main()
