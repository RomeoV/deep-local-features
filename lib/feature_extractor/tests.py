import torch
import matplotlib.pyplot as plt

from lib.feature_extractor import feature_extraction


def main():
    feature_map = torch.normal(0, 1, size=[16, 30, 40])
    attention_scores = torch.normal(0, 1, size=[30, 40])

    centers, features, scores = feature_extraction.ExtractFeatures(
        feature_map, attention_scores, rf=99.0, stride=16, padding=49.0, thresh=-1e6, sort=True, normalize_coords=False, image_height=480, image_width=640)

    print(centers.shape, centers)
    print(features.shape, features)
    print(scores.shape, scores)

    plt.scatter(centers[:, 0], centers[:, 1])
    plt.show()


if __name__ == "__main__":
    main()
