import torch


def ReceptiveBoxes(map_height, map_width, rf, stride, padding) -> torch.Tensor:
    """
    Pytorch adaption of https://github.com/tensorflow/models/blob/master/research/delf/delf/python/feature_extractor.py#L41
    To get the parameters have a look at Receptive_Field_Calculator.ipynb
    Calculate receptive boxes for each feature point.
    Args:
        height: The height of feature map.
        width: The width of feature map.
        rf: The receptive field size.
        stride: The effective stride between two adjacent feature points.
        padding: The effective padding size.
    Returns:
        rf_boxes: [N, 4] receptive boxes tensor. Here N equals to height x width.
        Each box is represented by [ymin, xmin, ymax, xmax].
    """
    x, y = torch.meshgrid(torch.arange(map_width), torch.arange(map_height))
    coords = torch.reshape(torch.stack([y, x], axis=2), [-1, 2])
    point_boxes = torch.cat([coords, coords], 1)
    bias = torch.FloatTensor(
        [-padding, -padding, -padding + rf - 1, -padding + rf - 1])
    return stride * point_boxes + bias


def KeypointCenters(boxes):
    """
    Pytorch adaption of https://github.com/tensorflow/models/blob/master/research/delf/delf/python/feature_extractor.py#L65
    Helper function to compute feature centers, from RF boxes.
    Args:
        boxes: [N, 4] float tensor.
    Returns:
        centers: [N, 2] float tensor.
    """
    return (boxes[:, :2] + boxes[:, 2:])*0.5


def ExtractFeatures(feature_map, attention_map, rf, stride, padding, thresh=None, sort=False, normalize_coords=True, image_height=None, image_width=None):
    if normalize_coords:
        assert(image_height is not None and image_width is not None)
    boxes = ReceptiveBoxes(
        feature_map.shape[0], feature_map.shape[1], rf, stride, padding)
    scores = torch.transpose(attention_map, 0, 1)
    scores = torch.reshape(scores, [-1])

    features = torch.transpose(feature_map, 0, 1)
    h, w, c = features.shape
    features = torch.reshape(features, [h*w, c])
    if thresh is not None:
        mask = scores >= thresh
        boxes = boxes[mask, :]
        scores = scores[mask]
        features = features[mask, :]

    if sort:
        result = torch.sort(scores, descending=True)
        scores = result.values
        boxes = boxes[result.indices, :]
        features = features[result.indices, :]

    centers = KeypointCenters(boxes)

    if normalize_coords:
        centers[:, 0] /= image_height
        centers[:, 1] /= image_width

    return centers, features, scores
