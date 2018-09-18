import tensorflow as tf
import torch
import torch.nn.functional as F


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.
    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the minimum.
    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums, _ = torch.max(data, dim, keepdim=True)
    masked_minimums, _ = torch.min(
        (data - axis_maximums) * mask, dim,
        keepdim=True)
    masked_minimums += axis_maximums
    return masked_minimums


def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.
    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the maximum.
    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums, _ = torch.min(data, dim, keepdim=True)
    masked_maximums, _ = torch.max(
        (data - axis_minimums) * mask, dim,
        keepdim=True)
    masked_maximums += axis_minimums
    return masked_maximums


def triplet_semihard_loss(labels, embeddings, margin=1.0):
    """Computes the triplet loss with semi-hard negative mining.
    The loss encourages the positive distances (between a pair of embeddings with
    the same labels) to be smaller than the minimum negative distance among
    which are at least greater than the positive distance plus the margin constant
    (called semi-hard negative) in the mini-batch. If no such negative exists,
    uses the largest negative distance instead.
    See: https://arxiv.org/abs/1503.03832.
    Args:
        labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
        multiclass integer labels.
        embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
        be l2 normalized.
        margin: Float, margin term in the loss definition.
    Returns:
        triplet_loss: tf.float32 scalar.
    """
    lshape = labels.size()
    labels = torch.reshape(labels, [lshape[0], 1])
    pdist_matrix = pairwise_distances(embeddings)

    # all anchor and positive pair, except diagonal
    adjacency = torch.eq(labels, torch.t(labels))
    # all anchor and negetive pair
    adjacency_not = torch.ne(labels, torch.t(labels))

    batch_size = torch.numel(labels)

    pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
    mask = adjacency_not.repeat(batch_size, 1) & (
        pdist_matrix_tile > torch.reshape(torch.t(pdist_matrix), [-1, 1]))
    mask_final = torch.reshape(
        (torch.sum(mask.float(), 1, keepdim=True) > 0.0), [
            batch_size, batch_size]
    )
    mask_final = torch.t(mask_final)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = torch.reshape(
        masked_minimum(pdist_matrix_tile, mask.float()), [
            batch_size, batch_size]
    )
    negatives_outside = torch.t(negatives_outside)
    # negatives_inside: largest D_an.
    negatives_inside = masked_maximum(
        pdist_matrix, adjacency_not.float()).repeat(1, batch_size)
    semi_hard_negatives = torch.where(
        mask_final, negatives_outside, negatives_inside)

    loss_mat = margin + pdist_matrix - semi_hard_negatives

    mask_positives = adjacency.float() - torch.diag(
        torch.ones([batch_size]))
    num_positives = torch.sum(mask_positives)

    triplet_loss = torch.sum(
        torch.max(loss_mat * mask_positives, torch.zeros_like(loss_mat))) / num_positives
    return triplet_loss


if __name__ == "__main__":
    # check tf function equal to pytorch fucntion or not
    import numpy as np

    for _ in range(10):
        labels = np.random.randint(0, 4, 10)
        embeddings = np.random.random((10, 5)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        # tensorflow
        sess = tf.Session()
        result_tf = tf.contrib.losses.metric_learning.triplet_semihard_loss(
            tf.constant(labels),
            tf.constant(embeddings)
        )
        v1 = float(sess.run(result_tf))
        # pytorch
        v2 = triplet_semihard_loss(torch.tensor(
            labels), torch.tensor(embeddings))
        v2 = float(v2.numpy())
        print(v1, type(v1))
        print(v2, type(v2))
        assert abs(v1 - v2) < 1e-6
