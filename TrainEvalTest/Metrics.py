from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
import torch
import torch.nn.functional as func
from jaxtyping import Float32 as F32
from typing import Tuple, List

Tensor = torch.Tensor

def hungarianMetric(batch_pred_segs: List[F32[Tensor, "P N_interp 2"]],
                    batch_target_segs: List[F32[Tensor, "Q N_interp 2"]]) -> Tuple[List[float], List[float]]:
    """
    Find hungarian matching between predicted and target segments.
    Then compute MAE and MSE between matched segments.

    Hungarian matching is used to find global optimal matching between predicted and target segments.

    :param batch_pred_segs: list of predicted segments
    :param batch_target_segs: list of target segments
    :return: MAE, MSE
    """

    B = len(batch_pred_segs)
    mae_list = []
    mse_list = []

    for b in range(B):
        pred_segs = batch_pred_segs[b].flatten(1)  # (P, N_interp*2)
        target_segs = batch_target_segs[b].flatten(1)  # (Q, N_interp*2)
        P = len(pred_segs)
        Q = len(target_segs)

        if P > Q:
            target_segs = func.pad(target_segs, (0, 0, 0, P - Q))
        elif Q > P:
            pred_segs = func.pad(pred_segs, (0, 0, 0, Q - P))

        cost_matrix = torch.cdist(pred_segs, target_segs, p=2).cpu().detach().numpy()  # (P, Q)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Matched segments, compute MAE and MSE normally for matched segments
        matched_pred_segs = pred_segs[row_ind]
        matched_target_segs = target_segs[col_ind]

        # Unmatched segments, The match target will be 0 if the segment is unmatched
        # unmatched_rows = [i for i in range(pred_segs.shape[0]) if i not in row_ind]
        # unmatched_cols = [i for i in range(target_segs.shape[0]) if i not in col_ind]
        # unmatched_pred_segs = pred_segs[unmatched_rows]
        # unmatched_target_segs = target_segs[unmatched_cols]

        mae = torch.abs(matched_pred_segs - matched_target_segs).mean().item()

        mse = ((matched_pred_segs - matched_target_segs) ** 2).mean().item()

        # if len(unmatched_pred_segs) > 0:
        #     mae += unmatched_pred_segs.abs().mean().item()
        #     mse += (unmatched_pred_segs ** 2).mean().item()
        #
        # if len(unmatched_target_segs) > 0:
        #     mae += unmatched_target_segs.abs().mean().item()
        #     mse += (unmatched_target_segs ** 2).mean().item()

        mae_list.append(mae)
        mse_list.append(mse)

    return mae_list, mse_list


def chamferMetric(batch_pred_segs: List[F32[Tensor, "P N_interp 2"]],
                  batch_target_segs: List[F32[Tensor, "Q N_interp 2"]]) -> Tuple[List[float], List[float]]:
    """
    Find chamfer matching between predicted and target segments.
    Then compute MAE and MSE between matched segments.

    Chamfer matching is to find the best match for each segment in the predicted segments.

    :param batch_pred_segs: list of predicted segments
    :param batch_target_segs: list of target segments
    :return: Chamfer distance
    """
    B = len(batch_pred_segs)
    mae_list = []
    mse_list = []

    for b in range(B):
        pred_segs = batch_pred_segs[b].flatten(1)  # (P, N_interp*2)
        target_segs = batch_target_segs[b].flatten(1)  # (Q, N_interp*2)

        if len(target_segs) == 0:
            target_segs = torch.zeros_like(pred_segs)

        # Compute pairwise distance matrix
        cost_matrix = torch.cdist(pred_segs, target_segs, p=2)  # (P, Q)

        # Find the best match for each segment in the predicted segments
        row_ind_p2t = torch.argmin(cost_matrix, dim=1)

        # Compute MAE and MSE (match the predicted segments to the target segments)
        mae_p2t = (torch.abs(pred_segs - target_segs[row_ind_p2t]).mean())
        mse_p2t = ((pred_segs - target_segs[row_ind_p2t]) ** 2).mean()

        # Compute MAE and MSE (match the target segments to the predicted segments)
        row_ind_t2p = torch.argmin(cost_matrix, dim=0)
        mae_t2p = (torch.abs(pred_segs[row_ind_t2p] - target_segs).mean())
        mse_t2p = ((pred_segs[row_ind_t2p] - target_segs) ** 2).mean()

        # Why do we compute p2t and also t2p?
        # When we match p to t, some segments in p may not have a match in t, they are not counted in the loss.
        # When we match t to p, some segments in t may not have a match in p, they are not counted in the loss.
        # So we need to compute both to make sure all segments are counted in the loss.
        mae = (mae_p2t + mae_t2p).item()
        mse = (mse_p2t + mse_t2p).item()

        mae_list.append(mae)
        mse_list.append(mse)

    return mae_list, mse_list


def segLengthMetric(batch_pred_segs: List[F32[Tensor, "P N_interp 2"]],
                    batch_target_segs: List[F32[Tensor, "Q N_interp 2"]]) -> List[float]:
    """
    Compute the difference in the length distribution of the predicted and target segments
    :param batch_pred_segs: the predicted segments
    :param batch_target_segs: the target segments
    :return: the difference in the length distribution of the predicted and target segments
    """
    B = len(batch_pred_segs)

    # Each batch item is of shape (P, N_interp, 2) or (Q, N_interp, 2)
    # It contains P or Q segments, each segment has N_interp points in 2D

    distribution_diffs = []
    for b in range(B):
        pred_segs = batch_pred_segs[b]  # (P, N_interp, 2)
        target_segs = batch_target_segs[b]  # (Q, N_interp, 2)

        # Compute the length of each segment

        # (P, N_interp, 2) -> (P, )
        pred_lengths = torch.linalg.norm(pred_segs[:, 1:] - pred_segs[:, :-1], dim=-1).sum(dim=-1)
        # (Q, N_interp, 2) -> (Q, )
        target_lengths = torch.linalg.norm(target_segs[:, 1:] - target_segs[:, :-1], dim=-1).sum(dim=-1)
        if len(target_lengths) == 0:
            target_lengths = torch.zeros_like(pred_lengths)

        # Compute the difference in the length distribution using Wasserstein distance
        distribution_diff = wasserstein_distance(pred_lengths.cpu().numpy(), target_lengths.cpu().numpy())
        distribution_diffs.append(distribution_diff)

    return distribution_diffs


def reportAllMetrics(batch_pred_segs: List[F32[Tensor, "P N_interp 2"]],
                     batch_target_segs: List[F32[Tensor, "Q N_interp 2"]]) -> List[List[float]]:
    """
    Compute all metrics for road network prediction

    :param log_path: path to save the log
    :param batch_pred_segs: list of predicted segments
    :param batch_target_segs: list of target segments
    :return: accuracy, precision, recall, f1, MAE, MSE
    """
    # heatmap_accuracy, heatmap_precision, heatmap_recall, heatmap_f1 = heatmapMetric(pred_heatmap, target_heatmap)
    hungarian_mae, hungarian_mse = hungarianMetric(batch_pred_segs, batch_target_segs)
    chamfer_mae, chamfer_mse = chamferMetric(batch_pred_segs, batch_target_segs)
    diff_seg_length = segLengthMetric(batch_pred_segs, batch_target_segs)

    return [hungarian_mae, hungarian_mse, chamfer_mae, chamfer_mse, diff_seg_length]