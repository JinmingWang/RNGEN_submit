from Dataset import RoadNetworkDataset
# Accuracy, Precision, Recall, F1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
import torch
import torch.nn.functional as func
import numpy as np
from jaxtyping import Float32 as F32
from typing import Tuple, List
from shapely.geometry import LineString
import cv2
import networkx as nx

Tensor = torch.Tensor


def heatmapMetric(pred_heatmap: F32[Tensor, "B 1 H W"],
                  target_heatmap: F32[Tensor, "B 1 H W"],
                  threshold: float = 0.5) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Compute the scores between road network projected to 2D (heatmaps)

    :param pred_heatmap: the predicted road network heatmap (after sigmoid)
    :param target_heatmap: the target road network heatmap (binary)
    :param threshold: threshold for binarization
    :return: accuracy, precision, recall, f1
    """
    B = pred_heatmap.shape[0]

    batch_accuracy = []
    batch_precision = []
    batch_recall = []
    batch_f1 = []

    pred_flatten = np.int32(pred_heatmap.view(B, -1).cpu().numpy() > threshold)
    target_flatten = np.int32(target_heatmap.view(B, -1).cpu().numpy() > threshold)

    for b in range(B):
        accuracy = accuracy_score(target_flatten[b], pred_flatten[b])
        precision = precision_score(target_flatten[b], pred_flatten[b])
        recall = recall_score(target_flatten[b], pred_flatten[b])
        f1 = f1_score(target_flatten[b], pred_flatten[b])

        batch_accuracy.append(accuracy)
        batch_precision.append(precision)
        batch_recall.append(recall)
        batch_f1.append(f1)

    return batch_accuracy, batch_precision, batch_recall, batch_f1


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


def segCountMetric(batch_pred_segs: List[F32[Tensor, "P N_interp 2"]],
                     batch_target_segs: List[F32[Tensor, "Q N_interp 2"]]) -> List[float]:
    """
    Compute the difference in the number of segments in the predicted and target segments
    :param batch_pred_segs: the predicted segments
    :param batch_target_segs: the target segments
    :return: the difference in the number of segments in the predicted and target segments
    """
    B = len(batch_pred_segs)

    pred_counts = [segs.shape[0] for segs in batch_pred_segs]
    target_counts = [segs.shape[0] for segs in batch_target_segs]

    diff_counts = [float(abs(pred_counts[i] - target_counts[i])) for i in range(B)]

    return diff_counts


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

def renderPlot(image, src, dst, current):
    rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = image
    rgb[..., 1] = image
    rgb[..., 2] = image

    cv2.putText(rgb, "s", (src[0], src[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(rgb, "d", (dst[0], dst[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.circle(rgb, current, 3, (0, 255, 0), 1)

    cv2.imwrite("search_map.png", cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_NEAREST))


def findAndAddPath(graph, heatmap, src, dst, visualize) -> LineString:
    # Reached another keynode
    search_grid = np.int32(heatmap > 0)
    H, W = search_grid.shape

    # skip if edge already exists
    if graph.has_edge(f"{src[0]}_{src[1]}", f"{dst[0]}_{dst[1]}"):
        return None
    # Try to find a path from (kx, ky) to (x, y) with greedy search
    path = [src]
    current = src
    while current != dst:
        # Take the pixel with the smallest distance to the target
        neighbors = [(current[0] + dx, current[1] + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
        neighbors.remove(current)
        # if len(path) > 1:
        #     neighbors.remove((path[-2][0], path[-2][1]))
        neighbors = list(filter(lambda p: 0 <= p[0] < W and 0 <= p[1] < H and search_grid[p[1], p[0]] != 0, neighbors))

        # Compute heuristic distance to the target
        # grid_values are the values of the heatmap at the neighbors
        # higher values means more visited
        grid_values = np.array([search_grid[p[1], p[0]] for p in neighbors])
        distance = np.linalg.norm(np.array(neighbors) - np.array([dst]), axis=1) + grid_values

        nearest_id = np.argmin(distance)
        current = neighbors[nearest_id]

        # update the search grid
        search_grid[current[1], current[0]] += 1

        path.append(current)

        if visualize:
            tmp = heatmap.copy()
            tmp[current[1], current[0]] = 255
            cv2.imshow("search_map", cv2.resize(tmp, (512, 512), interpolation=cv2.INTER_NEAREST))
            cv2.waitKey(1)

    geometry = LineString(path)

    length = geometry.length
    interp_times = np.linspace(0, length, 8)
    geometry = LineString([geometry.interpolate(i) for i in interp_times])

    graph.add_edge(f"{src[0]}_{src[1]}", f"{dst[0]}_{dst[1]}", geometry=geometry)


def heatmapsToSegments(pred_heatmaps: F32[Tensor, "B 1 H W"],
                       pred_nodemaps: F32[Tensor, "B 1 H W"],
                       visualize: bool = False) -> List[F32[Tensor, "P N_interp 2"]]:
    B, _, H, W = pred_heatmaps.shape

    segs = []

    pred_heatmaps = pred_heatmaps.cpu().numpy()
    local_max_map = torch.nn.functional.max_pool2d(pred_nodemaps, 3, 1, 1)
    nms_nodemaps = torch.where((pred_nodemaps == local_max_map) * (pred_nodemaps >= 0.2), 1, 0)
    nms_nodemaps = nms_nodemaps.cpu().numpy()

    for i in range(B):
        pred_heatmap = pred_heatmaps[i, 0]
        nodemap = nms_nodemaps[i, 0]

        x_list, y_list = np.where(nodemap == 1)
        keynodes = zip(x_list, y_list)

        corner_map = np.zeros_like(np.uint8(pred_heatmap))
        graph = nx.Graph()
        for (x, y) in keynodes:
            corner_map[y, x] = 255
            pred_heatmap[y, x] = 1.0
            graph.add_node(f"{x}_{y}", pos=(x, y))

        # Now extract the edges (1-pixel wide) from the predicted heatmap
        edge_map = np.uint8(pred_heatmap > 0.5) * 255

        cv2.imwrite("edge_map.png", edge_map)

        temp_map = edge_map.copy()

        # Flood fill from a keynode until it reaches another keynode
        for ki, (kx, ky) in enumerate(keynodes):
            frontier = {(kx, ky)}
            while frontier:
                new_frontier = set()
                for (x, y) in frontier:
                    temp_map[y, x] = 64

                    # If this pixel is close to another keynode, stop, connect this keynode and the reached keynode
                    distances = np.linalg.norm(keynodes - np.array([x, y]), 2, axis=1)  # (num_keynodes,)
                    # set distance to itself to infinity
                    distances[ki] = np.inf
                    nearest_id = np.argmin(distances)
                    if distances[nearest_id] < 2:
                        src = (kx, ky)
                        dst = (keynodes[nearest_id][0], keynodes[nearest_id][1])
                        findAndAddPath(graph, temp_map, src, dst, visualize)
                        continue

                    # Check neighbors
                    neighbors = [(x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
                    neighbors.remove((x, y))
                    for (newx, newy) in neighbors:
                        if 0 <= newx < W and 0 <= newy < H and edge_map[newy, newx] != 0 and temp_map[newy, newx] == 255:
                            new_frontier.add((newx, newy))

                frontier = new_frontier
                if visualize:
                    cv2.imshow("temp_map", temp_map)
                    cv2.waitKey(1)

        segs.append(
            torch.tensor([data["geometry"].coords for u, v, data in graph.edges(data=True)], dtype=torch.float32,
                         device=pred_nodemaps.device))

        for seg_i in range(len(segs[-1])):
            if segs[-1][seg_i, 0, 0] > segs[-1][seg_i, -1, 0]:
                segs[-1][seg_i] = segs[-1][seg_i].flip(0)

        try:
            # z-score normalize segments to 0-1
            mean_point = torch.mean(segs[-1].flatten(0, 1), dim=0)
            std_point = torch.std(segs[-1].flatten(0, 1), dim=0)
            segs[-1] = (segs[-1] - mean_point) / std_point

            # max_point = torch.max(segs[-1].flatten(0, 1), dim=0).values
            # min_point = torch.min(segs[-1].flatten(0, 1), dim=0).values
            # point_range = max_point - min_point
            # segs[-1] = ((segs[-1] - min_point) / point_range)
            if torch.any(torch.isnan(segs[-1])):
                print("nan")
        except:
            segs[-1] = torch.zeros(1, 8, 2, dtype=torch.float32, device=pred_nodemaps.device)

    return segs


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
    diff_seg_count = segCountMetric(batch_pred_segs, batch_target_segs)
    diff_seg_length = segLengthMetric(batch_pred_segs, batch_target_segs)

    return [hungarian_mae, hungarian_mse, chamfer_mae, chamfer_mse, diff_seg_count, diff_seg_length]

    # return [heatmap_accuracy, heatmap_precision, heatmap_recall, heatmap_f1,
    #         hungarian_mae, hungarian_mse, chamfer_mae, chamfer_mse]