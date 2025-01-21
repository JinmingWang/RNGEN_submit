import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import Literal, List
from enum import Enum

class HungarianMode(Enum):
    Seq = 0
    DoubleSeq = 1
    SeqMat = 2


class HungarianLoss(nn.Module):
    def __init__(self, mode: HungarianMode, base_loss: Literal['mse', 'l1'] = 'mse', feature_weight: List[float] = None):
        """
        The difference between this and HungarianLoss_Sequential is that this one is for double sequences.
        The cdist is computed using the first sequence, and both sequence use this cdist to find the best matches.
        :param base_loss:
        """
        super(HungarianLoss, self).__init__()
        self.base_loss = nn.MSELoss() if base_loss == 'mse' else nn.L1Loss()
        self.mode = mode

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if feature_weight is None:
            self.f_w = torch.ones(1, 1, 1, device=device, dtype=torch.float32)
        else:
            self.f_w = torch.tensor(feature_weight, device=device, dtype=torch.float32).view(1, 1, -1)

    def forward(self, pred_segs: list[torch.Tensor], target_segs):
        # pred_segs: B * (P, N, D)
        # target_segs: (B, Q, N, D)
        B, N, D = target_segs.shape  # Batch size, number of nodes, feature dimension
        losses = []

        for b in range(B):
            # (P, Q)
            cost_matrix = torch.cdist(pred_segs[b].flatten(1), target_segs[b].flatten(1), p=2).detach().cpu().numpy()

            # Step 2: Apply Hungarian algorithm to find the best node matches
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Step 3: Compute node feature loss for matched pairs
            losses.append(self.base_loss(pred_segs[b][row_ind], target_segs[b][col_ind]))

            pred_matches = pred_segs[b][row_ind][:, -1]
            target_matches = target_segs[b][col_ind][:, -1]

            losses.append(nn.functional.binary_cross_entropy(pred_matches, target_matches))

            # Step 4: Since we expect number of target <= number of pred
            # The unmatched pred nodes should all be zero
            # So now we need to find the unmatched pred nodes
            unmatched_ids = [i for i in range(N) if i not in row_ind]
            if len(unmatched_ids) > 0:
                pred_unmatched = pred_segs[b][unmatched_ids][:, -1]
                target_unmatched = torch.zeros_like(pred_unmatched)
                losses.append(nn.functional.binary_cross_entropy(pred_unmatched, target_unmatched))

        return torch.stack(losses).mean()
