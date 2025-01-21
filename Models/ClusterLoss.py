from .Basics import *

class ClusterLoss(nn.Module):
    def __init__(self):
        super(ClusterLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def getClusterMat(self, target_seq: Tensor) -> Tensor:
        # target_seq: (B, N, D)
        # Two tokens are in the same cluster if they are the same
        # so, generate a (B, N, N) matrix where cluster_mat[i, j] = 1 if target_seq[i] == target_seq[j]
        dist_mat = torch.cdist(target_seq, target_seq, 2)
        cluster_mat = (dist_mat < 1e-3).float()
        return cluster_mat

    def forward(self, pred_seq, pred_cluster_mat: Tensor, target_seq: Tensor) -> Tensor:
        # pred_seg: (B, N_trajs*L_route, N_interp, 2)
        # target_seq: (B, N_segs, N_interp, 2)
        B, M, D, _ = pred_seq.shape

        # Step 1. find the matching between input_seq and target_seq
        # pred_seq: (x1, y1, x2, y2)
        # pred_seqp: (x2, y2, x1, y1)
        # pred_swap = pred_seq.flip(-1)
        #cost_matrices_A = torch.cdist(pred_seq, target_seq, p=2)  # (B, M, N)
        #cost_matrices_B = torch.cdist(pred_swap, target_seq, p=2)    # (B, M, N)
        #cost_matrices = torch.min(cost_matrices_A, cost_matrices_B).detach().cpu().numpy()  # (B, M, N)
        cost_matrices = torch.cdist(pred_seq.flatten(2), target_seq.flatten(2), p=2)

        # Step 2. rearrange the target_seq with the matching, so input seq and target_seq have 1-1 correspondence
        # if target_seq[j] is the nearest to input_seq[i], then matched_target_seq[i] = target_seq[j]
        nearest_match = cost_matrices.argmin(axis=2)  # (B, M)
        matched_target_seq = torch.zeros_like(pred_seq)    # (B, M, L, 2)
        for b in range(B):
            matched_target_seq[b] = target_seq[b, nearest_match[b]]

        # Step 3. get the cluster matrix of the matched target_seq
        target_cluster_mat = self.getClusterMat(matched_target_seq.flatten(2))
        # Step 4. calculate the loss
        return self.bce_loss(pred_cluster_mat, target_cluster_mat)


    def getClusters(self, input_seq, cluster_mat):
        # input_seq: (M, D)
        # cluster_mat: (M, M) or (M, M)
        M, D = input_seq.shape

        # cluster_mat contains only triu, so we need to make it symmetric
        cluster_mat = cluster_mat + cluster_mat.t() + torch.eye(M, device=cluster_mat.device)

        # Rearrange the input seq and cluster mat, so that elements of the same cluster are next to each other
        clusters = []
        this_id = 0
        while M > 0:
            this_cluster_ids = torch.nonzero(cluster_mat[0] >= 0.5).squeeze(1)
            cluster_size = len(this_cluster_ids)
            # put the elements in the cluster to the front
            clusters.append(input_seq[this_cluster_ids])

            remaining_mask = torch.ones(M, device=cluster_mat.device, dtype=torch.bool)
            remaining_mask[this_cluster_ids] = False

            # remove the elements in the cluster from the cluster_mat
            cluster_mat = cluster_mat[remaining_mask][:, remaining_mask]
            input_seq = input_seq[remaining_mask]
            M -= cluster_size

            this_id += cluster_size

        return clusters