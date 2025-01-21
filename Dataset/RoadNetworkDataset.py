import torch
from .Utils import *
from tqdm import tqdm
import os
import numpy as np
import cv2


class RoadNetworkDataset():
    def __init__(self,
                 folder_path: str,
                 batch_size: int = 32,
                 drop_last: bool = True,
                 set_name: str = "train",
                 permute_seq: bool = True,
                 enable_aug: bool = False,
                 shuffle:bool = True,
                 img_H: int = 256,
                 img_W: int = 256,
                 need_image: bool = False,
                 need_heatmap: bool = False,
                 need_nodes: bool = False) -> None:
        """
        Initialize the dataset, this class loads data from a cache file
        The cache file is created by using LaDeDatasetCacheGenerator class
        :param path: the path to the cache file
        :param max_trajs: the maximum number of trajectories to use
        :param set_name: the name of the set, can be "train", "test", "debug" or "all"
        """
        print("Loading RoadNetworkDataset")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.set_name = set_name
        self.permute_seq = permute_seq
        self.enable_aug = enable_aug
        self.shuffle = shuffle
        self.img_H = img_H
        self.img_W = img_W
        self.need_image = need_image
        self.need_heatmap = need_heatmap
        self.need_nodes = need_nodes

        dataset = torch.load(os.path.join(folder_path, "dataset.pt"))

        # (N_data, N_trajs, L_traj, 2)
        self.trajs = dataset["trajs"]
        data_count = len(self.trajs)
        if set_name == "train":
            slicing = slice(data_count - 1000)
        elif set_name == "test":
            slicing = slice(data_count - 1000, None)
        elif set_name == "debug":
            slicing = slice(300)
        elif set_name == "all":
            slicing = slice(data_count)

        # Data Loading

        self.trajs = self.trajs[slicing]
        # (N_data, N_trajs, L_route, N_interp, 2)
        self.routes = dataset["routes"][slicing]
        # (N_data, N_segs, N_interp, 2)
        self.segs = dataset["segs"][slicing]
        if need_image:
            # (N_data, 3, H, W)
            self.images = dataset["images"][slicing]
            self.images = torch.nn.functional.interpolate(self.images, (img_H, img_W), mode="bilinear")
        if need_heatmap:
            # (N_data, 1, H, W)
            self.heatmaps = dataset["heatmaps"][slicing]
            self.heatmaps = torch.nn.functional.interpolate(self.heatmaps, (img_H, img_W), mode="nearest")

        self.L_traj = dataset["traj_lens"].to(torch.int32)
        self.L_route = dataset["route_lens"].to(torch.int32)
        self.N_segs = dataset["seg_nums"].to(torch.int32)

        self.mean_norm = dataset["point_mean"]
        self.std_norm = dataset["point_std"]

        self.bboxes = dataset["bboxes"]

        # Get the data dimensions

        self.N_data, self.N_trajs, self.max_L_traj = self.trajs.shape[:3]
        self.max_L_route = self.routes.shape[2]
        self.max_N_segs, self.N_interp = self.segs.shape[1:3]

        if need_heatmap:
            self.getTargetHeatmaps()
        if need_nodes:
            self.segmentsToNodesAdj()

        print(str(self))

    def __str__(self):
        return f"RoadNetworkDataset: {self.set_name} set with {self.N_data} samples packed to {len(self)} batches"


    def __repr__(self):
        return self.__str__().replace("\n", ", ")


    def __len__(self) -> int:
        if self.drop_last:
            return self.N_data // self.batch_size
        else:
            if self.N_data % self.batch_size == 0:
                return self.N_data // self.batch_size
            else:
                return self.N_data // self.batch_size + 1


    def augmentation(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Apply data augmentation to the given sample
        :return: The augmented sample
        """

        B = batch["trajs"].shape[0]
        point_shift = torch.randn(B, 1, 1, 2).to(DEVICE) * 0.05
        batch["trajs"] += point_shift * (batch["trajs"] != 0)
        batch["routes"] += point_shift.unsqueeze(1) * (batch["routes"] != 0)
        batch["segs"] += point_shift * (batch["segs"] != 0)
        if self.need_nodes:
            batch["nodes"] += point_shift.squeeze(1) * (batch["nodes"] != 0)
            batch["edges"] = batch["edges"].unflatten(-1, (-1, 2)) + point_shift.unsqueeze(1)

        if np.random.rand() < 0.5:
            batch["trajs"][..., 0] = -batch["trajs"][..., 0]
            batch["routes"][..., 0] = -batch["routes"][..., 0]
            batch["segs"][..., 0] = -batch["segs"][..., 0]
            if self.need_nodes:
                batch["nodes"][..., 0] = -batch["nodes"][..., 0]
                batch["edges"][..., 0] = -batch["edges"][..., 0]

        if np.random.rand() < 0.5:
            batch["trajs"][..., 1] = -batch["trajs"][..., 1]
            batch["routes"][..., 1] = -batch["routes"][..., 1]
            batch["segs"][..., 1] = -batch["segs"][..., 1]
            if self.need_nodes:
                batch["nodes"][..., 1] = -batch["nodes"][..., 1]
                batch["edges"][..., 1] = -batch["edges"][..., 1]

        # Random rotate trajs, routes and segs centered at (0, 0)
        # trajs: (B, N_traj, L_traj, 2)
        # routes: (B, N_traj, L_route, N_interp, 2)
        # segs: (B, N_segs, N_interp, 2)
        radian = torch.rand(B) * 2 * np.pi
        cos_theta = torch.cos(radian)
        sin_theta = torch.sin(radian)
        rot_matrix = torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=1).view(B, 2, 2).to(DEVICE)

        batch["trajs"] = torch.einsum("bij,bnlj->bnli", rot_matrix, batch["trajs"])
        batch["routes"] = torch.einsum("bij,bnlkj->bnlki", rot_matrix, batch["routes"])
        batch["segs"] = torch.einsum("bij,bnlj->bnli", rot_matrix, batch["segs"])
        if self.need_nodes:
            batch["nodes"] = torch.einsum("bij,bnj->bni", rot_matrix, batch["nodes"])
            batch["edges"] = torch.einsum("bij,bhwnj->bhwni", rot_matrix, batch["edges"]).flatten(-2, -1)

        return batch

    def __getitem__(self, idx) -> Dict[str, Tensor]:
        """
        Return the sample at the given index
        :param idx: the index of the sample
        :return: the sample at the given index
        """
        if isinstance(idx, int):
            idx = [idx]

        trajs = self.trajs[idx].to(DEVICE)
        routes = self.routes[idx].to(DEVICE)
        segs = self.segs[idx].to(DEVICE)
        L_traj = self.L_traj[idx].to(DEVICE)
        L_route = self.L_route[idx].to(DEVICE)

        if self.permute_seq:
            # permute the order of the trajectories and routes
            traj_perm = torch.randperm(trajs.shape[1])
            trajs = trajs[:, traj_perm]
            routes = routes[:, traj_perm]
            L_traj = L_traj[:, traj_perm]
            L_route = L_route[:, traj_perm]

            # permute the order of the segments
            segs_perm = torch.randperm(segs.shape[1])
            segs = segs[:, segs_perm]

        batch_data = {
            "trajs": trajs,
            "routes": routes,
            "segs": segs,
            "L_traj": L_traj,
            "L_route": L_route,
            "N_segs": self.N_segs[idx].to(DEVICE),
            "mean_point": self.mean_norm[idx].to(DEVICE),
            "std_point": self.std_norm[idx].to(DEVICE),
            "bbox": self.bboxes[idx].to(DEVICE)
        }

        if self.need_heatmap:
            batch_data["heatmap"] = self.heatmaps[idx].to(DEVICE)
            batch_data["target_heatmaps"] = self.target_heatmaps[idx].to(DEVICE)

        if self.need_image:
            batch_data["image"] = self.images[idx].to(DEVICE)

        if self.need_nodes:
            batch_data["nodes"] = self.nodes[idx].to(DEVICE)
            batch_data["adj_mat"] = self.adj_mats[idx].to(DEVICE)
            batch_data["edges"] = self.edges[idx].to(DEVICE)
            batch_data["N_nodes"] = self.N_nodes[idx].to(DEVICE)
            batch_data["degrees"] = self.degrees[idx].to(DEVICE)

        if self.enable_aug:
            return self.augmentation(batch_data)
        return batch_data


    def __iter__(self):
        if self.shuffle:
            shuffled_indices = torch.randperm(self.N_data)
            self.trajs = self.trajs[shuffled_indices].contiguous()
            self.routes = self.routes[shuffled_indices].contiguous()
            self.segs = self.segs[shuffled_indices].contiguous()
            self.L_traj = self.L_traj[shuffled_indices].contiguous()
            self.L_route = self.L_route[shuffled_indices].contiguous()
            self.N_segs = self.N_segs[shuffled_indices].contiguous()
            self.mean_norm = self.mean_norm[shuffled_indices].contiguous()
            self.std_norm = self.std_norm[shuffled_indices].contiguous()
            self.bboxes = self.bboxes[shuffled_indices].contiguous()

            if self.need_image:
                self.images = self.images[shuffled_indices].contiguous()

            if self.need_heatmap:
                self.heatmaps = self.heatmaps[shuffled_indices].contiguous()
                self.target_heatmaps = self.target_heatmaps[shuffled_indices].contiguous()

            if self.need_nodes:
                self.nodes = self.nodes[shuffled_indices].contiguous()
                self.adj_mats = self.adj_mats[shuffled_indices].contiguous()
                self.edges = self.edges[shuffled_indices].contiguous()
                self.N_nodes = self.N_nodes[shuffled_indices].contiguous()
                self.degrees = self.degrees[shuffled_indices].contiguous()

        if self.drop_last:
            end = self.N_data - self.N_data % self.batch_size
        else:
            end = self.N_data

        for i in range(0, end, self.batch_size):
            yield self[i:i+self.batch_size]


    @staticmethod
    def getNodeHeatmaps(batch: dict):
        """
        Draw heatmap for the nodes
        :param batch: the batch of data
        :return: the heatmap of the nodes
        """
        B, _, H, W = batch["heatmap"].shape

        node_maps = torch.zeros((B, 1, H, W), dtype=torch.float32, device=DEVICE)

        for i in range(B):
            # Get the bounding box of the segment
            trajs = batch["trajs"][i]
            L_traj = batch["L_traj"][i]
            points = torch.cat([trajs[j, :L_traj[j]] for j in range(trajs.shape[0])], dim=0)
            min_point = torch.min(points, dim=0, keepdim=True).values
            max_point = torch.max(points, dim=0, keepdim=True).values
            point_range = max_point - min_point

            segs = batch["segs"][i]     # (N_segs, N_interp, 2)
            segs = segs[torch.all(segs.flatten(1) != 0, dim=1)]

            segs = (segs - min_point.view(1, 1, 2)) / point_range.view(1, 1, 2)

            segs[..., 0] = torch.clip(segs[..., 0] * W, 0, W-1)
            segs[..., 1] = torch.clip(segs[..., 1] * H, 0, H-1)

            seg_end_points = segs[:, [0, -1], :].flatten(0, 1)  # (2 * N_segs, 2)

            # fill seg end points pixels
            node_map = torch.zeros((H, W), dtype=torch.float32, device=DEVICE)
            node_map[seg_end_points[:, 1].long(), seg_end_points[:, 0].long()] = 1

            node_maps[i, 0] = node_map

        return {"node_heatmap": node_maps}


    @staticmethod
    def sequencesToSegments(seqs: torch.Tensor, L_seg: int) -> torch.Tensor:
        # seqs: (B, N_seqs, L_seq, D_token)
        B, N_seqs, L_seq, D_token = seqs.shape

        result = torch.cat([
            seqs[:, :, :-1].view(B, N_seqs, -1, L_seg - 1, 2),  # (B, N_seqs, N_segs, L_seg-1, 2)
            seqs[:, :, L_seg - 1::L_seg - 1].unsqueeze(3)],  # (B, N_seqs, N_segs, 1, 2)
            dim=-2)

        # (B, N_seqs, N_segs, L_seg, 2)

        return result.flatten(-2, -1)  # (B, N_seqs, N_segs, L_seg * 2)


    def segmentsToNodesAdj(self) -> None:
        B, N_segs, N_interp, _ = self.segs.shape
        nodes_padded = []
        adj_padded = []
        edges_padded = []
        nodes_counts = []

        for i in range(self.N_data):
            segs = self.segs[i]  # Shape: (N_segs, N_interp, 2)
            # Reshape to (2 * N, 2) to get all endpoints
            end_points = segs[:, [0, -1], :].flatten(0, 1)  # (2N, 2)

            # Extract unique nodes (2D points)
            unique_nodes, inverse_indices = torch.unique(end_points, dim=0, return_inverse=True)
            num_nodes = unique_nodes.shape[0]
            nodes_counts.append(num_nodes)

        self.N_nodes = torch.tensor(nodes_counts, dtype=torch.long)
        max_node_count = self.N_nodes.max()
        self.max_N_nodes = max_node_count

        for i in tqdm(range(self.N_data), desc="Building Nodes and Adjacency Matrices"):
            segs = self.segs[i]  # Shape: (N_segs, N_interp, 2)
            # Reshape to (2 * N, 2) to get all endpoints
            end_points = segs[:, [0, -1], :].flatten(0, 1)  # (2N, 2)

            # Extract unique nodes (2D points)
            unique_nodes, inverse_indices = torch.unique(end_points, dim=0, return_inverse=True)
            num_nodes = unique_nodes.shape[0]

            nodes_counts.append(num_nodes)

            node_tensor = torch.zeros((max_node_count, 2))
            node_tensor[:num_nodes, :2] = unique_nodes

            nodes_padded.append(node_tensor)

            # Initialize adjacency matrix of size (nodes_pad_len, nodes_pad_len)
            adj_matrix = torch.zeros((max_node_count, max_node_count), dtype=torch.int32)
            edge_feature_matrix = torch.zeros((max_node_count, max_node_count, 16), dtype=torch.float32)

            # Fill adjacency matrix
            for j in range(N_segs):
                # Get the indices of the two points of the line segment in the unique nodes list
                p1_idx = inverse_indices[2 * j]  # First point of the line segment
                p2_idx = inverse_indices[2 * j + 1]  # Second point of the line segment
                adj_matrix[p1_idx, p2_idx] = 1
                adj_matrix[p2_idx, p1_idx] = 1  # Undirected segs

                # edge feature is the corresponding segment
                edge_feature_matrix[p1_idx, p2_idx] = segs[j].flatten()
                edge_feature_matrix[p2_idx, p1_idx] = segs[j].flip(0).flatten()

            adj_padded.append(adj_matrix)
            edges_padded.append(edge_feature_matrix)

        # Convert lists to tensors using torch.stack
        self.nodes = torch.stack(nodes_padded)  # Shape: (B, nodes_pad_len, 2)
        self.adj_mats = torch.stack(adj_padded).to(torch.float32)  # Shape: (B, nodes_pad_len, nodes_pad_len)
        self.edges = torch.stack(edges_padded)  # Shape: (B, nodes_pad_len, nodes_pad_len, 16)
        self.degrees = torch.sum(self.adj_mats, dim=-1)     # (B, nodes_pad_len)


    @staticmethod
    def getJointsFromSegments(segments) -> Dict[str, Tensor]:
        """
        Computes the adjacency (joint) matrix for a batch of line segments.

        Args:
            segments (torch.Tensor): A tensor of shape (B, N, D), where D=5 (x1, y1, x2, y2, flag).

        Returns:
            torch.Tensor: A joint matrix of shape (B, N, N) where each entry (i, j) is 1 if segments i and j are joint, 0 otherwise.
        """
        B, N, _ = segments.shape

        p1 = segments[:, :, 0:2]    # (B, N, 2)
        p2 = segments[:, :, 2:4]    # (B, N, 2)

        # p1p1_match[i, j] = 1 if p1[i] == p1[j]
        p1p1_match = torch.cdist(p1, p1) < 1e-5   # (B, N, N)
        p1p2_match = torch.cdist(p1, p2) < 1e-5   # (B, N, N)
        p2p1_match = torch.cdist(p2, p1) < 1e-5   # (B, N, N)
        p2p2_match = torch.cdist(p2, p2) < 1e-5   # (B, N, N)

        # Combine the matches
        joint_matrix = p1p1_match | p1p2_match | p2p1_match | p2p2_match

        return {"joints": joint_matrix.to(torch.float32)}

    def getTargetHeatmaps(self, line_width: int = 3):
        """
        Compute the target heatmaps for the given segments
        :param segs: the segments tensor
        :param bboxes: (min_lat, max_lat, min_lon, max_lon)
        :param H: the height of the heatmap
        :param W: the width of the heatmap
        :param line_width: the width of the line
        :return: the target heatmaps of shape (B, 1, H, W)
        """
        target_heatmaps = np.zeros((self.N_data, 1, self.img_H, self.img_W), dtype=np.uint8)

        for i in range(self.N_data):
            # Get the bounding box of the segment
            points = torch.cat([self.trajs[i, j, :self.L_traj[i, j]] for j in range(self.trajs[i].shape[0])], dim=0)
            if len(points) == 0:
                continue
            min_point = torch.min(points, dim=0, keepdim=True).values
            max_point = torch.max(points, dim=0, keepdim=True).values
            point_range = max_point - min_point

            segs = self.segs[i]     # (N_segs, N_interp, 2)
            segs = segs[torch.all(segs.flatten(1) != 0, dim=1)]

            segs = (segs - min_point.view(1, 1, 2)) / point_range.view(1, 1, 2)

            segs[..., 0] *= self.img_W
            segs[..., 1] *= self.img_H

            lons = segs[:, :, 0].cpu().numpy().astype(np.int32)
            lats = segs[:, :, 1].cpu().numpy().astype(np.int32)

            for j in range(len(segs)):
                # Draw the polyline
                cv2.polylines(target_heatmaps[i, 0],
                              [np.stack([lons[j], lats[j]], axis=-1)],
                                isClosed=False,
                                color=1,
                                thickness=line_width)

        self.target_heatmaps = torch.tensor(target_heatmaps, dtype=torch.float32, device=DEVICE)
