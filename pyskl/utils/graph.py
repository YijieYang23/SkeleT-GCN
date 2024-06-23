import numpy as np
import torch


def k_adjacency(A, k, with_self=False, self_factor=1):
    # A is a 2D square array
    if isinstance(A, torch.Tensor):
        A = A.data.cpu().numpy()
    assert isinstance(A, np.ndarray)
    Iden = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return Iden
    Ak = np.minimum(np.linalg.matrix_power(A + Iden, k), 1) - np.minimum(np.linalg.matrix_power(A + Iden, k - 1), 1)
    if with_self:
        Ak += (self_factor * Iden)
    return Ak


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A, dim=0):
    # A is a 2D square array
    Dl = np.sum(A, dim)
    h, w = A.shape
    Dn = np.zeros((w, w))

    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)

    AD = np.dot(A, Dn)
    return AD


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.eye(num_node)

    for i, j in edge:
        A[i, j] = 1
        A[j, i] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [
        np.linalg.matrix_power(A, d) for d in range(max_hop + 1)
    ]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


class Graph:

    def __init__(self,
                 layout='coco',
                 mode='spatial',
                 max_hop=1,
                 nx_node=1,
                 num_filter=3,
                 init_std=0.02,
                 init_off=0.04):

        self.max_hop = max_hop
        self.layout = layout
        self.mode = mode
        self.num_filter = num_filter
        self.init_std = init_std
        self.init_off = init_off
        self.nx_node = nx_node

        if layout.startswith('random_'):
            self.num_node = int(layout[7:])
            self.A = getattr(self, mode)()
        else:
            assert nx_node == 1 or mode == 'random', "nx_node can be > 1 only if mode is 'random'"
            assert layout in ['openpose', 'nturgb+d', 'coco', 'coco-wholebody', 'expressive-keypoints']

            self.get_layout(layout)
            self.hop_dis = get_hop_distance(self.num_node, self.inward, max_hop)

            assert hasattr(self, mode), f'Do Not Exist This Mode: {mode}'
            self.A = getattr(self, mode)()

    def __str__(self):
        return self.A

    def get_layout(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self.inward = [
                (4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9),
                (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0),
                (14, 0), (17, 15), (16, 14)
            ]
            self.center = 1
        elif layout == 'nturgb+d':
            self.num_node = 25
            neighbor_base = [
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)
            ]
            self.inward = [(i - 1, j - 1) for (i, j) in neighbor_base]
            self.center = 21 - 1
        elif layout == 'coco':
            self.num_node = 17
            self.inward = [
                (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
                (1, 0), (3, 1), (2, 0), (4, 2)
            ]
            self.center = 0
        elif layout.startswith('coco-wholebody'):
            assert layout in ['coco-wholebody', 'expressive-keypoints']
            body_mapping = [(i, j) for (i, j) in [
                [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                [8, 10], [1, 2], [0, 1], [0, 2],
                [1, 3], [2, 4], [0, 4], [0, 3],
                [0, 5], [0, 6]
            ]]
            face_mapping = [(i - 1, j - 1) for (i, j) in [
                (32, 31), (31, 30), (30, 29), (29, 28), (28, 27),
                (27, 26), (26, 25), (25, 24), (24, 41), (41, 42),
                (42, 43), (43, 44), (44, 45), (45, 46), (46, 47),
                (47, 48), (48, 49), (49, 50), (50, 40), (40, 39),
                (39, 38), (38, 37), (37, 36), (36, 35), (35, 34),
                (34, 33), (33, 32), (45, 51), (46, 51), (60, 61),
                (61, 62), (62, 63), (63, 64), (64, 65), (65, 60),
                (69, 68), (68, 67), (67, 66), (66, 71), (71, 70),
                (70, 69), (63, 51), (66, 51), (51, 52), (52, 53),
                (53, 54), (55, 56), (56, 57), (57, 58), (58, 59),
                (54, 57), (72, 73), (73, 74), (74, 75), (75, 76),
                (76, 77), (77, 78), (78, 79), (79, 80), (80, 81),
                (81, 82), (82, 83), (83, 72), (84, 85), (85, 86),
                (86, 87), (87, 88), (88, 89), (89, 90), (90, 91),
                (91, 84), (72, 84), (78, 88), (57, 75), (75, 86),
                (86, 90), (90, 81), (81, 32), (24, 60), (40, 69),
                (55, 54), (59, 54)
            ]]
            lhand_mapping = [(i, j) for (i, j) in [
                [91, 92], [92, 93], [93, 94], [94, 95], [91, 96],
                [96, 97], [97, 98], [98, 99], [91, 100], [100, 101],
                [101, 102], [102, 103], [91, 104], [104, 105],
                [105, 106], [106, 107], [91, 108], [108, 109],
                [109, 110], [110, 111],
            ]]
            rhand_mapping = [(i, j) for (i, j) in [
                [112, 113], [113, 114], [114, 115], [115, 116],
                [112, 117], [117, 118], [118, 119], [119, 120],
                [112, 121], [121, 122], [122, 123], [123, 124],
                [112, 125], [125, 126], [126, 127], [127, 128],
                [112, 129], [129, 130], [130, 131], [131, 132]
            ]]
            lfoot_mapping = [
                (17, 18), (18, 19), (19, 17)
            ]
            rfoot_mapping = [
                (20, 21), (21, 22), (22, 20)
            ]
            body2face_mapping = [(i - 1, j - 1) for (i, j) in [
                (1, 54), (2, 51), (3, 51), (4, 40), (5, 24)
            ]]
            body2feet_mapping = [
                (15, 19), (16, 22)
            ]
            body2hand_mapping = [
                (9, 91), (10, 112)
            ]
            if layout == 'coco-wholebody':
                self.num_node = 133
                self.inward = body_mapping + face_mapping + lhand_mapping + rhand_mapping + lfoot_mapping + \
                              rfoot_mapping + body2face_mapping + body2hand_mapping + body2feet_mapping
                self.center = 0
            elif layout == 'expressive-keypoints':
                # body  =0~16   (17 nodes)      body  =0~16     (17 nodes)
                # feet  =17~22  (6  nodes)      feet  =17~22    (6  nodes)
                # face  =23~90  (68 nodes)      hands =91-(91-23)~132-(91-23)
                # hands =91~132 (42 nodes)      hands =23~64    (42 nodes)
                self.num_node = 65
                face_index = range(23, 91)
                mapping = body_mapping + lhand_mapping + rhand_mapping + lfoot_mapping + rfoot_mapping \
                          + body2hand_mapping + body2feet_mapping
                self.inward = [
                    (i - len(face_index) if i >= face_index[0] else i, j - len(face_index) if j >= face_index[0] else j)
                    for (i, j) in mapping]
                self.center = 0
            else:
                raise ValueError(f'Do Not Exist This Layout: {layout}')
        else:
            raise ValueError(f'Do Not Exist This Layout: {layout}')
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward


    def spatial(self):
        Iden = edge2mat(self.self_link, self.num_node)
        In = normalize_digraph(edge2mat(self.inward, self.num_node))
        Out = normalize_digraph(edge2mat(self.outward, self.num_node))
        A = np.stack((Iden, In, Out))
        return A

    def skelet_spatial(self):
        Iden = edge2mat(self.self_link, self.num_node)
        In = edge2mat(self.inward, self.num_node)
        Out = edge2mat(self.outward, self.num_node)
        A = Iden + In + Out
        return A

    def binary_adj(self):
        A = edge2mat(self.inward + self.outward, self.num_node)
        return A[None]

    def random(self):
        num_node = self.num_node * self.nx_node
        return np.random.randn(self.num_filter, num_node, num_node) * self.init_std + self.init_off
