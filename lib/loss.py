import pdb
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from lib.warping import *
from externals.d2net.lib.utils import *

class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()

    def forward(self, attentions1, attentions2, correspondence):
        return F.mse_loss(attention1, attention2)

class CorrespondenceLoss(nn.Module):
    """ Adapted from 'D2-Net: A Trainable CNN for Joint Description and Detection of Local Features'

    https://arxiv.org/pdf/1905.03561.pdf
    https://github.com/mihaidusmanu/d2-net/blob/master/lib/loss.py
    """

    def __init__(self, margin=1, safe_radius=4, scaling_steps=3):
        super().__init__()
        self.margin = margin
        self.safe_radius = safe_radius
        self.scaling_steps = scaling_steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu") #torch.device('cpu') if not torch.cuda.is_available() else 
        self.plot = False

    def forward(self, x1_encoded, x2_encoded, correspondences):
        loss = torch.tensor(np.array([0], dtype=np.float32), device = self.device)
        #ToDo: only count valid samples!
        n_valid = 0.0
        for idx in range(x1_encoded.shape[0]):
            l = self.correspondence_loss(x1_encoded, x2_encoded, correspondences, idx)
            if l is not None:
                loss += l
                n_valid += 1.0
        return loss / n_valid

    def correspondence_loss(self, x1_encoded, x2_encoded, correspondences, idx):
        """
        x1_encoded: 1xBx48xWxH (dense_features1) (usually x32x32)
        x2_encoded: 1xBx48xWxH (dense_features2) (usually x32x32)
        attentions1: 1xBx1xWxH (scores1)
        attentions2: 1xBx1xWxH (scores2)
        correspondences (image1: 1xBx3x256x256 = 1xBx3xw1xh1, ...) -> 32 * 8 = 256 --> 3 scaling steps
        idx: batch index (basically loop over all images in the batch)
        out: scalar tensor (1)
        """

        depth1 = correspondences['depth1'][idx].to(self.device) # [h1, w1]???
        intrinsics1 = correspondences['intrinsics1'][idx].to(self.device)  # [3, 3]
        pose1 = correspondences['pose1'][idx].view(4, 4).to(self.device)  # [4, 4]
        bbox1 = correspondences['bbox1'][idx].to(self.device) # [2]

        depth2 = correspondences['depth2'][idx].to(self.device)
        intrinsics2 = correspondences['intrinsics2'][idx].to(self.device)
        pose2 = correspondences['pose2'][idx].view(4, 4).to(self.device)
        bbox2 = correspondences['bbox2'][idx].to(self.device)
        

        # Network output

        dense_features1 = x1_encoded[idx] #48x32x32
        c, h1, w1 = dense_features1.size()

        dense_features2 = x2_encoded[idx]
        _, h2, w2 = dense_features2.size()

        all_descriptors1 = F.normalize(dense_features1.view(c, -1), dim=0)# 48x1024, row-major
        descriptors1 = all_descriptors1

        all_descriptors2 = F.normalize(dense_features2.view(c, -1), dim=0)# 48x1024

        # Warp the positions from image 1 to image 2
        fmap_pos1 = grid_positions(h1, w1, self.device) #feature positions, 2x(32*32)=2x1024 [y,x]-format -> [[0,0],[0,1], ...[32,32]]
        pos1 = upscale_positions(fmap_pos1, scaling_steps=self.scaling_steps) # feature positions in 256x256, [y,x]-format -> [[0,0],[0,11.5], ...[256,256]]
        #ids: matching ids in sequence (256*256)
        #default pos1 has ids [0, ..., 1024]
        # now ids says which of these are valid based on relative transformation between them, e.g.
        # [5, 28, 32, ...,1020]
        # so a legal correspondence would be pos1[:,5]<-->pos2[:,0]
        try:
            pos1, pos2, ids = warp(pos1,
                depth1, intrinsics1, pose1, bbox1,
                depth2, intrinsics2, pose2, bbox2)
        except EmptyTensorError:
            return None
        fmap_pos1 = fmap_pos1[:, ids] #uv-positions on 32x32 grid, but in list format 2xlen(ids)
        descriptors1 = descriptors1[:, ids] #again as list 48xlen(ids)

        # Skip the pair if not enough GT correspondences are available
        if ids.size(0) < 128:
            return None

        # Descriptors at the corresponding positions
        fmap_pos2 = torch.round(
            downscale_positions(pos2, scaling_steps=self.scaling_steps)
        ).long()
        descriptors2 = F.normalize(
            dense_features2[:, fmap_pos2[0, :], fmap_pos2[1, :]],
            dim=0
        )
        positive_distance = 2 - 2 * (
            descriptors1.t().unsqueeze(1) @ descriptors2.t().unsqueeze(2)
        ).squeeze() #p(c) in paper, ||dA -dB||

        all_fmap_pos2 = grid_positions(h2, w2, self.device)
        position_distance = torch.max(
            torch.abs(
                fmap_pos2.unsqueeze(2).float() -
                all_fmap_pos2.unsqueeze(1)
            ),
            dim=0
        )[0] # all other distances within image2 (distance is feature-metric norm!)
        is_out_of_safe_radius = position_distance > self.safe_radius
        distance_matrix = 2 - 2 * (descriptors1.t() @ all_descriptors2) # ||dA -dN2||
        negative_distance2 = torch.min(
            distance_matrix + (1 - is_out_of_safe_radius.float()) * 10., #weird
            dim=1
        )[0] 

        all_fmap_pos1 = grid_positions(h1, w1, self.device)
        position_distance = torch.max(
            torch.abs(
                fmap_pos1.unsqueeze(2).float() -
                all_fmap_pos1.unsqueeze(1)
            ),
            dim=0
        )[0]
        is_out_of_safe_radius = position_distance > self.safe_radius
        distance_matrix = 2 - 2 * (descriptors2.t() @ all_descriptors1)  # ||dB -dN1||
        negative_distance1 = torch.min(
            distance_matrix + (1 - is_out_of_safe_radius.float()) * 10., #weird
            dim=1
        )[0]

        diff = positive_distance - torch.min(
            negative_distance1, negative_distance2
        ) # (n(c))

        return torch.mean(F.relu(self.margin + diff))


class TripletMarginLoss(nn.Module):
    """ Taken from 'D2-Net: A Trainable CNN for Joint Description and Detection of Local Features'

    https://arxiv.org/pdf/1905.03561.pdf
    https://github.com/mihaidusmanu/d2-net/blob/master/lib/loss.py
    """
    def __init__(self, margin=1, safe_radius=4, scaling_steps=3):
        super().__init__()
        self.margin = margin
        self.safe_radius = safe_radius
        self.scaling_steps = scaling_steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu") #torch.device('cpu') if not torch.cuda.is_available() else 

        self.plot = False

    def forward(self, x1_encoded, x2_encoded, attentions1, attentions2, correspondences):
        loss = torch.tensor(np.array([0], dtype=np.float32), device = self.device)
        #ToDo: only count valid samples!
        for idx in range(x1_encoded.shape[0]):
            loss += self.triplet_margin_loss(x1_encoded, x2_encoded, attentions1, attentions2, correspondences, idx)
        return loss / x1_encoded.shape[0]

    def triplet_margin_loss(self, x1_encoded, x2_encoded, attentions1, \
        attentions2, correspondences, idx):
        """
        x1_encoded: 1xBx48xWxH (dense_features1) (usually x32x32)
        x2_encoded: 1xBx48xWxH (dense_features2) (usually x32x32)
        attentions1: 1xBx1xWxH (scores1)
        attentions2: 1xBx1xWxH (scores2)
        correspondences (image1: 1xBx3x256x256 = 1xBx3xw1xh1, ...) -> 32 * 8 = 256 --> 3 scaling steps
        idx: batch index (basically loop over all images in the batch)
        out: scalar tensor (1)
        """

        
        loss = torch.tensor(np.array([0], dtype=np.float32), device=self.device)
        depth1 = correspondences['depth1'][idx].to(self.device) # [h1, w1]???
        intrinsics1 = correspondences['intrinsics1'][idx].to(self.device)  # [3, 3]
        pose1 = correspondences['pose1'][idx].view(4, 4).to(self.device)  # [4, 4]
        bbox1 = correspondences['bbox1'][idx].to(self.device) # [2]

        depth2 = correspondences['depth2'][idx].to(self.device)
        intrinsics2 = correspondences['intrinsics2'][idx].to(self.device)
        pose2 = correspondences['pose2'][idx].view(4, 4).to(self.device)
        bbox2 = correspondences['bbox2'][idx].to(self.device)
        

        # Network output

        dense_features1 = x1_encoded[idx] #48x32x32
        c, h1, w1 = dense_features1.size()
        scores1 = attentions1[idx].view(-1) #1x1024 (ids format)

        dense_features2 = x2_encoded[idx]
        _, h2, w2 = dense_features2.size()
        scores2 = attentions2[idx].squeeze(0)

        all_descriptors1 = F.normalize(dense_features1.view(c, -1), dim=0)# 48x1024, row-major
        descriptors1 = all_descriptors1

        all_descriptors2 = F.normalize(dense_features2.view(c, -1), dim=0)# 48x1024

        # Warp the positions from image 1 to image 2
        fmap_pos1 = grid_positions(h1, w1, self.device) #feature positions, 2x(32*32)=2x1024 [y,x]-format -> [[0,0],[0,1], ...[32,32]]
        pos1 = upscale_positions(fmap_pos1, scaling_steps=self.scaling_steps) # feature positions in 256x256, [y,x]-format -> [[0,0],[0,11.5], ...[256,256]]
        #ids: matching ids in sequence (256*256)
        #default pos1 has ids [0, ..., 1024]
        # now ids says which of these are valid based on relative transformation between them, e.g.
        # [5, 28, 32, ...,1020]
        # so a legal correspondence would be pos1[:,5]<-->pos2[:,0]
        try:
            pos1, pos2, ids = warp(pos1,
                depth1, intrinsics1, pose1, bbox1,
                depth2, intrinsics2, pose2, bbox2)
        except EmptyTensorError:
            return loss
        fmap_pos1 = fmap_pos1[:, ids] #uv-positions on 32x32 grid, but in list format 2xlen(ids)
        descriptors1 = descriptors1[:, ids] #again as list 48xlen(ids)
        scores1 = scores1[ids] #again as list 1xlen(ids)

        # Skip the pair if not enough GT correspondences are available
        if ids.size(0) < 128:
            return loss

        # Descriptors at the corresponding positions
        fmap_pos2 = torch.round(
            downscale_positions(pos2, scaling_steps=self.scaling_steps)
        ).long()
        descriptors2 = F.normalize(
            dense_features2[:, fmap_pos2[0, :], fmap_pos2[1, :]],
            dim=0
        )
        positive_distance = 2 - 2 * (
            descriptors1.t().unsqueeze(1) @ descriptors2.t().unsqueeze(2)
        ).squeeze() #p(c) in paper, ||dA -dB||

        all_fmap_pos2 = grid_positions(h2, w2, self.device)
        position_distance = torch.max(
            torch.abs(
                fmap_pos2.unsqueeze(2).float() -
                all_fmap_pos2.unsqueeze(1)
            ),
            dim=0
        )[0] # all other distances within image2 (distance is feature-metric norm!)
        is_out_of_safe_radius = position_distance > self.safe_radius
        distance_matrix = 2 - 2 * (descriptors1.t() @ all_descriptors2) # ||dA -dN2||
        negative_distance2 = torch.min(
            distance_matrix + (1 - is_out_of_safe_radius.float()) * 10., #weird
            dim=1
        )[0] 

        all_fmap_pos1 = grid_positions(h1, w1, self.device)
        position_distance = torch.max(
            torch.abs(
                fmap_pos1.unsqueeze(2).float() -
                all_fmap_pos1.unsqueeze(1)
            ),
            dim=0
        )[0]
        is_out_of_safe_radius = position_distance > self.safe_radius
        distance_matrix = 2 - 2 * (descriptors2.t() @ all_descriptors1)  # ||dB -dN1||
        negative_distance1 = torch.min(
            distance_matrix + (1 - is_out_of_safe_radius.float()) * 10., #weird
            dim=1
        )[0]

        diff = positive_distance - torch.min(
            negative_distance1, negative_distance2
        ) # (n(c))

        scores2 = scores2[fmap_pos2[0, :], fmap_pos2[1, :]].view(-1)

        if (self.plot):
            # We should put this in separate functions to visualize the attention maps
            pos1_aux = pos1.cpu().numpy()
            pos2_aux = pos2.cpu().numpy()
            # print(pos1_aux)
            # print(pos2_aux)
            k = pos1_aux.shape[1]
            col = np.random.rand(k, 3)
            n_sp = 4
            plt.figure(figsize=(15,15))
            plt.subplot(1, n_sp, 1)
            im1 = imshow_image(
                correspondences['image1'][idx].cpu().numpy(),
            )
            plt.imshow(im1)
            plt.scatter(
                pos1_aux[1, :], pos1_aux[0, :],
                s=1.0, c=col, marker=',', alpha=0.5
            )
            plt.axis('off')
            plt.subplot(1, n_sp, 2)

            plt.imshow(
                attentions1[idx].squeeze(0).data.cpu().numpy(),
                cmap='Reds'
            )
            plt.axis('off')
            plt.subplot(1, n_sp, 3)
            im2 = imshow_image(
                correspondences['image2'][idx].cpu().numpy()
            )
            plt.imshow(im2)
            plt.scatter(
                pos2_aux[1, :], pos2_aux[0, :],
                s=1.0, c=col, marker=',', alpha=0.5
            )
            plt.axis('off')
            plt.subplot(1, n_sp, 4)
            plt.imshow(
                attentions2[idx].squeeze(0).data.cpu().numpy(),
                cmap='Reds'
            )
            plt.axis('off')
            # savefig('train_vis/%s.%02d.%02d.%d.png' % (
            #     'train' if batch['train'] else 'valid',
            #     batch['epoch_idx'],
            #     batch['batch_idx'] // batch['log_interval'],
            #     idx_in_batch
            # ), dpi=300)
            # plt.close()

        return (torch.sum(scores1 * scores2 * F.relu(self.margin + diff)) /
            torch.sum(scores1 * scores2))



class PeakyLoss (nn.Module):
    """ Distinctiveness loss, adapted from 'R2D2: Repeatable and Reliable Detector and Descriptor'

    https://arxiv.org/pdf/1906.06195.pdf
    https://github.com/naver/r2d2/blob/master/nets/repeatability_loss.py

    Tries to make the repeatability locally peaky.
    Mechanism: we maximize, for each pixel, the difference between the local mean and the local max.
    """
    def __init__(self, N=16):
        nn.Module.__init__(self)
        self.name = f'peaky{N}'
        assert N % 2 == 0, 'N must be pair'
        self.preproc = nn.AvgPool2d(3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(N+1, stride=1, padding=N//2)
        self.avgpool = nn.AvgPool2d(N+1, stride=1, padding=N//2)

    def forward_one(self, sali):
        sali = self.preproc(sali) # remove super high frequency
        return 1 - (self.maxpool(sali) - self.avgpool(sali)).mean()

    def forward(self, repeatability, **kw):
        sali1, sali2 = repeatability
        return (self.forward_one(sali1) + self.forward_one(sali2)) /2

class DistinctivenessLoss(nn.Module):
    """ Distinctiveness loss, adapted from 'D2D: Learning to find good correspondences for image matching and manipulation'

    https://arxiv.org/pdf/2007.08480.pdf
    """

    def __init__(self, margin=1, safe_radius=4, scaling_steps=3, N=32):
        super().__init__()
        self.margin = margin
        self.safe_radius = safe_radius
        self.scaling_steps = scaling_steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu") #torch.device('cpu') if not torch.cuda.is_available() else 
        
        self.tau = 0.25
        self.K = 512

        self.name = f'peaky_dist{N}'
        assert N % 2 == 0, 'N must be pair'
        self.preproc = nn.AvgPool2d(3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(N+1, stride=1, padding=N//2)
        self.avgpool = nn.AvgPool2d(N+1, stride=1, padding=N//2)
        self.lambda_peaky = 0.2
        self.plot = False
        self.m = nn.ReLU(True)
    
    def forward(self, x1_encoded, x2_encoded, attentions1, attentions2, correspondences):
        loss = torch.tensor(np.array([0], dtype=np.float32), device = self.device)
        #ToDo: only count valid samples!
        n_valid = 0.0
        for idx in range(x1_encoded.shape[0]):
            l = self.distinctiveness_loss(x1_encoded, x2_encoded, attentions1, attentions2, correspondences, idx)
            if l is not None:
                loss += l
                n_valid += 1.0
        loss = loss / n_valid + self.lambda_peaky * (self.peaky_loss(attentions1) + self.peaky_loss(attentions2)) / 2.0
        return loss

    def pick_K_in_each_row(self, distances, safe = True):
        n, d = distances.shape

        if safe:
            sample = torch.cat([torch.randperm(d)[:self.K] for _ in range(n)]).view(n, self.K)
            mask = torch.zeros(n, d, dtype=torch.bool)
            mask.scatter_(dim=1, index=sample, value=True)
        else:
            rand_mat = torch.rand(n, d)
            k_th_quant = torch.topk(rand_mat, self.K, largest = False)[0][:,-1:]
            mask = rand_mat <= k_th_quant

        return torch.masked_select(distances, mask).view(n,self.K)

    def peaky_loss(self, attentions):
        """
        Measures distinctiveness within one image
        """
        sali = self.preproc(attentions) # remove super high frequency
        return self.m(1 - (self.maxpool(sali) - self.avgpool(sali)).mean())
    
    def distinctiveness_loss(self, x1_encoded, x2_encoded, attentions1, attentions2, correspondences, idx):
        """
        x1_encoded: 1xBx48xWxH (dense_features1) (usually x32x32)
        x2_encoded: 1xBx48xWxH (dense_features2) (usually x32x32)
        attentions1: 1xBx1xWxH (scores1)
        attentions2: 1xBx1xWxH (scores2)
        correspondences (image1: 1xBx3x256x256 = 1xBx3xw1xh1, ...) -> 32 * 8 = 256 --> 3 scaling steps
        idx: batch index (basically loop over all images in the batch)
        out: scalar tensor (1)
        """

        
        #loss = torch.tensor(np.array([0], dtype=np.float32), device=self.device)
        depth1 = correspondences['depth1'][idx].to(self.device) # [h1, w1]???
        intrinsics1 = correspondences['intrinsics1'][idx].to(self.device)  # [3, 3]
        pose1 = correspondences['pose1'][idx].view(4, 4).to(self.device)  # [4, 4]
        bbox1 = correspondences['bbox1'][idx].to(self.device) # [2]

        depth2 = correspondences['depth2'][idx].to(self.device)
        intrinsics2 = correspondences['intrinsics2'][idx].to(self.device)
        pose2 = correspondences['pose2'][idx].view(4, 4).to(self.device)
        bbox2 = correspondences['bbox2'][idx].to(self.device)
        

        # Network output

        dense_features1 = x1_encoded[idx] #48x32x32
        c, h1, w1 = dense_features1.size()
        scores1 = attentions1[idx].view(-1) #1x1024 (ids format)

        dense_features2 = x2_encoded[idx]
        _, h2, w2 = dense_features2.size()
        scores2 = attentions2[idx].squeeze(0)

        all_descriptors1 = F.normalize(dense_features1.view(c, -1), dim=0)# 48x1024, row-major
        descriptors1 = all_descriptors1

        all_descriptors2 = F.normalize(dense_features2.view(c, -1), dim=0)# 48x1024

        # Warp the positions from image 1 to image 2
        fmap_pos1 = grid_positions(h1, w1, self.device) #feature positions, 2x(32*32)=2x1024 [y,x]-format -> [[0,0],[0,1], ...[32,32]]
        pos1 = upscale_positions(fmap_pos1, scaling_steps=self.scaling_steps) # feature positions in 256x256, [y,x]-format -> [[0,0],[0,11.5], ...[256,256]]
        #ids: matching ids in sequence (256*256)
        #default pos1 has ids [0, ..., 1024]
        # now ids says which of these are valid based on relative transformation between them, e.g.
        # [5, 28, 32, ...,1020]
        # so a legal correspondence would be pos1[:,5]<-->pos2[:,0]
        try:
            pos1, pos2, ids = warp(pos1,
                depth1, intrinsics1, pose1, bbox1,
                depth2, intrinsics2, pose2, bbox2)
        except EmptyTensorError:
            return None
        fmap_pos1 = fmap_pos1[:, ids] #uv-positions on 32x32 grid, but in list format 2xlen(ids)
        descriptors1 = descriptors1[:, ids] #again as list 48xlen(ids)
        scores1 = scores1[ids] #again as list 1xlen(ids)

        # Skip the pair if not enough GT correspondences are available
        if ids.size(0) < 128:
            return None

        # Descriptors at the corresponding positions
        fmap_pos2 = torch.round(
            downscale_positions(pos2, scaling_steps=self.scaling_steps)
        ).long()
        descriptors2 = F.normalize(
            dense_features2[:, fmap_pos2[0, :], fmap_pos2[1, :]],
            dim=0
        )
        scores2 = scores2[fmap_pos2[0, :], fmap_pos2[1, :]].view(-1)

        indices1 = torch.randperm(all_descriptors1.shape[1], device = self.device)[self.K]
        indices2 = torch.randperm(all_descriptors2.shape[1], device = self.device)[self.K]
        
        distances1 = torch.cdist(descriptors1.t(), all_descriptors2.t()) #shape (338x1024)
        distances2 = torch.cdist(descriptors2.t(), all_descriptors1.t()) #shape (338x1024)
        
        # distances1 = self.pick_K_in_each_row(distances1).to(self.device)
        # distances2 = self.pick_K_in_each_row(distances2).to(self.device)

        mx1 = torch.sum(distances1 < self.margin,1)
        mx2 = torch.sum(distances2 < self.margin,1)
        
        target1 = 1.0 / (1.0 + mx1)**self.tau
        target2 = 1.0 / (1.0 + mx2)**self.tau
        
        loss = torch.mean(torch.abs(scores1 - target1)) + torch.mean(torch.abs(scores2 - target2))
        
        return loss

