import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def __init__(self, args, clip_extractor, using_pixel_list=False):
        super().__init__()
        self.extractor = clip_extractor
        self.using_pixel_list = using_pixel_list
        self.struc_lambda = getattr(args, "struc_lambda", 1)
        self.distri_lambda = getattr(args, "distri_lambda", 1)
    
    # def structure_loss(self, x, x_res):
    #     with torch.no_grad():
    #         target_self_sim = self.clip_extractor.get_self_sim(x)  # batch, 67, 67
    #     current_self_sim = self.clip_extractor.get_self_sim(x_res)
    #     loss = F.mse_loss(current_self_sim, target_self_sim, reduction="none")
    #     loss = loss.sum() / x.size(0)
    #     return loss
    
    def distribution_align_loss(self, x, x_res, pixel_list=None):
        """
        args:
            x: the images was augmentated
            x_res: reconstructed images
            pixel_list: meaning and std that extractor to use
        """
        if pixel_list and self.using_pixel_list:
            mean = pixel_list[0]
            std = pixel_list[1]
            with torch.no_grad():   # 0~255

                x = x * std + mean
                x_res = x_res * std + mean
            
        b = x.size(0)
        # contain a whole image embedding and (n - 1) random crop image embedding
        # [whole, random_crop1, random_crop2 ...]
        image_embedings = self.extractor.get_image_embedding(torch.cat([x, x_res], dim=0), aug=True)
        # import pdb;pdb.set_trace()
        loss = 0.0
        x_embeds = []
        x_res_embeds = []
        for im_embed in image_embedings:
            x_embed, x_res_embed = im_embed.split(split_size=b, dim=0)
            x_embeds.append(x_embed)
            x_res_embeds.append(x_res_embed)
        x_embeds = torch.cat(x_embeds, dim=0)
        x_res_embeds = torch.cat(x_res_embeds, dim=0)

        # import pdb; pdb.set_trace()
        # vector module
        x_l = x_embeds.norm(p=2, dim=-1)
        x_res_l = x_res_embeds.norm(p=2, dim=-1)
        
        # # L2 -> crossentropy
        # loss = loss + (- x_l * x_res_l.log()).mean()

        # length loss
        length_loss =  F.l1_loss(x_l, x_res_l, reduction='mean')
        # direction loss
        direction_loss = (1 - F.cosine_similarity(x_embeds, x_res_embeds, dim=-1)).mean()
        
        len_detach, direc_detach = length_loss.detach(), direction_loss.detach()
        total_detach = len_detach + direc_detach + 1e-4
        len_w = len_detach / total_detach
        direc_w = direc_detach / total_detach

        loss = loss + len_w * length_loss + direc_w * direction_loss
        # loss = loss + 1 * length_loss + 1 * direction_loss

        # loss = loss + direction_loss
        # loss = loss + length_loss

        # # distribution align loss
        # # cross-entropy
        # norm_x = torch.abs(x_embeds)
        # norm_x_res = torch.abs(x_res_embeds)
        # norm_x = norm_x / norm_x.sum(dim=1)[:, None]
        # norm_x_res = norm_x_res / norm_x_res.sum(dim=1)[:, None]
        # loss = loss + (- norm_x * norm_x_res.log()).sum(-1).mean()
        
        # # kl type
        # norm_x = F.log_softmax(x_embeds, dim=-1)
        # norm_x_res = F.softmax(x_res_embeds, dim=-1)
        # loss = loss + F.kl_div(norm_x_res, norm_x, reduction="batchmean")

        # # vector l1
        # loss = loss + F.l1_loss(x_embeds, x_res_embeds, reduction="mean")
        return loss
        
    def forward(self, x, x_res, pixel_list):
        total_loss = 0.0
        # calculate struct loss
        # total_loss += self.struc_lambda * self.structure_loss(x, x_res)
        # calculate distribution align loss
        total_loss += self.distri_lambda * self.distribution_align_loss(x, x_res, pixel_list)
        return total_loss