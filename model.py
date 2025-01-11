import open_clip
import torch
from utils import *




class WinCLIP_AD(torch.nn.Module):
    def __init__(self,model_name = 'ViT-B-16-plus-240'):
        super(WinCLIP_AD, self).__init__()
        self.model, _, self.preprocess = open_clip.create_customer_model_and_transforms(model_name, pretrained='laion400m_e31')
        # self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion400m_e31')
        self.mask = patch_scale((240,240))
    def multiscale(self):
        pass
    def encode_text(self, text):
        return self.model.encode_text(text)
    def encode_image(self, image, patch_size, mask=True):
        if mask:
            b, _, _, _ = image.shape
            large_scale = self.mask.make_mask(kernel_size=48, patch_size=patch_size).squeeze().cuda()
            mid_scale = self.mask.make_mask(kernel_size=32, patch_size=patch_size).squeeze().cuda()
            tokens_list, class_tokens, patch_tokens = self.model.encode_image(image, [large_scale,mid_scale], proj = False)
            large_scale_tokens, mid_scale_tokens = tokens_list[0], tokens_list[1]
            return large_scale_tokens, mid_scale_tokens, patch_tokens.unsqueeze(2), class_tokens, large_scale, mid_scale