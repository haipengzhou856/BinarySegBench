import torch
from torch import nn
from transformers.modeling_outputs import SemanticSegmenterOutput
from typing import Optional
from diffusers import AutoencoderKL, AutoencoderTiny
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F




VAE = AutoencoderTiny.from_pretrained("madebyollin/taesd")

# TODO
FoundationModel = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name_or_path='/home/haipeng/Pretrained/nvidia/segformer-b3-finetuned-ade-512-512',
        ignore_mismatched_sizes=True,
        num_labels=4)


class ICCV(nn.Module):
    def __init__(self, vae=VAE, fm=FoundationModel, adapter=None):
        super().__init__()
        self.vae = vae
        self.encoder = fm.segformer
        self.decoder = fm.decode_head
        self.adpter = adapter # TODO
        #self.upsampler = Upsample(2, 4)
        '''self.linear_proj1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        )
        self.linear_proj2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=128, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2,padding=1),
        )
        self.linear_proj3 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=320, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=4,padding=1),
        )
        self.linear_proj4 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=512, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=8,padding=1),
        )'''

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2,padding=1),
            nn.LeakyReLU(),
        )

    def forward_train(self, pixel_values: torch.FloatTensor,
                    labels: torch.LongTensor) -> SemanticSegmenterOutput:
        with torch.no_grad():
            z = self.vae.encode(labels)
            z = z.latents # scale factor: 8'''
            #z_img = self.vae.encode(pixel_values)
            #z_img = z_img.latents # scale factor: 8

        '''img_embed0 = self.linear_proj1(z_img)
        img_embed1 = self.linear_proj2(z_img)
        img_embed2 = self.linear_proj3(z_img)
        img_embed3 = self.linear_proj4(z_img)'''

        pixel_values = self.downsample(pixel_values)
        outputs = self.encoder(pixel_values,output_hidden_states=True)
        encoder_hidden_states = outputs.hidden_states

        z_analog = self.decoder(encoder_hidden_states)
        l1_loss = F.l1_loss(z_analog, z)


        loss = l1_loss

        return SemanticSegmenterOutput(loss=loss)

    def forward_eval(self, pixel_values: torch.FloatTensor) -> SemanticSegmenterOutput:

        pixel_values = self.downsample(pixel_values)
        outputs = self.encoder(pixel_values,output_hidden_states=True)
        encoder_hidden_states = outputs.hidden_states

        z_analog = self.decoder(encoder_hidden_states)

        #z_analog = self.fm(pixel_values)
        #z_analog = z_analog.logits
        #z_analog = F.interpolate(z_analog, scale_factor=2, mode='nearest')
        # TODO
        #cascade_z = self.adpter(z_analog)
        with torch.no_grad():
            rec_mask = self.vae.decoder(z_analog)
        rec_mask = (rec_mask).clamp(0, 1)
        return SemanticSegmenterOutput(logits=rec_mask)

    def forward(self, pixel_values: torch.FloatTensor,
                labels: Optional[torch.LongTensor] = None) -> SemanticSegmenterOutput:
        if self.training:
            return self.forward_train(pixel_values, labels)
        else:
            return self.forward_eval(pixel_values)




if __name__ == '__main__':
    model = ICCV()
    input = torch.randn(4,3,2048,2048).cuda()
    label = torch.ones(4,3,2048,2048).cuda()
    model.cuda()
    output = model(input, label)
    print("pause")