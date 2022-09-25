import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from omegaconf import OmegaConf
import segmentation_models_pytorch as smp
# import pretrainedmodels

import timm
class SMPUNet(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        CFG = OmegaConf.to_container(CFG)
        self.CFG = CFG
        if CFG.get("decoder_name") == 'smp_unet':
            if CFG.get("load_weights"):
                self.model = smp.Unet(
                    encoder_name = CFG.get("encoder_name"), 
                    encoder_depth = CFG.get("encoder_depth"), 
                    encoder_weights = "imagenet", 
                    decoder_channels = CFG.get("decoder_channels"),
                    decoder_attention_type = CFG.get("decoder_attention_type"),
                    in_channels = CFG.get("n_channels"), 
                    classes= CFG.get("num_classes")
                )
            else:
                self.model = smp.Unet(
                    encoder_name = CFG.get("encoder_name"), 
                    encoder_depth = CFG.get("encoder_depth"), 
                    encoder_weights = None, 
                    decoder_channels = CFG.get("decoder_channels"),
                    decoder_attention_type = CFG.get("decoder_attention_type"),
                    in_channels = CFG.get("n_channels"), 
                    classes= CFG.get("num_classes")
                )
            
        if CFG.get("encoder_name") == 'tu-tf_efficientnet_b7_ns':
            encoder_dims = (224, 80, 48, 32)
            
        if CFG.get("encoder_name") == 'tu-tf_efficientnet_b5_ns':
            encoder_dims = (176, 64, 40, 24)
            
        if CFG.get("aux"):
            self.auxes = nn.ModuleList([
                nn.Conv2d(encoder_dims[i], CFG.get("num_classes"), kernel_size=1, padding=0) for i in range(len(encoder_dims))
            ])
            
    def forward(self, x):
        ### encoder
        encoders = self.model.encoder(x)
        
        ### decoder
        encoders = encoders[1:]  # remove first skip with same spatial resolution
        features = encoders[::-1]  # reverse channels to start from head of encoder
        head = features[0]
        skips = features[1:]
        
            
        x = self.model.decoder.center(head)
        if 'convnext' in self.CFG.get("encoder_name"):
            y4 = self.model.decoder.blocks[0](x, skips[0])
            y3 = self.model.decoder.blocks[1](y4, skips[1])
            y2 = self.model.decoder.blocks[2](y3, skips[2])
            y1 = self.model.decoder.blocks[3](y2)
            y0 = F.interpolate(y1,scale_factor=2,mode='bilinear')
        else:
            y4 = self.model.decoder.blocks[0](x, skips[0])
            y3 = self.model.decoder.blocks[1](y4, skips[1])
            y2 = self.model.decoder.blocks[2](y3, skips[2])
            y1 = self.model.decoder.blocks[3](y2, skips[3])
            y0 = self.model.decoder.blocks[4](y1)
        x = y0
        ### head    
        logit = self.model.segmentation_head(x)
        r = {}
        r['logit'] = logit
        
        ### for aux loss
        if self.CFG.get("aux"):
            aux_logits = []
            for index, (feature,aux) in enumerate(zip(skips, self.auxes)):
                aux_logit = aux(feature)
                aux_logits.append(aux_logit)
            r['aux_logits'] = aux_logits
        return logit
        
from .encoders.timm_universal import *
from .encoders.swin_transformer_v1 import *
from .encoders.cswin_transformer import *
from .encoders.mix_transformer import mit_b0,mit_b1,mit_b2,mit_b3,mit_b4,mit_b5
from .encoders.cswin_transformer import CSWinTransformer
from .encoders.hila_mix_transformer import hila_mit_b2
from .encoders.coat import coat_lite_small, coat_lite_medium, coat_parallel_small_plus1
from .encoders.pvt_v2 import pvt_v2_b2, pvt_v2_b2_5level, pvt_v2_b4, pvt_v2_b4_5level, HybridPVTV2
from .encoders.crossformer_backbone import CrossFormer_S


from .decoders.upernet import UPerDecoder
from .decoders.segformer import SegformerDecoder
from .decoders.daformer import DaformerDecoder
from segmentation_models_pytorch.unet.model import UnetDecoder


class Net(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        CFG = OmegaConf.to_container(CFG)
        self.CFG = CFG
        
        ### encoder
        if CFG.get("encoder_name") == 'convnext_large':
            self.checkpoint = f'/home/user/.cache/torch/hub/checkpoints/{CFG.get("encoder_name")}_22k_224.pth'
            encoder_dim = [192, 384, 768, 1536]
            self.encoder = TimmUniversalEncoder(CFG.get("encoder_name"),depth=4)

        if CFG.get("encoder_name") == 'swin_small_patch4_window7_224_22k':
            self.checkpoint = f'/home/user/ai-competition/hubmap-organ-segmentation/weights/{CFG.get("encoder_name")}.pth'
            swin = dict(
                embed_dim=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,               
                ape=False,
                drop_path_rate=0.3,
                patch_norm=True,
                out_indices=(0, 1, 2, 3),
                use_checkpoint=False)
            
            encoder_dim = [96, 192, 384, 768]
            self.encoder = SwinTransformerV1(**{**swin, **{'out_norm':LayerNorm2d}})
                
        if CFG.get("encoder_name") == 'cswin_small_224':
            self.checkpoint = f'/home/user/ai-competition/hubmap-organ-segmentation/weights/{CFG.get("encoder_name")}.pth'            
            encoder_dim = [64, 128, 256, 512]           
            self.encoder = CSWinTransformer(
                img_size = CFG.get("image_size")[0],
                patch_size=4, 
                embed_dim=64, 
                depth=[2,4,32,2],
                split_size=[1,2,7,7], 
                num_heads=[2,4,8,16],
                mlp_ratio=4., 
                qkv_bias=True, 
                qk_scale=None, 
                drop_rate=0., 
                attn_drop_rate=0.,
                drop_path_rate=0., 
                hybrid_backbone=None, 
                norm_layer=nn.LayerNorm,
                use_chk=False)
            
        ### mix_transformer
        if CFG.get("encoder_name") == 'mit_b2':
            self.encoder = mit_b2()
            encoder_dim = self.encoder.embed_dims
            self.checkpoint = f'/home/user/ai-competition/hubmap-organ-segmentation/weights/{CFG.get("encoder_name")}.pth'

        if CFG.get("encoder_name") == 'mit_b3':
            self.encoder = mit_b3()
            encoder_dim = self.encoder.embed_dims
            self.checkpoint = f'/home/user/ai-competition/hubmap-organ-segmentation/weights/{CFG.get("encoder_name")}.pth'
            
        if CFG.get("encoder_name") == 'mit_b4':
            self.encoder = mit_b4()
            encoder_dim = self.encoder.embed_dims
            self.checkpoint = f'/home/user/ai-competition/hubmap-organ-segmentation/weights/{CFG.get("encoder_name")}.pth'
            
        if CFG.get("encoder_name") == 'mit_b5':
            self.encoder = mit_b5()
            encoder_dim = self.encoder.embed_dims
            self.checkpoint = f'/home/user/ai-competition/hubmap-organ-segmentation/weights/{CFG.get("encoder_name")}.pth'

 
        if CFG.get("encoder_name") == 'hila_mit_b2':
            self.encoder = hila_mit_b2()
            encoder_dim = self.encoder.embed_dims
            self.checkpoint = f'/home/user/ai-competition/hubmap-organ-segmentation/weights/mit_b2.pth'

        ### cswin_transformer
        if CFG.get("encoder_name") == 'cswin_small_224':
            encoder_dim = [64, 128, 256, 512]           
            self.encoder = CSWinTransformer(
                img_size = CFG.get("image_size")[0],
                patch_size=4, 
                embed_dim=64, 
                depth=[2,4,32,2],
                split_size=[1,2,7,7], 
                num_heads=[2,4,8,16],
                mlp_ratio=4., 
                qkv_bias=True, 
                qk_scale=None, 
                drop_rate=0., 
                attn_drop_rate=0.,
                drop_path_rate=0., 
                hybrid_backbone=None, 
                norm_layer=nn.LayerNorm,
                use_chk=False)
            self.checkpoint = f'/home/user/ai-competition/hubmap-organ-segmentation/weights/{CFG.get("encoder_name")}.pth'
            
        if CFG.get("encoder_name") == 'coat_lite_medium':
            self.encoder = coat_lite_medium()
            encoder_dim = self.encoder.embed_dims
            self.checkpoint = f'/home/user/ai-competition/hubmap-organ-segmentation/weights/coat_lite_medium_384x384_f9129688.pth'
            
        if CFG.get("encoder_name") == 'coat_parallel_small_plus1':
            self.encoder = coat_parallel_small_plus1()
            encoder_dim = self.encoder.embed_dims
            self.checkpoint = f'/home/user/ai-competition/hubmap-organ-segmentation/weights/coat_small_7479cf9b.pth'
            
        if CFG.get("encoder_name") == 'pvt_v2_b2':
            self.encoder = pvt_v2_b2()
            encoder_dim = self.encoder.embed_dims
            self.checkpoint = f'/home/user/ai-competition/hubmap-organ-segmentation/weights/pvt_v2_b2.pth'
        
        if CFG.get("encoder_name") == 'pvt_v2_b2_5level':
            self.encoder = pvt_v2_b2_5level()
            encoder_dim = self.encoder.embed_dims
            self.checkpoint = f'/home/user/ai-competition/hubmap-organ-segmentation/weights/pvt_v2_b2.pth'
        
        if CFG.get("encoder_name") == 'pvt_v2_b4':
            self.encoder = pvt_v2_b4()
            encoder_dim = self.encoder.embed_dims
            self.checkpoint = f'/home/user/ai-competition/hubmap-organ-segmentation/weights/pvt_v2_b4.pth'
            
        if CFG.get("encoder_name") == 'pvt_v2_b4_5level':
            self.encoder = pvt_v2_b4_5level()
            encoder_dim = self.encoder.embed_dims
            self.checkpoint = f'/home/user/ai-competition/hubmap-organ-segmentation/weights/pvt_v2_b4.pth'
            
        if CFG.get("encoder_name") == 'crossformer_s':
            self.encoder = CrossFormer_S()
            encoder_dim = [96, 192, 384, 768]  
            self.checkpoint = f'/home/user/ai-competition/hubmap-organ-segmentation/weights/crossformer-s.pth'
            
        if CFG.get("encoder_name") == 'hybrid_cnn_pvt_v2_b4':   
            #https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py
            conv_dim = 32
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, conv_dim, kernel_size=3, stride=1, padding=1, bias=False)
            ) 
            self.encoder = pvt_v2_b4()
            encoder_dim = self.encoder.embed_dims
            self.checkpoint = f'/home/user/ai-competition/hubmap-organ-segmentation/weights/pvt_v2_b4.pth'
        
        if CFG.get("encoder_name") == 'hybrid_cnn_pvt_v2_b4_5level':   
            #https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py
            conv_dim = 32
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, conv_dim, kernel_size=3, stride=1, padding=1, bias=False)
            ) 
            self.encoder = pvt_v2_b4_5level()
            encoder_dim = self.encoder.embed_dims
            self.checkpoint = f'/home/user/ai-competition/hubmap-organ-segmentation/weights/pvt_v2_b4.pth'
            
        if CFG.get("encoder_name") == 'hybrid_resnet50_pvt_v2_b4':
            conv_dim = 64
            embedder = timm.create_model('resnet50', pretrained=CFG.get("load_weights"), features_only=True, out_indices=CFG.get("out_indices"))
            self.encoder = HybridPVTV2(
                embedder,
                patch_size=4,
                embed_dims=[64, 128, 320, 512],
                num_heads=[1, 2, 5, 8],
                mlp_ratios=[8, 8, 4, 4],
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                depths=[3, 8, 27, 3], 
                sr_ratios=[8, 4, 2, 1],
            )
            encoder_dim = self.encoder.embed_dims
            self.checkpoint = f'/home/user/ai-competition/hubmap-organ-segmentation/weights/pvt_v2_b4.pth'
        
        if CFG.get("encoder_name") == 'hybrid_resnet50_pvt_v2_b4_5level':
            conv_dim = 64
            embedder = timm.create_model('resnet50', pretrained=CFG.get("load_weights"), features_only=True, out_indices=CFG.get("out_indices"))
            self.encoder = HybridPVTV2(
                embedder,
                patch_size=4,
                embed_dims=[64, 128, 320, 512, 768],
                num_heads=[1, 2, 5, 8, 8],
                mlp_ratios=[8, 8, 4, 4, 4],
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                depths=[3, 8, 27, 3, 3], 
                sr_ratios=[8, 4, 2, 1, 1],
                num_stages = 5,
            )
            encoder_dim = self.encoder.embed_dims
            self.checkpoint = f'/home/user/ai-competition/hubmap-organ-segmentation/weights/pvt_v2_b4.pth'
        
            
        if CFG.get("load_weights"):
            self.load_pretrain()
        
        
        ### decoder with upernet
        if CFG.get("decoder_name") == 'upernet':
            self.decoder = UPerDecoder(
                in_dim=encoder_dim,
                ppm_pool_scale=[1, 2, 3, 6],
                ppm_dim=512,
                fpn_out_dim=CFG.get("decoder_dim"))
        
        ###decoder with segformer
        if CFG.get("decoder_name") == 'segformer':
            self.decoder = SegformerDecoder(
                encoder_dim = encoder_dim, 
                decoder_dim = CFG.get("decoder_dim"))
        
        ###decoder with daformer
        if CFG.get("decoder_name") == 'daformer':
            self.decoder = DaformerDecoder(
                encoder_dim = encoder_dim, 
                decoder_dim = CFG.get("decoder_dim"),
                fuse = CFG.get("fuse"))
            
        #### decoder with smpunet
        if CFG.get("decoder_name") == 'unetdecoder':
            encoder_dim = [conv_dim] + self.encoder.embed_dims
            self.decoder = UnetDecoder(
                encoder_channels=[0] + encoder_dim,
                decoder_channels= CFG.get("decoder_channels"),
                n_blocks=CFG.get("encoder_depth"),
                use_batchnorm=True,
                center=False,
                attention_type=None,
            )
            CFG["decoder_dim"] = CFG.get("decoder_channels")[-1]     
        
        ###head v2
        self.dropout = nn.Dropout(CFG.get("dropout"))
        self.head = nn.Sequential(
                nn.Conv2d(CFG.get("decoder_dim"), CFG.get("num_classes"), kernel_size=1, padding=0))
        
        if CFG.get("aux"):
            self.auxes = nn.ModuleList([
                nn.Conv2d(encoder_dim[i], 1, kernel_size=1, padding=0) for i in range(len(encoder_dim))
            ])
            
    def load_pretrain(self):
        checkpoint = self.checkpoint
        print('loading %s ...'%checkpoint)
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
#         checkpoint = torch.load(checkpoint, map_location='cpu')
#         print(f'checkpoint keys is {checkpoint.keys()}')
        state_dict_key = ''
        if isinstance(checkpoint, dict):
            if checkpoint.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif checkpoint.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            elif 'model' in checkpoint:
                state_dict_key = 'model'
#         print(f'state_dict_key is {state_dict_key}')        
        if state_dict_key:
            state_dict = checkpoint[state_dict_key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
            
        if 0:
            skip = ['relative_coords_table','relative_position_index']
            filtered={}
            for k,v in checkpoint.items():
                if any([s in k for s in skip ]): continue
                filtered[k]=v
            checkpoint = filtered
        print(self.encoder.load_state_dict(state_dict,strict=False))  #True       

        
    def forward(self, x):
        B,C,H,W = x.shape
        encoder = self.encoder(x)
#         for i,e in enumerate(encoder):
#             print(f'{i} encoder shape is {e.shape}')
    
        if self.CFG.get("decoder_name") == 'unetdecoder':
            if self.CFG.get("encoder_name") == 'hybrid_cnn_pvt_v2_b4' or self.CFG.get("encoder_name") == 'hybrid_cnn_pvt_v2_b4_5level':
                conv = self.conv(x)
                feature = encoder[::-1]  # reverse channels to start from head of encoder
                head = feature[0]
                skip = feature[1:] + [conv, None]
                encoder = [conv] + encoder

            if self.CFG.get("encoder_name") == 'hybrid_resnet50_pvt_v2_b4' or self.CFG.get("encoder_name") == 'hybrid_resnet50_pvt_v2_b4_5level':
                feature = encoder[::-1]  # reverse channels to start from head of encoder
                head = feature[0]
                skip = feature[1:] + [None]
            
            d = self.decoder.center(head)
            decoder = []
            for i, decoder_block in enumerate(self.decoder.blocks):
#                 print(i, d.shape, skip[i].shape if skip[i] is not None else 'none')
#                 print(decoder_block.conv1[0])
#                 print('')
                s = skip[i]
                d = decoder_block(d, s)
                decoder.append(d)
            last = d
            logit = self.head(last)       
        else:
            last, decoder = self.decoder(encoder)
            last  = self.dropout(last)
            logit = self.head(last)
            logit = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)
        
        r ={}
        r['logit'] = logit
        ### for aux loss
        if self.CFG.get("aux"):
            aux_logits = []
            for index, (feature,aux) in enumerate(zip(encoder, self.auxes)):
                aux_logit = aux(feature)
                aux_logits.append(aux_logit)
            r['aux_logits'] = aux_logits                
        return logit