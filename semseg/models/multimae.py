import torch
from torch import Tensor
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import ConvNeXtAdapter

def tensor_check_fn(param, input_param, error_msgs):
    if param.shape != input_param.shape:
        return False
    return True
    
class MultiMAE(BaseModel):
    def __init__(self, backbone: str = 'ViT-B', num_classes: int = 40, modals: list = ['img', 'depth'], image_size = [768, 768]) -> None:
        super().__init__(backbone, num_classes, modals, image_size)
        self.decode_head = ConvNeXtAdapter(num_classes=num_classes, image_size=image_size)
        self.apply(self._init_weights)

    def forward(self, x: list) -> list:
        y = self.backbone(x)
        y = self.decode_head(y)
        return y
    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            checkpoint = torch.load(pretrained, map_location='cpu')
            if 'model' in checkpoint.keys():
                checkpoint = checkpoint['model']
            new_state_dict = {}
            state_dict= self.backbone.state_dict()

            for k,v in checkpoint.items():
                if k in state_dict.keys():
                    if state_dict[k].shape == checkpoint[k].shape:
                        new_state_dict[k] = v
                        if k == 'cls_token' and state_dict['global_tokens'].shape == checkpoint[k].shape:
                            state_dict['global_tokens'] = v#.detach()
            self.backbone.load_state_dict(new_state_dict, strict= False)
    def no_weight_decay(self):
        return {'backbone.cls_token', 'backbone.pos_embed', 'backbone.global_tokens', 'backbone.rgb_adapter.pos_emb', 'backbone.depth_adapter.pos_emb'}


if __name__ == '__main__':
    modals = ['img', 'depth']
    model = MultiMAE('ViT-B', 40, modals)
    with open('multimae_modified.txt', 'w') as f:
        for name, param in model.named_parameters():
            f.write(name+'\n')
    x = [torch.zeros(1, 3, 1024, 1024), torch.zeros(1, 3, 1024, 1024)]
    y = model(x)
    print(y.shape) 
