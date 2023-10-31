import numpy as np
import torch
from pyturbo import Stage
from torch.backends import cudnn
from torchvision.models import video as video_models
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms

class CNN3DFeature(Stage):

    """
    Input: a clip [T x H x W x C]
    Output: CNN feature [D]
    """

    def allocate_resource(self, resources, *, model_name, weight_name,
                          node_name, replica_per_gpu=1):
        self.model_name = model_name
        self.weight_name = weight_name
        self.node_name = node_name
        self.model = None
        gpus = resources.get('gpu')
        self.num_gpus = len(gpus)
        if len(gpus) > 0:
            return resources.split(len(gpus)) * replica_per_gpu
        return [resources]

    def reset(self):
        if self.model is None:
            gpu_ids = self.current_resource.get('gpu', 1)
            if len(gpu_ids) >= 1:
                self.device = 'cuda:%d' % (gpu_ids[0])
                cudnn.fastest = True
                cudnn.benchmark = True
            else:
                self.device = 'cpu'
                self.logger.warn('No available GPUs, running on CPU.')
           
           
            # build 3D CNN model with weights and input transforms
            weights = getattr(video_models, self.weight_name).DEFAULT
            # self.transforms = weights.transforms()
            base_model = getattr(video_models, self.model_name)(weights=weights)
            self.model = create_feature_extractor(
                base_model, {self.node_name: 'feature'})
            self.model = self.model.to(self.device).eval()

            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def extract_cnn3d_features(self, clip: torch.Tensor) -> torch.Tensor:
        """
        frame: [T x H x W x C] in uint8 [0, 255]

        Return: Feature, [D]
        """
        # train_nodes, eval_nodes = get_graph_node_names(self.model)
        # print(train_nodes)
        # First convert batch into [B x T x C x H x W] with B=1.
        clip = clip.permute(0, 3, 1, 2).unsqueeze(0).to(dtype=torch.float32)
        
        transformed_clip_list = []
        for t in range(clip.size(1)):
            frame = clip[0, t].to(dtype=torch.uint8).cpu().numpy()
            frame = np.transpose(frame, (1, 2, 0))
            transformed_frame = self.transforms(frame)
            transformed_clip_list.append(transformed_frame)
        transformed_clip = torch.stack(transformed_clip_list, dim=1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(transformed_clip)
            if isinstance(features, dict):
                features = features['feature']

        features = features.squeeze(0)

        return features
    

    def process(self, task):
        task.start(self)
        frames = task.content
        features = self.extract_cnn3d_features(frames).cpu().numpy()
        task.meta['sequence_id'] = task.meta['batch_id']
        return task.finish(features)
