import torch

from utils.geometry import get_neighbourhood_indices


class Translator(torch.nn.Module):

    def __init__(self, config):

        super(Translator, self).__init__()

        self.config = config

        try:
            self.n_features = config.n_features - (1 - int(config.use_count_renderer))
        except:
            self.n_features = config.n_features

        self.output_scale = config.RENDERER.output_scale

        activation = eval(config.RENDERER.activation)

        self.layer_context1 = torch.nn.Sequential(torch.nn.Linear((self.config.RENDERER.kernel ** 3) * self.n_features, self.n_features),
                                                  torch.nn.LayerNorm([self.n_features], elementwise_affine=False),
                                                  activation)

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(self.n_features + self.n_features, 32),
            activation)

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(self.n_features + 32 , 16),
            activation)

        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(self.n_features + 16, 8),
            activation)

        self.layer4 = torch.nn.Sequential(
            torch.nn.Linear(self.n_features + 8, self.n_features),
            activation)

        if self.config.RENDERER.sdf_head:
            self.sdf_head = torch.nn.Sequential(torch.nn.Linear(self.n_features, 1),
                                                torch.nn.Tanh())

        if self.config.RENDERER.occ_head:
            self.occ_head = torch.nn.Sequential(torch.nn.Linear(self.n_features, 1),
                                                torch.nn.Sigmoid())

        self.padding = torch.nn.ReplicationPad3d(self.config.RENDERER.kernel // 2)
        self.feature_dropout = torch.nn.Dropout2d(p=0.2)

        self.relu = torch.nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.hardtanh = torch.nn.Hardtanh(min_val=-0.06, max_val=0.06)
        self.softsign = torch.nn.Softsign()

        indices = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    indices.append(torch.Tensor([i, j, k]).view(1, 1, 3))
        indices = torch.cat(indices, dim=0)
        indices = indices.view(1, 27, 3)
        self.index_shift = indices.int().clone()


        # compute interpolation shift
        indices = []
        for i in range(0,  2):
            for j in range(0, 2):
                for k in range(0, 2):
                    indices.append(torch.Tensor([i, j, k]).view(1, 1, 3))
        indices = torch.cat(indices, dim=0)
        indices = indices.view(1, 8, 3)
        self.interpolation_shift = indices.int().clone()



    def forward(self, points, grid, padding=True):
        
        if padding:
            grid = self.padding(grid)
            points = points + int(self.config.RENDERER.kernel // 2)
        else:
            pass

        n_points = points.shape[0]

        indices = points.floor()

        df = 2. * (points - (indices + 0.5))

        neighbourhood = get_neighbourhood_indices(indices, size=(self.config.RENDERER.kernel,                                                 self.config.RENDERER.kernel,
                                                                 self.config.RENDERER.kernel))
        neighbourhood = neighbourhood.unsqueeze_(0)

        n_neighbourhood = neighbourhood.shape[1]

        indices = indices.unsqueeze_(1)
        indices_neighbourhood = indices + neighbourhood

        indices = indices.long()
        indices_neighbourhood = indices_neighbourhood.long()

        indices_neighbourhood = indices_neighbourhood.view(n_points * n_neighbourhood, 3)

        indices = indices.squeeze_(1)

        features = grid[:, :,
                        indices_neighbourhood[:, 0],
                        indices_neighbourhood[:, 1],
                        indices_neighbourhood[:, 2]]

        center_features = grid[:, :, indices[:, 0], indices[:, 1], indices[:, 2]]

        features = features.permute(-1, 1, 0)
        center_features = center_features.permute(-1, 1, 0)

        try:
            if not self.config.use_count_renderer:
                features = features[:, :-1, :]
                center_features = center_features[:, :-1, :]

            else:
                neighbourhood_count = features[:, -1, :].unsqueeze_(1)
                center_count = center_features[:, -1, :].unsqueeze_(1)

                max_count_neighbourhood = torch.max(neighbourhood_count)
                max_count_center = torch.max(center_count)
                max_count = torch.max(max_count_neighbourhood, max_count_center) + 1.e-09

                features = torch.cat([features[:, :-1, :],
                                  neighbourhood_count/max_count], dim=1)
                center_features = torch.cat([center_features[:, :-1, :],
                                         center_count/max_count], dim=1)
        except:
            pass

        features = features.contiguous().view(n_points, n_neighbourhood * self.n_features)
        center_features = center_features.squeeze_(-1)

        if self.config.minimal_gpu:
            df = df.to(self.config.device)
            features = features.to(self.config.device)
            center_features = center_features.to(self.config.device)

        center_features = center_features.unsqueeze_(-1).unsqueeze_(-1)
        center_features = self.feature_dropout(center_features)
        center_features = center_features.squeeze_(-1).squeeze_(-1)

        if self.config.RENDERER.superresolve:
            features = torch.cat([df, features], dim=1)

        context_features = self.layer_context1(features)
    
        input_features = torch.cat([center_features, context_features], dim=1)

        features = self.layer1(input_features)
        features = torch.cat([center_features, features], dim=1)

        #features = features.unsqueeze_(-1).unsqueeze_(-1)
        #features = self.feature_dropout(features)
        #features = features.squeeze_(-1).squeeze_(-1)

        if self.config.RENDERER.superresolve:
            features = torch.cat([df, features], dim=1)

        features = self.layer2(features)
        features = torch.cat([center_features, features], dim=1)

        #features = features.unsqueeze_(-1).unsqueeze_(-1)
        #features = self.feature_dropout(features)
        #features = features.squeeze_(-1).squeeze_(-1)

        if self.config.RENDERER.superresolve:
            features = torch.cat([df, features], dim=1)

        features = self.layer3(features)
        features = torch.cat([center_features, features], dim=1)

        if self.config.RENDERER.superresolve:
            features = torch.cat([df, features], dim=1)

        features = self.layer4(features)

        output = []
        sdf = self.output_scale * self.sdf_head(features)
        output.append(sdf)

        if self.config.RENDERER.occ_head:
            occ = self.occ_head(features)
            output.append(occ)

        del features, context_features, center_features, df, \
            indices, indices_neighbourhood, neighbourhood, \
            points, sdf

        return torch.cat(output, dim=1)






