import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



class NerfModel(nn.Module):
    def __init__(self):
        super().__init__()

        # CONFIGS
        # Input dimensions for Positional Encoding : pos and dir
        self.L_pos = 10
        self.L_dir = 4
        pos_enc_features = 3 + 3 * 2 * self.L_pos ## Well adding three neurons
        dir_enc_features = 3 + 3 * 2 * self.L_dir
        in_features = pos_enc_features  # 63
        num_neurons = 256
        # early mlp layers = 5, with 256 neurons each

        # MLP
        self.early_mlp = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        in_features = pos_enc_features + num_neurons  # 63 + 256
        # later mlp layers = 3, with 256 neurons each

        self.later_mlp = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.sigma_layer = nn.Linear(num_neurons, num_neurons + 1)
        self.pre_final_layer = nn.Sequential(
            nn.Linear(dir_enc_features + num_neurons, num_neurons // 2),  # output 128 neuron layer
            nn.ReLU(),
        )
        self.final_layer = nn.Sequential(nn.Linear(num_neurons // 2, 3), nn.Sigmoid())  # rgb output

    def forward(self, rays_samples, view_dirs):
        # POSITIONAL ENCODING

        # rays_samples - 3D points
        rays_samples_encoded = [rays_samples]
        for l_pos in range(self.L_pos):
            rays_samples_encoded.append(torch.sin(2 ** l_pos * torch.pi * rays_samples))
            rays_samples_encoded.append(torch.cos(2 ** l_pos * torch.pi * rays_samples))

        rays_samples_encoded = torch.cat(rays_samples_encoded, dim=-1)

        # view_dirs - viewing directions of rays
        view_dirs = view_dirs / view_dirs.norm(p=2, dim=-1).unsqueeze(-1)
        view_dirs_encoded = [view_dirs]
        for l_dir in range(self.L_dir):
            view_dirs_encoded.append(torch.sin(2 ** l_dir * torch.pi * view_dirs))
            view_dirs_encoded.append(torch.cos(2 ** l_dir * torch.pi * view_dirs))

        view_dirs_encoded = torch.cat(view_dirs_encoded, dim=-1)

        # Use the network to predict colors (c_is) and volume densities (sigma_is) for
        # 3D points (xs) along rays given the viewing directions (ds) of the rays

        outputs = self.early_mlp(rays_samples_encoded)
        outputs = self.later_mlp(torch.cat([rays_samples_encoded, outputs], dim=-1))
        outputs = self.sigma_layer(outputs)
        sigma_is = torch.relu(outputs[:, 0])  # volume densities
        outputs = self.pre_final_layer(torch.cat([view_dirs_encoded, outputs[:, 1:]], dim=-1))
        c_is = self.final_layer(outputs)  # predicted colors

        return {"c_is": c_is, "sigma_is": sigma_is}
