import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
import os
import random

from model.unet import (
    conv_nd,
    normalization,
    get_activation,
    zero_module,
    Upsample,
    Downsample,
    TimestepBlock,
)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: specified, the number of out channels.
    :param use_conv: True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines the signal is 1D, 2D, or 3D.
    :param up: True, use this block for upsampling.
    :param down: True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        dropout,
        activation,
        out_channels=None,
        use_conv=False,
        dims=2,
        up=False,
        down=False,
        dtype=None,
        scale_skip_connection=False,
    ):
        super().__init__()
        self.dtype = dtype
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.scale_skip_connection = scale_skip_connection

        self.in_layers = nn.Sequential(
            normalization(channels, dtype=self.dtype),
            get_activation(activation),
            conv_nd(dims, channels, self.out_channels, 3, padding=1, dtype=self.dtype),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, dtype=self.dtype)
            self.x_upd = Upsample(channels, False, dims, dtype=self.dtype)
        elif down:
            self.h_upd = Downsample(channels, False, dims, dtype=self.dtype)
            self.x_upd = Downsample(channels, False, dims, dtype=self.dtype)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(self.out_channels, dtype=self.dtype),
            get_activation(activation),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(
                    dims,
                    self.out_channels,
                    self.out_channels,
                    3,
                    padding=1,
                    dtype=self.dtype,
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1, dtype=self.dtype
            )
        else:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 1, dtype=self.dtype
            )

    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        h = self.out_layers(h)

        res = self.skip_connection(x) + h
        if self.scale_skip_connection:
            res *= 0.7071  # 1 / sqrt(2), https://arxiv.org/pdf/2104.07636.pdf
        return res


class Volume2Triplane(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        volume_size,
        sh_degree=0,
        model_channels=64,
        num_res_blocks=3,
        dropout=0.0,
    ):
        assert (
            in_channels == 3 * (sh_degree + 1) * (sh_degree + 1) + 3 + 8
        ), f"input channel is {in_channels}, but expected {3*(sh_degree+1)*(sh_degree+1)+3+8}"
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.volume_size = volume_size
        self.num_feature_dc = 3
        if sh_degree > 0:
            self.num_feature_rest = (sh_degree + 1) * (sh_degree + 1) * 3 - 3

        self.triplane_conv_yz = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=(volume_size, 1, 1),
            stride=(volume_size, 1, 1),
            bias=False,
        )
        self.triplane_conv_xz = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=(1, volume_size, 1),
            stride=(1, volume_size, 1),
            bias=False,
        )
        self.triplane_conv_xy = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=(1, 1, volume_size),
            stride=(1, 1, volume_size),
            bias=False,
        )

        self.conv_1x = nn.Conv2d(in_channels, model_channels, 3, 1, 1)
        self.conv_1y = nn.Conv2d(in_channels, model_channels, 3, 1, 1)
        self.conv_1z = nn.Conv2d(in_channels, model_channels, 3, 1, 1)

        model_x = nn.ModuleList()
        model_y = nn.ModuleList()
        model_z = nn.ModuleList()

        for i in range(num_res_blocks):
            model_x.append(ResBlock(model_channels, dropout=dropout, activation="silu"))
            model_y.append(ResBlock(model_channels, dropout=dropout, activation="silu"))
            model_z.append(ResBlock(model_channels, dropout=dropout, activation="silu"))

        self.model_x = model_x
        self.model_y = model_y
        self.model_z = model_z

        self.out_conv = nn.Conv2d(model_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        B, C, H, W, D = x.shape
        yz = self.triplane_conv_yz(x)  # (B, C, 1, H, W)
        xz = self.triplane_conv_xz(x)  # (B, C, H, 1, W)
        xy = self.triplane_conv_xy(x)  # (B, C, H, W, 1)

        yz = yz.reshape(B, C, H, W)  # (B, C, H, W)
        xz = xz.reshape(B, C, H, W)  # (B, C, H, W)
        xy = xy.reshape(B, C, H, W)  # (B, C, H, W)

        yz = self.conv_1x(yz)  # (B, model_channels, H, W)
        xz = self.conv_1y(xz)  # (B, model_channels, H, W)
        xy = self.conv_1z(xy)  # (B, model_channels, H, W)

        for res_x, res_y, res_z in zip(self.model_x, self.model_y, self.model_z):
            yz = res_x(yz)  # (B, model_channels, H, W)
            xz = res_y(xz)  # (B, model_channels, H, W)
            xy = res_z(xy)  # (B, model_channels, H, W)

        x = torch.cat(
            [yz.unsqueeze(-1), xz.unsqueeze(-1), xy.unsqueeze(-1)], dim=-1
        ).permute(
            0, 4, 1, 2, 3
        )  # (B, 3, model_channels, H, W)
        x = x.reshape(B * 3, self.model_channels, H, W)  # (B*3, model_channels, H, W
        x = self.out_conv(x)  # (B*3, out_channels, H, W)
        x = x.reshape(B, 3, self.out_channels, H, W)

        return x


# Classfier
class Classfier(nn.Module):
    def __init__(self, num_labels=1000, in_features=3 * 3 * 16 * 16):
        super(Classfier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_labels),
        )
        self.init_weights()

    def init_weights(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01**2)
                nn.init.constant_(layer.bias, val=0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01**2)
                nn.init.constant_(layer.bias, val=0)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


class OmniObject3D_Dataset(Dataset):
    def __init__(self, root_dir, test=False):
        super().__init__()
        self.root_dir = root_dir
        obj_names = [f for f in os.listdir(root_dir) if f.endswith(".pt")]
        classes = [f.split("_")[0] for f in obj_names]
        classes = list(set(classes))
        class_to_obj = dict()
        for obj_name in obj_names:
            class_name = obj_name.split("_")[0]
            if class_name not in class_to_obj.keys():
                class_to_obj[class_name] = []
            class_to_obj[class_name].append(os.path.join(root_dir, obj_name))

        if test:
            for class_name in class_to_obj.keys():
                class_to_obj[class_name] = class_to_obj[class_name][
                    len(class_to_obj[class_name]) * 4 // 5 :
                ]
        else:
            for class_name in class_to_obj.keys():
                class_to_obj[class_name] = class_to_obj[class_name][
                    : len(class_to_obj[class_name]) * 4 // 5
                ]
        self.classes = classes
        self.shuffle_classes = random.shuffle(self.classes.copy())
        self.class_to_obj = class_to_obj
        self.obj_names = obj_names

    def __len__(self):
        return len(self.obj_names)

    def __getitem__(self, idx):
        class_ = self.classes[idx % len(self.classes)]
        obj_path = random.choice(self.class_to_obj[class_])
        obj = torch.load(obj_path)
        target = self.classes.index(class_)
        target = torch.tensor(target)
        return obj, target

    def get_classes(self):
        return self.classes


device = "cuda:1"
batch_size = 128
device = "cuda:1"
learning_rate = 1e-3
epochs = 200
data_type = torch.float32
train_dataset = OmniObject3D_Dataset(
    root_dir="/data/OmniObject3D/GaussianCube/4096/volume_act"
)
test_dataset = OmniObject3D_Dataset(
    root_dir="/data/OmniObject3D/GaussianCube/4096/volume_act", test=True
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = Volume2Triplane(59, 3, 16, 3)
model = model.to(device)
classfier = Classfier(216, 3 * 3 * 16 * 16)
classfier = classfier.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(classfier.parameters()), lr=learning_rate
)

training_loss_list = []
test_loss_list = []
training_acc_list = []
test_acc_list = []
for epoch in range(epochs):
    training_loss = 0
    training_accuracy = 0
    iters = 0
    for data, target in train_dataloader:
        data = data.to(device, dtype=data_type)
        target = target.to(device, dtype=torch.long)
        data = data.permute(0, 4, 1, 2, 3)

        optimizer.zero_grad()
        out = model(data)
        out = classfier(out)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            training_loss += loss.item() / out.shape[0]
            training_accuracy += (
                torch.sum(torch.argmax(out, dim=1) == target).item() / out.shape[0]
            )
            iters += 1
    training_loss /= iters
    training_accuracy /= iters
    with torch.no_grad():
        test_accuracy = 0
        test_loss = 0
        iters = 0
        for data, target in test_dataloader:
            data = data.to(device, dtype=data_type)
            target = target.to(device)
            data = data.permute(0, 4, 1, 2, 3)
            out = model(data)
            out = classfier(out)
            loss = loss_fn(out, target)
            test_loss += loss.item() / out.shape[0]
            test_accuracy += (
                torch.sum(torch.argmax(out, dim=1) == target).item() / out.shape[0]
            )
            iters += 1
        test_loss /= iters
        test_accuracy /= iters
        print(
            f"epoch: {epoch}, training_loss: {training_loss}, test_loss: {test_loss}, training_accuracy: {training_accuracy}, test_accuracy: {test_accuracy}"
        )
        training_loss_list.append(training_loss)
        test_loss_list.append(test_loss)
        training_acc_list.append(training_accuracy)
        test_acc_list.append(test_accuracy)

import matplotlib.pyplot as plt

plt.plot(training_loss_list, label="Training Loss")
plt.plot(test_loss_list, label="Test Loss")
plt.plot(training_acc_list, label="Training Accuracy")
plt.plot(test_acc_list, label="Test Accuracy")
plt.legend()
plt.show()
plt.savefig("training_test.png")

model = model.to("cpu")
classfier = classfier.to("cpu")
torch.save(model.state_dict(), "model.pth")
torch.save(classfier.state_dict(), "classfier.pth")
