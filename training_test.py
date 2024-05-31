# %%
import torch
import torch.nn as nn

# %%
device = "cuda:1"

# %% [markdown]
# # Volume to Triplane

# %%
class Volume2Triplane(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        volume_size,
        sh_degree=0,
    ):
        assert (
            in_channels == 3 * (sh_degree + 1) * (sh_degree + 1) + 3 + 8
        ), f"input channel is {in_channels}, but expected {3*(sh_degree+1)*(sh_degree+1)+3+8}"
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.volume_size = volume_size
        self.num_feature_dc = 3
        if sh_degree > 0:
            self.num_feature_rest = (sh_degree + 1) * (sh_degree + 1) * 3 - 3

        self.volume2triplane_x = nn.Conv2d(volume_size * 3, 1, 3, padding=1)
        self.volume2triplane_y = nn.Conv2d(volume_size * 3, 1, 3, padding=1)
        self.volume2triplane_z = nn.Conv2d(volume_size * 3, 1, 3, padding=1)
        self.feature_dc_conv_x = nn.Conv2d(
            self.num_feature_dc * volume_size, 1, 3, padding=1
        )
        self.feature_dc_conv_y = nn.Conv2d(
            self.num_feature_dc * volume_size, 1, 3, padding=1
        )
        self.feature_dc_conv_z = nn.Conv2d(
            self.num_feature_dc * volume_size, 1, 3, padding=1
        )
        if self.num_feature_rest > 0:
            self.feature_rest_conv_x = nn.Conv2d(
                self.num_feature_rest * volume_size, 1, 3, padding=1
            )
            self.feature_rest_conv_y = nn.Conv2d(
                self.num_feature_rest * volume_size, 1, 3, padding=1
            )
            self.feature_rest_conv_z = nn.Conv2d(
                self.num_feature_rest * volume_size, 1, 3, padding=1
            )
        self.opacity_conv_x = nn.Conv2d(1 * volume_size, 1, 3, padding=1)
        self.opacity_conv_y = nn.Conv2d(1 * volume_size, 1, 3, padding=1)
        self.opacity_conv_z = nn.Conv2d(1 * volume_size, 1, 3, padding=1)
        self.scale_conv_x = nn.Conv2d(3 * volume_size, 1, 3, padding=1)
        self.scale_conv_y = nn.Conv2d(3 * volume_size, 1, 3, padding=1)
        self.scale_conv_z = nn.Conv2d(3 * volume_size, 1, 3, padding=1)
        self.rotation_conv_x = nn.Conv2d(4 * volume_size, 1, 3, padding=1)
        self.rotation_conv_y = nn.Conv2d(4 * volume_size, 1, 3, padding=1)
        self.rotation_conv_z = nn.Conv2d(4 * volume_size, 1, 3, padding=1)

        model_channels = 15 if self.num_feature_rest > 0 else 12
        self.out_conv = nn.Conv2d(model_channels, out_channels, 3, padding=1)

    def forward(self, x):
        B, C, D, H, W = x.size()
        assert (
            C == 3 + self.num_feature_dc + self.num_feature_rest + 8
        ), f"input channel is {C}, but expected {3+self.num_feature_dc+self.num_feature_rest+8}"
        # x: (B, pos_dim+feature_size, D, H, W)
        pos = x[:, :3, :, :, :]  # (B, pos_dim, D, H, W)
        feature_dc = x[:, 3 : 3 + self.num_feature_dc, :, :, :]
        feature_rest = (
            x[
                :,
                3
                + self.num_feature_dc : 3
                + self.num_feature_dc
                + self.num_feature_rest,
                :,
                :,
                :,
            ]
            if self.num_feature_rest > 0
            else None
        )
        opacity = x[:, -8:-7, :, :, :]
        scale = x[:, -7:-4, :, :, :]
        rotation = x[:, -4:, :, :, :]

        yz = pos.permute(0, 1, 2, 3, 4).reshape(B, -1, H, W)  # (B, pos_dim*D, H, W)
        xz = pos.permute(0, 1, 3, 2, 4).reshape(B, -1, D, W)  # (B, pos_dim*H, D, W)
        xy = pos.permute(0, 1, 4, 2, 3).reshape(B, -1, D, H)  # (B, pos_dim*W, D, H)
        yz = self.volume2triplane_x(yz)  # (B, 1, H, W)
        xz = self.volume2triplane_y(xz)  # (B, 1, D, W)
        xy = self.volume2triplane_z(xy)  # (B, 1, D, H)
        yz = yz.reshape(B, -1, H, W)  # (B, 1, H, W)
        xz = xz.reshape(B, -1, D, W)  # (B, 1, D, W)
        xy = xy.reshape(B, -1, D, H)  # (B, 1, D, H)

        feature_dc_x = feature_dc.permute(0, 1, 2, 3, 4).reshape(B, -1, H, W)
        feature_dc_y = feature_dc.permute(0, 1, 3, 2, 4).reshape(B, -1, D, W)
        feature_dc_z = feature_dc.permute(0, 1, 4, 2, 3).reshape(B, -1, D, H)
        feature_dc_x = self.feature_dc_conv_x(feature_dc_x)
        feature_dc_y = self.feature_dc_conv_y(feature_dc_y)
        feature_dc_z = self.feature_dc_conv_z(feature_dc_z)
        feature_dc_x = feature_dc_x.reshape(B, -1, H, W)
        feature_dc_y = feature_dc_y.reshape(B, -1, D, W)
        feature_dc_z = feature_dc_z.reshape(B, -1, D, H)

        if self.num_feature_rest > 0:
            feature_rest_x = feature_rest.permute(0, 1, 2, 3, 4).reshape(B, -1, H, W)
            feature_rest_y = feature_rest.permute(0, 1, 3, 2, 4).reshape(B, -1, D, W)
            feature_rest_z = feature_rest.permute(0, 1, 4, 2, 3).reshape(B, -1, D, H)
            feature_rest_x = self.feature_rest_conv_x(feature_rest_x)
            feature_rest_y = self.feature_rest_conv_y(feature_rest_y)
            feature_rest_z = self.feature_rest_conv_z(feature_rest_z)
            feature_rest_x = feature_rest_x.reshape(B, -1, H, W)
            feature_rest_y = feature_rest_y.reshape(B, -1, D, W)
            feature_rest_z = feature_rest_z.reshape(B, -1, D, H)

        opacity_x = opacity.permute(0, 1, 2, 3, 4).reshape(B, -1, H, W)
        opacity_y = opacity.permute(0, 1, 3, 2, 4).reshape(B, -1, D, W)
        opacity_z = opacity.permute(0, 1, 4, 2, 3).reshape(B, -1, D, H)
        opacity_x = self.opacity_conv_x(opacity_x)
        opacity_y = self.opacity_conv_y(opacity_y)
        opacity_z = self.opacity_conv_z(opacity_z)
        opacity_x = opacity_x.reshape(B, -1, H, W)
        opacity_y = opacity_y.reshape(B, -1, D, W)
        opacity_z = opacity_z.reshape(B, -1, D, H)

        scale_x = scale.permute(0, 1, 2, 3, 4).reshape(B, -1, H, W)
        scale_y = scale.permute(0, 1, 3, 2, 4).reshape(B, -1, D, W)
        scale_z = scale.permute(0, 1, 4, 2, 3).reshape(B, -1, D, H)
        scale_x = self.scale_conv_x(scale_x)
        scale_y = self.scale_conv_y(scale_y)
        scale_z = self.scale_conv_z(scale_z)
        scale_x = scale_x.reshape(B, -1, H, W)
        scale_y = scale_y.reshape(B, -1, D, W)
        scale_z = scale_z.reshape(B, -1, D, H)

        rotation_x = rotation.permute(0, 1, 2, 3, 4).reshape(B, -1, H, W)
        rotation_y = rotation.permute(0, 1, 3, 2, 4).reshape(B, -1, D, W)
        rotation_z = rotation.permute(0, 1, 4, 2, 3).reshape(B, -1, D, H)
        rotation_x = self.rotation_conv_x(rotation_x)
        rotation_y = self.rotation_conv_y(rotation_y)
        rotation_z = self.rotation_conv_z(rotation_z)
        rotation_x = rotation_x.reshape(B, -1, H, W)
        rotation_y = rotation_y.reshape(B, -1, D, W)
        rotation_z = rotation_z.reshape(B, -1, D, H)

        x = torch.cat(
            [
                yz,
                xz,
                xy,
                feature_dc_x,
                feature_dc_y,
                feature_dc_z,
                opacity_x,
                opacity_y,
                opacity_z,
                scale_x,
                scale_y,
                scale_z,
                rotation_x,
                rotation_y,
                rotation_z,
            ],
            dim=1,
        )

        x = self.out_conv(x)
        return x


# %%
import torch
from torch import nn


# Classfier
class Classfier(nn.Module):
    def __init__(self, num_labels=1000, in_channels=3):
        super(Classfier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_labels),
        )
        self.init_weights()

    def init_weights(self):
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01**2)
                nn.init.constant_(layer.bias, val=0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01**2)
                nn.init.constant_(layer.bias, val=0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def test_output_shape(self):
        # out_channel input_channel, image_size
        test_img = torch.rand(size=(1, 3, 227, 227), dtype=torch.float32)
        for layer in self.features:
            test_img = layer(test_img)
            print(layer.__class__.__name__, "output shape: \t", test_img.shape)

# %% [markdown]
# # Dataset and DataLoader

# %%
from torch.utils.data import DataLoader, Dataset
import os
import random

# %%
class OmniObject3D_Dataset(Dataset):
    def __init__(self, root_dir,test=False):
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
                class_to_obj[class_name]=class_to_obj[class_name][len(class_to_obj[class_name])*4//5:]
        else:
            for class_name in class_to_obj.keys():
                class_to_obj[class_name]=class_to_obj[class_name][:len(class_to_obj[class_name])*4//5]
        self.classes = classes
        self.shuffle_classes = random.shuffle(self.classes.copy())
        self.class_to_obj = class_to_obj
        self.obj_names = obj_names
    
    def __len__(self):
        return len(self.obj_names)
    
    def __getitem__(self, idx):
        class_=self.classes[idx%len(self.classes)]
        obj_path= random.choice(self.class_to_obj[class_])
        obj = torch.load(obj_path)
        target=self.classes.index(class_)
        target = torch.tensor(target)
        return obj, target
    
    def get_classes(self):
        return self.classes

# %%
train_dataset = OmniObject3D_Dataset(root_dir="/data/OmniObject3D/GaussianCube/4096/volume_act")
test_dataset = OmniObject3D_Dataset(root_dir="/data/OmniObject3D/GaussianCube/4096/volume_act", test=True)

# %% [markdown]
# # Training

# %%
batch_size=128
device='cuda:1'
learning_rate=1e-4
epochs=500
data_type=torch.float32

# %%
train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
model=Volume2Triplane(59, 3, 16, 3)
model=model.to(device)
classfier=Classfier(216, 3)
classfier=classfier.to(device)

# %%
loss_fn = nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(list(model.parameters())+list(classfier.parameters()), lr=learning_rate)

# %%
training_loss_list=[]
test_loss_list=[]
training_acc_list=[]
test_acc_list=[]
for epoch in range(epochs):
    training_loss=0
    training_accuracy=0
    for data, target in train_dataloader:
        data=data.to(device, dtype=data_type)
        target=target.to(device, dtype=torch.long)
        data=data.permute(0, 4, 1, 2, 3)
        
        optimizer.zero_grad()
        out=model(data)
        out=classfier(out)
        loss=loss_fn(out, target)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            training_loss+=loss.item()/len(train_dataloader)
            training_accuracy+=torch.sum(torch.argmax(out, dim=1)==target).item()/len(train_dataloader)
    with torch.no_grad():
        test_accuracy=0
        test_loss=0
        for data, target in test_dataloader:
            data=data.to(device, dtype=data_type)
            target=target.to(device)
            data=data.permute(0, 4, 1, 2, 3)
            out=model(data)
            out=classfier(out)
            loss=loss_fn(out, target)
            test_loss+=loss.item()/len(test_dataloader)
            test_accuracy+=torch.sum(torch.argmax(out, dim=1)==target).item()/len(test_dataloader)
        print(f"epoch: {epoch}, training_loss: {training_loss}, test_loss: {test_loss}, training_accuracy: {training_accuracy}, test_accuracy: {test_accuracy}")
        training_loss_list.append(training_loss)
        test_loss_list.append(test_loss)
        training_acc_list.append(training_accuracy)
        test_acc_list.append(test_accuracy)

# %%
import matplotlib.pyplot as plt
plt.plot(training_loss_list, label="Training Loss")
plt.plot(test_loss_list, label="Test Loss")
plt.plot(training_acc_list, label="Training Accuracy")
plt.plot(test_acc_list, label="Test Accuracy")
plt.legend()
plt.show()
plt.savefig("training_test.png")

