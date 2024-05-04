import torchvision.datasets as datasets
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import math

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class Reshape(torch.nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape
    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)
    
class FeedForwardClassifier(torch.nn.Module):
    def __init__(self, output_dim, embed_size) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(embed_size, 250)
        self.fc2 = torch.nn.Linear(250, output_dim)

    def forward(self, embedding):
        x = embedding.squeeze()
        x = torch.nn.ReLU()(self.fc1(x))
        output = self.fc2(x)
        return output

    def predict(self, embedding):
        logits = self.forward(embedding)
        probabilities = torch.nn.Softmax()(logits)
        return torch.argmax(probabilities, dim=1)
    
# Encoder and decoder use the DC-GAN architecture
class Encoder(torch.nn.Module):
    def __init__(self, z_dim, input_channels, input_dim):
        super(Encoder, self).__init__()
        self.input_channels = input_channels
        self.height, self.width = input_dim
        self.layer_height, self.layer_width = self.get_layer_size(2)
        self.model = torch.nn.ModuleList([
            torch.nn.Conv2d(
                in_channels=input_channels, 
                out_channels=64, 
                kernel_size=4, 
                stride=2, 
                padding=1
            ),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=64, 
                out_channels=128, 
                kernel_size=4, 
                stride=2, 
                padding=1
            ),
            torch.nn.LeakyReLU(),
            # torch.nn.Conv2d(
            #     in_channels=64, 
            #     out_channels=128, 
            #     kernel_size=4, 
            #     stride=2, 
            #     padding=1
            # ),
            # torch.nn.LeakyReLU(),
            Flatten(),
            torch.nn.Linear(128*self.layer_height*self.layer_width, 1024),
            torch.nn.LeakyReLU(),
        ])
        self.dense_mu = torch.nn.Linear(1024, z_dim)
        self.dense_logvar = torch.nn.Linear(1024, z_dim)
        
    def get_layer_size(self, layer_num):
        h = self.height
        w = self.width
        for _ in range(layer_num):
            h = math.floor(h/2)
            w = math.floor(w/2)
        return h, w
        
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        mu = self.dense_mu(x)
        logvar = self.dense_logvar(x)
        return mu, logvar
    
    
class Decoder(torch.nn.Module):
    def __init__(self, z_dim, input_channels, input_dim):
        super(Decoder, self).__init__()
        self.input_channels = input_channels
        self.height, self.width = input_dim
        self.layer_height, self.layer_width = self.get_layer_size(2)
        self.model = torch.nn.ModuleList([
            torch.nn.Linear(z_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, self.layer_height*self.layer_width*128),
            torch.nn.ReLU(),
            Reshape((128,self.layer_height,self.layer_width,)),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            torch.nn.ReLU(),
            # torch.nn.ConvTranspose2d(64, 32, 4, 2, padding=1),
            # torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, input_channels, 4, 2, padding=1),
            torch.nn.Sigmoid()
        ])

    def get_layer_size(self, layer_num):
        h = self.height
        w = self.width
        for _ in range(layer_num):
            h = math.floor(h/2)
            w = math.floor(w/2)
        return h, w
        
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

def save_weights(encoder, decoders):
    encoder_state_dict = encoder.state_dict()
    torch.save(encoder_state_dict, f"../../weights/mat/fashionmnist_encoder_mat_weights.pt")

    for i, decoder in enumerate(decoders):
        decoder_state_dict = decoder.state_dict()
        torch.save(decoder_state_dict, f"../../weights/mat/fashionmnist_decoder_{i}_weights.pt")

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

MNIST_training_dataset = datasets.MNIST(root="../../data/MNIST", train=True, download=True, transform=transforms.ToTensor())
MNIST_test_dataset = datasets.MNIST(root="../../data/MNIST", train=False, download=True, transform=transforms.ToTensor())

FashionMNIST_training_dataset = datasets.FashionMNIST(root="../../data/FashionMNIST", train=True, download=True, transform=transforms.ToTensor())
FashionMNIST_test_dataset = datasets.FashionMNIST(root="../../data/FashionMNIST", train=False, download=True, transform=transforms.ToTensor())

num_classes = len(FashionMNIST_training_dataset.classes)
image_shape = FashionMNIST_training_dataset.data[0].shape
if len(image_shape)==2:
    input_channels = 1
    input_dim = tuple(image_shape)
else:
    input_channels = image_shape[0]
    input_dim = tuple(image_shape[1:])

train_loader = DataLoader(FashionMNIST_training_dataset, shuffle=False, batch_size = 32)
test_loader = DataLoader(FashionMNIST_test_dataset, shuffle=False, batch_size = 32)

embed_list = [2,4,8,16,32,64,128]
z_max = max(embed_list)
epochs = 20
num_samples=200
beta=5
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
best_score = None

encoder = Encoder(z_dim=z_max, input_channels=input_channels, input_dim=input_dim).to(DEVICE)
encoder.train()

decoders = [Decoder(z_dim=z, input_channels=input_channels, input_dim=input_dim).to(DEVICE) for z in embed_list]
params = []
for decoder in decoders:
    decoder.train()
    params += list(decoder.parameters())

vae_optimizer = torch.optim.Adam(list(encoder.parameters()) + params)

# Training 
for e in range(epochs):
    epoch_loss = 0
    for image, label in tqdm(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        vae_optimizer.zero_grad()
        mu, logvar = encoder.forward(image)
        loss = 0
        for i, embed_size in enumerate(embed_list):
            decoder = decoders[i]
            current_mu = mu[:, :embed_size]
            current_logvar = logvar[:, :embed_size]
            current_std = torch.exp(current_logvar / 2)
            eps = torch.randn_like(current_std)
            z = current_mu + current_std * eps
            x_reconstructed = decoder.forward(z)

            # Mean Square Error Loss (Reconstruction loss in VAE Loss function)
            mse_loss = torch.nn.functional.mse_loss(x_reconstructed, image)

            # Maximum Mean Discrepancy
            true_samples =  torch.randn(num_samples, embed_size).to(DEVICE)
            mmd = compute_mmd(true_samples, z)

            # KL-Divergence Loss (Regularization loss in VAE Loss function)
            kl_loss = torch.mul(input=torch.sum(current_mu.pow(2) + current_logvar.exp() - current_logvar - 1), other=0.5)

            loss += (mse_loss + beta*mmd)
        loss.backward() # Perform Back-Propogation
        vae_optimizer.step() # Performs a single optimization step (parameter update)
        epoch_loss += loss
        del loss
    print(f'Epoch: {e}; Training Loss: {epoch_loss}')
    if best_score == None:
        best_score = epoch_loss
        save_weights(encoder, decoders)
    elif best_score>epoch_loss:
        best_score = epoch_loss
        save_weights(encoder, decoders)

encoder.load_state_dict(torch.load(f"../../weights/mat/fashionmnist_encoder_mat_weights.pt"))
encoder.eval()
classifier_set=[]

# Training Classifier
for i, embed_size in enumerate(embed_list):
    classifier = FeedForwardClassifier(output_dim=num_classes, embed_size=embed_size).to(DEVICE)
    classifier_set.append(classifier)
    classifier.train()
    classifier_optimizer = torch.optim.Adam(classifier.parameters())
    classifier_best_score = None

    for epoch in range(epochs):
        # Training Loop
        epoch_loss = 0
        for image, label in tqdm(train_loader):
            classifier_optimizer.zero_grad()
            image = image.to(DEVICE)
            label = label.type(torch.LongTensor).to(DEVICE)

            with torch.no_grad():
                # Passing the input image through VAE
                mu, logvar = encoder.forward(image)
                current_mu = mu[:, :embed_size]
                current_logvar = logvar[:, :embed_size]
                current_std = torch.exp(current_logvar / 2)
                eps = torch.randn_like(current_std)
                embedding = current_mu + current_std * eps

            # Passing the latent representation through Classifier
            prediction = classifier.forward(embedding)
            loss = torch.nn.CrossEntropyLoss()(prediction, label)

            loss.backward() # Perform Back-Propogation
            classifier_optimizer.step() # Performs a single optimization step (parameter update)
            epoch_loss += loss
            del loss
        print(f'Epoch: {epoch}; Training Loss: {epoch_loss}')
        if classifier_best_score == None:
            classifier_best_score = epoch_loss
            classifier_state_dict = classifier.state_dict()
            torch.save(classifier_state_dict, f"../../weights/mat/fashionmnist_classifier_{i}_weights.pt")
        elif classifier_best_score>epoch_loss:
            classifier_best_score = epoch_loss
            classifier_state_dict = classifier.state_dict()
            torch.save(classifier_state_dict, f"../../weights/mat/fashionmnist_classifier_{i}_weights.pt")

accuracies=[]

for i, embed_size in enumerate(embed_list):
    classifier = classifier_set[i]
    classifier.load_state_dict(torch.load(f"../../weights/mat/fashionmnist_classifier_{i}_weights.pt"))
    classifier.eval()

    correct = 0
    with torch.no_grad():
        for image, label in tqdm(test_loader):
            image = image.to(DEVICE)
            label = label.to(torch.int32).to(DEVICE)

            mu, logvar = encoder.forward(image)
            current_mu = mu[:, :embed_size]
            current_logvar = logvar[:, :embed_size]
            current_std = torch.exp(current_logvar / 2)
            eps = torch.randn_like(current_std)
            embedding = current_mu + current_std * eps

            prediction = classifier.predict(embedding).to(torch.int32)

            correct += int(sum(prediction==label))

    accuracy = 100*correct/len(test_loader.dataset)
    accuracies.append(accuracy)
    print(f"Accuracy for embed size {embed_size}: {accuracy:.2f}")


df = pd.DataFrame([], index=embed_list)
df["Accuracy"] = accuracies
df.to_csv("../../results/FashionMNIST_VAE_mat_accuracy.csv")