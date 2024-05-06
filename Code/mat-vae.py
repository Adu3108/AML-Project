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
    
    def encode(self, x, strategy, embed_size):
        mu, logvar = self.forward(x)
        current_mu = mu[:, :embed_size]
        current_logvar = logvar[:, :embed_size]
        current_std = torch.exp(current_logvar / 2)
        eps = torch.randn_like(current_std)
        if strategy=="Sampling":
            z = current_mu + eps * current_std
        elif strategy=="MAP":
            z = current_mu
        return z
    
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
    
class Trainer:
    def __init__(self, train_loader, val_loader, model_type, dataset_name, num_samples, epochs, patience, delta, beta, batch_size, embed_list, input_channels, input_dim):
        # Hyperparameters
        self.embed_list = embed_list
        self.max_embed_size = max(embed_list)
        self.input_channels = input_channels
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.epochs = epochs
        self.patience = patience
        self.delta = delta
        self.beta = beta

        # Dataloader
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.best_score = None
        self.num_bad_epochs = 0
        self.DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"

        assert model_type=="VAE" or model_type=="MMD"
        self.model_type = model_type
        assert dataset_name=="MNIST" or dataset_name=="FashionMNIST" or dataset_name=="CIFAR10"
        self.dataset_name = dataset_name

        # Model
        self.encoder = Encoder(z_dim=self.max_embed_size, input_channels=self.input_channels, input_dim=self.input_dim).to(self.DEVICE)
        self.decoders = [Decoder(z_dim=z, input_channels=self.input_channels, input_dim=self.input_dim).to(self.DEVICE) for z in self.embed_list]

    def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
        return torch.exp(-kernel_input) # (x_size, y_size

    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
        return mmd
    
    def save_weights(self, encoder, decoders, fold):
        encoder_state_dict = encoder.state_dict()
        torch.save(encoder_state_dict, f"../../weights/mat/{self.dataset_name}_{self.model_type}_encoder_mat_{fold}_weights.pt")

        for i, decoder in enumerate(decoders):
            decoder_state_dict = decoder.state_dict()
            torch.save(decoder_state_dict, f"../../weights/mat/{self.dataset_name}_{self.model_type}_decoder_{self.embed_list[i]}_{fold}_weights.pt")

    def early_stopping(self, fold):
        # Validation Loop
        val_loss = 0
        with torch.no_grad():
            for image, label in tqdm(self.val_loader):
                image = image.to(self.DEVICE)
                label = label.type(torch.LongTensor).to(self.DEVICE)

                mu, logvar = self.encoder.forward(image)
                for i, embed_size in enumerate(self.embed_list):
                    decoder = self.decoders[i]
                    current_mu = mu[:, :embed_size]
                    current_logvar = logvar[:, :embed_size]
                    current_std = torch.exp(current_logvar / 2)
                    eps = torch.randn_like(current_std)
                    z = current_mu + current_std * eps
                    x_reconstructed = decoder.forward(z)

                    # Mean Square Error Loss (Reconstruction loss in VAE Loss function)
                    mse_val_loss = torch.nn.functional.mse_loss(x_reconstructed, image)

                    # Maximum Mean Discrepancy
                    true_samples =  torch.randn(self.num_samples, embed_size).to(self.DEVICE)
                    mmd_val_loss = self.compute_mmd(true_samples, z)

                    # KL-Divergence Loss (Regularization loss in VAE Loss function)
                    kl_val_loss = torch.mul(input=torch.sum(current_mu.pow(2) + current_logvar.exp() - current_logvar - 1), other=0.5)

                    if self.model_type=="VAE":
                        val_loss += (mse_val_loss + self.beta*kl_val_loss)
                    elif self.model_type=="MMD":
                        val_loss += (mse_val_loss + self.beta*mmd_val_loss)
        
        # Early Stopping Condition
        if self.best_score == None:
            self.best_score = val_loss
            self.save_weights(self.encoder, self.decoders, fold)
        elif self.best_score-val_loss>self.delta:
            self.best_score = val_loss
            self.num_bad_epochs = 0
            self.save_weights(self.encoder, self.decoders, fold)
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs==self.patience:
            return True, val_loss
        else:
            return False, val_loss
        
    def training(self, fold):
        params = []
        for decoder in self.decoders:
            decoder.train()
            params += list(decoder.parameters())

        optimizer = torch.optim.Adam(list(self.encoder.parameters()) + params)
        self.encoder.train()

        for epoch in range(self.epochs):
            # Training Loop
            epoch_loss = 0
            for image, label in tqdm(self.train_loader):
                optimizer.zero_grad()
                image = image.to(self.DEVICE)
                label = label.type(torch.LongTensor).to(self.DEVICE)

                # print(image.shape)
                # if self.dataset_name=="CIFAR10":
                #     image = torch.reshape(image, (image.shape[0], image.shape[3], image.shape[1], image.shape[2]))

                # print(image.shape)
                mu, logvar = self.encoder.forward(image)
                loss = 0
                for i, embed_size in enumerate(self.embed_list):
                    decoder = self.decoders[i]
                    current_mu = mu[:, :embed_size]
                    current_logvar = logvar[:, :embed_size]
                    current_std = torch.exp(current_logvar / 2)
                    eps = torch.randn_like(current_std)
                    z = current_mu + current_std * eps
                    x_reconstructed = decoder.forward(z)

                    # Mean Square Error Loss (Reconstruction loss in VAE Loss function)
                    mse_train_loss = torch.nn.functional.mse_loss(x_reconstructed, image)

                    # Maximum Mean Discrepancy
                    true_samples =  torch.randn(self.num_samples, embed_size).to(self.DEVICE)
                    mmd_train_loss = self.compute_mmd(true_samples, z)

                    # KL-Divergence Loss (Regularization loss in VAE Loss function)
                    kl_train_loss = torch.mul(input=torch.sum(current_mu.pow(2) + current_logvar.exp() - current_logvar - 1), other=0.5)

                    # Total Loss Function
                    if self.model_type=="VAE":
                        loss += (mse_train_loss + self.beta*kl_train_loss)       # Reconstruction Loss + Regularization Loss
                    elif self.model_type=="MMD":
                        loss += (mse_train_loss + self.beta*mmd_train_loss)           # Reconstruction Loss + Maximum Mean Discrepancy

                loss.backward() # Perform Back-Propogation
                optimizer.step() # Performs a single optimization step (parameter update)
                epoch_loss += loss
                del loss
                
            early_stopping, val_loss = self.early_stopping(fold)
            print(f'Epoch: {epoch}; Training Loss: {epoch_loss}; Validation Loss: {val_loss}')
            
            if early_stopping:
                print(f'Early Stopping at epoch {epoch}')
                break

class Classifier:
    def __init__(self, model_type, train_loader, val_loader, test_loader, dataset_name, epochs, num_classes, embed_list, sampling_strategy, delta, patience, input_channels, input_dim):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.dataset_name = dataset_name
        self.model_type = model_type,

        self.input_channels = input_channels
        self.input_dim = input_dim
        self.epochs = epochs
        self.num_classes = num_classes
        self.embed_list = embed_list
        self.max_embed_size = max(embed_list)
        self.delta = delta
        self.patience = patience
        self.DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"

        assert sampling_strategy=="Sampling" or sampling_strategy=="MAP"
        self.sampling_strategy = sampling_strategy

        self.encoder = Encoder(z_dim=self.max_embed_size, input_channels=self.input_channels, input_dim=self.input_dim).to(self.DEVICE)
        self.encoder.load_state_dict(torch.load(f"../../weights/mat/{self.dataset_name}_{self.model_type[0]}_encoder_mat_{fold}_weights.pt"))
        self.encoder.eval()
        self.classifiers = [FeedForwardClassifier(output_dim=self.num_classes, embed_size=z).to(self.DEVICE) for z in self.embed_list]

    def early_stopping(self, embed_idx, fold):
        # Validation Loop
        val_loss = 0
        with torch.no_grad():
            for image, label in tqdm(self.val_loader):
                image = image.to(self.DEVICE)
                label = label.type(torch.LongTensor).to(self.DEVICE)

                embedding = self.encoder.encode(image, self.sampling_strategy, self.embed_list[embed_idx])
                prediction = self.classifiers[embed_idx].forward(embedding)
                classification_loss = torch.nn.CrossEntropyLoss()(prediction, label)
                val_loss += classification_loss
        
        # Early Stopping
        if self.best_score == None:
            self.best_score = val_loss
            classifier_state_dict = self.classifiers[embed_idx].state_dict()
            torch.save(classifier_state_dict, f"../../weights/mat/{self.dataset_name}_classifier_{self.embed_list[embed_idx]}_{fold}_weights.pt")
        elif self.best_score-val_loss>self.delta:
            self.best_score = val_loss
            classifier_state_dict = self.classifiers[embed_idx].state_dict()
            self.num_bad_epochs = 0
            torch.save(classifier_state_dict, f"../../weights/mat/{self.dataset_name}_classifier_{self.embed_list[embed_idx]}_{fold}_weights.pt")
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs==self.patience:
            return True, val_loss
        else:
            return False, val_loss

    def training(self, fold):
        for i, embed_size in enumerate(self.embed_list):
            classifier = self.classifiers[i]
            classifier.train()
            classifier_optimizer = torch.optim.Adam(classifier.parameters())
            self.best_score = None
            self.num_bad_epochs = 0

            for epoch in range(self.epochs):
                # Training Loop
                epoch_loss = 0
                for image, label in tqdm(self.train_loader):
                    classifier_optimizer.zero_grad()
                    image = image.to(self.DEVICE)
                    label = label.type(torch.LongTensor).to(self.DEVICE)

                    with torch.no_grad():
                        # Passing the input image through VAE
                        mu, logvar = self.encoder.forward(image)
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
                
                early_stopping, val_loss = self.early_stopping(i, fold)
                print(f'Epoch: {epoch}; Training Loss: {epoch_loss}; Validation Loss: {val_loss}')
                
                if early_stopping:
                    print(f'Early Stopping at epoch {epoch}')
                    break
            

    def predict(self, fold):
        accuracies = []
        for i, embed_size in enumerate(self.embed_list):
            classifier = self.classifiers[i]
            classifier.load_state_dict(torch.load(f"../../weights/mat/{self.dataset_name}_classifier_{embed_size}_{fold}_weights.pt"))
            classifier.eval()

            correct = 0
            with torch.no_grad():
                for image, label in tqdm(self.test_loader):
                    image = image.to(self.DEVICE)
                    label = label.to(torch.int32).to(self.DEVICE)

                    mu, logvar = self.encoder.forward(image)
                    current_mu = mu[:, :embed_size]
                    current_logvar = logvar[:, :embed_size]
                    current_std = torch.exp(current_logvar / 2)
                    eps = torch.randn_like(current_std)
                    embedding = current_mu + current_std * eps

                    prediction = classifier.predict(embedding).to(torch.int32)

                    correct += int(sum(prediction==label))

            accuracy = 100*correct/len(self.test_loader.dataset)
            accuracies.append(accuracy)
            print(f"Accuracy for embed size {embed_size}: {accuracy:.2f}")
        return accuracies

MNIST_training_dataset = datasets.MNIST(root="../../data/MNIST", train=True, download=True, transform=transforms.ToTensor())
MNIST_test_dataset = datasets.MNIST(root="../../data/MNIST", train=False, download=True, transform=transforms.ToTensor())

FashionMNIST_training_dataset = datasets.FashionMNIST(root="../../data/FashionMNIST", train=True, download=True, transform=transforms.ToTensor())
FashionMNIST_test_dataset = datasets.FashionMNIST(root="../../data/FashionMNIST", train=False, download=True, transform=transforms.ToTensor())

CIFAR10_training_dataset = datasets.CIFAR10(root="../../data/CIFAR10", train=True, download=True, transform=transforms.ToTensor())
CIFAR10_test_dataset = datasets.CIFAR10(root="../../data/CIFAR10", train=False, download=True, transform=transforms.ToTensor())

batch_size = 32
num_samples = 200
model_epochs = 20
classifier_epochs = 20
patience = 5
delta = 1e-4
beta = 5
embed_list = [2,4,8,16,32,64,128]
sampling_strategies = ["Sampling", "MAP"]
model_types = ["VAE", "MMD"]

train_datasets = [CIFAR10_training_dataset]
test_datasets = [CIFAR10_test_dataset]
dataset_names = ["CIFAR10"]

for i in range(len(train_datasets)):
    training_dataset = train_datasets[i]
    test_dataset = test_datasets[i]
    dataset_name = dataset_names[i]

    num_classes = len(training_dataset.classes)
    image_shape = training_dataset.data[0].shape
    if len(image_shape)==2:
        input_channels = 1
        input_dim = tuple(image_shape)
    else:
        input_channels = 3
        if image_shape[2]==3:
            input_dim = tuple(image_shape[:2])
        else:
            input_dim = tuple(image_shape[1:])

    for model_type in model_types:
        for sampling_strategy in sampling_strategies:
            print(f"Model type - {model_type}; Sampling Strategy - {sampling_strategy}")
            k_folds = 5
            kf = KFold(n_splits=k_folds, shuffle=True)
            test_loader = DataLoader(test_dataset, shuffle=False, batch_size = batch_size)

            accuracy = np.zeros((len(embed_list), k_folds))
            for fold, (train_idx, val_idx) in enumerate(kf.split(training_dataset)):
                print(f"Fold {fold + 1}")
                print("-------")

                # Define the data loaders for the current fold
                train_loader = DataLoader(
                    dataset=torch.utils.data.Subset(training_dataset, train_idx),
                    batch_size=200
                )
                val_loader = DataLoader(
                    dataset=torch.utils.data.Subset(training_dataset, val_idx),
                    batch_size=200,
                )

                # Train VAE Network
                trainer = Trainer(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model_type=model_type,
                    dataset_name=dataset_name,
                    num_samples=num_samples,
                    epochs=model_epochs,
                    patience=patience,
                    delta=delta,
                    beta=beta,
                    batch_size=batch_size,
                    embed_list=embed_list,
                    input_channels=input_channels,
                    input_dim=input_dim
                )
                trainer.training(fold)

                # Train Classifier Network
                classifier = Classifier(
                    model_type=model_type,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    dataset_name=dataset_name,
                    epochs=classifier_epochs,
                    num_classes=num_classes,
                    embed_list=embed_list,
                    sampling_strategy=sampling_strategy,
                    delta=delta,
                    patience=patience,
                    input_channels=input_channels,
                    input_dim=input_dim
                )
                classifier.training(fold)
                accuracy[:,fold] = classifier.predict(fold)

            mean_accuracy = np.mean(accuracy, axis=1)
            best_accuracy = np.max(accuracy, axis=1)

            df = pd.DataFrame([], index=embed_list)
            df["Mean Accuracy"] = mean_accuracy
            df["Best Accuracy"] = best_accuracy
            df.to_csv(f"../../results/{dataset_name}_{model_type}_mat_accuracy.csv")