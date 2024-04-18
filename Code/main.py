import torchvision.datasets as datasets
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

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
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.model = torch.nn.ModuleList([
            torch.nn.Conv2d(1, 64, 4, 2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 128, 4, 2, padding=1),
            torch.nn.LeakyReLU(),
            Flatten(),
            torch.nn.Linear(6272, 1024),
            torch.nn.LeakyReLU(),
        ])
        self.dense_mu = torch.nn.Linear(1024, z_dim)
        self.dense_logvar = torch.nn.Linear(1024, z_dim)
        
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        mu = self.dense_mu(x)
        logvar = self.dense_logvar(x)
        return mu, logvar
    
    
class Decoder(torch.nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        self.model = torch.nn.ModuleList([
            torch.nn.Linear(z_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 7*7*128),
            torch.nn.ReLU(),
            Reshape((128,7,7,)),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 1, 4, 2, padding=1),
            torch.nn.Sigmoid()
        ])
        
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x
    
class Model(torch.nn.Module):
    def __init__(self, z_dim, strategy):
        super(Model, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        self.strategy = strategy
        
    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)

        if self.strategy=="Sampling":
            z = mu + eps * std
        elif self.strategy=="MAP":
            z = mu

        x_reconstructed = self.decoder(z)
        return z, x_reconstructed, mu, logvar

class Trainer:
    def __init__(self, train_loader, val_loader, model_type, sampling_strategy, num_samples, epochs, patience, delta, batch_size, embed_size):
        # Hyperparameters
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.epochs = epochs
        self.patience = patience
        self.delta = delta

        # Dataloader
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.best_score = None
        self.num_bad_epochs = 0
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        assert model_type=="VAE" or model_type=="MMD"
        self.model_type = model_type
    
        assert sampling_strategy=="Sampling" or sampling_strategy=="MAP"
        self.sampling_strategy = sampling_strategy

        # Model
        self.model = Model(z_dim=self.embed_size, strategy=self.sampling_strategy).to(self.DEVICE)

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
    
    def early_stopping(self, fold):
        # Validation Loop
        val_loss = 0
        with torch.no_grad():
            for image, label in tqdm(self.val_loader):
                image = image.to(self.DEVICE)
                label = label.type(torch.LongTensor).to(self.DEVICE)

                true_samples =  torch.randn(self.num_samples, self.embed_size).to(self.DEVICE)
                embedding, out, mu, logvar = self.model.forward(image)

                mse_val_loss = torch.nn.functional.mse_loss(out, image)
                kl_val_loss = torch.mul(input=torch.sum(mu.pow(2) + logvar.exp() - logvar - 1), other=0.5)
                mmd_val_loss = self.compute_mmd(true_samples, embedding)

                if self.model_type=="VAE":
                    val_loss += (mse_val_loss + kl_val_loss)
                elif self.model_type=="MMD":
                    val_loss += (mse_val_loss + mmd_val_loss)
        
        # Early Stopping Condition
        if self.best_score == None:
            self.best_score = val_loss
            model_state_dict = self.model.state_dict()
            torch.save(model_state_dict, f"./weights/VAE/VAE_{fold}_weights.pt")
        elif self.best_score-val_loss>self.delta:
            self.best_score = val_loss
            model_state_dict = self.model.state_dict()
            self.num_bad_epochs = 0
            torch.save(model_state_dict, f"./weights/VAE/VAE_{fold}_weights.pt")
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs==self.patience:
            return True, val_loss
        else:
            return False, val_loss
        
    def training(self, fold):
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.model.train()

        for epoch in range(self.epochs):
            # Training Loop
            epoch_loss = 0
            for image, label in tqdm(self.train_loader):
                self.optimizer.zero_grad()
                image = image.to(self.DEVICE)
                label = label.type(torch.LongTensor).to(self.DEVICE)

                # Passing the input image through VAE
                embedding, out, mu, logvar = self.model.forward(image)

                # Mean Square Error Loss (Reconstruction loss in VAE Loss function)
                mse_loss = torch.nn.functional.mse_loss(out, image)

                # Maximum Mean Discrepancy
                true_samples =  torch.randn(self.num_samples, self.embed_size).to(self.DEVICE)
                mmd = self.compute_mmd(true_samples, embedding)

                # KL-Divergence Loss (Regularization loss in VAE Loss function)
                kl_loss = torch.mul(input=torch.sum(mu.pow(2) + logvar.exp() - logvar - 1), other=0.5)

                # Total Loss Function
                if self.model_type=="VAE":
                    loss = mse_loss + kl_loss       # Reconstruction Loss + Regularization Loss
                elif self.model_type=="MMD":
                    loss = mse_loss + mmd           # Reconstruction Loss + Maximum Mean Discrepancy

                loss.backward() # Perform Back-Propogation
                self.optimizer.step() # Performs a single optimization step (parameter update)
                epoch_loss += loss
                del loss, out
                
            early_stopping, val_loss = self.early_stopping(fold)
            print(f'Epoch: {epoch}; Training Loss: {epoch_loss}; Validation Loss: {val_loss}')
            
            if early_stopping:
                print(f'Early Stopping at epoch {epoch}')
                break

class Classifier:
    def __init__(self, model, train_loader, val_loader, test_loader, epochs, num_classes, embed_size, delta, patience):
        self.training_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.epochs = epochs
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.delta = delta
        self.patience = patience
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model
        self.classifier = FeedForwardClassifier(output_dim=self.num_classes, embed_size=self.embed_size).to(self.DEVICE)

    def early_stopping(self, fold):
        # Validation Loop
        val_loss = 0
        with torch.no_grad():
            for image, label in tqdm(self.val_loader):
                image = image.to(self.DEVICE)
                label = label.type(torch.LongTensor).to(self.DEVICE)

                embedding, _, _, _ = self.model.forward(image)
                prediction = self.classifier.forward(embedding)
                classification_loss = torch.nn.CrossEntropyLoss()(prediction, label)
                val_loss += classification_loss
        
        # Early Stopping
        if self.best_score == None:
            self.best_score = val_loss
            classifier_state_dict = self.classifier.state_dict()
            torch.save(classifier_state_dict, f"./weights/classifier/classifier_{fold}_weights.pt")
        elif self.best_score-val_loss>self.delta:
            self.best_score = val_loss
            classifier_state_dict = self.classifier.state_dict()
            self.num_bad_epochs = 0
            torch.save(classifier_state_dict, f"./weights/classifier/classifier_{fold}_weights.pt")
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs==self.patience:
            return True, val_loss
        else:
            return False, val_loss

    def training(self, fold):
        self.optimizer = torch.optim.Adam(self.classifier.parameters())
        self.best_score = None
        self.num_bad_epochs = 0
        
        for epoch in range(self.epochs):
            # Training Loop
            epoch_loss = 0
            for image, label in tqdm(self.training_loader):
                self.optimizer.zero_grad()
                image = image.to(self.DEVICE)
                label = label.type(torch.LongTensor).to(self.DEVICE)

                with torch.no_grad():
                    # Passing the input image through VAE
                    embedding, _, _, _ = self.model.forward(image)

                # Passing the latent representation through Classifier
                prediction = self.classifier.forward(embedding)
                loss = torch.nn.CrossEntropyLoss()(prediction, label)

                loss.backward() # Perform Back-Propogation
                self.optimizer.step() # Performs a single optimization step (parameter update)
                epoch_loss += loss
                del loss
            
            early_stopping, val_loss = self.early_stopping(fold)
            print(f'Epoch: {epoch}; Training Loss: {epoch_loss}; Validation Loss: {val_loss}')
            
            if early_stopping:
                print(f'Early Stopping at epoch {epoch}')
                break

    def predict(self, fold):
        self.model.load_state_dict(torch.load(f"./weights/VAE/VAE_{fold}_weights.pt"))

        classifier_network = FeedForwardClassifier(output_dim=self.num_classes, embed_size=self.embed_size).to(self.DEVICE)
        classifier_network.load_state_dict(torch.load(f"./weights/classifier/classifier_{fold}_weights.pt"))

        self.model.eval()
        classifier_network.eval()

        correct = 0
        with torch.no_grad():
            for image, label in tqdm(self.test_loader):
                image = image.to(self.DEVICE)
                label = label.to(torch.int32).to(self.DEVICE)

                embedding, _, _, _ = self.model.forward(image)
                prediction = classifier_network.predict(embedding).to(torch.int32)

                correct += int(sum(prediction==label))
                
        accuracy = 100*correct/len(self.test_loader.dataset)
        print(f"Accuracy : {accuracy:.2f}")
        return accuracy
            
MNIST_training_dataset = datasets.MNIST(root="data/MNIST", train=True, download=True, transform=transforms.ToTensor())
MNIST_test_dataset = datasets.MNIST(root="data/MNIST", train=False, download=True, transform=transforms.ToTensor())

CIFAR10_training_dataset = datasets.CIFAR10(root="data/CIFAR10", train=True, download=True, transform=transforms.ToTensor())
CIFAR10_test_dataset = datasets.CIFAR10(root="data/CIFAR10", train=False, download=True, transform=transforms.ToTensor())

CIFAR100_training_dataset = datasets.CIFAR100(root="data/CIFAR100", train=True, download=True, transform=transforms.ToTensor())
CIFAR100_test_dataset = datasets.CIFAR100(root="data/CIFAR100", train=False, download=True, transform=transforms.ToTensor())

FashionMNIST_training_dataset = datasets.FashionMNIST(root="data/FashionMNIST", train=True, download=True, transform=transforms.ToTensor())
FashionMNIST_test_dataset = datasets.FashionMNIST(root="data/FashionMNIST", train=False, download=True, transform=transforms.ToTensor())

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
um_samples = 200
model_epochs = 10
classifier_epochs = 20
patience = 5
delta = 1e-4
embed_sizes = [2, 8, 16, 32, 64, 128]
sampling_strategies = ["Sampling", "MAP"]
model_types = ["VAE", "MMD"]

train_datasets = [FashionMNIST_training_dataset]
test_datasets = [FashionMNIST_test_dataset]
dataset_names = ["FashionMNIST"]

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
        input_channels = image_shape[0]
        input_dim = tuple(image_shape[1:])

    for model_type in model_types:
        for sampling_strategy in sampling_strategies:
            accuracies = []
            best_accuracies = []
            for embed_size in embed_sizes:
                k_folds = 5
                kf = KFold(n_splits=k_folds, shuffle=True)
                test_loader = DataLoader(test_dataset, shuffle=False, batch_size = batch_size)
                accuracy = []

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
                        sampling_strategy=sampling_strategy,
                        model_type=model_type,
                        num_samples=num_samples,
                        epochs=model_epochs,
                        patience=patience,
                        delta=delta,
                        batch_size=batch_size,
                        embed_size=embed_size
                    )
                    trainer.training(fold)

                    # Load the trained VAE Network
                    model = Model(z_dim=embed_size, strategy=sampling_strategy).to(DEVICE)
                    model.load_state_dict(torch.load(f"./weights/VAE/VAE_{fold}_weights.pt"))

                    # Train Classifier Network
                    classifier = Classifier(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        epochs=classifier_epochs,
                        num_classes=num_classes,
                        embed_size=embed_size,
                        delta=delta,
                        patience=patience
                    )
                    classifier.training(fold)
                    accuracy.append(classifier.predict(fold))

                accuracies.append(round(np.mean(accuracy),3))
                best_accuracies.append(round(np.max(accuracy),3))

                print(f"Mean Accuracy : {np.mean(accuracy):.2f}")
                print(f"Best Accuracy : {np.max(accuracy):.2f} at {np.argmax(accuracy)} fold")
                print()
            
            df = pd.DataFrame(list(zip(*[embed_sizes, accuracies, best_accuracies])), columns=["Embedding Dimension", "Mean Accuracy", "Best Accuracy"])
            df.to_csv(f"./results/{dataset_name}_{model_type}_{sampling_strategy}.csv")
