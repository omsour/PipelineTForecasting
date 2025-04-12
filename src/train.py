import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.model.LSTMForecast import LSTMForecaster
from src.dataset import TimeSeriesDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_model(data_folder, input_window=30, forecast_horizon=10, epochs=10, batch_size=128, learning_rate=0.01):
    dataset = TimeSeriesDataset(data_folder, input_window, forecast_horizon)
    train_loader = DataLoader(dataset.get_split_dataset("train"), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset.get_split_dataset("val"), batch_size=batch_size, shuffle=False)

    total_samples = len(dataset.train_data) - input_window - forecast_horizon
    num_batches = total_samples // batch_size
    print(f"Total samples in training set: {total_samples}")
    print(f"Number of batches per epoch: {num_batches}")

    model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2, output_size=forecast_horizon).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    	
    try:
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_bar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}]", leave=True)

            for inputs, targets in train_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_bar.set_postfix(loss=loss.item())

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_inputs, val_targets in tqdm(val_loader, desc="Validating", leave=False):
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    val_outputs = model(val_inputs)
                    loss = criterion(val_outputs, val_targets)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'best_model.pth')
    except KeyboardInterrupt:
        print("\nTraining interrupted. Plotting training and validation losses up to this point.")
    
    print("Training complete")

    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.grid(True)
    plt.show()

    return model
