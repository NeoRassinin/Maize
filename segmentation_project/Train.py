import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_model(loader, model, optimizer, device):
    """Обучает модель на одном проходе по train loader."""
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits, loss = model(images, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def eval_model(loader, model, device):
    """Оценивает модель на валидации или тесте."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(loader):
            images = images.to(device)
            masks = masks.to(device)

            logits, loss = model(images, masks)
            total_loss += loss.item()
    return total_loss / len(loader)


def train_model_epochs(model, train_loader, val_loader, device, best_model_name="best_model.pt", epochs=30, lr=1e-5):
    """Обучает модель несколько эпох, сохраняет лучшую по валидационной потере."""
    val_loss_lst = []
    train_loss_lst = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss = train_model(train_loader, model, optimizer, device)
        val_loss = eval_model(val_loader, model, device)

        train_loss_lst.append(train_loss)
        val_loss_lst.append(val_loss)

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), best_model_name)
            best_val_loss = val_loss
            print("MODEL SAVED")

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    return {'train_loss': train_loss_lst, 'val_loss': val_loss_lst}


def visualize_train_results(history, model_name):
    """Строит график потерь для обучения и валидации."""
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], 'r', label="Training loss")
    plt.plot(epochs, history['val_loss'], 'b', label="Validation loss")
    plt.title(model_name)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(model_name + ".jpg")
    plt.show()
