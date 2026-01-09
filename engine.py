import torch

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    One epoch of training.
    Returns: average_loss (float)
    """
    model.train()
    running_loss = 0.0
    total_ex = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.shape[0] # loss.item - avg loss over all images in batch -> loss.item() x batch_size
        total_ex += labels.shape[0] #count all examples in batch + add over all batches in loader
    avg_loss = running_loss / total_ex
    
    return avg_loss
    
def evaluate(model, loader, criterion, device):
    """
    Evaluation loop (no gradients)
    Returns: avg_loss, accuracy
    """
    model.eval()
    with torch.no_grad():
        total_ex = 0
        correct = 0
        running_loss = 0.0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * labels.shape[0]
            preds = logits.argmax(dim=1)
            correct_in_batch = (preds == labels).sum().item()
            correct += correct_in_batch
            total_ex += labels.shape[0]
        
        avg_loss = running_loss / total_ex
        acc = correct / total_ex

        return avg_loss, acc



