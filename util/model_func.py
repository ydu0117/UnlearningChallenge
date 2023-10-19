from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter


def train(model, loader, root, epoch, device, optim, loss_fn, writer):
    # Set model to training mode
    model.train()
    iter_count = 0
    epoch_loss = 0
    epoch_acc = 0

    # Add tqdm progress bar
    pbar = tqdm(loader)

    for batch in pbar:
        # Perform training here
        images, labels = batch['image'].to(device), batch['age_group'].to(device)
        # Generate predictions
        outputs = model(images)

        # Calculate loss
        loss = loss_fn(outputs, labels)

        # Backpropagate
        optim.zero_grad()
        loss.backward()
        optim.step()

        # calculate accuracy and loss
        epoch_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        epoch_acc += torch.sum(preds == labels.data).item()
        acc = torch.sum(preds == labels.data).item() / labels.data.numel()
        # Update tqdm progress bar
        pbar.set_description(f'Epoch: {epoch}, Loss: {loss:.4f}, Acc: {acc:.4f}')

        iter_count += labels.data.numel()
        writer.add_scalar('Training loss (iteration)', loss.item(), iter_count+epoch*len(loader.dataset))
        writer.add_scalar('Training accuracy (iteration)', acc, iter_count+epoch*len(loader.dataset))
    pbar.close()
    # Record loss for each epoch
    writer.add_scalar('Training loss (epoch)', epoch_loss/len(loader), epoch)
    writer.add_scalar('Training accuracy (epoch)', epoch_acc/len(loader.dataset), epoch)
    return epoch_loss/len(loader), epoch_acc/len(loader.dataset)

def evaluate(model, loader, root, epoch, device, optim, loss_fn, writer):
    # Set model to training mode
    model.eval()
    iter_count = 0
    epoch_loss = 0
    epoch_acc = 0

    # Add tqdm progress bar
    pbar = tqdm(loader)
    with torch.no_grad():
        for batch in pbar:

            # Perform training here
            images, labels = batch['image'].to(device), batch['age_group'].to(device)
            # Generate predictions
            outputs = model(images)

            # Calculate loss
            loss = loss_fn(outputs, labels)

            # calculate accuracy and loss
            epoch_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            epoch_acc += torch.sum(preds == labels.data).item()
            acc = torch.sum(preds == labels.data).item() / labels.data.numel()
            # Update tqdm progress bar
            pbar.set_description(f'Epoch: {epoch}, Loss: {loss:.4f}, Acc: {acc:.4f}')

            iter_count += labels.data.numel()
            # Record loss for each iteration
            writer.add_scalar('Evaluation loss (iteration)', loss.item(), iter_count+epoch*len(loader.dataset))
            writer.add_scalar('Evaluation accuracy (iteration)', acc, iter_count+epoch*len(loader.dataset))

    # Record loss for each epoch
    writer.add_scalar('Evaluation loss (epoch)', epoch_loss/len(loader), epoch)
    writer.add_scalar('Evaluation accuracy (epoch)', epoch_acc/len(loader.dataset), epoch)
    pbar.close()

    return epoch_loss/len(loader), epoch_acc/len(loader.dataset)


