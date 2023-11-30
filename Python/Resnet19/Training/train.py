import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time


def train(model, trainloader, optimizer, criterion, device, writer, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(trainloader, desc="Training")
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item()
        loop.set_postfix(loss = loss.item())

    avg_loss = total_loss / len(trainloader)
    accuracy = correct / total
    loop.set_postfix(avg_loss = avg_loss, accuracy = accuracy)

    writer.add_scalar("LOSS/Training", avg_loss, epoch)
    writer.add_scalar("ACCURACY/Training", accuracy, epoch)

    return avg_loss, accuracy


def evaluate(model, testloader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Evaluation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(testloader)
    accuracy = correct / total
    
    writer.add_scalar("LOSS/Evaluation", avg_loss, epoch)
    writer.add_scalar( "ACCURACY/Evaluation", accuracy, epoch)

    return avg_loss, accuracy


def train_model(model, optimizer, criterion, num_epochs, trainloader, testloader, scheduler = None, device = "cpu"):
    writer = SummaryWriter("./Logs")
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train(model, trainloader, optimizer, criterion, device, writer, epoch)
        eval_loss, eval_acc = evaluate(model, testloader, criterion, device, writer, epoch)
        if scheduler != None:
            lr = scheduler.get_last_lr()[0]
            scheduler.step()
            writer.add_scalar("LR/Training", lr, epoch)

        end_time = time.time()
        print(f"Epoch {epoch + 1} - Training Loss: {train_loss:.3f}, Accuracy: {train_acc:.2f}%")
        print(f"Epoch {epoch + 1} - Evaluation Loss: {eval_loss:.3f}, Accuracy: {eval_acc:.2f}%")
        print(f"Epoch {epoch + 1} - LR: {lr}, Time: {end_time - start_time:.2f} sec\n")
        if hasattr(model,'epoch'):
            model.epoch += 1

    print("Finished Training")
    save_model(model)

def save_model(model):
    if hasattr(model,'epoch'):
        torch.save(model.state_dict(), model.__class__.__name__ +"_epoch_"+ str(model.epoch) +".pth")
    else:
        torch.save(model.state_dict(), model.__class__.__name__ +".pth")
    print('model saved.')