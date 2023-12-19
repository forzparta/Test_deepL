import torch
from Python.Resnet19.Utils import utils
from Python.Resnet19.Training import train
from Python.Resnet19.Data import dataloader
from Python.Resnet19.Models import resnet19


def training():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    data_loader = dataloader.DatasetsSelector(download = True)
    trainloader, testloader, classes = dataloader.DatasetLoader(data_loader).getDataloaders()
    
    model_name = 'snn'
    num_classes = len(classes)
    model = resnet19.select_model(model_type=model_name, num_classes=num_classes, device=device)
    #model_1 = resnet19.select_model('fc', num_classes, device)
    #model_2 = resnet19.select_model('conv', num_classes, device)

    if model_name == 'snn':
        criterion, optimizer, scheduler = utils.setup_training_components(model, scheduler_type="StepLR", step_size= 10)
    else:
        criterion, optimizer, scheduler = utils.setup_training_components(model, scheduler_type="StepLR", momentum = 0.9, step_size= 10)

    num_epoch = 1
    try:
        train.train_model(model, optimizer, criterion, num_epoch, trainloader, testloader, scheduler= scheduler, device= device)
    except KeyboardInterrupt:
        print('manually interrupt')
        train.save_model(model)

if __name__ == '__main__':
    training()