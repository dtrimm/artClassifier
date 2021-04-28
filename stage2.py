# Common modules/packages
import matplotlib.pyplot as plt
import math
import numpy as np
import pathlib
import sys, shutil, time
import warnings

# PyTorch modules/packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from torchvision import datasets, models, transforms
from PIL import ImageFile



# Retrieves the list of files with a directory
def getFilesInDirectory(pathToDir, extension = "*.*"):
    if not isinstance(pathToDir, pathlib.PurePath):
        pathToDir = pathlib.Path(pathToDir)

    return list(pathToDir.glob(extension))

# Retrieves the list of folders with a directory
def getFoldersInDirectory(pathToDir, prefix = ""):
    if not isinstance(pathToDir, pathlib.PurePath):
        pathToDir = pathlib.Path(pathToDir)

    return sorted([fld for fld in pathToDir.iterdir() if fld.is_dir() and not fld.name.lower().startswith(prefix)])

# Retrieves the list of folders with a directory
def getFolderNamesInDirectory(pathToDir, prefix = ""):
    if not isinstance(pathToDir, pathlib.PurePath):
        pathToDir = pathlib.Path(pathToDir)
    return sorted([fld.name for fld in pathToDir.iterdir() if fld.is_dir() and not fld.name.lower().startswith(prefix)])


#%matplotlib inline
warnings.filterwarnings('ignore')

class Device():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def train_on_gpu(self):
        return self.device == 'cuda'
    def getDevice(self):
        return self.device
    def getCUDAProperties():
        if self.train_on_gpu():
            return torch.cuda.get_device_properties(0)


################################################################
# GLOBAL VARIABLES
#
device = Device()

# sets folders for image sets and model
pathToDataset = pathlib.Path.cwd().joinpath('..', 'dataset')
pathToTrain = pathToDataset.joinpath('train')
pathToTest = pathToDataset.joinpath('test')
pathToValid = pathToDataset.joinpath('valid')
pathToModel = None
theModel = None


# list the folders required under 'dataset' folder (using a list to reduce the lines of code)
artCategories = getFolderNamesInDirectory(pathToTrain, ".")  #collects the list of folders

# Model parameters
batchSize = 25
numWorkers = 4
n_epochs = 40 #  number of epochs to train the model
valid_loss_min = np.Inf # tracker for minimum validation loss
train_losses, valid_losses, accuracies, classes_accuracies=[],[],[], []
training_loss = 0.0
validation_loss = 0.0
accuracy = 0.0
ImageFile.LOAD_TRUNCATED_IMAGES = False  # Some images in dataset were truncated (maybe corrupted)
criterion, optimizer = None, None
train_dataset, test_dataset, valid_dataset = None, None, None
train_dataloader, test_dataloader, valid_dataloader = None, None, None




def global_summary():
    global pathToDataset, pathToTrain, pathToTest, pathToValid, pathToModel, batchSize, numWorkers 
    print(f"Total no. of categories = {len(artCategories)}")  #displays the number of classes (= Art categories)
    print(f"Categories: {artCategories}")  #displays the list of classes
    print(f"Device to use is {device.getDevice}")
    print(f"Path to dataset is {pathToDataset}")
    print(f"Path to training is {pathToTrain}")
    print(f"Path to test is {pathToTest}")
    print(f"Path to valid is {pathToValid}")
    print(f"Batch size is {batchSize}")
    print(f"Number of workers is {numWorkers}")
    print(f"Number of epochs {n_epochs}")
    


####################################################################################

    
# Define the data-augmentation transforms including normalisations

def define_and_apply_transforms():
    global train_dataset, test_dataset, valid_dataset
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                std=(0.229, 0.224, 0.225)),
                                           ])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                std=(0.229, 0.224, 0.225)),
                                           ])     

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                std=(0.229, 0.224, 0.225)),
                                           ])

    # load and apply above transforms on dataset using ImageFolder
    train_dataset = torchvision.datasets.ImageFolder(root=pathToTrain, transform=train_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root=pathToTest, transform=test_transforms)
    valid_dataset = torchvision.datasets.ImageFolder(root=pathToValid, transform=valid_transforms)

    
def print_dataset_stats():
    # Print out data stats
    print('Training  images: ', len(train_dataset))
    print('Testing   images: ', len(test_dataset))
    print('Validation images:', len(valid_dataset))


def generate_dataloaders():
    global batchSize, numWorkers
    global train_dataloader, test_dataloader, valid_dataloader
    global train_dataset, test_dataset, valid_dataset
    
    # Prepare data loaders
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True, num_workers=numWorkers)
    test_dataloader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batchSize, shuffle=True, num_workers=numWorkers)
    valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batchSize, shuffle=True, num_workers=numWorkers)

    # Print the batches stats
    print('Number of  training  batches:', len(train_dataloader))
    print('Number of  testing   batches:', len(test_dataloader))
    print('Number of validation batches:', len(valid_dataloader))


##################Optional Data Visualization################################

def data_visualizer():
    global batchSize, numWorkers
    # Ignore normalization and turn shuffle ON to visualize different art categories together
    visual_transforms  = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])

    # Load and apply above transform on dataset using ImageFolder
    # Use test directory images can be used for visualization
    visual_dataset  = torchvision.datasets.ImageFolder(root=pathToTest, transform=visual_transforms)

    # Prepare data loaders
    visualization_dataloader = torch.utils.data.DataLoader(dataset=visual_dataset,
                                                           batch_size=batchSize,
                                                           shuffle=True,
                                                           num_workers=numWorkers)

    # Obtain one batch of testing images
    dataiter = iter(visualization_dataloader)
    images, labels = dataiter.next()

    # Convert images to numpy for display
    images = np.asarray(images)

    # Plot the images in the batch along with the corresponding labels
    plotCols = 4
    plotRows = math.ceil(batchSize/plotCols) # SqRoot could be used as well: math.ceil(math.sqrt(batch_size))
    fig = plt.figure(figsize=(25, 25))
    for idx in np.arange(batchSize):
        ax = fig.add_subplot(plotRows, plotCols, idx+1, xticks=[], yticks=[])
        plt.imshow(np.transpose(images[idx], (1, 2, 0)))
        ax.set_title(artCategories[labels[idx]])

    plt.show()


############################### Stage 3: Building the Model #####################################

def freeze_training(model, model_name):
    """ Freeze training for all "features" layers"""
    if model_name == 'resnet50': #ResNet50 model
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.features.parameters():
            param.requires_grad = False

def set_outputs(model, model_name):
    global artCategories
    n_inputs = len(artCategories)
    
    if model_name == 'resnet50': #ResNet50 model
        model.fc = nn.linear(1024,n_inputs)
    else: #AlexNet, Vgg19
        #model.classifier[last_layer] = nn.Linear(1024,n_inputs)
        model.classifier[6] = nn.Linear(4096,n_inputs)

def check_output_features(model, model_name):
    """ Check to see that your last layer produces the expected number of outputs"""
    out_feat = model.fc.out_features if model_name=='resnet50' else model.classifier[6].out_features
    print(f"In this {model_name} model there are {out_feat} outputs")


    
def initialize_model(model_name, progress = False):
    """Returns model"""

    print(f"About to load model {model_name}")
    # Load the pretrained models from pytorch
    model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True, progress=progress)
    print("Model Loaded")

    freeze_training(model, model_name)
    set_outputs(model, model_name)

    # if GPU is available, move the model to GPU
    if device.train_on_gpu():
        model=model.to(device)

    # Print the new model architecture
    print(model)
    check_output_features(model, model_name)
    return model

    

def set_path_to_model(model_name):
    """set the saved model filename"""
    global pathToModel
    model_filename = 'trained_' + model_name + '.pt'
    pathToModel = pathlib.Path.cwd().joinpath('models', model_filename)
    print('File name for saved model: ', pathToModel)


def make_model(model_name, load_from_file=False):
    global theModel
    theModel = initialize_model(model_name)
    set_path_to_model(model_name)
    if load_from_file:
        theModel.load_state_dict(torch.load(pathToModel))



############################## TRAINING ##################################

   

def specify_loss_function_and_optimizer(model_name):
    global criterion, optimizer
    # specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()

    # specifying optimizer and learning rate
    if model_name == 'resnet50':
        optimizer = optim.Adam(theModel.fc.parameters(), lr=0.001)
    else:
        optimizer = optim.Adam(theModel.classifier.parameters(), lr=0.001)


def train_and_validate():
    global n_epochs
    global pathToModel
    global train_dataloader, test_dataloader, valid_dataloader
    global train_losses, valid_losses, accuracies
    global training_loss, validation_loss, accuracy, valid_loss_min
    
    a = time.time()  #Start-time for training

    for epoch in range(1, n_epochs+1):
        c = time.time()  #Start-time for epoch

        ###############
        # TRAIN MODEL #
        ###############
        
        for batch_i, (images, labels) in enumerate(train_dataloader):  #Getting one batch of images and their corresponding true labels

            # move tensors to GPU if CUDA is available
            if device.train_on_gpu():
                images = images.to(device.getDevice())
                labels = labels.to(device.getDevice())
            
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = theModel.forward(images)
            
            # calculate the batch loss
            loss = criterion(outputs, labels)
            
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # clear the previous/buffer gradients of all optimized variables
            optimizer.zero_grad()

            # perform a single optimization step (parameter update)
            optimizer.step()
            
            # update training loss 
            training_loss += loss.item()   
            
        ##################
        # VALIDATE MODEL #
        ##################
         
    # validation loss and accuracy
        validation_loss = 0.0
        accuracy = 0.0
        
        theModel.eval() #model is put to evalution mode i.e. dropout is switched off

        with torch.no_grad():  #Turning off calculation of gradients (not required for validaiton)  {saves time}
            for images, labels in valid_dataloader:   #Getting one batch of validation images
                
                if device.train_on_gpu():   #moving data to GPU if available
                    images = images.to(device.getDevice())
                    labels = labels.to(device.getDevice())

                outputs = theModel.forward(images)

                # calculate the batch loss
                batch_loss = criterion(outputs, labels)
                validation_loss += batch_loss.item()
                
                # Calculating accuracy
                ps = torch.exp(outputs)

                #getting top one probablilty and its corresponding class for batch of images
                top_p, top_class = ps.topk(1, dim=1) 

                #Comparing our predictions to true labels
                equals = top_class == labels.view(*top_class.shape)   #equals is a list of values
                
                #incrementing values of 'accuracy' with equals
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                #taking average of equals will give number of true-predictions
                #equals if of ByteTensor (boolean), changing it to FloatTensor for taking mean...
        
        
        train_losses.append(training_loss/len(train_dataloader))    
        valid_losses.append(validation_loss/len(valid_dataloader))
        accuracies.append(((accuracy/len(valid_dataloader))*100.0))
        d = time.time() #end-time for epoch
        
        print(f"Epoch {epoch} "
              f"Time: {int((d-c)/60)} min {int(d-c)%60} sec "
              f"Train loss: {training_loss/len(train_dataloader):.3f}.. "
              f"Validation loss: {validation_loss/len(valid_dataloader):.3f}.. "
              f"Validation accuracy: {((accuracy/len(valid_dataloader))*100.0):.3f}% "
              )
        
        training_loss = 0.0
        
        # save model if validation loss has decreased
        if ( validation_loss/len(valid_dataloader) <= valid_loss_min):
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min ,
                                                                                            validation_loss/len(valid_dataloader)))
            torch.save(theModel.state_dict(), pathToModel) #Saving model 
            valid_loss_min = validation_loss/len(valid_dataloader)   #Minimum validation loss updated
        
        
        #After validation, model is put to training mode i.e. dropout is again switched on
        theModel.train()
              
        ################
        # END OF EPOCH #
        ################
                
    b = time.time()  #end-time for training
    print('\n\n\tTotal training time: ' ,
          int((b-a)/(60*60)), "hour(s) " ,
          int(((b-a)%(60*60))/60),"minute(s) ",
          int(((b-a)%(60*60))%60) , "second(s)")


################ Testing ########################

def test_model():
    global test_dataloader
    global classes_accuracies
    
    test_loss = 0.0
    counter = 0
    class_correct = list(0. for i in range(len(artCategories)))
    class_total = list(0. for i in range(len(artCategories)))
    classes_accuracies=[]

    # evaluation mode (switching off dropout)
    theModel.eval()
    y_true = []
    y_pred = []

    # iterate over test data - get one batch of data from testloader
    for images, labels in test_dataloader:
        
        # move tensors to GPU if CUDA is available
        if device.train_on_gpu():
            images = images.to(device.getDevice())
            labels = labels.to(device.getDevice())
        
        # compute predicted outputs by passing inputs to the model
        output = theModel(images)
        
        # calculate the batch loss
        loss = criterion(output, labels)
        
        # update test loss 
        test_loss += loss.item()*images.size(0)
        
        # convert output probabilities to predicted class
        ps, pred = torch.max(output, 1)    
        
        # compare model's predictions to true labels
        for i in range(len(images)):
            y_true.append( artCategories[labels[i]] )
            y_pred.append( artCategories[pred[i]] )

            class_total[labels[i]] += 1
            if pred[i] == labels[i]:
                class_correct[pred[i]] += 1
        counter += 1

    # calculate avg test loss
    test_loss = test_loss/len(test_dataloader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(len(artCategories)):
        classes_accuracies.append(100 * class_correct[i] / class_total[i])
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                artCategories[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (artCategories[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % 
                              (100. * np.sum(class_correct) / np.sum(class_total),np.sum(class_correct), np.sum(class_total)))




def save_checkpoint():
    # Save checkpoint
    checkpoint = {'training_losses': train_losses,
              'valid_losses': valid_losses,
              'accuracies': accuracies,
              'classes_accuracies': classes_accuracies,
              'state_dict': theModel.state_dict()}

    torch.save(checkpoint, pathToModel)




############### Build and run model ###################

def runit(model_name = 'alexnet', train=True):
    """if train=Flase, then model will just be tested, not trained"""
    if model_name not in ['alexnet', 'resnet50', 'vgg19']:
        print(f"Model {model_name} not known")
        return None

    global_summary()
    define_and_apply_transforms()
    print_dataset_stats()
    generate_dataloaders()
    #data_visualizer() #optional task
    
    make_model(model_name, train) # model is now in global variable theModel
    specify_loss_function_and_optimizer(model_name)
    if train: # only do the training if specifically required
        train_and_validate()

    test_model()
    save_checkpoint()
    
    

