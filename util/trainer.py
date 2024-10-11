import torch
from tqdm.auto import tqdm
from util.common import *

# Helper function to train the classification networks
def train_model(model, device, training_loader, test_loader, epochs,
                 loss_fn, opt, scheduler, gradient_clipping=False):

    # History to show loss over time 
    history = []
    epoch_loss = float("inf")
    epoch_size = len(training_loader)

    model.to(device)
    
    # Training loop over epochs
    epoch_loop = tqdm(range(epochs), desc="Total progress", position=0)
    epoch_loop.set_postfix(lr = opt.param_groups[0]['lr'], loss = epoch_loss, accuracy = '0%', test_accuracy = '0%')
    for i in epoch_loop:
        
        # Training
        model.train()

        correct_pred, num_examples = 0, 0

        # Training loop over each step in an epoch
        for j, (inputs, labels) in enumerate(training_loader):
            
            # Zero the parameter gradients
            opt.zero_grad()

            # Transfer Images and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Training loss
            train_loss = loss_fn(outputs, labels)
            history.append(train_loss.item())

            # Training Accuarcy
            _, predicted_labels = torch.max(outputs, 1)
            num_examples += inputs.size(0)
            correct_pred += (predicted_labels == labels).sum()
            
            # Backward pass
            train_loss.backward()
            
            # Gradient clipping if enabled
            if gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                
            opt.step()
            
        scheduler.step()

        # Evaluation
        model.eval()
        epoch_loss = sum(history[-epoch_size:]) / epoch_size
        epoch_loop.set_postfix(lr = opt.param_groups[0]['lr'],
                             loss = epoch_loss, training_accuracy = f'{100*correct_pred.float()/num_examples:.3f}%',
                             test_accuracy = f'{compute_accuracy(model, test_loader, device):.3f}%')
    return history

# Helper function to train using knowledge distillation
def distillation_training(teacher, student, device, training_loader, test_loader, epochs,
                 loss_fn, opt, scheduler, temperature):

    # History to show loss over time 
    history = []
    epoch_loss = float("inf")
    epoch_size = len(training_loader)

    # Transfering models to device
    student.to(device)
    teacher.to(device)
    teacher.eval()
    
    # Training loop over epochs
    epoch_loop = tqdm(range(epochs), desc="Total progress", position=0)
    epoch_loop.set_postfix(lr = opt.param_groups[0]['lr'], loss = epoch_loss, accuracy = '0%', test_accuracy = '0%')
    for i in epoch_loop:
        
        # Training
        student.train()

        correct_pred, num_examples = 0, 0

        # Training loop over each step in an epoch
        for j, (inputs, labels) in enumerate(training_loader):
            
            # Zero the parameter gradients
            opt.zero_grad()

            # Transfer Images and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = student(inputs)
            
            with torch.no_grad():
                pred = (teacher(inputs)/temperature).softmax(dim=1)
            
            # Training loss
            train_loss = loss_fn(outputs, labels, pred)
            history.append(train_loss.item())

            # Training Accuarcy
            _, predicted_labels = torch.max(outputs, 1)
            num_examples += inputs.size(0)
            correct_pred += (predicted_labels == labels).sum()
            
            # Backward pass
            train_loss.backward()
                
            opt.step()
            
        scheduler.step()

        # Evaluation
        student.eval()
        epoch_loss = sum(history[-epoch_size:]) / epoch_size
        epoch_loop.set_postfix(lr = opt.param_groups[0]['lr'],
                             loss = epoch_loss, training_accuracy = f'{100*correct_pred.float()/num_examples:.3f}%',
                             test_accuracy = f'{compute_accuracy(student, test_loader, device):.3f}%')
    return history