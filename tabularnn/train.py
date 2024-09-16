import os
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score

##### train #####
def train_model(model, dataloader, num_epochs=10, criterion=None, optimizer=None, lr=0.001):
    if criterion is None: criterion = nn.CrossEntropyLoss()  # multiclass cross entropy loss
    if optimizer is None: optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer


    epoches_loss = []
    epoches_f1 = []
    epoches_auc = []
    epoches_acc = []
    epoches_precision = []

    ##### Training loop #####
    for epoch in range(num_epochs):
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        for batch_X, batch_y in dataloader:
            
            optimizer.zero_grad() # clear gradients from last batch

            outputs = model(batch_X) # predicted outputs

            # compute loss
            loss = criterion(outputs, batch_y)
            running_loss += loss.item()


            # back propagation
            loss.backward() # compute loss gradient wrt parameters      
            optimizer.step() # update parameters

            _, preds = torch.max(outputs, 1)  # class with highest logit

            
            # store predictions for metric computations
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())  # softmax to convert to probas (detach because tensor requires gradient)


    

        # compute performance metrics
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')  # 'weighted' handles class imbalance
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_precision = precision_score(all_labels, all_preds, average='weighted')

        
        if len(all_probs[0]) == 2:  # if binary
            epoch_auc = roc_auc_score(all_labels, [prob[1] for prob in all_probs])  # only the positive class probabilities
        else:  # if multi-class do one-v-rest
            epoch_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}, F1 Score: {epoch_f1:.3f}, Accuracy: {epoch_acc:.3f}, Precision: {epoch_precision:.3f}, AUROC: {epoch_auc:.3f}')
        
        # Store metrics for potential further use
        epoches_loss.append(running_loss / len(dataloader))
        epoches_f1.append(epoch_f1)
        epoches_acc.append(epoch_acc)
        epoches_precision.append(epoch_precision)
        epoches_auc.append(epoch_auc)

    return model, epoches_loss, epoches_f1, epoches_acc, epoches_precision, epoches_auc



##### evaluate #####
def evaluate(model, dataloader, criterion):
    model.eval()  # model in evaluation mode
    all_preds = []
    all_labels = []
    all_probs = []  # To store predicted probabilities
    running_loss = 0.0

    with torch.no_grad():  # no gradient tracking for inference
        for batch_X, batch_y in dataloader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            running_loss += loss.item()

        
            _, preds = torch.max(outputs, 1)  # class with highest logit
            probs = torch.softmax(outputs, dim=1) # softmax to convert to probas 

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # calculate metrics
    average_loss = running_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')

    if len(all_probs[0]) == 2:  # if binary
        auroc = roc_auc_score(all_labels, [prob[1] for prob in all_probs])  # only the positive class probabilities
    else:  # if multi-class do one-v-rest
        auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

    print(f'Loss: {average_loss:0.4f}, f1: {f1:0.3f}, accuracy: {accuracy:0.4f}, percision: {precision:0.4f}, auroc: {auroc:0.4f}')
    return average_loss, f1, accuracy, precision, auroc


##### plots #####
def plot_metrics(epoches_loss, epoches_f1, epoches_acc, epoches_precision, epoches_auc):
    epochs = range(1, len(epoches_loss) + 1)
    
    plt.figure(figsize=(12, 10))
    
    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, epoches_loss, label='Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss by Epoch')
    plt.legend()
    
    # F1 Score
    plt.subplot(2, 2, 2)
    plt.plot(epochs, epoches_f1, label='F1 Score', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score by Epoch')
    plt.legend()
    
    # Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(epochs, epoches_acc, label='Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Epoch')
    plt.legend()
    
    # Precision
    plt.subplot(2, 2, 4)
    plt.plot(epochs, epoches_precision, label='Precision', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Precision by Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()