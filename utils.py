  
import torch
from sklearn import metrics
import numpy as np

def _train_model(model, train_loader, epoch, num_epochs, batch_size, optimizer, criterion, writer, current_lr, log_every=100):
    
    print("Now Training...")
    # Set to train mode
    model.train()

    preds = []
    gts = []
    losses = []

    confusion_matrix = torch.zeros([2,2])

    for i, (images, label) in enumerate(train_loader):
        optimizer.zero_grad()

        if torch.cuda.is_available():
            images = images.cuda()
            label = label.cuda()

        output = model(images)

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.softmax(output,dim=1)

        for i in range(len(output)):
            gts.append(int(label[i, 1].item()))
            preds.append(probas[i, 1].item())

            gg = torch.argmax(label[i]).item()
            pp = torch.argmax(probas[i]).item()

            confusion_matrix[pp, gg] += 1


        try:
            auc = metrics.roc_auc_score(gts, preds)
        except:
            auc = 0.5

        writer.add_scalar('Train/Loss', loss_value,
                          epoch * len(train_loader) + i)

        if (i % log_every == 0):
            precision = confusion_matrix[1,1].item() / (confusion_matrix[1,0] + confusion_matrix[1,1] + 1e-8).item()
            recall = confusion_matrix[1,1].item() / (confusion_matrix[0,1] + confusion_matrix[1,1] + 1e-8).item()
            train_f1 = (precision*recall) / (precision + recall + 1e-8)
            print("[Epoch: {0} / {1} | Batch : {2} / {3} ]| Batch Loss : {4} | Train AUC : {5} | Train Accuracy : {6} | F1 Score : {7} | lr : {8}".
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(train_loader),
                      np.round(loss_value, 4),
                      np.round(auc, 4),
                      np.round((confusion_matrix[0,0].item() + confusion_matrix[1,1].item()) / confusion_matrix.sum().item(),4),
                      np.round(train_f1,4),
                      current_lr
                  )
                  )

    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(auc, 4)
    train_accuracy = (confusion_matrix[0,0].item() + confusion_matrix[1,1].item()) / confusion_matrix.sum().item()

    precision = confusion_matrix[1,1].item() / (confusion_matrix[1,0] + confusion_matrix[1,1] + 1e-8).item()
    recall = confusion_matrix[1,1].item() / (confusion_matrix[0,1] + confusion_matrix[1,1] + 1e-8).item()
    train_f1 = (precision*recall) / (precision + recall + 1e-8)

    writer.add_scalar('Train/AUC', auc, epoch)
    writer.add_scalar('Train/Loss_Epoch', train_loss_epoch, epoch)
    writer.add_scalar('Train/Accuracy', train_accuracy, epoch)
    writer.add_scalar('Train/F1_Score', train_f1, epoch)

    return confusion_matrix, train_loss_epoch, train_auc_epoch

def _eval_model(model, train_loader, epoch, num_epochs, batch_size, optimizer, criterion, writer, log_every=10):
    
    print("Now Validating...")
    # Set to eval mode
    model.eval()

    preds = []
    gts = []
    losses = []

    confusion_matrix = torch.zeros([2,2])

    for i, (images, label) in enumerate(train_loader):
        optimizer.zero_grad()

        if torch.cuda.is_available():
            images = images.cuda()
            label = label.cuda()

        with torch.no_grad():
            output = model(images)
            loss = criterion(output, label)

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.softmax(output,dim=1)

        for i in range(len(output)):
            gts.append(int(label[i, 1].item()))
            preds.append(probas[i, 1].item())

            gg = torch.argmax(label[i]).item()
            pp = torch.argmax(probas[i]).item()

            confusion_matrix[pp, gg] += 1


        try:
            auc = metrics.roc_auc_score(gts, preds)
        except:
            auc = 0.5

        writer.add_scalar('Val/Loss', loss_value,
                          epoch * len(train_loader) + i)

        if (i % log_every == 0):
            precision = confusion_matrix[1,1].item() / (confusion_matrix[1,0] + confusion_matrix[1,1] + 1e-8).item()
            recall = confusion_matrix[1,1].item() / (confusion_matrix[0,1] + confusion_matrix[1,1] + 1e-8).item()
            train_f1 = (precision*recall) / (precision + recall + 1e-8)
            print("[Epoch: {0} / {1} | Batch : {2} / {3} ]| Batch Loss : {4} | Val AUC : {5} | Val Accuracy : {6} | F1 Score : {7}".
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(train_loader),
                      np.round(loss_value, 4),
                      np.round(auc, 4),
                      np.round((confusion_matrix[0,0].item() + confusion_matrix[1,1].item()) / confusion_matrix.sum().item(),4),
                      np.round(train_f1,4),
                  )
                  )

    val_loss_epoch = np.round(np.mean(losses), 4)
    val_auc_epoch = np.round(auc, 4)
    val_accuracy = (confusion_matrix[0,0].item() + confusion_matrix[1,1].item()) / confusion_matrix.sum().item()

    precision = confusion_matrix[1,1].item() / (confusion_matrix[1,0] + confusion_matrix[1,1] + 1e-8).item()
    recall = confusion_matrix[1,1].item() / (confusion_matrix[0,1] + confusion_matrix[1,1] + 1e-8).item()
    val_f1 = (precision*recall) / (precision + recall + 1e-8)

    writer.add_scalar('Val/AUC', auc, epoch)
    writer.add_scalar('Val/Loss_Epoch', val_loss_epoch, epoch)
    writer.add_scalar('Val/Accuracy', val_accuracy, epoch)
    writer.add_scalar('Val/F1_Score', val_f1, epoch)

    return confusion_matrix, val_loss_epoch, val_auc_epoch

def _get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']