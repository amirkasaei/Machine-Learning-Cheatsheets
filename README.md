# Meachine-Learning-Cheatsheet-Codes

## Mount Google Drive
```
from google.colab import drive
drive.mount('/content/drive')
```

## Read file from paths
```
for d in os.listdir(path):
  pass
```

## Multi-line Visualization
```
fig,axs = plt.subplots(2,3, figsize=(20,5))

axs[0][0].title.set_text("image1")
axs[0][0].imshow(img)

axs[1][1].title.set_text("image2")
axs[1][1].imshow(mask, cmap='gray')

fig.tight_layout()
```


## train one epoch
```
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, phase):

  model.train()  # Set model to training mode

  running_loss = 0.0
  running_accuracy = 0.0
  running_precision = 0.0
  running_recall = 0.0
  running_f1_score = 0.0

  # Iterate over data.
  for inputs, labels in loader:
    inputs = inputs.to(device)
    labels = labels.type(torch.LongTensor).to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward
    with torch.set_grad_enabled(phase == 'train'):
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      loss = criterion(outputs, labels)

      if phase == 'train':
        loss.backward()
        optimizer.step()

    # statistics
    running_loss += loss.item()
    running_accuracy += sklearn.metrics.accuracy_score(labels.cpu(), preds.cpu())
    running_precision += sklearn.metrics.precision_score(labels.cpu(), preds.cpu(), average='weighted', zero_division=0)
    running_recall += sklearn.metrics.recall_score(labels.cpu(), preds.cpu(), average='weighted', zero_division=0)
    running_f1_score += sklearn.metrics.f1_score(labels.cpu(), preds.cpu(), average='weighted', zero_division=0)

    # if phase == 'train' and not scheduler:
    #     scheduler.step()

  loss = running_loss / len(loader)
  accuracy = running_accuracy / len(loader)
  precision = running_precision / len(loader)
  recall = running_recall / len(loader)
  f1_score = running_f1_score / len(loader)

  return loss, accuracy, precision, recall, f1_score
```

## train
```
def train(model, loaders, criterion, optimizer, num_epochs, device, scheduler=None):

  accuracy_dic, loss_dic = {}, {}
  loss_dic['train'], loss_dic['validation'] = [], []
  accuracy_dic['train'], accuracy_dic['validation'] = [], []

  for epoch in range(num_epochs):
      train_loss, train_acc, train_precision, train_recall, train_f1 = train_one_epoch(model, loaders['train'], criterion, optimizer, scheduler, device, phase='train' )
      val_loss, val_acc, val_precision, val_recall, val_f1 = one_epoch(model, loaders['validation'], criterion, optimizer, scheduler, device, phase='validation')

      loss_dic['train'].append(train_loss)
      loss_dic['validation'].append(val_loss)
      accuracy_dic['train'].append(train_acc)
      accuracy_dic['validation'].append(val_acc)

      wandb.log({"Train Loss": train_loss, "Train Accuracy":train_acc, "Validation Loss": val_loss, "Validation Accuracy":val_acc})

      print(f'Epoch [{epoch+1}/{num_epochs}] - '
            f'Train Loss: {train_loss:.4f} - '
            f'Train Accuracy: {train_acc:.4f} - '
            f'Validation Loss: {val_loss:.4f} - '
            f'Validation Accuracy {val_acc:.4f}% - '
            f'Validation Precision {val_precision:.4f}% - '
            f'Validation Recall {val_recall:.4f}% - '
            f'Validation F1-score {val_f1:.4f}% ')

  return loss_dic, accuracy_dic
```
