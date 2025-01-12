# Meachine-Learning-Cheatsheet-Codes

## Setup GPU Torch
```
import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

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

## Plot
```
def show_plots(num_epochs, data, metric):
  e = np.arange(num_epochs)
  plt.plot(e, data['train'], label='train '+metric)
  plt.plot(e, data['valid'], label='validation '+metric)
  plt.xlabel('epoch')
  plt.ylabel(metric)
  plt.legend()
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

```
fig,axs = plt.subplots(2,5, figsize=(15,7))
idx=0

for i, ax in enumerate(axs.flatten()):
  sample = train_dataset[indicies[idx]]
  ax.imshow(vis)
  ax.axis("off")
  ax.title.set_text(label[sample[1]])
  idx += 1

plt.tight_layout()
plt.show()
```
## Custom Pytorch Dataset
```
class Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.dataframe = df
        self.trasnform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.dataframe[['image_path', 'label']].iloc[index]


        # patch extraction
        image = cv2.imread(image_path)
        image = self.trasnform(image)

        return image, label
```

## train one epoch
```
def one_epoch(model, loader, criterion, optimizer, scheduler, device, phase):

  if phase == 'train':
    model.train() # Set model to training mode
  else:
    model.eval()

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

    # if phase == 'train' and scheduler:
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

  best_val_loss = float('inf')
  best_val_acc = 0
  # best_model = None

  accuracy_dic, loss_dic = {}, {}
  loss_dic['train'], loss_dic['validation'] = [], []
  accuracy_dic['train'], accuracy_dic['validation'] = [], []

  for epoch in range(num_epochs):
      train_loss, train_acc, train_precision, train_recall, train_f1 = one_epoch(model, loaders['train'], criterion, optimizer, scheduler, device, phase='train' )
      val_loss, val_acc, val_precision, val_recall, val_f1 = one_epoch(model, loaders['validation'], criterion, optimizer, scheduler, device, phase='validation')

      loss_dic['train'].append(train_loss)
      loss_dic['validation'].append(val_loss)
      accuracy_dic['train'].append(train_acc)
      accuracy_dic['validation'].append(val_acc)

      wandb.log({"Train Loss": train_loss, "Train Accuracy":train_acc, "Validation Loss": val_loss, "Validation Accuracy":val_acc})

      if val_loss < best_val_loss:
        best_val_acc = val_acc
        best_val_loss = val_loss
        # best_model = model.state_dict()
        torch.save(model.state_dict(), model_path)

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

## Save and Load Model
```
torch.save(model.state_dict(), model_path)
```
```
model.load_state_dict(torch.load(model_path))
```

## Confusion Matrix
```
def plot_confusionmatrix(y_pred, y_true, classes):
  print('Confusion matrix')
  cf = sklearn.metrics.confusion_matrix(y_pred, y_true, labels=np.arange(len(classes)))
  sns.heatmap(cf,annot=True,yticklabels=classes, xticklabels=classes, cmap='Blues', fmt='g')
  plt.tight_layout()
  plt.show()
```
```
classes = ['not cancerous', 'cancerous']

def report(model, loader, device, classes):

  # Each epoch has a training and validation phase
  model.eval()   # Set model to evaluate mode

  y_pred = []
  y_true = []

  # Iterate over data.
  for inputs, labels in loader:
    inputs = inputs.to(device)
    labels = labels.type(torch.LongTensor)

    # forward
    # track history if only in train
    with torch.set_grad_enabled(False):
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      y_pred.extend(preds.cpu())
      y_true.extend(labels)

  plot_confusionmatrix(y_pred, y_true, classes)
```

## Interpretability (GradCam)
```
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms.functional import to_pil_image


def plot_GradCam(model, image):
    target_layers = [model.layer4[-1]]
    input_tensor = torch.unsqueeze(image, dim=0)
    # Note: input_tensor can be a batch tensor with several images!

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers)

    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
    #   ...

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.

    targets = [ClassifierOutputTarget(1)] # one class

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    img = np.array(to_pil_image(image))/255

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    return grayscale_cam, visualization
```
```
k=15
indicies = np.random.choice(len(datasets['test']), k)
np.random.shuffle(indicies)

fig,axs = plt.subplots(k//5,5, figsize=(18,10))
idx=0

for i, ax in enumerate(axs.flatten()):
  image, label = datasets['train'][indicies[idx]]
  grayscale_cam, vis = plot_GradCam(model, image)
  ax.imshow(vis)
  ax.axis("off")
  ax.title.set_text(classes[label])
  idx += 1

plt.tight_layout()
plt.show()
```


## Add System Path

```
import sys
sys.path.append('/content/PATH/')
```
