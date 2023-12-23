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
