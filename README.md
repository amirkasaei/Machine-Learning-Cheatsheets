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
fig,axs = plt.subplots(2,3, figsize=(20,50))

img = io.imread(mri_df.image_path[i])
axs[count][0].title.set_text("Brain MRI")
axs[count][0].imshow(img)

mask = io.imread(mri_df.mask_path[i])
axs[count][1].title.set_text("Mask")
axs[count][1].imshow(mask, cmap='gray')

img[mask==255] = (0,255,150)
axs[count][2].title.set_text("MRI with Mask")
axs[count][2].imshow(img)

fig.tight_layout()
```
