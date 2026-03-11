from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

data = fetch_openml('mnist_784',version=1,as_frame=False)

x = data.data
y= data.target

fig,ax = plt.subplots(nrows=2,ncols=5,figsize=(10,5))

axes = ax.flatten()

for i in range(10):
    image = x[i].reshape(28,28)
    axes[i].imshow(image,cmap='gray')

    axes[i].set_title(f"Label : {y[i]}")
    print("Label : ",y[i])
    axes[i].axis('off')
plt.tight_layout()
plt.show()
