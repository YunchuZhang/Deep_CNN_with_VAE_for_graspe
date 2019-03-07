from __future__ import print_function
import torch
import torchvision
import matplotlib.pyplot as plt



# get a viusalization 
imgfile = "pcd0100r"
img = plt.imread("{}.jpg".format(imgfile))
box = annotation[imgfile]
print box
# Create figure and axes
fig,ax = plt.subplots(1)
ax.imshow(img)
for eachbox in box:
    cx, cy, w, h = eachbox 

    # Create a Rectangle patch
    rect = patches.Rectangle((cx-w/2,cy-h/2),w,h,linewidth=1,edgecolor='r',facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

plt.show()

print "The box is {}".format(box)
print "image size is {}".format(img.shape)
