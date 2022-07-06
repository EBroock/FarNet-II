"""
Display script:

You have to select a batch, a sequence and an index, 
and the associated mask and output will be displayed.
"""

# Imports
import matplotlib.pyplot as plt
import os 
import h5py
import numpy as np


# Selection of item
x = input('Index for file selection (0 or 1) :')
while int(x) not in range(2):
    print('Index needs to be 0 or 1')
    x = input('Index for file selection (0 or 1) :')
y = input('Index for sequence selection (0 to 9) :')
while int(y) not in range(10):
    print('Index needs to be on the 0 to 9 range')
    y = input('Index for sequence selection (0 to 9) :')
z = input('Index for element selection (0 to 10) :')
while int(z) not in range(11):
    print('Index needs to be on the 0 to 10 range')
    z = input('Index for element selection (0 to 10) :')

# Outputs list
list_outputs = []
for file in sorted(os.listdir('./outputs/')):
    if file.endswith('.h5'):
        list_outputs.append(file)

# Masks list
list_masks = []
for file in sorted(os.listdir('./masks/')):
    if file.endswith('.npy'):
        list_masks.append(file)

# Output and mask read
output = h5py.File('./outputs/'+list_outputs[int(x)],'r')['magnetogram'][int(y),int(z),:,:]
mask = np.load('./masks/'+list_masks[int(x)])[int(y),int(z),:,:]

# Figure
fig = plt.figure(1,figsize=(12,5))
plt.subplot(121)
plt.suptitle('output VS mask Batch '+str(x)+' Sequence '+str(y)+' Index '+str(z))
plt.title('output')
plt.xlabel('long (deg)')
plt.ylabel('lat (deg)')
plt.yticks([0,72,143],[-72,0,71])
plt.imshow(output,origin='lower')
plt.colorbar(fraction=0.0365)
plt.subplot(122)
plt.title('mask')
plt.xlabel('long (deg)')
plt.yticks([0,72,143],[-72,0,71])
plt.imshow(mask,origin='lower')
plt.colorbar(fraction=0.0365)
fig.tight_layout()
plt.savefig('output_mask_'+str(x)+'_'+str(y)+'_'+str(z)+'.pdf')

plt.show()

