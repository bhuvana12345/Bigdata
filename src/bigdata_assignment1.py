#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib


# In[2]:


lime_color = cv2.imread("/Users/bhuvanakorrapati/Desktop/data/lime.jpg")

orange_color = cv2.imread("/Users/bhuvanakorrapati/Desktop/data/orange.jpg")

apple_color = cv2.imread("/Users/bhuvanakorrapati/Desktop/data/apple.jpg")


# In[3]:


plt.imshow(lime_color[:,:,0])
plt.imshow(lime_color[:,:,1])
plt.imshow(lime_color[:,:,2])


# In[4]:


plt.imshow(orange_color[:,:,0])
plt.imshow(orange_color[:,:,1])
plt.imshow(orange_color[:,:,2])


# In[5]:


plt.imshow(apple_color[:,:,0])
plt.imshow(apple_color[:,:,1])
plt.imshow(apple_color[:,:,2])


# In[6]:


limeG = cv2.cvtColor(lime_color, cv2.COLOR_BGR2GRAY)
heightcG, widthcG = limeG.shape
print(limeG)


# In[7]:


orangeG = cv2.cvtColor(orange_color, cv2.COLOR_BGR2GRAY)
heightcG, widthcG = orangeG.shape
print(orangeG)


# In[8]:


appleG = cv2.cvtColor(apple_color, cv2.COLOR_BGR2GRAY)
heightcG, widthcG = appleG.shape
print(appleG)


# In[9]:


# Resize function.
def resize(width_val,height_val,path):
    img = cv2.imread(path)
    print(img.shape)
    width,height=width_val,height_val
    imgresize = cv2.resize(img,(width.height))
    print(imgresize.shape)
    plt.imshow(imgresize)
    


# ### Resizing and normalizing the images

# In[10]:


lime = cv2.resize(limeG, dsize=(192, 264), interpolation=cv2.INTER_CUBIC)
lime = cv2.normalize(lime.astype('float'), None, 0.0, 1.0, 
cv2.NORM_MINMAX)*255
heightl, widthl = lime.shape


# In[11]:


orange = cv2.resize(orangeG, dsize=(192, 264), interpolation=cv2.INTER_CUBIC)
orange = cv2.normalize(orange.astype('float'), None, 0.0, 1.0, 
cv2.NORM_MINMAX)*255
heighto, widtho = orange.shape


# In[12]:


apple = cv2.resize(appleG, dsize=(192, 264), interpolation=cv2.INTER_CUBIC)
apple = cv2.normalize(apple.astype('float'), None, 0.0, 1.0, 
cv2.NORM_MINMAX)*255
heighta, widtha = apple.shape


# ### Grayscale images

# In[13]:


plt.imshow(lime, cmap=plt.get_cmap('gray'))
plt.axis('off')


# In[14]:


plt.imshow(orange, cmap=plt.get_cmap('gray'))
plt.axis('off')


# In[15]:


plt.imshow(apple, cmap=plt.get_cmap('gray'))
plt.axis('off')


# In[16]:


cv2.imwrite('/Users/bhuvanakorrapati/Desktop/output/lime1.png', lime)
cv2.imwrite('/Users/bhuvanakorrapati/Desktop/output/orange1.png', orange)
cv2.imwrite('/Users/bhuvanakorrapati/Desktop/output/apple1.png', apple)


# ### Binarize the images

# In[17]:


tmpl = np.zeros((heightl, widthl), np.uint8)
th1 = lime.mean()
for i in range(heightl):
    for j in range(widthl):
        if(lime[i][j]<th1):
            tmpl[i][j] = 0
        else:
            tmpl[i][j] = 255
plt.imshow(tmpl, cmap=plt.get_cmap('gray'))


# In[18]:


tmpo = np.zeros((heighto, widtho), np.uint8)
th1 = orange.mean()
for i in range(heighto):
    for j in range(widtho):
        if(orange[i][j]<th1):
            tmpo[i][j] = 0
        else:
            tmpo[i][j] = 255
plt.imshow(tmpo, cmap=plt.get_cmap('gray'))


# In[19]:


tmpa = np.zeros((heighta, widtha), np.uint8)
th1 = apple.mean()
for i in range(heighta):
    for j in range(widtha):
        if(apple[i][j]<th1):
            tmpa[i][j] = 0
        else:
            tmpa[i][j] = 255
plt.imshow(tmpa, cmap=plt.get_cmap('gray'))


# ### Binarizing the image, read the image in zeros and ones and print the data file.

# In[20]:


fspacebinarya = pd.DataFrame(tmpa)  #panda object
fspacebinarya.to_csv('/Users/bhuvanakorrapati/Desktop/output/fspacebinarya.csv', index=False)


# In[21]:


cv2.imwrite('/Users/bhuvanakorrapati/Desktop/output/lime2.png', tmpl)
cv2.imwrite('/Users/bhuvanakorrapati/Desktop/output/orange2.png', tmpo)
cv2.imwrite('/Users/bhuvanakorrapati/Desktop/output/apple2.png', tmpa)


# ### Create feature space of 12X12 matrix and labels - 0 for lime, 1 for orange and 2 for apple

# In[22]:


lim = round(((heightl)*(widthl))/144)
flatlim = np.zeros((lim, 145), np.uint8)
k = 0
for i in range(0,heightl,12):
    for j in range(0,widthl,12):
        tmp1 = lime[i:i+12,j:j+12]
        flatlim[k,0:144] = tmp1.flatten()
        k = k + 1
fspaceL = pd.DataFrame(flatlim)  #panda object
fspaceL.to_csv('/Users/bhuvanakorrapati/Desktop/output/fspaceL.csv', index=False)


# In[23]:


print(fspaceL.shape)


# In[24]:


oran = round(((heighto)*(widtho))/144)
flatoran = np.ones((oran, 145), np.uint8)
k = 0
for i in range(0,heighto,12):
    for j in range(0,widtho,12):
        tmp2 = orange[i:i+12,j:j+12]
        flatoran[k,0:144] = tmp2.flatten()
        k = k + 1
fspaceO = pd.DataFrame(flatoran)  #panda object
fspaceO.to_csv('/Users/bhuvanakorrapati/Desktop/output/fspaceO.csv', index=False)


# In[25]:


app = round(((heighta)*(widtha))/144)
flatapp = np.full((app, 145), 2, np.uint8)
k = 0
for i in range(0,heighta,12):
    for j in range(0,widtha,12):
        tmp3 = apple[i:i+12,j:j+12]
        flatapp[k,0:144] = tmp3.flatten()
        k = k + 1
fspaceA = pd.DataFrame(flatapp)  #panda object
fspaceA.to_csv('/Users/bhuvanakorrapati/Desktop/output/fspaceA.csv', index=False)


# ### Create feature vectors using sliding window 12X12 matrix and labels - 0 for lime, 1 for orange and 2 for apple
# 

# In[26]:



lime_size = round((heightl-12)*(widthl-12))
sw_flatl = np.zeros((lime_size, 145), np.uint8)
k = 0
for i in range(0,heightl-12):
    for j in range(0,widthl-12):
        sw_crop_tmp1 = lime[i:i+12,j:j+12]
        sw_flatl[k,0:144] = sw_crop_tmp1.flatten()
        k = k + 1
sw_fspaceL = pd.DataFrame(sw_flatl)  #panda object
sw_fspaceL.to_csv('/Users/bhuvanakorrapati/Desktop/output/sw_fspaceL.csv', index=False)


# In[27]:


print(sw_fspaceL.shape)


# In[28]:


orange_size = round((heighto-12)*(widtho-12))
sw_flato = np.ones((orange_size, 145), np.uint8)
k = 0
for i in range(0,heighto-12):
    for j in range(0,widtho-12):
        sw_crop_tmp2 = orange[i:i+12,j:j+12]
        sw_flato[k,0:144] = sw_crop_tmp2.flatten()
        k = k + 1
sw_fspaceO = pd.DataFrame(sw_flato)  #panda object
sw_fspaceO.to_csv('/Users/bhuvanakorrapati/Desktop/output/sw_fspaceO.csv', index=False)


# In[29]:


print(sw_fspaceO.shape)


# In[30]:


apple_size = round((heighta-12)*(widtha-12))
sw_flata = np.full((apple_size, 145), 2, np.uint8)
k = 0
for i in range(0,heighta-12):
    for j in range(0,widtha-12):
        sw_crop_tmp2 = apple[i:i+12,j:j+12]
        sw_flatl[k,0:144] = sw_crop_tmp2.flatten()
        k = k + 1
sw_fspaceA = pd.DataFrame(sw_flatl)  #panda object
sw_fspaceA.to_csv('/Users/bhuvanakorrapati/Desktop/output/sw_fspaceA.csv', index=False)


# In[31]:


print(sw_fspaceA.shape)


# ### Statistical information(no.of observations, dimensions, mean and standard deviation)

# In[32]:


number_observations_fspaceL=fspaceL.shape[0]
number_observations_fspaceO=fspaceO.shape[0]
number_observations_fspaceA=fspaceA.shape[0]


# In[33]:


print(number_observations_fspaceL)
print(number_observations_fspaceO)
print(number_observations_fspaceA)


# In[34]:


dimension_fspaceL=fspaceL.shape[1]-1
print(dimension_fspaceL)


# In[35]:


dimension_fspaceO=fspaceO.shape[1]-1
print(dimension_fspaceO)


# In[36]:


dimension_fspaceA=fspaceA.shape[1]-1
print(dimension_fspaceA)


# In[37]:


mean_fspaceL=fspaceL[fspaceL.columns[0:144]].mean()
sd_fspaceL=fspaceL[fspaceL.columns[0:144]].std()


# In[38]:


print(mean_fspaceL)
print(sd_fspaceL)


# In[39]:


mean_fspaceO=fspaceO[fspaceO.columns[0:144]].mean()
sd_fspaceO=fspaceO[fspaceO.columns[0:144]].std()


# In[40]:


print(mean_fspaceO)
print(sd_fspaceO)


# In[41]:


mean_fspaceA=fspaceA[fspaceA.columns[0:144]].mean()
sd_fspaceA=fspaceA[fspaceA.columns[0:144]].std()


# In[42]:


print(mean_fspaceA.head())
print(sd_fspaceA.head())


# In[43]:


mean_fspaceL.plot(color = 'blue')
mean_fspaceO.plot(color = 'green')
mean_fspaceA.plot(color = 'red')


# In[44]:


sd_fspaceL.plot(color = 'blue')
sd_fspaceO.plot(color = 'green')
sd_fspaceA.plot(color = 'red')


# In[45]:


mean_fspaceL.hist(color = 'blue')
mean_fspaceO.hist(color = 'green')
mean_fspaceA.hist(color = 'red')


# ### SCATER PLOT AND HISTOGRAM FOR FEATURE SPACE VALUES

# In[46]:


s= np.random.randint(144)
t= np.random.randint(144)
fspaceL.plot.scatter(x=s,y=t, c='blue')
fspaceO.plot.scatter(x=s,y=t, c='Green')
fspaceA.plot.scatter(x=s,y=t, c='red')


# In[47]:


random_value=np.random.randint(144)
fspaceL[random_value].hist(bins=144)
fspaceO[random_value].hist(bins=144)
fspaceA[random_value].hist(bins=144)


# ### STATISTICAL INFORMATION, SCATTER PLOT AND HISTOGRAM FOR SLIDING BLOCK FEATURE

# In[48]:


number_observations_sw_fspaceL=sw_fspaceL.shape[0]
number_observations_sw_fspaceO=sw_fspaceO.shape[0]
number_observations_sw_fspaceA=sw_fspaceA.shape[0]


# In[49]:


print(number_observations_sw_fspaceL)
print(number_observations_sw_fspaceO)
print(number_observations_sw_fspaceA)


# In[50]:


mean_sw_fspaceL=sw_fspaceL[sw_fspaceL.columns[0:144]].mean()
mean_sw_fspaceO=sw_fspaceO[sw_fspaceO.columns[0:144]].mean()
mean_sw_fspaceA=sw_fspaceA[sw_fspaceA.columns[0:144]].mean()


# In[51]:


print(mean_sw_fspaceL)
print(mean_sw_fspaceO)
print(mean_sw_fspaceA)


# In[52]:


sd_sw_fspaceL=sw_fspaceL[sw_fspaceL.columns[0:144]].std()
sd_sw_fspaceO=sw_fspaceO[sw_fspaceO.columns[0:144]].std()
sd_sw_fspaceA=sw_fspaceA[sw_fspaceA.columns[0:144]].std()


# In[53]:


print(sd_sw_fspaceL)
print(sd_sw_fspaceO)
print(sd_sw_fspaceA)


# In[54]:


s= np.random.randint(144)
t= np.random.randint(144)
sw_fspaceL.plot.scatter(x=s,y=t, c='Red')
sw_fspaceO.plot.scatter(x=s,y=t, c='Green')
sw_fspaceA.plot.scatter(x=s,y=t, c='Blue')


# In[55]:


random_value=np.random.randint(144)
sw_fspaceL[random_value].hist(bins=144)
sw_fspaceO[random_value].hist(bins=144)
sw_fspaceA[random_value].hist(bins=144)


# ### MERGE FEATURE SPACE OF LIME AND ORANGE.

# In[56]:


frames = [fspaceL,fspaceO]
merged = pd.concat(frames)

M = np.arange(len(merged))
frame_merged = np.random.permutation(M)

frame_merged=merged.sample(frac=1).reset_index(drop=True)
frame_merged.to_csv('/Users/bhuvanakorrapati/Desktop/output/merge_LO.csv', index=False)


# In[57]:


frame_merged.shape


# In[58]:


s= np.random.randint(144)
t= np.random.randint(144)
frame_merged.plot.scatter(x=s,y=t, c='blue')


# In[59]:


random_value=np.random.randint(144)
frame_merged[random_value].hist(bins=144)


# ### MERGE FEATURE SPACE OF ALL THREE FRUITS AND CREATE A CSV FILE.

# In[60]:


frames_3 = [fspaceL,fspaceO,fspaceA]
merged_3 = pd.concat(frames_3)

indx = np.arange(len(merged_3))
frame_merged_3 = np.random.permutation(indx)

frame_merged_3=merged_3.sample(frac=1).reset_index(drop=True)
frame_merged_3.to_csv('/Users/bhuvanakorrapati/Desktop/output/merge_LOA.csv', index=False)


# In[61]:


frame_merged_3.shape


# In[62]:


s= np.random.randint(144)
t= np.random.randint(144)
frame_merged_3.plot.scatter(x=s,y=t, c='blue')


# In[63]:


random_values=np.random.randint(144)
frame_merged_3[random_values].hist(bins=144)


# In[64]:


x = np.array([fspaceL])

# the first scatter plot
y1 = np.array([fspaceO])
plt.scatter(x, y1, color = 'blue')

# the second scatter plot
y2 = np.array([fspaceA])
plt.scatter(x,y2, color = 'green')

# displaying both plots
plt.show()


# In[65]:


import glob
import cv2

images = [cv2.imread(file) for file in glob.glob("/Users/bhuvanakorrapati/Desktop/data/*.jpg")]


# In[66]:


import glob
cv_img = []
for img in glob.glob("/Users/bhuvanakorrapati/Desktop/data/*.jpg"):
    n= cv2.imread(img)
    cv_img.append(n)






