import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from sklearn.ensemble import RandomForestRegressor
from PIL import Image
from tqdm import tqdm


#first we need to import the image of heightlevels of europa_nosealevel#this was generated using
#https://tangrams.github.io/heightmapper/

eu = np.asarray(Image.open('europa_nosealevel.png'))[:,:,0]
eu = eu[::10,::10] #the map is somewhat large so we reduce the data by 100x
eu = eu/np.max(eu)#scale to a max of 1
max = eu.shape
eu = eu.T.flatten()
test_ix =  [i for i in range(len(eu))]
train_ix = [8803]

#as input data we simply generate a grid of x,y indices
#we therefore should be able to learn a map
x,y = np.meshgrid([i for i in range(max[0])],[i for i in range(max[1])])
x = x.flatten()/100.
y = y.flatten()/100.

#RF are typically not used for geospatial data ... but this is faster than GP
regr = RandomForestRegressor(n_estimators=50,random_state=1337)

#we run this for 1# of the data
for i_ in tqdm(range(1161)):
    regr.fit(np.array([x[train_ix],y[train_ix]]).T,eu[train_ix])
    pred = regr.predict(np.array([x[test_ix],y[test_ix]]).T)

    y_var = np.zeros([50,len(x[test_ix])])
    for j in range(50):
        y_var[j,:] = regr.estimators_[j].predict(np.array([x[test_ix],y[test_ix]]).T)

    aqf = pred+np.var(y_var,axis=0)
    ix = np.where(aqf==np.max(aqf))[0]
    i = np.random.choice(ix)

    train_ix.append(test_ix.pop(i))

#show which points were selected
rot = transforms.Affine2D().rotate_deg(-90)
plt.figure(figsize=(10,11.4))
base = plt.gca().transData
plt.scatter(x,y,c=eu,transform=rot+base,cmap='Greys',label='Map data')
plt.plot(x[train_ix],y[train_ix],'-o',transform=rot+base,color='red',alpha=0.5,label='Selected points')
#plt.scatter(x[train_ix],y[train_ix],c=eu[train_ix],transform=rot+base)
plt.axis('equal')
plt.axis('off')
plt.legend(frameon=False)

#show the distribution of values
plt.figure(figsize=(7,7))
_ = plt.hist(eu,123,range=(0,1),log=True,density=True,alpha=0.5,label='Map data',color='grey')
_ = plt.hist(eu[train_ix],123,range=(0,1),log=True,density=True,alpha=0.5,label='Selected points',color='red')
plt.legend()

pred = regr.predict(np.array([x,y]).T)


#show the prediction
rot = transforms.Affine2D().rotate_deg(-90)
plt.figure(figsize=(10,11.4))
base = plt.gca().transData
plt.scatter(x,y,c=pred,cmap='Greys',transform=rot+base,label='error')
plt.axis('equal')
plt.axis('off')
plt.colorbar()
plt.legend(frameon=False)

#show the error
rot = transforms.Affine2D().rotate_deg(-90)
plt.figure(figsize=(10,11.4))
base = plt.gca().transData
plt.scatter(x,y,c=pred-eu,cmap='seismic',transform=rot+base,label='error')
plt.axis('equal')
plt.axis('off')
plt.colorbar()
plt.legend(frameon=False)
