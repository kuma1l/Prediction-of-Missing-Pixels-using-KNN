import skimage.io as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def getStretchPaddedImage(imgs, fltshape):
    allpdds=[]
    for img in imgs:
        inshape = img.shape
        padshape = (fltshape[0]//2, fltshape[1]//2)

        padimg = np.zeros(shape=(inshape[0] +(2*padshape[0]),inshape[1] + (2*padshape[1]))).astype(int)

        padimg.fill(255)
        for r in range(0,inshape[0]):
            for c in range(0,inshape[1]):
                padimg[padshape[0] + r, padshape[1] + c] = img[r,c]


        for i in range(padshape[0] - 1, -1, -1):
            for j in range(padshape[1], inshape[1] + padshape[1]):#new imaghe column
                padimg[i,j]=img[0, j - padshape[1]]

        for i in range(inshape[0], inshape[0] + padshape[0]):
            for j in range(padshape[1], inshape[1] + padshape[1]):
                padimg[i + padshape[0], j]=img[inshape[0] - 1, j - padshape[1]]

        for i in range(0, len(padimg)):
            for j in range(padshape[1] - 1, -1, -1):
                padimg[i, j]=padimg[i, j + 1]

        for i in range(0, len(padimg)):
            for j in range(inshape[1] + padshape[1], inshape[1] + (2 * padshape[1])):
                padimg[i, j]=padimg[i, j - 1]
        
        allpdds.append(padimg)

    return allpdds


def getNeighbors(img):
    allneigh=[]
    for img1 in img:
        neigh=[]
        for row in range (len(img1)):
            for col in range(len(img1[0])):
                if img1[row][col]!=255 and img1[row, col]not in neigh:
                    neigh.append(img1[row,col])
            
        allneigh.append(neigh)
    
    return allneigh
            
def getwindow(pimg, row, col, winshape):
    offsetshape = (winshape[0]//2, winshape[1]//2)
    window= pimg[row-offsetshape[0]: row+offsetshape[0]+1,col-offsetshape[0]: col+offsetshape[0]+1]
    return window                
        
    
def manhattan_distance(array, index):
    center = (len(array) // 2, len(array[0]) // 2)
    distance = abs(index[0] - center[0]) + abs(index[1] - center[1])
    return distance

def euclidean_distance(array, index):
    center = (len(array) // 2, len(array[0]) // 2)
    distance=((index[0] - center[0])**2) + ((index[1]-center[1])**2)
    return distance**0.5
    

def calculate_knn(array, Class_List, k, distsel):
    distances = []
    if distsel=='Man':
        for i in range(len(array)):
            for j in range(len(array[0])):
                if array[i][j] in Class_List:
                    distances.append((manhattan_distance(array, (i, j)), (i, j)))
        distances.sort()
        knn = distances[:k]
        nearest_neighbors = [array[i][j] for distance, (i, j) in knn]
        if len(nearest_neighbors) <k :
            for i in range(len(nearest_neighbors), k):
                nearest_neighbors.append(255)
        return nearest_neighbors
    else:
        for i in range(len(array)):
            for j in range(len(array[0])):
                if array[i][j] in Class_List:
                    distances.append((euclidean_distance(array, (i, j)), (i, j)))
        distances.sort()
        knn = distances[:k]
        nearest_neighbors = [array[i][j] for distance, (i, j) in knn]
        if len(nearest_neighbors) < k :
            for i in range(len(nearest_neighbors), k):
                nearest_neighbors.append(255)
        return nearest_neighbors
        
    
def mode(lst, k):
    dct={}
    
    for i in range (k):
        dct[lst[i]]=0
    
    for i in lst:
        if i in dct:
            dct[i]+=1
    Key_max = max(zip(dct.values(), dct.keys()))[1]
    return Key_max
    
def KNN(img, neighbors,k, distance_Type):
    knn_col=calculate_knn(img, neighbors, k, distance_Type)
    if k==1:
        return knn_col[0]
    else:
        return mode(knn_col, k)
    
    
def knn_Window(imgs, Neighbors, fshape, k, distance_Type):
    allpadded=getStretchPaddedImage(imgs, fshape)
    n=0
    outimgs=[]
    for img in allpadded:
        pshape=img.shape
        
        rowoff=fshape[0]//2
        coloff=fshape[1]//2
        
        retimg=np.zeros(shape=imgs[n].shape).astype(int)
        retimg[0: imgs[n].shape[0],0 : imgs[n].shape[1]]=imgs[n]
        neighborFeat=Neighbors[n]
        
        nrow, ncol=retimg.shape
        for row in range(nrow):
            for col in range(ncol):
                if(retimg[row,col]==255):
                    window=getwindow(img, row+rowoff, col+coloff, fshape)
                    retimg[row, col]=KNN(window, neighbors[n], k, distance_Type)
        
        n+=1
        outimgs.append(retimg)
    return outimgs
        
        

img2=sk.imread("italy50.png")
orgimg=sk.imread("italy.png")
newimg = np.zeros(img2.shape).astype(int)
img_r = img2[:,:,0]
img_g = img2[:,:,1]
img_b = img2[:,:,2]
fshape=[21,21]
neighbors=getNeighbors([img_r, img_g, img_b])
limg=knn_Window([img_r, img_g,img_b],neighbors, fshape,9, 'eu')
newimg[:,:,0]=limg[0]
newimg[:,:,1]=limg[1]
newimg[:,:,2]=limg[2]


cm = confusion_matrix(newimg.flatten(), orgimg.flatten())

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(newimg), yticklabels=np.unique(img2))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('Confusion Matrix 50', dpi=300, bbox_inches='tight')
plt.show()

TP = np.diag(cm)  
TN = np.sum(cm) - np.sum(cm, axis=0) - np.sum(cm, axis=1) + np.diag(cm)
FP = np.sum(cm, axis=0) - np.diag(cm)  
FN = np.sum(cm, axis=1) - np.diag(cm)


accuracy = (np.sum(TP) / np.sum(cm))*100

specificity = (np.sum(TN) / (np.sum(TN) + np.sum(FP)))*100

sensitivity = (np.sum(TP) / (np.sum(TP) + np.sum(FN)))*100

print("Accuracy:", accuracy)
print("Specificity:", specificity)
print("Sensitivity:", sensitivity)
plt.grid(False)
plt.figtext(0.5, 0.01, accuracy, ha='center')
plt.imshow(newimg)
plt.savefig('Italy_50-Eucl-9.png', dpi=300, bbox_inches='tight')





