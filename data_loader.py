import scipy.signal
import scipy.misc
from glob import glob
import numpy as np
from sklearn.utils import shuffle

class DataLoader():
    def __init__(self):
       self.o = 0

    def load_data(self, samples_per_class=1, patch_size=64, mode=None, is_testing=False):
        data_type = "train" if not is_testing else "testi"
        if mode == 'train':
            Dataset_Orignal = "path to original dataset"
            Dataset_AMF = "path to anti-forensic dataset"
        if mode == 'test':
            Dataset_Orignal = "path to original dataset"
            Dataset_AMF = "path to anti-forensic dataset"

        D_path = [Dataset_Orignal,  Dataset_AMF]
        imgs_or = []
        imgs_mf = []
        imgs_amf = []
        imgs = []
        labels = []
        batch_size = samples_per_class
        count = 0
        c_p = 0
        for cur_path in D_path:
            path = glob(cur_path+'/*tiff')
            #print(cur_path)
            patch_nos = (512//(patch_size))**2
            #print(patch_nos)
            batch_images = np.random.choice(path, size=batch_size//patch_nos, replace = False)
            count = 0            
            #print(batch_images)
            for img_path in batch_images:
                img = self.imread(img_path)
                c = np.asarray(img).shape[-1]
                if c == 3:
                    img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
                #img = scipy.misc.imresize(img,(64,64))
                img_Mf = scipy.signal.medfilt(img, kernel_size=3)
                #img = scipy.misc.imresize(img,(64,64))
                #img_MF = scipy.misc.imresize(img_MF,(64,64))
                #print('MF',img.mf.shape)                
                for i in range(512//patch_size):   #512//2
                #if count==batch_size:
                #    break
                    for j in range(512//patch_size):
                        
			#count = count + 1
                        if c_p == 0: 
                            count = count+1
                            img_patch = img[(i*patch_size):(((i+1)*patch_size)),(j*patch_size):(((j+1)*patch_size))]
                            #print('img',img_patch.shape)                         
                            img_mf_patch = img_Mf[(i*patch_size):(((i+1)*patch_size)),(j*patch_size):(((j+1)*patch_size))]
                            
                            img_or = scipy.signal.medfilt(img_patch, kernel_size = 3) - img_patch
                            
                            img_mf = scipy.signal.medfilt(img_mf_patch, kernel_size = 3) - img_mf_patch
                            

                            imgs.append(img_or)
	
                           
                            labels.append(0)
                            if (count%2 ==0):
                                imgs.append(img_mf)
                                labels.append(1)       #for three classification label = 1 else label = 0 pertubed as original images
                           
                        else:
                           count = count+1
                           if (count%2==0):
                               continue;
                           img_patch = img[(i*patch_size):(((i+1)*patch_size)),(j*patch_size):(((j+1)*patch_size))]
                           #img_amf = scipy.misc.imresize(img_patch,(64,64),interp ='nearest')
                           img_amf = scipy.signal.medfilt(img_patch, kernel_size = 3) - img_patch
                           #img_amf = scipy.misc.imresize(img_amf,(64,64))
                           imgs.append(img_amf)
                                                     
                           labels.append(1)
            c_p = 1
             
        imgs = np.array(imgs) / 127.5 - 1.
        
        
        
        imgs = imgs.reshape(imgs.shape[0], patch_size, patch_size, 1)
        labels = np.asarray(labels)
        
        imgs, labels = shuffle(imgs,labels)      
        #print("Training Data Generated..\n Data shape :", imgs.shape, "\n Label shape: ", labels.shape)
        return imgs, labels 

    def imread(self, path):
        return scipy.misc.imread(path).astype(np.float)
    

    def crop_center(self,img,cropx,cropy):
        # print('to crop')
        y,x,z = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
	#print('cropped')    
        return img[starty:starty+cropy,startx:startx+cropx]

