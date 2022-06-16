import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import numpy as np

from PIL import Image

import time

import math 

import imageio 

#supplemental functions 


#starting by implementing convolution function focusing on kernel of the original image

def ConvolutionFunction(OriginalImage, Kernel):
    
    #the image.shape[0] of the original image represents its height 
    
    ImageHeight = OriginalImage.shape[0]
    
    #the image.shape[1] of the original image represents its width
    
    ImageWidth = OriginalImage.shape[1]
    
    #since kernel here represents a small 2d matrix to blur the original image 
    
    #We represents Kernel.shape[0] as its height, and Kernel.shape[1] as its width 
    
    KernelHeight = Kernel.shape[0]
    
    KernelWidth =  Kernel.shape[1]
    
    #pad numpy arrays within the image
    
    #we consider OriginalImage as an array 
    
    #if the grayscale image gives three element as the number of channels.
    
    
    if(len(OriginalImage.shape) == 3):
        
        PaddedImage = np.pad(OriginalImage, pad_width = ((KernelHeight // 2, KernelHeight // 2), 
        (KernelWidth//2, KernelWidth//2), (0,0)), mode='constant', constant_values=0).astype(np.float32)
        
        
        #if the grayscale image gives two element as the number of channels.
        
        
    elif (len(OriginalImage.shape) == 2):
        
        PaddedImage = np.pad(OriginalImage, pad_width = (( KernelHeight // 2,  KernelHeight // 2),
            (KernelWidth//2, KernelWidth//2)), mode='constant', constant_values=0).astype(np.float32)
        
        
    #floor division result quotient of Kernel height and width divides by 2 
        
    height = KernelHeight // 2
    
    width = KernelWidth // 2
      
    
    #initialize a new array of given shape and type, filled with zeros from padded image 
    
    ConvolvedImage = np.zeros(PaddedImage.shape)
    
    #sum = 0
    
    #iterate the image convolution as 2d array as well 
    
    for i in range(height, PaddedImage.shape[0] - height):
        
        for j in range(width, PaddedImage.shape[1] - width):
            
            
            #2D matrix indexes 
            
            x = PaddedImage[i - height:i-height + KernelHeight, j-width:j-width + KernelWidth]
            
            #use flaten() to return a copy of the array collapsed into one dimension.
            
            x = x.flatten() * Kernel.flatten()
            
            #pass the sum of the array elements into the convolved image matrix
            
            ConvolvedImage[i][j] = x.sum()
            
    #assign endpoints of height and width in the 2D matrix 
            
    HeightEndPoint = -height
    
    WidthEndPoint  = -width 
    
    #when there is no height, return [height, width = width end point] 
    
    if(height == 0):
        
        return ConvolvedImage[height:, width : WidthEndPoint]
    
    #when there is no width, return [height = height end point, width ] 
    
    if(width  == 0):
        
        return ConvolvedImage[height: HeightEndPoint, width:]
    
    #return the convolved image
    
    return ConvolvedImage[height: HeightEndPoint,  width: WidthEndPoint]

#SeparableGaussianBlurImage function implementation 


def SeparableGaussianBlurImage(image, sigma):
    
    #replace imread(image) with imageio.imread(image) since scipy is not supported 
    
    image = imageio.imread(image)
    
    
    FilterSize = 2 * int(4 * sigma + 0.5) + 1
    
    
    GaussianFilterY = np.zeros((FilterSize, 1), np.float32)
    
    
    GaussianFilterX = np.zeros((1, FilterSize), np.float32)
    
    
    m = FilterSize//2
    
    n = FilterSize//2
    
    
    for y in range(-n, n+1):
        
        x1 = np.sqrt(2*np.pi*(sigma**2))
        
        x2 = np.exp(-(y**2)/(2* sigma**2))
        
        GaussianFilterX[:, y+n] = (1/x1)*x2
        
    for x in range(-m, m):
        
            x1 = np.sqrt(2*np.pi*(sigma**2))
            
            x2 = np.exp(-(x**2)/(2* sigma**2))
            
            GaussianFilterY[x+m, :] = (1/x1)*x2
            
    #since gaussian filter = X * Y in the 2d array 
    
    
    FilteredImage = np.zeros_like(image, dtype=np.float32)
    
    for k in range(3):
        
        FilteredImage[:, :, k] = ConvolutionFunction(image[:, :, k], GaussianFilterX)
        
    for c in range(3):
        
        FilteredImage[:, :, k] = ConvolutionFunction(FilteredImage[:, :, c], GaussianFilterY)
    
    return (FilteredImage.astype(np.uint8))


#similarly to the implementation of gaussian blur image 


#Driver/Testing code


#Required test:

#Gaussian blur the image ”Seattle.jpg” with a sigma of 4.0, and save as ”2.png”
    

a = SeparableGaussianBlurImage('Seattle.jpg', 4.0)

plt.imshow(a)
    
plt.imsave('2.png', a)

    
#extra tests

    
#try sigma = 2.0 

b = SeparableGaussianBlurImage('Seattle.jpg', 2.0)
    
plt.imshow(b)
    
plt.imsave('SigmaIs2Sep.png', b)


#try sigma = 8.0
    
c = SeparableGaussianBlurImage('Seattle.jpg', 8.0)
    
plt.imshow(c)
    
plt.imsave('SigmaIs8Sep.png', c)


#try sigma = 32.0 
    
d = SeparableGaussianBlurImage('Seattle.jpg', 32.0)
    
plt.imshow(d)
    
plt.imsave('SigmaIs32Sep.png', d)
