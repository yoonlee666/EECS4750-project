###############################################
########### Fingerprint Recognition ###########
###############################################
from PIL import Image
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import time
import scipy
import scipy.misc
from scipy import signal
import numpy as np
import pyopencl as cl
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
	if platform.name == NAME:
		devs = platform.get_devices()
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

### PyOpenCL kernel
kernel="""
//gray to binary kernel
__kernel void gray_to_binary(const int M, const int N, const int K, __global int *a, __global int *b, __global int *c, const int P){
        int i = get_global_id(0);
	int j = get_global_id(1);
	int tmp=0;
        for(int k=0;k<K;k++){
                for(int l=0;l<K;l++){
                        if(((i+k-(K-1)/2)<0)||((i+k-(K-1)/2)>=M)||((j+l-(K-1)/2)<0)||((j+l-(K-1)/2)>=N)){
                                tmp+=0;
                        }
                       	else{
                                tmp+=a[(i+k-(K-1)/2)*N+j+l-(K-1)/2]*b[k*K+l];
                        }
                }
        }
       	c[i*N+j]=tmp/P;
}
"""

######################### gray image to binary image ########################
fig = plt.figure(figsize=(10,25))
im = Image.open('./thrones.jpg').convert('L') #converts the image to grayscale
image = np.array(im).astype(np.int32)
im_gray = Image.fromarray(image)
plt.subplot(921).set_xticks(())
plt.subplot(921).set_yticks(())
plt.title('pyopencl', fontsize= 30)
plt.imshow(im_gray)
plt.subplot(922).set_xticks(())
plt.subplot(922).set_yticks(())
plt.title('python', fontsize= 30)
plt.imshow(im_gray)
filters = {
                'identity':np.array([ [0.,0.,0.],[0.,1.,0.],[0.,0.,0.]  ]),
                'sharpen':np.array([[0.,-1.,0.],[-1.,5.,-1.],[0.,-1.,0.]]),
                'blur':np.array([[1.,1.,1.],[1.,1.,1.],[1.,1.,1.]]),
                'edge_det':np.array([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]]),
                'emboss':np.array([[2.,1.,0.],[1.,1.,-1.],[0.,-1.,-2.]]),
                'sob_x':np.array([[-1., 0. , 1.], [-2., 0., 2.], [-1., 0., 1.]]),
                'sob_y':np.array([[-1., -2. , -1.], [0., 0., 0.], [1., 2., 1.]]),
                'smooth_5x5':np.array([[0., 1., 2. , 1., 0.], [1., 4., 8., 4., 1.],[2.,8.,16.,8.,2.],[1.,4.,8.,4.,1.], [0.,1., 2., 1.,0.]])
        }

##### define a filter function
def filter(name, k, p, n):
        fil = filters[name].astype(np.int32)
        fil_flip = np.empty_like(fil).astype(np.int32)
        for i in range(0,k):
                for j in range(0,k):
                        fil_flip[i][j] = fil[k-1-i][k-1-j]
        fil_flip = fil_flip.astype(np.int32)

	a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image)
        b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fil_flip)
        c = np.empty_like(image).astype(np.int32)
        c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)
        prg = cl.Program(ctx, kernel).build()
        prg.convolution(queue, image.shape, None, np.int32(380), np.int32(612), np.int32(k), a_buf, b_buf, c_buf, np.int32(p))
        cl.enqueue_copy(queue, c, c_buf)

        plt.subplot(9,2,n).set_xticks(())
        plt.subplot(9,2,n).set_yticks(())
        im_after = Image.fromarray(c)
        plt.title(name, fontsize= 30)
        plt.imshow(im_after, extent=[0,612,0,380])
        d = scipy.signal.convolve2d(image,fil,mode='same')
        d = d/p
        plt.subplot(9,2,n+1).set_xticks(())
        plt.subplot(9,2,n+1).set_yticks(())
        im_after = Image.fromarray(d)
        plt.title(name, fontsize= 30)
        plt.imshow(im_after, extent=[0,612,0,380])
##### identity
filter('identity', 3, 1, 3)
######################### minutiae extraction #########################
