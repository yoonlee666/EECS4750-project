##############################################
########### Fingerprint Recognition ###########
###############################################
from PIL import Image
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import time
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
//this kernel is for showing image
__kernel void show(const int M, const int N, __global int *a, __global int *b){
        int i = get_global_id(0);
        int j = get_global_id(1);
        if (a[i*N+j]==0)      b[i*N+j]=0;
        else    b[i*N+j]=255;
}
//gray to binary kernel
__kernel void globalthreshold(const int M, const int N, __global int *a, __global int *b){
	int i = get_global_id(0);
	int j = get_global_id(1);
	if (a[i*N+j]<=205)	b[i*N+j]=0;
	else	b[i*N+j]=1;
}
__kernel void Bernsen(const int M, const int N, __global int *a, __global int *b){
        int i = get_global_id(0);
        int j = get_global_id(1);
	__local int bins[9];
	for(int k=i-1;k<=i+1;k++){
		for(int l=j-1;l<=j+1;l++){
			b[(k-i+1)*3+l-j+1]=a[k*N+l];
		}
	}
//extract max and min
	int min = 255;
	int max = 0;
	for (int k=0;k<9;k++){
		if(bins[k]<min)	    min=bins[k];
		if(bins[k]>max)     max=bins[k];
	}
        if (a[i*N+j]<=(min+max)/2)      b[i*N+j]=0;
        else    b[i*N+j]=255;
}

"""

######################### gray image to binary image ########################
# print original grayscale image
fig = plt.figure(figsize=(15,20))
im = Image.open('./1133*784.jpg').convert('L') #converts the image to grayscale
M = 1133
N = 784
image = np.array(im).astype(np.int32)
image_gray = Image.fromarray(image)
plt.subplot(2,2,1)
plt.title('gray(original) image', fontsize= 30)
plt.imshow(image_gray, extent=[0,N,0,M])
print image
# print image function
def showimage(imagename):
	binary_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=imagename)
	binary_show = np.empty_like(image).astype(np.int32)
	binary_show_buf = cl.Buffer(ctx, mf.WRITE_ONLY, binary_show.nbytes)
	prg.show(queue, image.shape, None, np.int32(M), np.int32(N), binary_buf, binary_show_buf)
	cl.enqueue_copy(queue, binary_show, binary_show_buf)
	return binary_show  
# do binary using a global threshold, output image is "binary_global"
image_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image)
binary_global = np.empty_like(image).astype(np.int32)
binary_buf = cl.Buffer(ctx, mf.WRITE_ONLY, binary_global.nbytes)
prg = cl.Program(ctx, kernel).build()
prg.globalthreshold(queue, image.shape, None, np.int32(M), np.int32(N), image_buf, binary_buf)
cl.enqueue_copy(queue, binary_global, binary_buf)
# show image
binary_show = showimage(binary_global)
im_after = Image.fromarray(binary_show)
plt.subplot(2,2,3)
plt.title("binary image using global threshold", fontsize= 30)
plt.imshow(im_after, extent=[0,N,0,M])
print binary_show
# do binary using Bernsen algorithm, output image is "binary_Bernsen"
image_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image)
binary_Bernsen = np.empty_like(image).astype(np.int32)
binary_buf = cl.Buffer(ctx, mf.WRITE_ONLY, binary_Bernsen.nbytes)
prg.Bernsen(queue, image.shape, None, np.int32(M), np.int32(N), image_buf, binary_buf)
cl.enqueue_copy(queue, binary_Bernsen, binary_buf)
# show image
binary_show = showimage(binary_Bernsen)
im_after = Image.fromarray(binary_show)
plt.subplot(2,2,4)
plt.title("binary image with Bersen algorithm", fontsize= 30)
plt.imshow(im_after, extent=[0,N,0,M])
print binary_show

fig.tight_layout()
plt.savefig('FPRS_pyopencl.jpg', dpi=500)

######################### minutiae extraction #########################
