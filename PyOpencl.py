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
import pyopencl.array
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
//this kernel is for showing image(converting 0,1 image to 0,255 image)
__kernel void show(const int M, const int N, __global int *a, __global int *b){
        int i = get_global_id(0);
        int j = get_global_id(1);
        if (a[i*N+j]==0)      b[i*N+j]=0;
        else    b[i*N+j]=255;
}
//gray to binary kernel using a global threshold
__kernel void globalthreshold(const int M, const int N, __global int *a, __global int *b){
	int i = get_global_id(0);
	int j = get_global_id(1);
	if (a[i*N+j]<=205)	b[i*N+j]=0;
	else	b[i*N+j]=1;
}

//gray to binary kernel using adaptive threshold --- Bernsen algorithm
__kernel void Bernsen(const int M, const int N, __global int *a, __global int *b){
        int i = get_global_id(0);
        int j = get_global_id(1);
	__local int bins[25];
	for(int k=i-2;k<=i+2;k++){
		for(int l=j-2;l<=j+2;l++){
			if((k<0)||(k>=1133)||(l<0)||(l>=784))	bins[(k-i+2)*5+l-j+2]=0;
			else bins[(k-i+2)*5+l-j+2]=a[k*N+l];
		}
	}
//extract max and min(can do it with reduction)
	int min = 255;
	int max = 0;
	for (int k=0;k<25;k++){
		if(bins[k]<min)	    min=bins[k];
		if(bins[k]>max)     max=bins[k];
	}
        if (((a[i*N+j]>=(min+max)/2)&&(max-min>70))||((a[i*N+j]>=205)&&(max-min<=70)))      b[i*N+j]=1;
        else    b[i*N+j]=0;
}

//Kernel for thinning
__kernel void thinning(__global unsigned int *img, __global unsigned int *y, __global unsigned int *flag,
                   __global unsigned int *table, const unsigned int col) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    __local int neighbor[8];
    __local unsigned int t[256];

    neighbor[0] = 1-img[i*col+j+1];
    neighbor[1] = 1-img[i*col+j+2];
    neighbor[2] = 1-img[(i+1)*col+j+2];
    neighbor[3] = 1-img[(i+2)*col+j+2];
    neighbor[4] = 1-img[(i+2)*col+j+1];
    neighbor[5] = 1-img[(i+2)*col+j];
    neighbor[6] = 1-img[(i+1)*col+j];
    neighbor[7] = 1-img[i*col+j];

    for (int n=0; n<8; n++) {
        if (neighbor[n] <0) neighbor[n] = 0; //Remove this for-loop if the input is a real 1-0 binary image
    }

    for (int n=0; n<256; n++) {
        t[n] = table[n];
    }

    int low_bit = neighbor[0]+neighbor[1]*2+neighbor[2]*4+neighbor[3]*8;
    int high_bit = neighbor[4]+neighbor[5]*2+neighbor[6]*4+neighbor[7]*8;

    int temp = img[(i+1)*col+j+1];
    if (temp == 0) {
        if (t[high_bit*16+low_bit] == 0) {
            temp = 1;
            flag[0] = 1;
        }
    }
    y[i*col+j] = temp;
}
"""

######################### gray image to binary image ########################
# print original grayscale image
fig = plt.figure(figsize=(20,30))
im = Image.open('./1133_784.jpg').convert('L') #converts the image to grayscale
M = 1133
N = 784
image = np.array(im).astype(np.int32)
image_gray = Image.fromarray(image)
plt.subplot(3,2,1)
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
plt.subplot(3,2,3)
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
plt.subplot(3,2,4)
plt.title("binary image with Bersen algorithm", fontsize= 30)
plt.imshow(im_after, extent=[0,N,0,M])
print binary_show

print np.allclose(binary_global,binary_Bernsen)

######################### Thinning #########################
# Table mapping
# First iteration table
table0 = np.array([[1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                   [1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                   [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                   [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1]], np.uint32)
# Second iteration table
table1 = np.array( [[1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0],
                   [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                   [1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                   [0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1]], np.uint32)


bi_img = binary_global # Use the output of global thresholding
M = len(bi_img)
N = len(bi_img[0])
k = 0
flag = np.array([0]).astype(np.uint32)

image_padding = np.lib.pad(bi_img,1,'constant', constant_values = 1) # Pad the input matrix with ones (white pixels)
img_gpu = cl.array.to_device(queue, image_padding)
y_gpu = cl.array.zeros(queue,bi_img.shape,bi_img.dtype)
table0_gpu = cl.array.to_device(queue, table0)
table1_gpu = cl.array.to_device(queue, table1)
flag_gpu = cl.array.to_device(queue, flag)

while(True):
    if (k == 0):
        prg.thinning(queue, bi_img.shape, None, img_gpu.data, y_gpu.data, flag_gpu.data, table0_gpu.data, np.int32(N))
    else:
        prg.thinning(queue, bi_img.shape, None, img_gpu.data, y_gpu.data, flag_gpu.data, table1_gpu.data, np.int32(N))
    y = y_gpu.get()
    if (flag[0] != 0):
        k = 1-k;
        flag[0] = 0;
        img_gpu = cl.array.to_device(queue, np.lib.pad(y,1,'constant', constant_values = 1))
    else:
        break;

skeleton =  y_gpu.get()

skeleton_show = showimage(skeleton)
im_after = Image.fromarray(skeleton_show)
plt.subplot(3,2,5)
plt.title("Skeleton", fontsize= 30)
plt.imshow(im_after, extent=[0,N,0,M])
print skeleton_show

######################### minutiae extraction #########################













################ show image ###############
fig.tight_layout()
plt.savefig('FPRS_pyopencl.jpg', dpi=500)
