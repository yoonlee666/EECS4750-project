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
//this kernel is for showing image(converting 0,1 image to 0,255 image)
__kernel void show(const int M, const int N, __global int *a, __global int *b){
        int i = get_global_id(0);
        int j = get_global_id(1);
        if (a[i*N+j]==0)      b[i*N+j]=0;
        else if (a[i*N+j]==1)    b[i*N+j]=255;
	else b[i*N+j]=a[i*N+j];
}

//gray to binary kernel using a global threshold
__kernel void globalthreshold(const int M, const int N, __global const int *image, __global int *histogram, __global int *binary){
	int i = get_global_id(0);
	int j = get_global_id(1);
	atomic_add(&histogram[image[i*N+j]],1);
	barrier(CLK_GLOBAL_MEM_FENCE);
	int globalthreshold = 0;
	for (int k=0;k<256;k++){
		if (histogram[k]!=0)	{globalthreshold = (255+k)/2; break;}
	}
	if (image[i*N+j]<=globalthreshold)	binary[i*N+j]=0;
	else	binary[i*N+j]=1;
}
//gray to binary kernel using adaptive threshold --- Bernsen algorithm
__kernel void Bernsen33(const int M, const int N, __global const int *image, __global int *histogram, __global int *binary){
        int i = get_global_id(0);
        int j = get_global_id(1);
        atomic_add(&histogram[image[i*N+j]],1);
        barrier(CLK_GLOBAL_MEM_FENCE);
        int globalthreshold = 0;
        for (int k=0;k<256;k++){
                if (histogram[k]!=0)    {globalthreshold = (255+k)/2; break;}
        }
		__local int bins[9];
		for(int k=i-1;k<=i+1;k++){
			for(int l=j-1;l<=j+1;l++){
				if((k<0)||(k>=1133)||(l<0)||(l>=784))	bins[(k-i+2)*3+l-j+2]=0;
				else bins[(k-i+2)*3+l-j+2]=image[k*N+l];
		}
	}
//extract max and min(can do it with reduction)
	int min = 255;
	int max = 0;
	for (int k=0;k<9;k++){
		if(bins[k]<min)	    min=bins[k];
		if(bins[k]>max)     max=bins[k];
	}
        if (((image[i*N+j]>=(min+max)/2)&&(max-min>50))||((image[i*N+j]>=globalthreshold)&&(max-min<=50)))      binary[i*N+j]=1;
        else    binary[i*N+j]=0;
}
__kernel void Bernsen55(const int M, const int N, __global const int *image, __global int *histogram, __global int *binary){
        int i = get_global_id(0);
        int j = get_global_id(1);
        atomic_add(&histogram[image[i*N+j]],1);
        barrier(CLK_GLOBAL_MEM_FENCE);
        int globalthreshold = 0;
        for (int k=0;k<256;k++){
                if (histogram[k]!=0)    {globalthreshold = (255+k)/2; break;}
        }
		__local int bins[25];
		for(int k=i-2;k<=i+2;k++){
			for(int l=j-2;l<=j+2;l++){
				if((k<0)||(k>=1133)||(l<0)||(l>=784))	bins[(k-i+2)*5+l-j+2]=0;
				else bins[(k-i+2)*5+l-j+2]=image[k*N+l];
		}
	}
//extract max and min(can do it with reduction)
	int min = 255;
	int max = 0;
	for (int k=0;k<25;k++){
		if(bins[k]<min)	    min=bins[k];
		if(bins[k]>max)     max=bins[k];
	}
        if (((image[i*N+j]>=(min+max)/2)&&(max-min>50))||((image[i*N+j]>=globalthreshold)&&(max-min<=50)))      binary[i*N+j]=1;
        else    binary[i*N+j]=0;
}
__kernel void Bernsen77(const int M, const int N, __global const int *image, __global int *histogram, __global int *binary){
        int i = get_global_id(0);
        int j = get_global_id(1);
        atomic_add(&histogram[image[i*N+j]],1);
        barrier(CLK_GLOBAL_MEM_FENCE);
        int globalthreshold = 0;
        for (int k=0;k<256;k++){
                if (histogram[k]!=0)    {globalthreshold = (255+k)/2; break;}
        }
		__local int bins[49];
		for(int k=i-3;k<=i+3;k++){
			for(int l=j-3;l<=j+3;l++){
				if((k<0)||(k>=1133)||(l<0)||(l>=784))	bins[(k-i+2)*7+l-j+2]=0;
				else bins[(k-i+2)*7+l-j+2]=image[k*N+l];
		}
	}
//extract max and min(can do it with reduction)
	int min = 255;
	int max = 0;
	for (int k=0;k<49;k++){
		if(bins[k]<min)	    min=bins[k];
		if(bins[k]>max)     max=bins[k];
	}
        if (((image[i*N+j]>=(min+max)/2)&&(max-min>50))||((image[i*N+j]>=globalthreshold)&&(max-min<=50)))      binary[i*N+j]=1;
        else    binary[i*N+j]=0;
}
"""

fig = plt.figure(figsize=(25,30))

im1 = Image.open('./fingerprint1.jpg').convert('L') #converts the image to grayscale
image1 = np.array(im1).astype(np.int32)
M = len(image1)
N = len(image1[0])

image1_gray = Image.fromarray(image1)
plt.subplot(4,2,1)
plt.title('gray(original) image1', fontsize= 25)
plt.imshow(image1_gray, extent=[0,N,0,M])
plt.subplot(4,2,2)
plt.title('gray(original) image1', fontsize= 25)
plt.imshow(image1_gray, extent=[0,N,0,M])

def showimage(imagename, xth, title):
	binary_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=imagename)
	binary_show = np.empty_like(image1).astype(np.int32)
	binary_show_buf = cl.Buffer(ctx, mf.WRITE_ONLY, binary_show.nbytes)
	prg.show(queue, image1.shape, None, np.int32(M), np.int32(N), binary_buf, binary_show_buf)
	cl.enqueue_copy(queue, binary_show, binary_show_buf)
	im_after = Image.fromarray(binary_show)
	plt.subplot(4,2,xth)
	plt.title(title, fontsize= 25)
	plt.imshow(im_after, extent=[0,N,0,M])
# compile kernel
prg = cl.Program(ctx, kernel).build()



######################### STEP 1: gray image to binary image ########################
##### do binary using a global threshold, output image is "binary1_global" and "binary2_global"
# image1: binary1_global
times=[]
for i in range(10):
	image1_gpu = cl.array.to_device(queue, image1)
	histogram1 = np.zeros((1,256)).astype(np.int32)
	histogram1_gpu = cl.array.to_device(queue, histogram1)
	binary1_gpu = cl.array.zeros(queue,image1.shape,image1.dtype)
	start=time.time()
	prg.globalthreshold(queue, image1.shape, None, np.int32(M), np.int32(N), image1_gpu.data, histogram1_gpu.data, binary1_gpu.data)
	times.append(time.time()-start)
print 'global threshold time: ',
print np.average(times)
binary1_global=binary1_gpu.get()
# show image
showimage(binary1_global, 3, "binary image1 using global threshold")

##### do binary using Bernsen algorithm, output image is "binary1_Bernsen" and "binary2_Bernsen"
# 3*3
times=[]
for i in range(10):
	image1_gpu = cl.array.to_device(queue, image1)
	histogram1 = np.zeros((1,256)).astype(np.int32)
	histogram1_gpu = cl.array.to_device(queue, histogram1)
	binary1_gpu = cl.array.zeros(queue,image1.shape,image1.dtype)
	start=time.time()
	prg.Bernsen33(queue, image1.shape, None, np.int32(M), np.int32(N), image1_gpu.data, histogram1_gpu.data, binary1_gpu.data)
	times.append(time.time()-start)
print 'local threshold with 3*3 mask time: ',
print np.average(times)
binary1_Bernsen=binary1_gpu.get()
# show image
showimage(binary1_Bernsen, 4, "binary image1 with Bersen algorithm applying 3*3 mask")
# 5*5
times=[]
for i in range(10):
	image1_gpu = cl.array.to_device(queue, image1)
	histogram1 = np.zeros((1,256)).astype(np.int32)
	histogram1_gpu = cl.array.to_device(queue, histogram1)
	binary1_gpu = cl.array.zeros(queue,image1.shape,image1.dtype)
	start=time.time()
	prg.Bernsen55(queue, image1.shape, None, np.int32(M), np.int32(N), image1_gpu.data, histogram1_gpu.data, binary1_gpu.data)
	times.append(time.time()-start)
print 'local threshold with 5*5 mask time: ',
print np.average(times)
binary1_Bernsen=binary1_gpu.get()
# show image
showimage(binary1_Bernsen, 6, "binary image1 with Bersen algorithm applying 5*5 mask")
# 7*7
#times=[]
#for i in range(10):
#	image1_gpu = cl.array.to_device(queue, image1)
#	histogram1 = np.zeros((1,256)).astype(np.int32)
#	histogram1_gpu = cl.array.to_device(queue, histogram1)
#	binary1_gpu = cl.array.zeros(queue,image1.shape,image1.dtype)
#	prg.Bernsen77(queue, image1.shape, None, np.int32(M), np.int32(N), image1_gpu.data, histogram1_gpu.data, binary1_gpu.data)
#	times.append(time.time()-start)
#print 'local threshold with 7*7 mask time: ',
#print np.average(times)
#binary1_Bernsen=binary1_gpu.get()
# show image
#showimage(binary1_Bernsen, 8, "binary image1 with Bersen algorithm applying 7*7 mask")

############### show image ###############
fig.tight_layout()
plt.savefig('step1.jpg', dpi=600)
