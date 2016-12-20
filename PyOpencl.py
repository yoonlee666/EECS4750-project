#############################################
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
__kernel void Bernsen(const int M, const int N, __global const int *image, __global int *histogram, __global int *binary){
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

//Kernel for thinning
__kernel void thinning(__global int *img, __global int *y, __global int *flag,
                   __global int *table, const int row, const int col) {
    unsigned int i = get_group_id(0)*get_local_size(0)+get_local_id(0);
    unsigned int j = get_group_id(1)*get_local_size(1)+get_local_id(1);
    __private int neighbor[8];
    __local int t[256];
    
    int n = get_local_size(1)*get_local_id(0)+get_local_id(1);
    int stride = get_local_size(1)*get_local_size(0);
    while (n<256) {
        t[n] = table[n];
        n += stride;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
        
    if (i<row && j<col)
    {
        for (int k=0; k<8; k++)
        {
            neighbor[k] = 0;
        }
        if (i>0) 
            neighbor[0] = 1-img[(i-1)*col+j];
        if (i>0 && j<col-1) 
            neighbor[1] = 1-img[(i-1)*col+j+1];
        if (j<col-1) 
            neighbor[2] = 1-img[i*col+j+1];
        if (i<row-1 && j<col-1) 
            neighbor[3] = 1-img[(i+1)*col+j+1];
        if (i<row-1) 
            neighbor[4] = 1-img[(i+1)*col+j];
        if (i<row-1 && j>0) 
            neighbor[5] = 1-img[(i+1)*col+j-1];
        if (j>0) 
            neighbor[6] = 1-img[i*col+j-1];
        if (i>0 && j>0) 
            neighbor[7] = 1-img[(i-1)*col+j-1];

        for (n=0; n<8; n++) {
            if (neighbor[n] <0) neighbor[n] = 0; //Remove this for-loop if the input is a real 1-0 binary image
        }    

        int low_bit = neighbor[0]+neighbor[1]*2+neighbor[2]*4+neighbor[3]*8;
        int high_bit = neighbor[4]+neighbor[5]*2+neighbor[6]*4+neighbor[7]*8;

        int temp = img[i*col+j];
        flag[i*col+j] = 0;
        if (temp == 0) {
            if (t[high_bit*16+low_bit] == 0) {
                temp = 1;
                flag[i*col+j] = 1;
            }
        }
        else temp = 1;
        y[i*col+j] = temp;
    }
}
//minutiae extraction using crossing number
__kernel void minutiae_extraction(const int M, const int N, __global const int *image, __global int *minutiae, __global int *number) {
	int i = get_global_id(0);
        int j = get_global_id(1);
        __local int bins[9];
	if(image[i*N+j]==0){
		if ((i>0)&&(i<1133)&&(j>0)&&(j<784)){
 	      		bins[0]=image[i*N+j+1];
			bins[1]=image[(i-1)*N+j+1];
			bins[2]=image[(i-1)*N+j];
			bins[3]=image[(i-1)*N+j-1];
			bins[4]=image[i*N+j-1];
			bins[5]=image[(i+1)*N+j-1];
			bins[6]=image[(i+1)*N+j];
			bins[7]=image[(i+1)*N+j+1];
			bins[8]=image[i*N+j+1];
		}
		int CN=0;
		for (int k=0;k<8;k++){
			CN+=abs(bins[k]-bins[k+1]);
		}
		if(CN==2){
			minutiae[i*N+j]=2;    //endpoint has value 2
			atomic_add(&number[0],1);
		}
		if(CN==6){
			minutiae[i*N+j]=3;    //burification has value 3
			atomic_add(&number[0],1);
		}
	}
	else	minutiae[i*N+j]=1;
}

// This kernel is for showing minutiae points in RGB
__kernel void rgbshow(const int N, __global char *a, __global char *b) {
        int i = get_global_id(0);
        int j = get_global_id(1);

        if (a[i*N+j]==1) {
            b[(i*N+j)*3]=255;
            b[(i*N+j)*3+1]=255;
            b[(i*N+j)*3+2]=255;
        }
        if (a[i*N+j]==2) {
            b[(i*N+j)*3]=255;
            b[(i*N+j)*3+1]=0;
            b[(i*N+j)*3+2]=0;
        }
        if (a[i*N+j]==3) {
            b[(i*N+j)*3]=0;
            b[(i*N+j)*3+1]=255;
            b[(i*N+j)*3+2]=0;
        }
}

// calculating matching score kernel
__kernel void matching_score(const int M, const int N, const int core1_y, const int core1_x, const int core2_y, const int core2_x, __global int *image1, __global int *image2, __global unsigned int *score) {
        int i = get_global_id(0);
        int j = get_global_id(1);
	if ((image2[i*N+j]==2)||(image2[i*N+j]==3)){
		int y = i-core2_y+core1_y;
		int x = j-core2_x+core1_x;
		__private int bins[25];
		for (int k=y-2;k<=y+2;k++){
			for (int l=x-2;l<=x+2;l++){
				if((k>=0)&&(k<M)&&(l>=0)&&(l<N)){
					bins[(k-y+2)*5+l-x+2]=image1[k*N+l];
				}
				else{
					bins[(k-y+2)*5+l-x+2]=0;
				}
			}
		}
		for (int k=0;k<25;k++){
			if(bins[k]==image2[i*N+j])	{atomic_add(&score[0],1);break;}
		}
	}
}
"""
################ time array #################
naivetime = [0]*10
improvetime = [0]*10

# print original grayscale image
fig = plt.figure(figsize=(25,30))

# image1
im1 = Image.open('./fingerprint1.jpg').convert('L') #converts the image to grayscale
image1 = np.array(im1).astype(np.int32)
M = len(image1)
N = len(image1[0])
# image2
im2 = Image.open('./fingerprint2.jpg').convert('L')
image2 = np.array(im2).astype(np.int32)
M = len(image2)
N = len(image2[0])

# print original image
# image1
image1_gray = Image.fromarray(image1)
plt.subplot(4,4,1)
plt.title('gray(original) image1', fontsize= 20)
plt.imshow(image1_gray, extent=[0,N,0,M])
# image2
image2_gray = Image.fromarray(image2)
plt.subplot(4,4,3)
plt.title('gray(original) image2', fontsize= 20)
plt.imshow(image2_gray, extent=[0,N,0,M])

# print image function
def showimage(imagename, xth, title):
	binary_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=imagename)
	binary_show = np.empty_like(image1).astype(np.int32)
	binary_show_buf = cl.Buffer(ctx, mf.WRITE_ONLY, binary_show.nbytes)
	prg.show(queue, image1.shape, None, np.int32(M), np.int32(N), binary_buf, binary_show_buf)
	cl.enqueue_copy(queue, binary_show, binary_show_buf)
	im_after = Image.fromarray(binary_show)
	plt.subplot(4,4,xth)
	plt.title(title, fontsize= 20)
	plt.imshow(im_after, extent=[0,N,0,M])
# compile kernel
prg = cl.Program(ctx, kernel).build()

######################### STEP 1: gray image to binary image ########################
##### do binary using a global threshold, output image is "binary1_global" and "binary2_global"
# image1: binary1_global
image1_gpu = cl.array.to_device(queue, image1)
histogram1 = np.zeros((1,256)).astype(np.int32)
histogram1_gpu = cl.array.to_device(queue, histogram1)
binary1_gpu = cl.array.zeros(queue,image1.shape,image1.dtype)
prg.globalthreshold(queue, image1.shape, None, np.int32(M), np.int32(N), image1_gpu.data, histogram1_gpu.data, binary1_gpu.data)
binary1_global=binary1_gpu.get()
# image2: binary2_global
image2_gpu = cl.array.to_device(queue, image2)
histogram2 = np.zeros((1,256)).astype(np.int32)
histogram2_gpu = cl.array.to_device(queue, histogram2)
binary2_gpu = cl.array.zeros(queue,image2.shape,image2.dtype)
prg.globalthreshold(queue, image2.shape, None, np.int32(M), np.int32(N), image2_gpu.data, histogram2_gpu.data, binary2_gpu.data)
binary2_global=binary2_gpu.get()
# show image
showimage(binary1_global, 5, "binary image1 using global threshold")
showimage(binary2_global, 7, "binary image2 using global threshold")

##### do binary using Bernsen algorithm, output image is "binary1_Bernsen" and "binary2_Bernsen"
# image1: binary1_Bernsen
image1_gpu = cl.array.to_device(queue, image1)
histogram1 = np.zeros((1,256)).astype(np.int32)
histogram1_gpu = cl.array.to_device(queue, histogram1)
binary1_gpu = cl.array.zeros(queue,image1.shape,image1.dtype)
prg.Bernsen(queue, image1.shape, None, np.int32(M), np.int32(N), image1_gpu.data, histogram1_gpu.data, binary1_gpu.data)
binary1_Bernsen=binary1_gpu.get()
# image2: binary2_Bernsen
image2_gpu = cl.array.to_device(queue, image2)
histogram2 = np.zeros((1,256)).astype(np.int32)
histogram2_gpu = cl.array.to_device(queue, histogram2)
binary2_gpu = cl.array.zeros(queue,image2.shape,image2.dtype)
prg.Bernsen(queue, image2.shape, None, np.int32(M), np.int32(N), image2_gpu.data, histogram2_gpu.data, binary2_gpu.data)
binary2_Bernsen=binary2_gpu.get()
# show image
showimage(binary1_Bernsen, 6, "binary image1 with Bersen algorithm")
showimage(binary2_Bernsen, 8, "binary image2 with Bersen algorithm")

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
                   [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1]], np.int32)
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
                   [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1]], np.int32)

# Use the output of global thresholding, the corresponding skeleton image is "skeleton1_global" and "skeleton2_global"
img1_gpu = cl.array.to_device(queue, binary1_global)
img2_gpu = cl.array.to_device(queue, binary2_global)
y1_gpu = cl.array.zeros(queue,binary1_global.shape,binary1_global.dtype)
y2_gpu = cl.array.zeros(queue,binary2_global.shape,binary2_global.dtype)
table0_gpu = cl.array.to_device(queue, table0)
table1_gpu = cl.array.to_device(queue, table1)
flag1_gpu = cl.array.zeros(queue,binary1_global.shape,binary1_global.dtype)
flag2_gpu = cl.array.zeros(queue,binary2_global.shape,binary2_global.dtype)
L = 8
M1 = ((M-1)/L+1)*L
N1 = ((N-1)/L+1)*L
k = 0
iteration_global = 1
while(True):
	if (k == 0):
		prg.thinning(queue, (M1,N1), (L,L), img1_gpu.data, y1_gpu.data, flag1_gpu.data, table0_gpu.data, np.int32(M), np.int32(N))
	else:
		prg.thinning(queue, (M1,N1), (L,L), img1_gpu.data, y1_gpu.data, flag1_gpu.data, table1_gpu.data, np.int32(M), np.int32(N))
	flag1 = flag1_gpu.get()
	if (np.count_nonzero(flag1) > 0):
		k = 1-k
		img1_gpu = y1_gpu
	else:
		break
	iteration_global += 1

skeleton1_global =  y1_gpu.get()
print 'iterations(global1) =', iteration_global
k = 0
iteration_global = 1
while(True):
	if (k == 0):
		prg.thinning(queue, (M1,N1), (L,L), img2_gpu.data, y2_gpu.data, flag2_gpu.data, table0_gpu.data, np.int32(M), np.int32(N))
	else:
		prg.thinning(queue, (M1,N1), (L,L), img2_gpu.data, y2_gpu.data, flag2_gpu.data, table1_gpu.data, np.int32(M), np.int32(N))	
	flag2 = flag2_gpu.get()
	if (np.count_nonzero(flag2) > 0):
		k = 1-k
		img2_gpu = y2_gpu
	else:
		break
	iteration_global += 1

skeleton2_global =  y2_gpu.get()
print 'iterations(global2) =', iteration_global

# show image
showimage(skeleton1_global, 9, "Skeleton of global binary image1")
showimage(skeleton2_global, 11, "Skeleton of global binary image2")

# Use the output of Bernsen thresholding, the corresponding skeleton image is "skeleton1_Bernsen" and "skeleton2_Bernsen"
img1_gpu = cl.array.to_device(queue, binary1_Bernsen)
img2_gpu = cl.array.to_device(queue, binary2_Bernsen)
y1_gpu = cl.array.zeros(queue,binary1_Bernsen.shape,binary1_Bernsen.dtype)
y2_gpu = cl.array.zeros(queue,binary2_Bernsen.shape,binary2_Bernsen.dtype)
table0_gpu = cl.array.to_device(queue, table0)
table1_gpu = cl.array.to_device(queue, table1)
flag1_gpu = cl.array.zeros(queue,binary1_Bernsen.shape,binary1_Bernsen.dtype)
flag2_gpu = cl.array.zeros(queue,binary2_Bernsen.shape,binary2_Bernsen.dtype)
k = 0
iteration_Bernsen = 1
while(True):
	if (k == 0):
		prg.thinning(queue, (M1,N1), (L,L), img1_gpu.data, y1_gpu.data, flag1_gpu.data, table0_gpu.data, np.int32(M), np.int32(N))
	else:
		prg.thinning(queue, (M1,N1), (L,L), img1_gpu.data, y1_gpu.data, flag1_gpu.data, table1_gpu.data, np.int32(M), np.int32(N))	
	flag1 = flag1_gpu.get()
	if (np.count_nonzero(flag1) > 0):
		k = 1-k
		flag1_gpu = cl.array.zeros(queue,binary1_Bernsen.shape,binary1_Bernsen.dtype)
		img1_gpu = y1_gpu
	else:
		break
	iteration_Bernsen += 1

skeleton1_Bernsen =  y1_gpu.get()
print 'iterations(Bernsen1) =', iteration_Bernsen

k = 0
iteration_Bernsen = 1
while(True):
	if (k == 0):
		prg.thinning(queue, (M1,N1), (L,L), img2_gpu.data, y2_gpu.data, flag2_gpu.data, table0_gpu.data, np.int32(M), np.int32(N))
	else:
		prg.thinning(queue, (M1,N1), (L,L), img2_gpu.data, y2_gpu.data, flag2_gpu.data, table1_gpu.data, np.int32(M), np.int32(N))	
	flag2 = flag2_gpu.get()
	if (np.count_nonzero(flag2) > 0):
		k = 1-k
		flag2_gpu = cl.array.zeros(queue,binary2_Bernsen.shape,binary2_Bernsen.dtype)
		img2_gpu = y2_gpu
	else:
		break
	iteration_Bernsen += 1

skeleton2_Bernsen =  y2_gpu.get()
print 'iterations(Bernsen2) =', iteration_Bernsen

# show image
showimage(skeleton1_Bernsen, 10, "Skeleton of Bernsen binary image1")
showimage(skeleton2_Bernsen, 12, "Skeleton of Bernsen binary image2")

######################### minutiae extraction #########################
# output file is extraction1_global and extraction2_global
image1_gpu = cl.array.to_device(queue, skeleton1_global)
extraction1_gpu = cl.array.zeros(queue,image1.shape,image1.dtype)
number1_gpu = cl.array.zeros(queue,(1,1),np.int32)
prg.minutiae_extraction(queue, image1.shape, None, np.int32(M), np.int32(N), image1_gpu.data, extraction1_gpu.data, number1_gpu.data)
extraction1_global=extraction1_gpu.get()
number1 = number1_gpu.get()

image2_gpu = cl.array.to_device(queue, skeleton2_global)
extraction2_gpu = cl.array.zeros(queue,image2.shape,image2.dtype)
number2_gpu = cl.array.zeros(queue,(1,1),np.int32)
prg.minutiae_extraction(queue, image2.shape, None, np.int32(M), np.int32(N), image2_gpu.data, extraction2_gpu.data, number2_gpu.data)
extraction2_global=extraction2_gpu.get()
number2 = number2_gpu.get()

# show RGB image
img = extraction1_global.astype(np.uint8)
channels = 3 # RGB channels
result = np.zeros((M, N, channels), dtype = np.uint8) # Creat an empty array for RGB image
img_gpu = cl.array.to_device(queue, img)
result_gpu = cl.array.to_device(queue, result)
prg.rgbshow(queue, img.shape, None, np.int32(N), img_gpu.data, result_gpu.data)
result = result_gpu.get()
img_rgb = Image.fromarray(result, 'RGB')
plt.subplot(4,4,13)
plt.title("minutiae extraction of image1 using global threshold", fontsize= 20)
plt.imshow(img_rgb, extent=[0,N,0,M])
img = extraction2_global.astype(np.uint8)
result = np.zeros((M, N, 3), dtype = np.uint8) # Creat an empty array for RGB image
img_gpu = cl.array.to_device(queue, img)
result_gpu = cl.array.to_device(queue, result)
prg.rgbshow(queue, img.shape, None, np.int32(N), img_gpu.data, result_gpu.data)
result = result_gpu.get()
img_rgb = Image.fromarray(result, 'RGB')
plt.subplot(4,4,15)
plt.title("minutiae extraction of image2 using global threshold", fontsize= 20)
plt.imshow(img_rgb, extent=[0,N,0,M])

################### calculate matching score #####################
image1_gpu = cl.array.to_device(queue, extraction1_global)
image2_gpu = cl.array.to_device(queue, extraction2_global)
score_gpu = cl.array.zeros(queue,(1,1),np.int32)
core1_y=250
core1_x=170
core2_y=250
core2_x=170
prg.matching_score(queue, image1.shape, None, np.int32(M), np.int32(N), np.int32(core1_y), np.int32(core1_x), np.int32(core2_y), np.int32(core2_x),image1_gpu.data, image2_gpu.data, score_gpu.data)
score = score_gpu.get()
print "matching score: "
print float(score[0][0])/float(number2[0][0])


################ show image ###############
fig.tight_layout()
plt.savefig('FPRS_pyopencl.jpg', dpi=600)
