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
//gray to binary kernel
__kernel void gray_to_binary(const int M, const int N, __global int *a, __global int *b){
	int i = get_global_id(0);
	int j = get_global_id(1);
	if (a[i*N+j]<=205)	b[i*N+j]=0;
	else	b[i*N+j]=255;
}
"""

######################### gray image to binary image ########################
fig = plt.figure(figsize=(23,20))
im = Image.open('./1133*784.jpg').convert('L') #converts the image to grayscale
M = 1133
N = 784
image = np.array(im).astype(np.int32)
im_gray = Image.fromarray(image)
plt.subplot(1,2,1)
plt.title('gray(original) image', fontsize= 30)
plt.imshow(im_gray, extent=[0,N,0,M])
print image

image_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image)
binary = np.empty_like(image).astype(np.int32)
binary_buf = cl.Buffer(ctx, mf.WRITE_ONLY, binary.nbytes)
prg = cl.Program(ctx, kernel).build()
prg.gray_to_binary(queue, image.shape, None, np.int32(M), np.int32(N), image_buf, binary_buf)
cl.enqueue_copy(queue, binary, binary_buf)
im_after = Image.fromarray(binary)
plt.subplot(1,2,2)
plt.title("binary image", fontsize= 30)
plt.imshow(im_after, extent=[0,N,0,M])
print binary

fig.tight_layout()
plt.savefig('FPRS_pyopencl.jpg', dpi=500)

######################### minutiae extraction #########################
