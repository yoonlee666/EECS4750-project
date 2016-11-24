###############################################
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
	if (a[i*N+j]<=235)	b[i*N+j]=0;
	else	b[i*N+j]=1;
}
"""

######################### gray image to binary image ########################
fig = plt.figure(figsize=(10,25))
im = Image.open('./1133*784.jpg').convert('L') #converts the image to grayscale
image = np.array(im).astype(np.int32)
im_gray = Image.fromarray(image)
plt.subplot(2,1,1)
plt.title('gray(original) image', fontsize= 30)
plt.imshow(im_gray, extent=[0,783,0,1132])

image_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image)
binary = np.empty_like(image).astype(np.int32)
binary_buf = cl.Buffer(ctx, mf.WRITE_ONLY, binary.nbytes)
prg = cl.Program(ctx, kernel).build()
prg.gray_to_binary(queue, image.shape, None, np.int32(1133), np.int32(784), image_buf, binary_buf)
cl.enqueue_copy(queue, binary, binary_buf)
im_after = Image.fromarray(binary)
plt.subplot(2,1,2)
plt.title("binary image", fontsize= 30)
plt.imshow(im_after, extent=[0,783,0,1132])

fig.tight_layout()
plt.savefig('FPRS_pyopencl.jpg', dpi=300)

######################### minutiae extraction #########################
