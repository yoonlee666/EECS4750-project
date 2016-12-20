# EECS4750-project

Project Name:
Fingerprint Recognition

Platform:
NVIDIA GPU

Technology:
parallel computing


Input
file: "fingerprint1.jpg", "fingerprint2.jpg"

Output
file: "FPRS_pyopencl.jpg"

Command:
sbatch --gres=gpu:1 --time=3 --wrap="python PyOpencl.py"

