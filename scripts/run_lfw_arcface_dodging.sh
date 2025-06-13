export CUDA_VISIBLE_DEVICES=0

python run.py target=arcface attack=dodging n_iter=20 batch_size=32 test_dataset=lfw device=cuda:0