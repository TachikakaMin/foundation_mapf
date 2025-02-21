python train.py -fd 3
python train.py -fd 4
python train.py -fd 5
python train.py -fd 6
python train.py -m cnn

# 如果时间太慢了，改下面train.py里面的下面两行
# test_list = [files[i] for i in test_indices][:1000]
# train_list = [files[i] for i in train_indices][:10000]
