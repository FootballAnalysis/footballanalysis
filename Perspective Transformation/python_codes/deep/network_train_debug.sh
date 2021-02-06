python network_train.py \
--train-file '../../data/train_data_10k.mat' \
--cuda-id 0 \
--lr 0.01 \
--num-epoch 100 \
--batch-size 2 \
--num-batch 4 \
--random-seed 0 \
--resume '' \
--save-name 'debug.pth'


#python network_train.py --train-file '../../data/train_data_10k.mat' --cuda-id 0 --lr 0.01 --num-epoch 100 --batch-size 2 --num-batch 4 --random-seed 0 --resume '' --save-name 'debug.pth'