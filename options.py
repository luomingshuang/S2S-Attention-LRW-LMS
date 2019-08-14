vid_path = '/home/share/DATASET/LRW/crop_img/crop'
duration_path = '/home/share/DATASET/LRW/lipread_txt/ABOUT/train'
# val_vid_path = '/home/lms/GRID/val'
# anno_path = '/home/lms/GRID/align'
vid_pad = 29
txt_pad = 15
max_epoch = 1000
lr = 1e-7
num_workers = 8
display = 10
test_iter = 1000
img_padding = 75
text_padding = 200
teacher_forcing_ratio = 0.01
# save_dir = 'weights'
save_dir1 = 'weights-val3'
save_dir2 = 'weights-test3'
mode = 'backendGRU'
if('finetune' in mode):
    batch_size = 64
else:
    batch_size = 32
# weights = 'iteration_54001_epoch_30_cer_0.0616691552036915_wer_0.12696610312053158.pt'
weights = 'weights-val3/iteration_10000_epoch_0_cer_0.16196862197002343_wer_0.2554256426683751.pt'
