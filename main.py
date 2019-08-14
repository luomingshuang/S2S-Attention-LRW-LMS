import torch
from torch.utils.data import Dataset, DataLoader
import tensorboardX
from data import MyDataset
from model import lipreading
import torch.nn as nn
from torch import optim
import os
import time
from tensorboardX import SummaryWriter
import tensorflow as tf

if(__name__ == '__main__'):
    torch.manual_seed(55)
    torch.cuda.manual_seed_all(55)
    opt = __import__('options')


def data_from_opt(vid_path, folds):
    dataset = MyDataset(folds,vid_path, 
        opt.vid_pad,
        opt.txt_pad)
    print('vid_path:{},num_data:{}'.format(vid_path,len(dataset.data)))
    loader = DataLoader(dataset, 
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        drop_last=False,
        shuffle=True)   
    return (dataset, loader)

if(__name__ == '__main__'):
    model = lipreading(mode=opt.mode, nClasses=30).cuda()
    writer =  SummaryWriter()

    if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)        
            
    (train_dataset, train_loader) = data_from_opt(opt.vid_path, 'train')
    (val_dataset, val_loader) = data_from_opt(opt.vid_path, 'val')
    (tst_dataset, tst_loader) = data_from_opt(opt.vid_path, 'test')
    
    criterion = nn.NLLLoss() 

    optimizer = optim.Adam(model.parameters(),
             lr=opt.lr,
             weight_decay=0)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

    # optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    iteration = 0
    n = 0
    k = 0
    for epoch in range(opt.max_epoch):
        start_time = time.time()
        exp_lr_scheduler.step()

        for (i, batch) in enumerate(train_loader):
            (encoder_tensor, decoder_tensor) = batch['encoder_tensor'].cuda(), batch['decoder_tensor'].cuda()

            outputs = model(encoder_tensor, decoder_tensor, opt.teacher_forcing_ratio)            
            flatten_outputs = outputs.view(-1, outputs.size(2))

            loss1 = criterion(flatten_outputs, decoder_tensor.view(-1))
            loss = loss1

            optimizer.zero_grad()  

            loss.backward()
            optimizer.step()
            tot_iter = epoch*len(train_loader)+i
            writer.add_scalar('data/CE_loss', loss.detach().cpu().numpy(), iteration)
            train_loss = loss.item()
            
            print('iteration:%d, epoch:%d, train_loss:%.6f'%(iteration, epoch, train_loss))

            iteration += 1
            # if(tot_iter % opt.test_iter == 0):
            # if (iteration % 20 == 0):
            if (iteration % 5000) == 0:
                with torch.no_grad():
                    predict_txt_total = []
                    truth_txt_total = []
                    for (idx,batch) in enumerate(val_loader):
                        (encoder_tensor, decoder_tensor) \
                            = batch['encoder_tensor'].cuda(), batch['decoder_tensor'].cuda()
                        outputs = model(encoder_tensor)
                        predict_txt = MyDataset.arr2txt(outputs.argmax(-1))
                        truth_txt = MyDataset.arr2txt(decoder_tensor)
                        predict_txt_total.extend(predict_txt)
                        truth_txt_total.extend(truth_txt)
                
                print(''.join(101*'-'))                
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101*'-')) 
                
                for (predict, truth) in list(zip(predict_txt_total, truth_txt_total))[:10]:
                    # print('{:<50}|{:>50}'.format(predict, truth))
                    print('{:<50}|{:>50}'.format(predict.lower(), truth.lower()))                
                print(''.join(101 *'-'))
                wer = MyDataset.wer(predict_txt_total, truth_txt_total)
                cer = MyDataset.cer(predict_txt_total, truth_txt_total)

                writer.add_scalar('data/cer_val', cer, n)
                writer.add_scalar('data/wer_val', wer, n)

                print('cer:{}, wer:{}'.format(cer, wer))          
                print(''.join(101*'-'))
                savename = os.path.join(opt.save_dir1, 'iteration_{}_epoch_{}_cer_{}_wer_{}.pt'.format(iteration, epoch, cer, wer))
                savepath = os.path.split(savename)[0]
                if(not os.path.exists(savepath)): os.makedirs(savepath)
                torch.save(model.state_dict(), savename)
                n += 1
        if epoch % 1 == 0:
                with torch.no_grad():
                    predict_txt_total = []
                    truth_txt_total = []
                    for (idx,batch) in enumerate(tst_loader):
                        (encoder_tensor, decoder_tensor) \
                            = batch['encoder_tensor'].cuda(), batch['decoder_tensor'].cuda()
                        outputs = model(encoder_tensor)
                        predict_txt = MyDataset.arr2txt(outputs.argmax(-1))
                        truth_txt = MyDataset.arr2txt(decoder_tensor)
                        predict_txt_total.extend(predict_txt)
                        truth_txt_total.extend(truth_txt)
                
                print(''.join(101*'-'))                
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101*'-')) 
                
                for (predict, truth) in list(zip(predict_txt_total, truth_txt_total))[:10]:
                    # print('{:<50}|{:>50}'.format(predict, truth))
                    print('{:<50}|{:>50}'.format(predict.lower(), truth.lower()))                
                print(''.join(101 *'-'))
                wer = MyDataset.wer(predict_txt_total, truth_txt_total)
                cer = MyDataset.cer(predict_txt_total, truth_txt_total)                
                writer.add_scalar('data/cer_tst', cer, n)
                writer.add_scalar('data/wer_tst', wer, n)

                print('cer:{}, wer:{}'.format(cer, wer))          
                print(''.join(101*'-'))
                savename = os.path.join(opt.save_dir2, 'iteration_{}_epoch_{}_cer_{}_wer_{}.pt'.format(iteration, epoch, cer, wer))
                savepath = os.path.split(savename)[0]
                if(not os.path.exists(savepath)): os.makedirs(savepath)
                torch.save(model.state_dict(), savename) 
                k += 1
