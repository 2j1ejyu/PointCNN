import time
import torch
from torch import optim
import torch.nn.functional as F
import statistics as stat
import pdb
import wandb
import os


def run(args, num_classes, train_loader, test_loader, model, augmentor, device):
    if args.use_wandb == 'True':
        wandb.init(project=args.project_title)
        wandb.config.update(args)
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_epoch, gamma=args.lr_decay)
    
    dataset_type = "augmented" if args.use_augmentation=='True' else 'vanilla'
    
    best_test_acc, best_test_acc_epoch = 0.,0.
    
    for i in range(1,args.num_epochs+1):
        start_time = time.time()
        
        # train
        train_result = train_one_epoch(args, num_classes, train_loader, model, augmentor, optimizer, device)
        print("Epoch {}, loss_clf: {:.4f}, acc_clf: {:.4f}, duration : {:.4f}".format(i, train_result[0], train_result[1], time.time()-start_time))
        
        
        # test
        test_acc = test(args, test_loader, model, device)
        print("test acc : {}".format(test_acc))
        if best_test_acc < test_acc:
            best_test_acc = test_acc
            best_test_acc_epoch = i
        
        # wandb
        if args.use_wandb == 'True':
            log_dict = {'Loss_clf': train_result[0], 'Acc_clf': train_result[1], 'Test_acc': test_acc, 'Best Test_acc': best_test_acc, 'Best Test epoch': best_test_acc_epoch}
            wandb.log(log_dict)
        
        #save model
        if(i%10==0):
            if args.model_save == 'True':
                foldername = "./saves/{}/{}/{}".format(args.dataset, args.model, dataset_type)
                os.makedirs(foldername, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(foldername, 'classifier_{}.pt'.format(i)))
                
        scheduler.step()

def train_one_epoch(args, num_classes, train_loader, model, augmentor, optimizer, device):
    P_correct = 0
    losses_classifier = []
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        if args.use_augmentation=='True':
            pos = data.pos.reshape(-1,args.num_points,3)  #(B*N,3) -> (B,N,3)
            pos = augmentor(pos).view(-1,3)
            data.pos = pos
        #forward
        pred = model(data, get_feature = False)
        loss = F.nll_loss(pred, data.y)

        losses_classifier.append(loss.item())
        
        #backward
        loss.backward()
        optimizer.step()

        P_pred = pred.max(1)[1]
        P_correct += P_pred.eq(data.y).sum().item()

    return stat.mean(losses_classifier), P_correct / len(train_loader.dataset)

def test(args, test_loader, model, device):
    model.eval()
    correct = 0
    for data in test_loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    test_acc = correct / len(test_loader.dataset)
    return test_acc