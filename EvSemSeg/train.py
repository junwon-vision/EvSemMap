import os
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Adam

def train(model_dir, dataset, model, epoch_start, writer, args):
    trainloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16, shuffle=True, drop_last=True)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    if args.model == 'evidential':
        model.set_max_iter(len(trainloader))
    
    model.cuda()
    
    # Setup optimizer, scheduler objects
    optimizer = Adam([{'params' : model.encoder.parameters(), 'lr' : args.l_rate}], lr=args.l_rate, betas=(0.5, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    total_iters = 0
    for epoch in range(epoch_start, args.n_epoch+1):
        model.train()
        epoch_iter = 0 

        # epoch_acc, epoch_ece = 0.0, 0.0
        for i, data in enumerate(trainloader, start=1):
            optimizer.zero_grad()

            img, lbl = data
            img, lbl = img.cuda(), lbl.cuda()

            if args.model == 'vanilla':
                # loss, acc, ece = model(img, lbl, with_acc_ece = True)
                loss = model(img, lbl, with_acc_ece = False)
            else:
                # loss, acc, ece = model(img, lbl, i, epoch, with_acc_ece = True)
                loss = model(img, lbl, i, epoch, with_acc_ece = False)
            # epoch_acc += acc
            # epoch_ece += ece
                
            loss = loss.mean()
            loss.backward()
            optimizer.step()
        
            total_iters += args.batch_size
            epoch_iter += args.batch_size

            if i % 10 == 0:
                print(f"[{args.remark}][EPOCH {epoch}, ({i} / {len(trainloader)}) loss {loss.item() :.4f}")

        # writer.add_scalar("Epoch_loss/acc", epoch_acc.item() / len(trainloader), epoch)
        # writer.add_scalar("Epoch_loss/ece", epoch_ece.item() / len(trainloader), epoch)
        writer.add_scalar("Epoch_loss/loss", loss.item(), epoch)
        scheduler.step()
    
        if epoch % args.save_freq == 0:
            print("Saving the model")
            torch.save({ 'epoch': epoch, 'network' : model.state_dict() }, os.path.join(model_dir, f"{epoch}.pth"))
        writer.flush()