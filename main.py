import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize,
    Resize,
    RandomHorizontalFlip,
)
from Convface import convformer_s18 as creat_model
from utils_all import read_train_data, read_val_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate,train_one_epoch_age,evaluate_age,MyDataSet

def check_for_nan_inf(data_loader):
    for inputs, labels in data_loader:
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            print("Input contains NaN or Inf values")
        if torch.isnan(labels).any() or torch.isinf(labels).any():
            print("Label contains NaN or Inf values")
        break


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    train_images_path, train_images_label = read_train_data(args.train_data_path, args.task)
    val_images_path, val_images_label = read_val_data(args.val_data_path, args.task)
    data_transform = {
        "train": transforms.Compose(
            [
                RandomHorizontalFlip(),
                Resize((224, 224)),
                ToTensor(),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
        "val": transforms.Compose(
            [
                Resize((224, 224)),
                ToTensor(),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
    }

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = creat_model(num_classes=args.num_classes).to(device)

    #   model=create_model(num_classes=args.num_classes).to(device)
    #   model.apply(init_weights)

    #   check_for_nan_inf(train_loader)

    if not args.RESUME:
        if args.weights != "":
            assert os.path.exists(args.weights), "权重文件 '{}' 不存在.".format(args.weights)

            # 加载权重字典
            weights_dict = torch.load(args.weights, map_location=device)

            # 移除与头部相关的键
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]

            # 加载模型权重（不严格匹配模型结构）
            model.load_state_dict(weights_dict, strict=False)
        '''
                if args.weights != " ":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load(args.weights, map_location=device)
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            model.load_state_dict(weights_dict, strict=False)
        '''

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=10, hold_epoch=1,end_factor=args.end_factor)

    best_acc = 0.
    start_epoch = 0
    best_mae = float('inf')  # Initialize to infinity so any real MAE will be better
    #  print('len',len(train_loader))

    if args.RESUME:
        path_checkpoint = "./model_weight/checkpoint/ckpt_best_30_numclass_5.pth"
        print("model continue train")
        checkpoint = torch.load(path_checkpoint)
        print('keys\n', checkpoint.keys())
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        lr_scheduler.load_state_dict(checkpoint['lr_schedule'])
    if args.task == 'age':
        for epoch in range(start_epoch + 1, args.epochs + 1):
            train_loss, train_mae = train_one_epoch_age(
                model=model,
                optimizer=optimizer,
                data_loader=train_loader,
                device=device,
                epoch=epoch,
                lr_scheduler=lr_scheduler,
                total_steps=args.epochs * len(train_loader)
                # Apply gradient clipping
            )
            val_loss, val_mae = evaluate_age(
                model=model,
                data_loader=val_loader,
                device=device,
                epoch=epoch,
                total_steps=args.epochs * len(val_loader)
            )

            tags = ["train_loss", "train_mae", "val_loss", "val_mae", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_mae, epoch)  # 记录训练集的 MAE
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_mae, epoch)  # 记录验证集的 MAE
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

            if best_mae > val_mae:
                if not os.path.isdir("./model_weight"):
                    os.mkdir("./model_weight")
                torch.save(model.state_dict(), "./model_weight/best_model_by_mae.pth")
                print("Saved epoch {} as new best model by MAE".format(epoch))
                best_mae = val_mae
            if epoch % 1 == 0:
                print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
            if epoch % 15 == 0:
                print('epoch:', epoch)
                print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
                checkpoint = {
                    "net": model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "epoch": epoch,
                    'lr_schedule': lr_scheduler.state_dict()
                }
                if not os.path.isdir("./model_weight/checkpoint"):
                    os.mkdir("./model_weight/checkpoint")
                torch.save(checkpoint, './model_weight/checkpoint/ckpt_best_%s_numclass_%s.pth' % (
                    str(epoch), str(args.num_classes)))

            print("[epoch {}] MAE: {}".format(epoch, round(val_mae, 5)))
    else:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch,
                                                    lr_scheduler=lr_scheduler
                                                    # Apply gradient clipping
                                                    )
            val_loss, val_acc = evaluate(model=model,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch)

            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

            if best_acc < val_acc:
                if not os.path.isdir("./model_weight"):
                    os.mkdir("./model_weight")
                torch.save(model.state_dict(), "./model_weight/best_model5141.pth")
                print("Saved epoch{} as new best model".format(epoch))
                best_acc = val_acc
            if epoch % 1 == 0:
                print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
            if epoch % 15 == 0:
                print('epoch:', epoch)
                print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
                checkpoint = {
                    "net": model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "epoch": epoch,
                    'lr_schedule': lr_scheduler.state_dict()
                }
                if not os.path.isdir("./model_weight/checkpoint"):
                    os.mkdir("./model_weight/checkpoint")
                torch.save(checkpoint, './model_weight/checkpoint/ckpt_best_%s_numclass_%s.pth' % (
                    str(epoch), str(args.num_classes)))

            print("[epoch {}] accuracy: {}".format(epoch, round(val_acc, 5)))

    total = sum([param.nelement() for param in model.parameters()])

    num_params_message = "Number of parameters: %.2fM" % (total / 1e6)
    best_acc_message = "Best validation accuracy: {:.5f}".format(best_acc)
    best_mae_message = "Best validation MAE: {:.5f}".format(best_mae)

    # Print to console
    print(num_params_message)
    print(best_acc_message)
    print(best_mae_message)

    # Save to file
    output_path = "./model_weight/training_summary.txt"
    with open(output_path, "w") as f:
        f.write(num_params_message + "\n")
        f.write(best_acc_message + "\n")
        f.write(best_mae_message + "\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--task', type=str,default='race')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--end_factor', type=float, default=1e-1)
    parser.add_argument('--RESUME', type=bool, default=False)
    parser.add_argument('--weights', type=str, default='convformer_s18.pth', help='initial weights path')
    parser.add_argument('--train_data_path', type=str, default='./tt')
    parser.add_argument('--val_data_path', type=str, default='./vv')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--cross_attn', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)
