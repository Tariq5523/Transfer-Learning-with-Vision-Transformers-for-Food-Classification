

import argparse
import utils
from pathlib import Path
def main(args):
    
    
    import datetime
    import numpy as np
    import time
    import torch
    import torch.backends.cudnn as cudnn
    import json

   

    from timm.data import Mixup
    from timm.models import create_model
    from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
    from timm.scheduler import create_scheduler

    from optimizer import build_optimizer
    from timm.utils import get_state_dict, ModelEma

    from datasets import build_dataset, split_dataset, count_samples_per_class
    from engine import train_one_epoch, evaluate, NativeScalerAccum
    from losses import DistillationLoss
    from samplers import RASampler
    
    import pkg_resources
    from torchinfo import summary

    # The following code is adapted for PyTorch version 2.1.0 and above.
    # This method may not work for future versions.
    torch_version = pkg_resources.get_distribution("torch").version
    if torch_version >= '2.1.0':
        import torch._dynamo

        torch._dynamo.config.automatic_dynamic_shapes = False

    import transnext



    # print('All packages loaded fine')
    if args.world_size > 1:
        utils.init_distributed_mode(args)
    else:
        print("Running on a single GPU. Skipping distributed setup.")

    Freeze_and_train_classifer_only = True
    print(f'Freeze_and_train_classifer_only: {Freeze_and_train_classifer_only}')
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Loading the dataset

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    # dataset_train = split_dataset(dataset_train, split_size=0.2, seed=42)
    # dataset_val = split_dataset(dataset_val, split_size=0.2, seed=42)
    # Print dataset sizes and number of classes
    print(f"Number of training samples: {len(dataset_train)}")
    print(f"Number of validation samples: {len(dataset_val)}")
    print(f"Number of classes: {args.nb_classes}")

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # Count the number of samples per class
    # train_class_counts = count_samples_per_class(dataset_train)
    # val_class_counts = count_samples_per_class(dataset_val)

    # # Print the counts
    # print("Training samples per class:")
    # for cls, count in train_class_counts.items():
    #     print(f"Class {cls}: {count} images")

    # print("\nValidation samples per class:")
    # for cls, count in val_class_counts.items():
    #     print(f"Class {cls}: {count} images")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # Optionally, check a batch of data from the training loader
    print("Checking a batch of training data...")
    # for images, labels in data_loader_train:
    #     print(f"Batch of images shape: {images.shape}")
    #     print(f"Batch of labels shape: {labels.shape}")
    #     print(f"Labels: {labels}")
    #     # Optional visualization (requires matplotlib and torchvision)
    #     utils.visualize_batch(images, labels, args)
    #     break  # Only need to check one batch

    # Similarly, check a batch of data from the validation loader
    print("Checking a batch of validation data...")
    # for images, labels in data_loader_val:
    #     print(f"Batch of images shape: {images.shape}")
    #     print(f"Batch of labels shape: {labels.shape}")
    #     print(f"Labels: {labels}")
    #     utils.visualize_batch(images, labels, args)
    #     break  # Only need to check one batch

    print(f'start epoch is {args.start_epoch} last epoch is {args.epochs}')
    
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        img_size=args.input_size,
        pretrain_size=args.pretrain_size,
        fixed_pool_size=args.fixed_pool_size,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        
        # Remove the classification head weights from the checkpoint
        checkpoint_model.pop('head.weight', None)
        checkpoint_model.pop('head.bias', None)
        print("Removed 'head.weight' and 'head.bias' from the pre-trained checkpoint.")
        
        # for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        #         print(f"Removing key {k} from pretrained checkpoint")
        #         del checkpoint_model[k]

        assert args.pretrain_size is not None, 'In finetune,you need input the pretrain size of your model'
        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    # Freeze all parameters
    if Freeze_and_train_classifer_only:
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the classification head
        for param in model.head.parameters():
            param.requires_grad = True
        
        in_features = model.head.weight.shape[1]

        # Define the size of the new intermediate layer
        hidden_units = 512  # You can choose any size you prefer

        # Store the old head
        old_head = model.head

        # Replace the head with a small MLP: Dense -> ReLU -> Original Head
        model.head = torch.nn.Sequential(
            torch.nn.Linear(in_features, 576),
            torch.nn.ReLU(),
            old_head
        )

        
    # Print the model summary
    print("Model summary:")
    # If using DataParallel or DistributedDataParallel, access the underlying module
    model_for_summary = model.module if hasattr(model, 'module') else model
    # Move the model to CPU for summary to save GPU memory (optional)
    model_for_summary.to('cpu')

    summary(
        model_for_summary,
        input_size=(args.batch_size, 3, args.input_size, args.input_size),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=3  # Adjust depth as needed to control the verbosity
    )

    # Move the model back to the original device
    model_for_summary.to(device)

    model_ema = None

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module
    # else:
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of trainable params:', n_parameters)
   
    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    optimizer = build_optimizer(args, model_without_ddp)
    loss_scaler = NativeScalerAccum()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    criterion = DistillationLoss(
        criterion, None, 'none', 0, 0
    )

    output_dir = Path(args.output_dir)

    max_accuracy = 0.0

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        # Remove incompatible keys for the classification head
        checkpoint_model.pop('head.weight', None)
        checkpoint_model.pop('head.bias', None)
        print("Removed 'head.weight' and 'head.bias' from the checkpoint.")

        # Load the rest of the model parameters (ignoring missing keys for the head)
        msg = model_without_ddp.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            print('Loading optimizer checkpoint')
            optimizer.load_state_dict(checkpoint['optimizer'])
            if lr_scheduler is not None:
                print('Loading lr_scheduler checkpoint')
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                print('Loading scaler checkpoint')
                loss_scaler.load_state_dict(checkpoint['scaler'])
            if 'max_accuracy' in checkpoint:
                max_accuracy = checkpoint['max_accuracy']
                print(f'Previous max accuracy record is {max_accuracy:.2f}%')


    if args.compile_model:
        model = torch.compile(model)
        torch._dynamo.config.cache_size_limit = args.cache_size_limit

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.fp32_resume and epoch > args.start_epoch + 1:
            args.fp32_resume = False
        loss_scaler._scaler = torch.amp.GradScaler('cuda', enabled=not args.fp32_resume)#torch.cuda.amp.GradScaler(enabled=not args.fp32_resume)

        # if args.distributed:
        #     data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
            fp32=args.fp32_resume,
            grad_accum_steps=args.update_freq,
        )

        if lr_scheduler is not None:
            lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'last.pth']
            checkpoint_dict = {'model': model_without_ddp.state_dict(),
                               'optimizer': optimizer.state_dict(),
                               'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                               'epoch': epoch,
                               # 'model_ema': get_state_dict(model_ema),
                               'scaler': loss_scaler.state_dict(),
                               'args': args,
                               'max_accuracy': max_accuracy}

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(checkpoint_dict, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        if test_stats['acc1'] >= max_accuracy and args.output_dir:
            checkpoint_paths = [output_dir / 'best.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(checkpoint_dict, checkpoint_path)
                print(f'Saving best performance checkpoint')
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))




if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[utils.get_args_parser()])
    args = parser.parse_args()

    args = utils.update_from_config(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

