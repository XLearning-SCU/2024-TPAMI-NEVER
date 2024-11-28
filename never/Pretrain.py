"""
Pre-training script.
"""

import argparse
import datetime
import json
import os
import time
from os.path import join
from pathlib import Path

import torch
import torch.distributed as dist

import utils
from dataset import create_dataset, create_loader, create_sampler
from models.model_pretrain import ALBEF
from models.tokenization_bert import BertTokenizer
from models.vit import interpolate_pos_embed
from optim import create_optimizer
from scheduler import create_scheduler


def get_metric_logger():
    # terminal metric logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=50, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "loss_mlm", utils.SmoothedValue(window_size=50, fmt="{value:.4f}")
    )
    metric_logger.add_meter(
        "loss_ita", utils.SmoothedValue(window_size=50, fmt="{value:.4f}")
    )
    metric_logger.add_meter(
        "loss_itm", utils.SmoothedValue(window_size=50, fmt="{value:.4f}")
    )
    return metric_logger


def train(
    model,
    data_loader,
    optimizer,
    tokenizer,
    epoch,
    warmup_steps,
    device,
    scheduler,
    config,
    max_epoch,
):
    # train
    model.train()

    metric_logger = get_metric_logger()

    header = "Train Epoch: [{}]".format(epoch)
    print_freq = 100
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (image, image_aug, text) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        optimizer.zero_grad()

        image = image.to(device, non_blocking=True)
        image_aug = image_aug.to(device, non_blocking=True)

        text_input = tokenizer(
            text, padding="longest", truncation=True, max_length=25, return_tensors="pt"
        ).to(device)

        if epoch > 0:
            alpha = config["alpha"]
        else:
            alpha = config["alpha"] * min(1, i / len(data_loader))

        # compute the negative percentage
        neg_thresh = epoch * config["neg_thresh"] / (max_epoch - 1)

        loss_mlm, loss_ita, loss_itm = model(
            image,
            image_aug,
            text_input,
            alpha=alpha,
            neg_thresh=neg_thresh,
            epoch=epoch,
        )

        loss = loss_mlm + loss_ita + loss_itm

        loss.backward()
        optimizer.step()

        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    train_stats = {
        k: "{:.3f}".format(meter.global_avg)
        for k, meter in metric_logger.meters.items()
    }
    return train_stats


def main(args, config):
    utils.init_distributed_mode(args)
    utils.print_config(config)
    device = torch.device(args.device)

    # set seed for reproducibility
    utils.set_seed(config["seed"])

    start_epoch = 0
    max_epoch = config["schedular"]["epochs"]
    warmup_steps = config["schedular"]["warmup_epochs"]

    #### Dataset ####

    print("Creating dataset")
    datasets = [create_dataset("pretrain", config)]

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True], num_tasks, global_rank)
    else:
        samplers = [None]

    data_loader = create_loader(
        datasets,
        samplers,
        batch_size=[config["batch_size"]],
        num_workers=[4],
        is_trains=[True],
        collate_fns=[None],
    )[0]

    tokenizer = BertTokenizer.from_pretrained(config["text_encoder"])

    #### Model ####
    print("Creating model")
    model = ALBEF(
        config=config,
        text_encoder=config["text_encoder"],
        tokenizer=tokenizer,
        init_deit=True,
    )

    model = model.to(device)

    arg_opt = utils.AttrDict(config["optimizer"])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config["schedular"])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint["model"]
        if args.resume:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            start_epoch = checkpoint["epoch"] + 1
        else:
            pos_embed_reshaped = interpolate_pos_embed(
                state_dict["visual_encoder.pos_embed"], model.visual_encoder
            )
            m_pos_embed_reshaped = interpolate_pos_embed(
                state_dict["visual_encoder_m.pos_embed"], model.visual_encoder_m
            )
            state_dict["visual_encoder.pos_embed"] = pos_embed_reshaped
            state_dict["visual_encoder_m.pos_embed"] = m_pos_embed_reshaped
        model.load_state_dict(state_dict)
        print("load checkpoint from %s" % args.checkpoint)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):

        if epoch > 0:
            # epoch 0 will quickly step {warmup_steps} steps, then step once in each epoch
            lr_scheduler.step(epoch + warmup_steps)

        train_stats = train(
            model,
            data_loader,
            optimizer,
            tokenizer,
            epoch,
            warmup_steps,
            device,
            lr_scheduler,
            config,
            max_epoch,
        )
        if utils.is_main_process():
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
            save_obj = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "config": config,
                "epoch": epoch,
            }
            torch.save(save_obj, join(args.output_dir, "checkpoint_%02d.pth" % epoch))

            remove_previous_ckpts(args.output_dir, epoch)
            with open(join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


@utils.rank_zero_only
def remove_previous_ckpts(save_dir, last_epoch):
    # remove the previous checkpoints except the last one
    if os.path.isfile(join(save_dir, "checkpoint_%02d.pth" % last_epoch)):
        for e in range(last_epoch):
            if os.path.isfile(join(save_dir, "checkpoint_%02d.pth" % e)):
                os.remove(join(save_dir, "checkpoint_%02d.pth" % e))


def get_train_arrows():
    # we store the images in arrow format.
    coco_files = [
        "arrow/coco/coco_karpathy_train.arrow",
        "arrow/coco/coco_karpathy_restval.arrow",
    ]
    cc_files = [f"arrow/cc/cc_{i}.arrow" for i in range(32)]
    sbu_files = [f"arrow/sbu/sbu_{i}.arrow" for i in range(9)]
    vg_files = ["arrow/vg/vg_albef.arrow"]
    pretrain_4m_files = coco_files + cc_files + sbu_files + vg_files

    # 12m
    cc12m_files = [f"arrow/cc_12m/cc_12m_{i}.arrow" for i in range(102)]

    train_arrows = pretrain_4m_files + cc12m_files
    return train_arrows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="./never/configs/Pretrain.yaml")
    parser.add_argument(
        "-o", "--override_params", nargs="+", help="override configuration params"
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume", default=False, type=bool)
    parser.add_argument("--output_dir", default="Pretrain/")

    # env setting
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--distributed", default=True, type=bool)

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = utils.load_config(args.config, args.override_params, print_conf=False)

    # generate arrow paths for pretraining:
    config["train_file"] = get_train_arrows()
    utils.save_config(config, join(args.output_dir, "config.yaml"))

    main(args, config)
