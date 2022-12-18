import datetime
import logging
import time
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from smoke.utils.metric_logger import MetricLogger
from smoke.utils.comm import get_world_size


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
        cfg,
        distributed,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
):
    logger = logging.getLogger("smoke.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter=" ")
    max_epoch = cfg.SOLVER.MAX_EPOCH
    start_iter = arguments["iteration"]
    start_epoch = arguments["epoch"]
    accumulate = max(round(64 / cfg.SOLVER.IMS_PER_BATCH), 1)  # accumulate n times before optimizer update (bs 64)

    start_training_time = time.time()
    nb = len(data_loader)
   
    for epoch in range(start_epoch, max_epoch):  # epoch ------------------------------------------------------------------
        model.train()
        end = time.time()
        pbar = tqdm(enumerate(data_loader), total=nb)  # progress bar
        for i, data in pbar:  # batch -------------------------------------------------------------
            iteration = i + nb * epoch
            arguments["iteration"] = iteration
            
            images = data["images"].to(device)
            targets = [target.to(device) for target in data["targets"]]
    
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
    
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)
    
            # Backward
            losses *= cfg.SOLVER.IMS_PER_BATCH / 64  # scale loss
            losses.backward()
            
            # Optimize
            if iteration % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()  
                

        # Update scheduler
        scheduler.step()  
        arguments['epoch'] = epoch
                
        epoch_time = time.time() - end
        end = time.time()
        meters.update(time=epoch_time)

        eta_seconds = meters.time.global_avg * (max_epoch - epoch)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        print('')
        if epoch % 1 == 0 or epoch + 1 == max_epoch:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "epoch: {epoch}",
                        "{meters}",
                        "lr: {lr:.8f}",
                        "max men: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    epoch=epoch,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                )
            )
                    
            checkpointer.save("checkpoint/model_{:07d}".format(epoch), **arguments)
              
        # reset meters
        meters.reset_all()         
        # fixme: do we need checkpoint_period here
        if epoch + 1 == max_epoch:
            checkpointer.save("model_final", **arguments)            
        
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / ep)".format(
            total_time_str, total_training_time / (max_epoch)
        )
    )

def do_train_(
        cfg,
        distributed,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
):
    logger = logging.getLogger("smoke.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter=" ")
    max_iter = cfg.SOLVER.MAX_ITERATION
    start_iter = arguments["iteration"]

    model.train()
    start_training_time = time.time()
    end = time.time()

    for data, iteration in zip(data_loader, range(start_iter, max_iter)):
        data_time = time.time() - end
        iteration += 1
        arguments["iteration"] = iteration

        images = data["images"].to(device)
        targets = [target.to(device) for target in data["targets"]]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 10 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.8f}",
                        "max men: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                )
            )
        # fixme: do we need checkpoint_period here
        if iteration in cfg.SOLVER.STEPS:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
        # todo: add evaluations here
        # if iteration % evaluate_period == 0:
        # test_net.main()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / ep)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
