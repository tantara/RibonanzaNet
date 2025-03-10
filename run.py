import argparse
import json
import os
import pickle
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from dataset import Custom_Collate_Obj, RNADataset
from functions import (
    CSVLogger,
    dataset_dropout,
    drop_pk5090_duplicates,
    load_config_from_yaml,
)
from network import RibonanzaNet
from ranger import Ranger
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

# from torch.cuda.amp import GradScaler
# from torch import autocast


def train(args):
    # load config
    config = load_config_from_yaml(args.config_path)

    # init accelerator & wandb
    accelerator = Accelerator(mixed_precision="fp16")
    if accelerator.is_local_main_process:
        wandb.init(
            project="ribonanza",
            config=config,
        )

    # init logger
    logger = CSVLogger(
        ["epoch", "train_loss", "val_loss"], f"logs/fold{config.fold}.csv"
    )

    # init seed
    start_time = time.time()
    np.random.seed(0)

    # set environment variables
    # os.environ["POLARS_MAX_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    # create directories
    os.system("mkdir logs")
    os.system("mkdir models")
    os.system("mkdir oofs")

    # load data
    data = pl.read_csv(f"{config.input_dir}/train_data.csv")
    pl.Config.set_fmt_str_lengths(100)
    data = drop_pk5090_duplicates(data)

    print("before dropping duplicates data shape is:", data.shape)
    data = data.unique(subset=["sequence_id", "experiment_type"]).sort(
        ["sequence_id", "experiment_type"]
    )
    print("after dropping duplicates data shape is:", data.shape)
    # data=data.sort(["signal_to_noise"],descending=True).unique(subset=["sequence_id", "experiment_type"]).sort(["sequence_id", "experiment_type"])

    n_sequences_total = len(data) // 2
    # get necessary data as lists and numpy arrays
    seq_length = 206

    # filter out a sequence if min SN is smaller than 1
    SN = data["signal_to_noise"].to_numpy().astype("float32").reshape(-1, 2)
    SN = SN.min(-1)
    SN = np.repeat(SN, 2)
    print("before filtering data shape is:", data.shape)
    dirty_data = data.filter((SN <= 1))
    data = data.filter(SN > 1)
    print("after filtering data shape is:", data.shape)
    print("dirty data shape is:", dirty_data.shape)

    # get sequences where one of 2A3/DMS has SN>1
    dirty_SN = dirty_data["signal_to_noise"].to_numpy().astype("float32").reshape(-1, 2)
    dirty_SN = dirty_SN.max(-1)
    dirty_SN = np.repeat(dirty_SN, 2)
    dirty_data = dirty_data.filter(dirty_SN > 1)
    print("after filtering dirty_data shape is:", dirty_data.shape)

    label_names = [
        "reactivity_{:04d}".format(number + 1) for number in range(seq_length)
    ]
    error_label_names = [
        "reactivity_error_{:04d}".format(number + 1) for number in range(seq_length)
    ]

    sequences = data.unique(subset=["sequence_id"], maintain_order=True)[
        "sequence"
    ].to_list()
    sequence_ids = data.unique(subset=["sequence_id"], maintain_order=True)[
        "sequence_id"
    ].to_list()
    labels = (
        data[label_names]
        .to_numpy()
        .astype("float32")
        .reshape(-1, 2, 206)
        .transpose(0, 2, 1)
    )
    errors = (
        data[error_label_names]
        .to_numpy()
        .astype("float32")
        .reshape(-1, 2, 206)
        .transpose(0, 2, 1)
    )
    SN = data["signal_to_noise"].to_numpy().astype("float32").reshape(-1, 2)
    dataset_name = data["dataset_name"].to_list()
    dataset_name = [
        dataset_name[i * 2].replace("2A3", "NULL").replace("DMS", "NULL")
        for i in range(len(data) // 2)
    ]

    data_dict = {
        "sequences": sequences,
        "sequence_ids": sequence_ids,
        "labels": labels,
        "errors": errors,
        "SN": SN,
    }

    # StratifiedKFold on dataset
    kfold = StratifiedKFold(n_splits=config.nfolds, shuffle=True, random_state=0)
    fold_indices = {}
    for i, (train_index, test_index) in enumerate(
        kfold.split(np.arange(len(data) // 2), dataset_name)
    ):
        fold_indices[i] = (train_index, test_index)

    train_indices = fold_indices[config.fold][0]
    val_indices = fold_indices[config.fold][1]

    # for data scaling experiments
    if config.use_data_percentage < 1:
        print(f"Only using {config.use_data_percentage:.02f} of data")
        size = int(config.use_data_percentage * len(train_indices))
        train_indices = np.random.choice(train_indices, size, replace=False)
        print(f"number of sequences in train {len(train_indices)} after subsampling")

    if config.use_dirty_data:
        print("using sequences where one of 2A3/DMS has SN>1")
        data_dict["sequences"] += dirty_data.unique(
            subset=["sequence_id"], maintain_order=True
        )["sequence"].to_list()
        data_dict["sequence_ids"] += dirty_data.unique(
            subset=["sequence_id"], maintain_order=True
        )["sequence_id"].to_list()
        data_dict["labels"] = np.concatenate(
            [
                data_dict["labels"],
                dirty_data[label_names]
                .to_numpy()
                .astype("float32")
                .reshape(-1, 2, 206)
                .transpose(0, 2, 1),
            ]
        )
        data_dict["errors"] = np.concatenate(
            [
                data_dict["errors"],
                dirty_data[error_label_names]
                .to_numpy()
                .astype("float32")
                .reshape(-1, 2, 206)
                .transpose(0, 2, 1),
            ]
        )
        data_dict["SN"] = np.concatenate(
            [
                data_dict["SN"],
                dirty_data["signal_to_noise"]
                .to_numpy()
                .astype("float32")
                .reshape(-1, 2),
            ]
        )

        print(f"number of sequences in train {len(train_indices)}")
        train_indices = np.concatenate(
            [
                train_indices,
                np.arange(len(data) // 2, len(data) // 2 + len(dirty_data) // 2),
            ]
        )
        print(
            f"number of sequences in train {len(train_indices)} after using dirty data"
        )

    if hasattr(config, "dataset2drop"):
        train_indices = dataset_dropout(
            dataset_name, train_indices, config.dataset2drop
        )

    # if accelerator.is_local_main_process:
    #     pl.Config.set_fmt_str_lengths(100)
    #     print(data[np.concatenate([train_indices*2,train_indices*2+1])]['dataset_name'].value_counts(sort=True))
    #     print(data[np.concatenate([val_indices*2,val_indices*2+1])]['dataset_name'].value_counts(sort=True))
    # print(data[val_indices*2]['dataset_name'].value_counts(sort=True))
    # train_indices=np.concatenate([train_indices,np.arange(len(train_indices),len(train_indices)+len(dirty_data)//2)])

    val_datasets_names = data[np.concatenate([val_indices * 2])][
        "dataset_name"
    ].to_list()
    with open("oofs/val_dataset_names.p", "wb+") as f:
        pickle.dump(val_datasets_names, f)

    del data
    del dirty_data

    print(f"train shape: {train_indices.shape}")
    print(f"val shape: {val_indices.shape}")

    val_dataset_name = [dataset_name[i] for i in val_indices]

    # pl_train=pl.read_parquet()

    train_dataset = RNADataset(
        train_indices, data_dict, k=config.k, flip=config.use_flip_aug
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=Custom_Collate_Obj(),
        num_workers=min(config.batch_size, 16),
    )

    sample = train_dataset[0]

    val_dataset = RNADataset(val_indices, data_dict, train=False, k=config.k)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        collate_fn=Custom_Collate_Obj(),
        num_workers=min(config.batch_size, 16),
    )

    print(accelerator.distributed_type)

    model = RibonanzaNet(config)  # .cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {total_params}")

    optimizer = Ranger(
        model.parameters(), weight_decay=config.weight_decay, lr=config.learning_rate
    )

    criterion = torch.nn.L1Loss(reduction="none")
    val_criterion = torch.nn.L1Loss(reduction="none")

    cos_epoch = int(config.epochs * 0.75) - 1
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        (config.epochs - cos_epoch)
        * len(train_loader)
        // config.gradient_accumulation_steps,
    )

    model, optimizer, train_loader, val_loader, lr_schedule = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_schedule
    )

    best_val_loss = np.inf
    step = 0
    for epoch in range(config.epochs):
        # training loop
        model.train()

        tbar = tqdm(train_loader)
        total_loss = 0
        for idx, batch in enumerate(tbar):
            step += 1
            src = batch["sequence"]  # .cuda()
            masks = batch["masks"].bool()  # .cuda()
            labels = batch["labels"]  # .cuda()
            SN = batch["SN"]

            bs = len(labels)
            # batch_attention_mask=batch['attention_mask'].unsqueeze(1)[:,:,:src.shape[-1],:src.shape[-1]]

            loss_masks = batch["loss_masks"]  # .cuda()
            errors = batch["errors"]  # .cuda()#.un
            # SSH FS test
            SN = SN.reshape(SN.shape[0], 1, SN.shape[1]) >= 1
            loss_masks = loss_masks * SN

            # batch_attention_mask=batch['attention_mask']
            # batch_attention_mask=torch.stack([batch_attention_mask[:,:src.shape[-1],:src.shape[-1]],bpp],1)
            SN = batch["SN"]
            with accelerator.autocast():
                output = model(src, masks)
                loss = criterion(output, labels)  # *loss_weight BxLxC
                loss = loss[loss_masks]
                loss = loss.mean()

            accelerator.backward(loss / config.gradient_accumulation_steps)

            # loss.backward()
            if (idx + 1) % config.gradient_accumulation_steps == 0:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()
                if epoch > cos_epoch:
                    lr_schedule.step()

            local_loss = loss.item()
            total_loss += local_loss
            tbar.set_description(f"Epoch {epoch + 1} Loss: {total_loss/(idx+1)}")

            if accelerator.is_local_main_process:
                if step % 1000 == 0:
                    wandb.log(
                        {
                            "train_loss": total_loss / (idx + 1),
                            "local_loss": local_loss,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        },
                        step=step,
                    )

        train_loss = total_loss / (idx + 1)
        if epoch == cos_epoch:
            torch.save(
                accelerator.unwrap_model(model).state_dict(),
                f"models/model{config.fold}_pl_only.pt",
            )
        torch.save(
            accelerator.unwrap_model(optimizer).state_dict(),
            f"models/optimizer{config.fold}.pt",
        )

        # validation loop
        model.eval()

        tbar = tqdm(val_loader)
        val_loss = 0
        preds = []
        gts = []
        print("doing val")
        val_loss_masks = []
        for idx, batch in enumerate(tbar):
            src = batch["sequence"]  # .cuda()
            masks = batch["masks"].bool()  # .cuda()
            labels = batch["labels"]  # .cuda()
            bs = len(labels)
            loss_masks = batch["loss_masks"]  # .cuda()
            # bpp=batch['bpps'].float()#.cuda().float()
            # batch_attention_mask=batch['attention_mask']
            # batch_attention_mask=torch.stack([batch_attention_mask[:,:src.shape[-1],:src.shape[-1]],bpp],1)

            # flipped
            # batch_attention_mask=batch['attention_mask'].unsqueeze(1)[:,:,:src.shape[-1],:src.shape[-1]]
            src_flipped = src.clone()

            length = batch["length"]
            for batch_idx in range(len(src)):
                src_flipped[batch_idx, : length[batch_idx]] = src_flipped[
                    batch_idx, : length[batch_idx]
                ].flip(0)

            # with accelerator.autocast():
            with torch.no_grad():
                with accelerator.autocast():
                    output = model(src, masks)
                    if config.use_flip_aug:
                        flipped_output = model(src_flipped, masks)
                        for batch_idx in range(len(flipped_output)):
                            flipped_output[batch_idx, : length[batch_idx]] = (
                                flipped_output[batch_idx, : length[batch_idx]].flip(0)
                            )

                        output = (flipped_output + output) / 2
            loss = val_criterion(output, labels)[loss_masks]

            L = src.shape[1]
            to_pad = seq_length - L
            # output=output#[loss_masks]
            # labels=labels#[loss_masks]

            output = F.pad(output, (0, 0, 0, to_pad), value=0)
            labels = F.pad(labels, (0, 0, 0, to_pad), value=0)
            loss_masks = F.pad(loss_masks, (0, 0, 0, to_pad), value=0)

            all_output = accelerator.gather(output)
            all_labels = accelerator.gather(labels)
            all_masks = accelerator.gather(loss_masks)

            preds.append(all_output)
            gts.append(all_labels)
            val_loss_masks.append(all_masks)

            loss = loss.mean()
            # loss=torch.sqrt(loss)
            val_loss += loss.item()

            tbar.set_description(f"Epoch {epoch + 1} Val Loss: {val_loss/(idx+1)}")

        # val_loss=val_loss/len(tbar)

        preds = torch.cat(preds)
        gts = torch.cat(gts)
        val_loss_masks = torch.cat(val_loss_masks)

        if accelerator.is_local_main_process:
            val_loss = (
                val_criterion(preds[val_loss_masks], gts[val_loss_masks]).mean().item()
            )

            logger.log([epoch, train_loss, val_loss])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    accelerator.unwrap_model(model).state_dict(),
                    f"models/model{config.fold}.pt",
                )
                # accelerator.save_model(model, f"models/model{config.fold}.pt")
                data_dict = {
                    "preds": preds.cpu().numpy(),
                    "gts": gts.cpu().numpy(),
                    "val_loss_masks": val_loss_masks.cpu().numpy(),
                }

                # Save to pickle file
                with open(f"oofs/{config.fold}.pkl", "wb+") as file:
                    pickle.dump(data_dict, file)

        if accelerator.is_local_main_process:
            wandb.log(
                {
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                },
                step=step,
            )

    if accelerator.is_local_main_process:
        torch.save(
            accelerator.unwrap_model(model).state_dict(),
            f"models/model{config.fold}_lastepoch.pt",
        )

        end_time = time.time()
        elapsed_time = end_time - start_time

        with open("run_stats.json", "w") as file:
            json.dump({"total_execution_time": elapsed_time}, file, indent=4)

    if accelerator.is_local_main_process:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/pairwise.yaml")
    args = parser.parse_args()

    train(args)
