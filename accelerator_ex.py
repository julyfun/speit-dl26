if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from torch import nn
import torch.optim as optim
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
import pickle
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env_runner.real_pusht_image_runner import RealPushTImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.common.lr_decay import param_groups_lrd
from accelerate import Accelerator

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetTimmWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetTimmPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetTimmPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state

        if 'timm' in cfg.policy.obs_encoder._target_:
            if cfg.training.layer_decay < 1.0:
                assert not cfg.policy.obs_encoder.use_lora
                assert not cfg.policy.obs_encoder.share_rgb_model
                obs_encorder_param_groups = param_groups_lrd(self.model.obs_encoder,
                                                             shape_meta=cfg.shape_meta,
                                                             weight_decay=cfg.optimizer.encoder_weight_decay,
                                                             no_weight_decay_list=self.model.obs_encoder.no_weight_decay(),
                                                             layer_decay=cfg.training.layer_decay)
                count = 0
                for group in obs_encorder_param_groups:
                    count += len(group['params'])
                if cfg.policy.obs_encoder.feature_aggregation == 'map':
                    obs_encorder_param_groups.extend([{'params': self.model.obs_encoder.attn_pool.parameters()}])
                    for _ in self.model.obs_encoder.attn_pool.parameters():
                        count += 1
                print(f'obs_encorder params: {count}')
                param_groups = [{'params': self.model.model.parameters()}]
                param_groups.extend(obs_encorder_param_groups)
            else:
                obs_encorder_lr = cfg.optimizer.lr
                if cfg.policy.obs_encoder.pretrained and not cfg.policy.obs_encoder.use_lora:
                    obs_encorder_lr *= cfg.training.encoder_lr_coefficient
                    print('==> reduce pretrained obs_encorder\'s lr')
                obs_encorder_params = list()
                for param in self.model.obs_encoder.parameters():
                    if param.requires_grad:
                        obs_encorder_params.append(param)
                print(f'obs_encorder params: {len(obs_encorder_params)}')
                param_groups = [
                    {'params': self.model.model.parameters()},
                    {'params': obs_encorder_params, 'lr': obs_encorder_lr}
                ]
            optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
            optimizer_cfg.pop('_target_')
            if 'encoder_weight_decay' in optimizer_cfg.keys():
                optimizer_cfg.pop('encoder_weight_decay')
            self.optimizer = torch.optim.AdamW(
                params=param_groups,
                **optimizer_cfg
            )
        else:
            optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
            optimizer_cfg.pop('encoder_weight_decay')
            # hack: use larger learning rate for multiple gpus
            accelerator = Accelerator()
            cuda_count = accelerator.num_processes
            print(f"Number of available CUDA devices: {cuda_count}.")
            print(f"Original learning rate: {optimizer_cfg['lr']}")
            optimizer_cfg['lr'] = optimizer_cfg['lr'] * cuda_count
            print(f"Updated learning rate: {optimizer_cfg['lr']}")
            self.optimizer = hydra.utils.instantiate(
                optimizer_cfg, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        accelerator = Accelerator(log_with='wandb')
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # [mkdir if output_dir does not exist]
        if accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)

        # normalizer = dataset.get_normalizer()
        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')

        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            with open(normalizer_path, 'wb') as f:
                pickle.dump(normalizer, f)

        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # TODO: remove this hacking for simulation experiments
        # Hack env_runner
        env_runner = RealPushTImageRunner(output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # accelerator
        train_dataloader, val_dataloader, self.model, self.ema_model, self.optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.ema_model, self.optimizer, lr_scheduler
        )
        if accelerator.state.num_processes > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                accelerator.unwrap_model(self.model),
                device_ids=[self.model.device],
                find_unused_parameters=True
            )

        # device transfer
        # device = self.model.device
        # if self.ema_model is not None:
            # raise NotImplementedError
            # self.ema_model = accelerator.unwrap_model(self.ema_model)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        # batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        accelerator.backward(loss)

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(accelerator.unwrap_model(self.model))

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch

                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if cfg.task.dataset.val_ratio > 0 and (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            num = 0
                            loss = None
                            for batch_idx, batch in enumerate(tepoch):
                                if loss is None:
                                    loss = self.model(batch)
                                else:
                                    loss += self.model(batch)
                                num += 1

                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break

                        loss = loss / num
                        all_loss = accelerator.gather_for_metrics(loss)
                        if accelerator.is_main_process:
                            step_log['val_loss'] = all_loss.mean().item()

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        batch = train_sampling_batch
                        obs_dict = batch['obs']
                        gt_action = batch['action']

                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']

                        all_preds, all_gt = accelerator.gather_for_metrics((pred_action, gt_action))

                        if accelerator.is_main_process:
                            mse = torch.nn.functional.mse_loss(all_preds, all_gt)
                            step_log['train_action_mse_error'] = mse.item()

                accelerator.wait_for_everyone()

                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and accelerator.is_main_process:
                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                    # recover the DDP model
                    self.model = model_ddp

                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        accelerator.end_training()
