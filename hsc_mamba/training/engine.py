import os
import torch
import csv
import time
import glob
import math
import soundfile as sf
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from loguru import logger
from ..utils import util_engine, functions
from ..utils.decorators import *
from .checkpointing import average_validation_checkpoints

PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, ".."))
def _log_split_weights(lightning_module):
    for stage, weight in lightning_module.model.separator.get_split_weights().items():
        lightning_module.log(
            f"Split_Weight/{stage}",
            weight.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )


def _combine_training_losses(main_loss, stage_losses, num_spks):
    alpha = 0.4
    stage_weights = (0.1, 0.2, 0.3, 0.4)
    weighted_aux_loss = sum(
        weight * loss
        for weight, loss in zip(stage_weights, stage_losses)
    )
    return ((1.0 - alpha) * main_loss + alpha * weighted_aux_loss) / num_spks


def _compute_validation_main_loss(
    criterion,
    final_output,
    input_sizes,
    targets,
    num_spks,
):
    main_loss = criterion(
        estims=final_output,
        input_sizes=input_sizes,
        target_attr=targets,
    )
    return main_loss / num_spks


class Engine(object):
    def __init__(self, args, config, model, dataloaders, criterions, optimizers, schedulers, gpuid, device):
        self.args = args
        self.config = config
        self.model = model
        self.dataloaders = dataloaders
        self.criterions = criterions
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.gpuid = gpuid
        self.device = device
        
        self.engine_mode = args.engine_mode
        self.out_wav_dir = args.out_wav_dir
        self.model_dir = PACKAGE_DIR
        self.output_dir = self._resolve_output_dir()
        self.log_dir = os.path.join(self.output_dir, "logs")
        self.results_dir = os.path.join(self.output_dir, "metrics")
        self.separated_audio_dir = os.path.join(self.output_dir, "separated_audios")
        self.checkpoint_dir = self._resolve_checkpoint_dir()
        for directory in (
            self.log_dir,
            self.checkpoint_dir,
            self.results_dir,
            self.separated_audio_dir,
        ):
            os.makedirs(directory, exist_ok=True)

    def _resolve_output_dir(self):
        output_dir = self.config.get("dataset", {}).get("output_dir")
        if output_dir is None:
            dataset_config = self.config.get("dataset", {})
            output_dir = os.path.join(
                "outputs",
                dataset_config.get("condition", "unknown"),
                dataset_config.get("name", dataset_config.get("data_type", "default")),
            )
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(PROJECT_DIR, output_dir)
        return output_dir

    def _resolve_checkpoint_dir(self):
        checkpoint_dir = self.config.get("dataset", {}).get("weight_dir")
        if checkpoint_dir is None:
            return os.path.join(self.output_dir, "weights")
        if not os.path.isabs(checkpoint_dir):
            checkpoint_dir = os.path.join(PROJECT_DIR, checkpoint_dir)
        return checkpoint_dir

    def run(self):
        class _InnerLightningModule(pl.LightningModule):
            def __init__(self, parent_engine):
                super().__init__()
                self.parent = parent_engine
                self.config = parent_engine.config
                self.model = parent_engine.model
                self.engine_mode = parent_engine.engine_mode
                self.out_wav_dir = parent_engine.out_wav_dir
                
                self.PIT_SISNR_mag_loss, self.PIT_Hybrid_loss, self.PIT_SISNRi_loss, self.PIT_SDRi_loss = parent_engine.criterions
                self.main_optimizer = parent_engine.optimizers[0]
                self.main_scheduler, self.warmup_scheduler = parent_engine.schedulers
                
                self.model_ema = None
                self.loaded_ema_state = None
                self.valid_sisnri_total = 0.0
                self.test_idx = 0
                self.train_loss_total = 0.0
                self.train_sample_count = 0
                self.valid_loss_total = 0.0
                self.valid_sample_count = 0

            def _distributed_average(self, total, count):
                stats = torch.tensor(
                    [float(total), float(count)], device=self.device, dtype=torch.float64
                )
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
                return float(stats[0].item() / stats[1].item()) if stats[1] > 0 else None

            # 强制将 EMA 权重存入 checkpoint
            def on_save_checkpoint(self, checkpoint):
                if self.model_ema is not None:
                    checkpoint["ema_state_dict"] = self.model_ema.ema.state_dict()

            # 从 checkpoint 中提取 EMA 权重
            def on_load_checkpoint(self, checkpoint):
                state_dict = checkpoint.get("state_dict", {})
                keys_to_delete = [k for k in state_dict.keys() if "total_ops" in k or "total_params" in k]
                for k in keys_to_delete:
                    del state_dict[k]
                
                if "ema_state_dict" in checkpoint:
                    self.loaded_ema_state = checkpoint["ema_state_dict"]
                    logger.info("✅ Found EMA weights in checkpoint.")

            def on_train_start(self):
                self.model_ema = util_engine.ModelEMA(self.model, decay=0.999, device=self.device)
                if self.loaded_ema_state is not None:
                    self.model_ema.ema.load_state_dict(self.loaded_ema_state)

            def on_train_epoch_start(self):
                self.train_loss_total = 0.0
                self.train_sample_count = 0
                completed_epoch = int(self.current_epoch)
                if (
                    hasattr(self.main_scheduler, "will_decay_after")
                    and self.main_scheduler.will_decay_after(completed_epoch)
                ):
                    checkpoint_path = os.path.join(
                        self.parent.checkpoint_dir,
                        f"pre_lr_decay_epoch{completed_epoch}.ckpt",
                    )
                    self.trainer.save_checkpoint(checkpoint_path)
                    if self.trainer.is_global_zero:
                        logger.info(
                            f"Saved epoch {completed_epoch} weights before LR decay: "
                            f"{os.path.basename(checkpoint_path)}"
                        )
                if hasattr(self.model, "set_epoch"):
                    self.model.set_epoch(int(self.current_epoch + 1))
                _log_split_weights(self)
                    
            def training_step(self, batch, batch_idx):
                input_sizes, mixture, src, _ = batch
                nnet_input = functions.apply_cmvn(mixture) if self.config['engine']['mvn'] else mixture
                
                if self.current_epoch == 0:
                    self.warmup_scheduler.step()

                estim_src, estim_src_bn = self.model(nnet_input)

                cur_loss_s_bn = []
                for idx, estim_src_value in enumerate(estim_src_bn):
                    loss_val = self.PIT_SISNR_mag_loss(estims=estim_src_value, idx=idx, input_sizes=input_sizes, target_attr=src)
                    cur_loss_s_bn.append(loss_val)
                    self.log(
                        f"Aux_Loss/Train_Stage_{idx}",
                        loss_val.detach() / self.config['model']['num_spks'],
                        on_step=False,
                        on_epoch=True,
                    )
                    
                cur_loss_s = self.PIT_Hybrid_loss(estims=estim_src, input_sizes=input_sizes, target_attr=src)
                cur_loss = _combine_training_losses(
                    cur_loss_s,
                    cur_loss_s_bn,
                    self.config['model']['num_spks'],
                )
                
                if not torch.isfinite(cur_loss).all():
                    logger.error(f"🚨 Batch {batch_idx}: Final Loss is NaN/Inf. Skipped.")
                    return None

                self.log("Step_Loss/Train_Total", cur_loss, on_step=True, on_epoch=False)
                self.log("Step_Loss/Train_Hybrid", cur_loss_s / self.config['model']['num_spks'], on_step=True, on_epoch=False)
                self.log("Epoch_Loss/Train", cur_loss, on_step=False, on_epoch=True, prog_bar=True)
                batch_size = int(mixture.shape[0])
                self.train_loss_total += float(cur_loss.detach().item()) * batch_size
                self.train_sample_count += batch_size
                
                return cur_loss

            def on_train_batch_end(self, outputs, batch, batch_idx):
                if self.model_ema is not None:
                    self.model_ema.update(self.model)
                if batch_idx > 0 and batch_idx % 1000 == 0:
                    torch.cuda.empty_cache()

            def on_validation_epoch_start(self):
                self.valid_loss_total = 0.0
                self.valid_sample_count = 0
                self.valid_sisnri_total = 0.0
                if self.model_ema is not None:
                    self.model_ema.swap_weights(self.model)

            def validation_step(self, batch, batch_idx):
                input_sizes, mixture, src, _ = batch
                nnet_input = functions.apply_cmvn(mixture) if self.config['engine']['mvn'] else mixture
                
                estim_src, _ = self.model(nnet_input)
                cur_loss_s = _compute_validation_main_loss(
                    self.PIT_Hybrid_loss,
                    estim_src,
                    input_sizes,
                    src,
                    self.config['model']['num_spks'],
                )
                
                self.log("Epoch_Loss/Valid", cur_loss_s, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                score, _ = self.PIT_SISNRi_loss(
                    estims=estim_src, mixture=mixture, input_sizes=input_sizes,
                    target_attr=src, eps=1.0e-15,
                )
                score = score / self.config['model']['num_spks']
                if not torch.isfinite(score):
                    raise RuntimeError('Validation SI-SNRi is not finite; checkpoint selection aborted.')
                batch_size = int(mixture.shape[0])
                self.log('valid_sisnri', score, on_step=False, on_epoch=True,
                         sync_dist=True, batch_size=batch_size)
                self.valid_sisnri_total += float(score.detach().item()) * batch_size
                self.valid_loss_total += float(cur_loss_s.detach().item()) * batch_size
                self.valid_sample_count += batch_size
                return cur_loss_s

            def on_validation_epoch_end(self):
                if self.model_ema is not None:
                    self.model_ema.swap_weights(self.model)

                if self.trainer.sanity_checking:
                    return

                ep = self.current_epoch + 1
                train_loss = self._distributed_average(
                    self.train_loss_total, self.train_sample_count
                )
                valid_loss = self._distributed_average(
                    self.valid_loss_total, self.valid_sample_count
                )
                valid_score = self._distributed_average(
                    self.valid_sisnri_total, self.valid_sample_count
                )
                if self.trainer.is_global_zero:
                    logger.info(f'Epoch {ep} | train_loss={train_loss} | '
                                f'valid_loss={valid_loss} | valid_SI-SNRi={valid_score}')

            def on_test_start(self):
                paths = glob.glob(os.path.join(self.parent.checkpoint_dir, 'best_valid_*.ckpt'))
                state, selected = average_validation_checkpoints(paths, count=5)
                self.model.load_state_dict(state, strict=True)
                logger.info(f'Testing average of validation Top-5 EMA states: {selected}')

            def on_test_epoch_start(self):
                self.csv_file_sisnr = open(os.path.join(self.parent.results_dir, 'test_SISNRi_value.csv'), 'w', newline='')
                self.csv_file_sdr = open(os.path.join(self.parent.results_dir, 'test_SDRi_value.csv'), 'w', newline='')
                self.writer_sisnr = csv.writer(self.csv_file_sisnr, quotechar='|', quoting=csv.QUOTE_MINIMAL)
                self.writer_sdr = csv.writer(self.csv_file_sdr, quotechar='|', quoting=csv.QUOTE_MINIMAL)
                self.test_idx = 0
                self.test_sisnri_total = 0.0
                self.test_sample_count = 0

            def test_step(self, batch, batch_idx):
                input_sizes, mixture, src, key = batch
                if len(key) > 1:
                    raise ValueError("Test batch size must be one!")
                    
                try:
                    estim_src, _ = self.model(mixture)
                    
                    cur_loss_SISNRi, cur_loss_SISNRi_src = self.PIT_SISNRi_loss(estims=estim_src, mixture=mixture, input_sizes=input_sizes, target_attr=src, eps=1.0e-15)
                    cur_loss_SDRi, cur_loss_SDRi_src = self.PIT_SDRi_loss(estims=estim_src, mixture=mixture, input_sizes=input_sizes, target_attr=src)
                    if not torch.isfinite(cur_loss_SISNRi).all() or not math.isfinite(float(cur_loss_SDRi)):
                        raise RuntimeError('Test metric is NaN or Inf.')
                except Exception as error:
                    logger.exception(f'Test failed for utterance {key}; evaluation stopped.')
                    raise RuntimeError(f'Test failed for utterance {key}') from error

                spks = self.config['model']['num_spks']
                self.writer_sisnr.writerow([key[0][:-4]] + [cur_loss_SISNRi_src[i].item() for i in range(spks)])
                self.writer_sdr.writerow([key[0][:-4]] + [cur_loss_SDRi_src[i].item() for i in range(spks)])
                
                if self.engine_mode == "test_save":
                    wav_dir = self.out_wav_dir if self.out_wav_dir else self.parent.separated_audio_dir
                    os.makedirs(wav_dir, exist_ok=True)
                    mix_np = torch.squeeze(mixture).float().cpu().numpy()
                    sf.write(os.path.join(wav_dir, f"{key[0][:-4]}_mixture.wav"), 0.5 * mix_np / max(abs(mix_np)), 8000)
                    for i in range(spks):
                        src_np = torch.squeeze(estim_src[i]).float().cpu().numpy()
                        sf.write(os.path.join(wav_dir, f"{key[0][:-4]}_out_{i}.wav"), 0.5 * src_np / max(abs(src_np)), 8000)
                        
                self.test_idx += 1
                test_sisnri = cur_loss_SISNRi.item() / spks
                self.test_sisnri_total += test_sisnri
                self.test_sample_count += 1
                self.log("Metrics/Test_SISNRi", test_sisnri, on_epoch=True)
                self.log("Metrics/Test_SDRi", cur_loss_SDRi.item() / spks, on_epoch=True)

            def on_test_epoch_end(self):
                self.csv_file_sisnr.close()
                self.csv_file_sdr.close()
                test_sisnri = self._distributed_average(
                    self.test_sisnri_total, self.test_sample_count
                )
                if self.trainer.is_global_zero and test_sisnri is not None:
                    logger.info(f"Test Summary | SI-SNRi={test_sisnri:.4f} dB")

            def configure_optimizers(self):
                return {
                    "optimizer": self.main_optimizer,
                    "lr_scheduler": {
                        "scheduler": self.main_scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                        "name": "epoch_lr",
                    },
                }

            def lr_scheduler_step(self, scheduler, metric):
                scheduler.step()

        lightning_module = _InnerLightningModule(self)

        dataset_config = self.config.get("dataset", {})
        dataset_name = dataset_config.get("name", dataset_config.get("data_type", "dataset"))
        run_name = self.config.get("engine", {}).get(
            "wandb_name",
            time.strftime("HSC-Mamba_%Y%m%d_%H%M%S"),
        )
        os.environ["WANDB_MODE"] = "offline"
        wandb_logger = WandbLogger(
            project="HSC-Mamba",
            name=f"{dataset_name}_{run_name}",
            save_dir=self.log_dir,
            config=self.config,
            log_model=False,
            offline=True,
        )

        checkpoint_dir = self.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_last = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="last_{epoch}",
            monitor="epoch",
            mode="max",
            save_top_k=1,
            auto_insert_metric_name=False
        )

        checkpoint_valid = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best_valid_epoch{epoch}_score{valid_sisnri:.8f}',
            monitor='valid_sisnri', mode='max', save_top_k=5,
            auto_insert_metric_name=False, save_on_train_epoch_end=False,
        )
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        if isinstance(self.gpuid, int):
            devices = [self.gpuid]
        elif isinstance(self.gpuid, (list, tuple)):
            devices = []
            for d in self.gpuid:
                if isinstance(d, (list, tuple)):
                    devices.extend([int(x) for x in d])
                else:
                    devices.append(int(d))
        else:
            devices = "auto"

        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=[checkpoint_last, checkpoint_valid, lr_monitor],
            max_epochs=self.config['engine']['max_epoch'],
            accelerator="gpu",
            precision=self.config['engine'].get('precision', 'bf16-mixed'),
            devices=devices,
            strategy="ddp_find_unused_parameters_true" if isinstance(devices, list) and len(devices) > 1 else "auto",
            gradient_clip_val=self.config['engine']['clip_norm'],
            check_val_every_n_epoch=1,
            enable_model_summary=False,  
        )

        if "test" in self.engine_mode:
            logger.info("🧪 Mode: TEST | Evaluation Start...")
            
            trainer.test(model=lightning_module, dataloaders=self.dataloaders['test'])
        else:
            logger.info("🚀 Training Started (Lightning Managed)")
            
            last_ckpts = glob.glob(os.path.join(checkpoint_dir, "last_*.ckpt"))
            if last_ckpts:
                last_ckpts.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
                ckpt_path = last_ckpts[-1]
                logger.info(f"🔄 Auto-resuming from {os.path.basename(ckpt_path)}")
            else:
                ckpt_path = None
            
            trainer.fit(
                model=lightning_module,
                train_dataloaders=self.dataloaders['train'],
                val_dataloaders=self.dataloaders['valid'],
                ckpt_path=ckpt_path
            )
            
            logger.info("🎉 All training tasks finished. Running final test...")
            trainer.test(model=lightning_module, dataloaders=self.dataloaders['test'])
