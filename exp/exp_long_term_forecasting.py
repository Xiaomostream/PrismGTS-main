from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, visual, LargeScheduler, attn_map
from utils.metrics import metric
import torch
import torch.nn as nn
import os
import time
import re
import warnings
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers.trainer_pt_utils import get_parameter_names

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
       
        H_hat = 0
        self.J = []
        while H_hat < args.pred_len:
            for j in reversed(args.horizon_lengths):
                if H_hat + j <= args.pred_len:
                    H_hat += j
                    self.J.append(j)
                    break
        self.idx = {i for i, h in enumerate(args.horizon_lengths) if h in self.J}

    def _build_model(self):
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = self.model_dict[self.args.model].Model(self.args).to(self.device)
            model = DDP(model, device_ids=[self.args.local_rank], find_unused_parameters=True)
        else:
            self.args.device = self.device
            model = self.model_dict[self.args.model].Model(self.args)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
       
        optim_groups = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and p.requires_grad],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        model_optim = torch.optim.AdamW(
            optim_groups,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=1e-8,
        )
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _select_physical_criterion(self):
        """PrismGTS: Physical constraint loss for semantic disentanglement."""
        z_dim = self.args.d_model
        lambda_smooth = getattr(self.args, 'lambda_smooth', 0.1)
        lambda_periodic = getattr(self.args, 'lambda_periodic', 0.1)
        lambda_mi = getattr(self.args, 'lambda_mi', 0.0)
        
        self.physical_criterion = PhysicalConstraintLossV3(
            z_dim=z_dim,
            lambda_smooth=lambda_smooth,
            lambda_periodic=lambda_periodic,
            lambda_mi=lambda_mi,
            verbose=True
        ).to(self.device)
        return self.physical_criterion

    def vali(self, vali_data, vali_loader, criterion, flag='train'):
        total_loss = []
        total_count = []
        self.model.eval()
        with torch.no_grad():
            if flag == 'train':
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                   
                    if self.args.head_type == 'prediction':
                        # Handle list format from UTSD dataloader
                        if isinstance(batch_y, list):
                            batch_y = [y.float() for y in batch_y]
                            batch_y_mark = [y_mark.float().to(self.device) for y_mark in batch_y_mark]
                            dec_inp = [torch.zeros_like(y[:, -h:, :]).float() for y, h in zip(batch_y, self.args.horizon_lengths)]
                            dec_inp = [torch.cat([y[:, :self.args.label_len, :], d], dim=1).float().to(self.device) for y, d in zip(batch_y, dec_inp)]
                        else:
                            batch_y = batch_y.float()
                            batch_y_mark = batch_y_mark.float().to(self.device)
                            # decoder input
                            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    else:
                        batch_y = [y.float() for y in batch_y]
                        batch_y_mark = [y_mark.float().to(self.device) for y_mark in batch_y_mark]
                       
                        dec_inp = [torch.zeros_like(y[:, -h:, :]).float() for y, h in zip(batch_y, self.args.horizon_lengths)]
                        dec_inp = [torch.cat([y[:, :self.args.label_len, :], d], dim=1).float().to(self.device) for y, d in zip(batch_y, dec_inp)]
                   
                    # encoder - decoder
                    if self.args.output_attention:
                        model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    # Handle PrismGTS V3 6-output format
                    if self.args.model == 'PrismGTS' and len(model_output) == 6:
                        outputs, recons, _, _, _, _ = model_output
                    elif self.args.model == 'PrismGTS' and len(model_output) == 4:
                        outputs, recons, _, _ = model_output
                    else:
                        outputs, recons = model_output
                       
                    f_dim = -1 if self.args.features == 'MS' else 0  
                    recons = recons[:, :, f_dim:]
                    batch_x_proc = batch_x[:, :, f_dim:]
                    
                    # Align recon length
                    recon_len = recons.shape[1]
                    if recon_len != batch_x_proc.shape[1]:
                        batch_x_proc = batch_x_proc[:, :recon_len, :]
                    
                    recons = recons.detach().cpu()
                    batch_x_proc = batch_x_proc.detach().cpu()
                   
                    if self.args.head_type == 'prediction':
                        # Handle both list and tensor batch_y
                        if isinstance(batch_y, list):
                            pre_loss = torch.tensor(0.)
                            for j, (o, y) in enumerate(zip(outputs, batch_y)):
                                o = o[:, -self.args.horizon_lengths[j]:, f_dim:]
                                y = y[:, -self.args.horizon_lengths[j]:, f_dim:].to(self.device)
                                o = o.detach().cpu()
                                y = y.detach().cpu()
                                pre_loss = pre_loss + criterion(o, y)
                            loss = pre_loss + criterion(recons, batch_x_proc)
                        else:
                            outputs = torch.cat([outputs[j][:, -h:, f_dim:] for j, h in enumerate(self.args.horizon_lengths) if j in self.idx], dim=1)
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            pred = outputs.detach().cpu()
                            true = batch_y.detach().cpu()
                            loss = criterion(pred, true) + criterion(recons, batch_x_proc)
                    else:
                        pre_loss = torch.tensor(0.)
                        for j, (o, y) in enumerate(zip(outputs, batch_y)):
                            o = o[:, -self.args.horizon_lengths[j]:, f_dim:]
                            y = y[:, -self.args.horizon_lengths[j]:, f_dim:].to(self.device)
                            o = o.detach().cpu()
                            y = y.detach().cpu()
                            pre_loss = pre_loss + criterion(o, y)
                        loss = pre_loss + criterion(recons, batch_x_proc)
                   
                    loss = loss.detach().cpu()
                    total_loss.append(loss)
                    total_count.append(batch_x.shape[0])
        
                    torch.cuda.empty_cache()
                   
            elif flag == 'pretrain':
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()
                   
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
               
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                       
                    # encoder - decoder
                    if self.args.output_attention:
                        model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    # Handle PrismGTS V3 6-output format: (y_list, x_recon, x_recon_t, x_recon_s, z_t, z_s)
                    if self.args.model == 'PrismGTS' and len(model_output) == 6:
                        _, outputs, _, _, _, _ = model_output  # Use x_recon for pretrain validation
                    elif self.args.model == 'PrismGTS' and len(model_output) == 4:
                        _, outputs, _, _ = model_output
                    else:
                        outputs, _ = model_output
                        
                    f_dim = -1 if self.args.features == 'MS' else 0            
                    outputs = outputs[:, :, f_dim:]
                    batch_x_proc = batch_x[:, :, f_dim:]
                    
                    # Align lengths
                    recon_len = outputs.shape[1]
                    if recon_len != batch_x_proc.shape[1]:
                        batch_x_proc = batch_x_proc[:, :recon_len, :]

                    pred = outputs.detach().cpu()
                    true = batch_x_proc.detach().cpu()
                   
                    loss = criterion(pred, true)
                    loss = loss.detach().cpu()
                    total_loss.append(loss)
                    total_count.append(batch_x.shape[0])
                   
                    torch.cuda.empty_cache()

        if self.args.use_multi_gpu:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
   
    def pretrain(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # Stage 1: Save best checkpoint to setting/stage1/
        path = os.path.join(self.args.checkpoints, setting, 'stage1')
        if not os.path.exists(path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        # PrismGTS: Initialize physical constraint loss
        if self.args.model == 'PrismGTS':
            self._select_physical_criterion()
            # Freeze MoE in Stage 1 to reduce compute
            model_ref = self.model.module if hasattr(self.model, 'module') else self.model
            if hasattr(model_ref, 'freeze_moe_stage1'):
                model_ref.freeze_moe_stage1()
        
        # ============== Detailed Logging ==============
        print("\n" + "=" * 70)
        print("PrismGTS Stage 1: Pretraining (Reconstruction)")
        print("=" * 70)
        
        # Data loading stats
        print(f"\n[Data Loading]")
        print(f"  Train samples: {len(train_data):,} | batches: {len(train_loader)}")
        print(f"  Vali  samples: {len(vali_data):,} | batches: {len(vali_loader)}")
        print(f"  Test  samples: {len(test_data):,} | batches: {len(test_loader)}")
        print(f"  Batch size: {self.args.batch_size}")
        print(f"  Seq len: {self.args.seq_len} | Pred len: {self.args.pred_len}")
        print(f"  Patch len: {self.args.patch_len} | Stride: {self.args.stride}")
        
        # Parameter stats (after freeze)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n[Model Parameters]")
        print(f"  Total params: {total_params:,}")
        print(f"  Stage 1 trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        
        if self.args.model == 'PrismGTS':
            model_ref = self.model.module if hasattr(self.model, 'module') else self.model
            modules = {
                'semantic_moe': getattr(model_ref, 'semantic_moe', None),
                'trend_encoder': getattr(model_ref, 'trend_encoder', None),
                'seasonal_encoder': getattr(model_ref, 'seasonal_encoder', None),
                'decoder': getattr(model_ref, 'decoder', None),
                'trend_heads': getattr(model_ref, 'trend_heads', None),
                'seasonal_heads': getattr(model_ref, 'seasonal_heads', None),
            }
            print(f"\n  [Module Breakdown]")
            for name, module in modules.items():
                if module is not None:
                    params = sum(p.numel() for p in module.parameters())
                    print(f"    {name}: {params:,}")
        
        # GPU memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"\n[GPU Memory]")
            print(f"  Allocated: {allocated:.0f} MB | Reserved: {reserved:.0f} MB")
        
        # Physical loss config
        if self.args.model == 'PrismGTS' and hasattr(self, 'physical_criterion'):
            print(f"\n[Physical Constraint Loss V3]")
            print(f"  λ_smooth: {self.physical_criterion.lambda_smooth}")
            print(f"  λ_periodic: {self.physical_criterion.lambda_periodic}")
            print(f"  λ_mi: {self.physical_criterion.lambda_mi}")
        
        print("=" * 70 + "\n")
       
        scheduler = LargeScheduler(self.args, model_optim)
        global_step = 0
       
        for epoch in range(self.args.pretrain_epochs):
            iter_count = 0

            loss_val = torch.tensor(0.).to(self.device)
            count = torch.tensor(0.).to(self.device)

            self.model.train()
            epoch_time = time.time()
           
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
               
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
               
                # encoder - decoder
                if self.args.output_attention:
                    model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Handle PrismGTS V3 6-output format: (y_list, x_recon, x_recon_t, x_recon_s, z_t, z_s)
                if self.args.model == 'PrismGTS' and len(model_output) == 6:
                    _, outputs, x_recon_trend, x_recon_seasonal, trend_feat, seasonal_feat = model_output
                elif self.args.model == 'PrismGTS' and len(model_output) == 4:
                    _, outputs, x_recon_trend, x_recon_seasonal = model_output
                    trend_feat, seasonal_feat = None, None
                else:
                    outputs, _ = model_output
                    x_recon_trend, x_recon_seasonal = None, None
                    trend_feat, seasonal_feat = None, None
                   
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_x_proc = batch_x[:, :, f_dim:]
                
                # Align lengths if needed
                recon_len = outputs.shape[1]
                if recon_len != batch_x_proc.shape[1]:
                    batch_x_proc = batch_x_proc[:, :recon_len, :]
                
                # Reconstruction loss
                loss = criterion(outputs, batch_x_proc)
                
                # PrismGTS V3: Add physical constraint loss
                if self.args.model == 'PrismGTS' and hasattr(self, 'physical_criterion') and self.physical_criterion is not None:
                    if x_recon_trend is not None and x_recon_seasonal is not None:
                        # Align trend/seasonal recon to match batch_x_proc
                        t_len = x_recon_trend.shape[1]
                        s_len = x_recon_seasonal.shape[1]
                        target_len = batch_x_proc.shape[1]
                        
                        # Slice or pad to match
                        if t_len > target_len:
                            trend_recon_aligned = x_recon_trend[:, :target_len, :]
                        else:
                            trend_recon_aligned = x_recon_trend
                            
                        if s_len > target_len:
                            seasonal_recon_aligned = x_recon_seasonal[:, :target_len, :]
                        else:
                            seasonal_recon_aligned = x_recon_seasonal
                        
                        # Use the minimum length for x_true
                        min_len = min(trend_recon_aligned.shape[1], seasonal_recon_aligned.shape[1], target_len)
                        x_true_aligned = batch_x_proc[:, :min_len, :]
                        trend_recon_aligned = trend_recon_aligned[:, :min_len, :]
                        seasonal_recon_aligned = seasonal_recon_aligned[:, :min_len, :]
                        
                        # Get z representations from trend_feat/seasonal_feat or pooled recon
                        if trend_feat is not None and seasonal_feat is not None:
                            z_trend = trend_feat.mean(dim=1)  # [B*C, D] -> mean gives [B*C, D]
                            z_seasonal = seasonal_feat.mean(dim=1)
                        else:
                            z_trend = trend_recon_aligned.mean(dim=1)  # [B, C]
                            z_seasonal = seasonal_recon_aligned.mean(dim=1)  # [B, C]
                        
                        # Physical loss V3: uses PhysicalConstraintLossV3 with new loss formulation
                        loss_phy = self.physical_criterion(
                            trend_recon_aligned, seasonal_recon_aligned, x_true_aligned,
                            z_trend, z_seasonal, stage=1
                        )
                        loss = loss + loss_phy
               
                loss_val += loss
                count += 1
                   
                if i % 100 == 0:
                    cost_time = time.time() - time_now
                    # Print with physical loss breakdown if available
                    phy_info = ""
                    if self.args.model == 'PrismGTS' and hasattr(self, 'physical_criterion') and self.physical_criterion is not None:
                        phy_info = " | " + self.physical_criterion.get_loss_string()
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.4f}{3} | cost: {4:.0f}s | mem: {5:.0f}MB"
                        .format(i, epoch + 1, loss.item(), phy_info, cost_time,
                                torch.cuda.memory_allocated() / 1024 / 1024))
                    time_now = time.time()


                loss.backward()
                model_optim.step()
                scheduler.schedule_step(global_step)
                global_step += 1
               
                torch.cuda.empty_cache()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if self.args.use_multi_gpu:
                dist.barrier()
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            train_loss = loss_val.item() / count.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion, flag='pretrain')
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion, flag='pretrain')
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            dist.barrier()
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def train(self, setting, train=0):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
               
        if train:
            print('loading model for Stage 2 / Few-shot')
            
            # Custom load path (few-shot/zero-shot)
            if hasattr(self.args, 'pretrained_path') and self.args.pretrained_path:
                ckpt_path = self.args.pretrained_path
                print(f'Loading from custom path: {ckpt_path}')
            else:
                # Default from Stage1
                original_setting = setting
                setting = re.sub(r'_' + self.args.data + '_', '_UTSD_', setting)
                setting = re.sub(r'_pl' + str(self.args.pred_len) + '_', '_pl96_', setting)
                ckpt_path = os.path.join('./checkpoints/' + setting, 'stage1', 'checkpoint.pth')
                print(f'Loading from Stage 1: {ckpt_path}')
                setting = original_setting
            
            self.model.load_state_dict(torch.load(ckpt_path))
            print(f'Loaded checkpoint successfully')
           
            if self.args.model == 'PrismGTS':
                model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                if hasattr(model_ref, 'freeze_backbone'):
                    model_ref.freeze_backbone()


        # Stage 2: Save checkpoints to setting/stage2/
        path = os.path.join(self.args.checkpoints, setting, 'stage2')
        if not os.path.exists(path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        # PrismGTS: Initialize physical constraint loss for Stage 2
        if self.args.model == 'PrismGTS':
            self._select_physical_criterion()

        print('Model parameters: ', sum(param.numel() for param in self.model.parameters() if param.requires_grad))
        scheduler = LargeScheduler(self.args, model_optim)
        global_step = 0

        for epoch in range(self.args.train_epochs):
            iter_count = 0

            loss_val = torch.tensor(0.).to(self.device)
            count = torch.tensor(0.).to(self.device)

            self.model.train()
            epoch_time = time.time()
           
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, ref_x, ref_y) in enumerate(train_loader):
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                             
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                if self.args.head_type == 'prediction':
                    # Handle list format from UTSD dataloader
                    if isinstance(batch_y, list):
                        batch_y = [y.float() for y in batch_y]
                        batch_y_mark = [y_mark.float().to(self.device) for y_mark in batch_y_mark]
                        dec_inp = [torch.zeros_like(y[:, -h:, :]).float() for y, h in zip(batch_y, self.args.horizon_lengths)]
                        dec_inp = [torch.cat([y[:, :self.args.label_len, :], d], dim=1).float().to(self.device) for y, d in zip(batch_y, dec_inp)]
                    else:
                        batch_y = batch_y.float()
                        batch_y_mark = batch_y_mark.float().to(self.device) 
                        # decoder input (tensor case)
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    batch_y = [y.float() for y in batch_y]
                    batch_y_mark = [y_mark.float().to(self.device) for y_mark in batch_y_mark]
                   
                    dec_inp = [torch.zeros_like(y[:, -h:, :]).float() for y, h in zip(batch_y, self.args.horizon_lengths)]
                    dec_inp = [torch.cat([y[:, :self.args.label_len, :], d], dim=1).float().to(self.device) for y, d in zip(batch_y, dec_inp)]                

                # encoder - decoder
                if self.args.output_attention:
                    model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Handle PrismGTS 6-output format: (y_list, x_recon, x_recon_t, x_recon_s, z_t, z_s)
                if self.args.model == 'PrismGTS' and len(model_output) == 6:
                    outputs, recons, x_recon_trend, x_recon_seasonal, trend_feat, seasonal_feat = model_output
                elif self.args.model == 'PrismGTS' and len(model_output) == 4:
                    outputs, recons, x_recon_trend, x_recon_seasonal = model_output
                    trend_feat, seasonal_feat = None, None
                else:
                    outputs, recons = model_output
                    x_recon_trend, x_recon_seasonal = None, None
                    trend_feat, seasonal_feat = None, None
                             
                f_dim = -1 if self.args.features == 'MS' else 0
                recons = recons[:, :, f_dim:]
                batch_x_proc = batch_x[:, :, f_dim:]
                
                # Align recon length
                recon_len = recons.shape[1]
                if recon_len != batch_x_proc.shape[1]:
                    batch_x_proc_recon = batch_x_proc[:, :recon_len, :]
                else:
                    batch_x_proc_recon = batch_x_proc
                
                if self.args.head_type == 'prediction':
                    # Handle both list (UTSD) and tensor formats
                    if isinstance(batch_y, list):
                        pre_loss = torch.tensor(0.).to(self.device)
                        for j, (o, y) in enumerate(zip(outputs, batch_y)):
                            o = o[:, -self.args.horizon_lengths[j]:, f_dim:]
                            y = y[:, -self.args.horizon_lengths[j]:, f_dim:].to(self.device)
                            pre_loss = pre_loss + criterion(o, y)
                        loss = pre_loss + criterion(recons, batch_x_proc_recon)
                    else:
                        outputs = torch.cat([outputs[j][:, -h:, f_dim:] for j, h in enumerate(self.args.horizon_lengths) if j in self.idx], dim=1)
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y) + criterion(recons, batch_x_proc_recon)
                else:
                    pre_loss = torch.tensor(0.).to(self.device)
                    for j, (o, y) in enumerate(zip(outputs, batch_y)):
                        o = o[:, -self.args.horizon_lengths[j]:, f_dim:]
                        y = y[:, -self.args.horizon_lengths[j]:, f_dim:].to(self.device)
                        pre_loss = pre_loss + criterion(o, y)
                    loss = pre_loss + criterion(recons, batch_x_proc_recon)
                               
                loss_val += loss
                count += 1
                   
                if i % 100 == 0:
                    cost_time = time.time() - time_now
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f} | cost_time: {3:.0f} | memory: allocated {4:.0f}MB, reserved {5:.0f}MB, cached {6:.0f}MB "
                        .format(i, epoch + 1, loss.item(), cost_time,
                                torch.cuda.memory_allocated() / 1024 / 1024,
                                torch.cuda.memory_reserved() / 1024 / 1024,
                                torch.cuda.memory_cached() / 1024 / 1024))
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                scheduler.schedule_step(global_step)
                global_step += 1
                torch.cuda.empty_cache()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if self.args.use_multi_gpu:
                dist.barrier()
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            train_loss = loss_val.item() / count.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion, flag='train')
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion, flag='train')
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))

            # Stage 2: Save checkpoint for EVERY epoch
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                epoch_ckpt_path = path + f'/checkpoint_epoch_{epoch + 1}.pth'
                torch.save(self.model.state_dict(), epoch_ckpt_path)
                print(f"Epoch {epoch + 1} saved: {epoch_ckpt_path}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Only load best checkpoint if training actually happened
        if self.args.train_epochs > 0:
            best_model_path = path + '/' + 'checkpoint.pth'
            if self.args.use_multi_gpu:
                dist.barrier()
            if os.path.exists(best_model_path):
                self.model.load_state_dict(torch.load(best_model_path))
            else:
                print(f"Warning: Best model checkpoint not found at {best_model_path}, using current model weights")
        else:
            print("Skipping checkpoint load (train_epochs=0)")

        return self.model


    def test(self, setting, test=0):
        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        attns = []
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            if self.args.pretrained_path:
                checkpoint_file = os.path.basename(self.args.pretrained_path)
                print(f'Loading pretrained model from: {self.args.pretrained_path}')
                print(f'Checkpoint: "{checkpoint_file}"')
                self.model.load_state_dict(torch.load(self.args.pretrained_path))
                self.loaded_checkpoint = checkpoint_file  # Store for result file
            else:
                original_setting = setting
                if self.args.is_zeroshot:
                    setting = re.sub(r'_CI_', '_UTSD_', setting)
                    setting = re.sub(r'_pl' + str(self.args.pred_len) + '_', '_pl96_', setting)
                checkpoint_path = os.path.join('./checkpoints/' + setting, 'stage2', 'checkpoint.pth')
                checkpoint_file = os.path.basename(checkpoint_path)
                print(f'Checkpoint: "{checkpoint_file}"')
                self.model.load_state_dict(torch.load(checkpoint_path))
                self.loaded_checkpoint = checkpoint_file  # Store for result file
                setting = original_setting

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(folder_path)

        # ============== Efficiency Measurement ==============
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        test_start_time = time.time()
        n_batches = 0
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)       
               
                batch_y = batch_y.float()
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
               
                # encoder - decoder
                if self.args.output_attention:
                    outputs, recons, attns = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # PrismGTS V3 returns (y_list, x_recon, x_recon_t, x_recon_s, z_t, z_s)
                    if isinstance(model_output, tuple) and len(model_output) == 6:
                        outputs, recons, _, _, _, _ = model_output
                    elif isinstance(model_output, tuple) and len(model_output) == 4:
                        outputs, recons, _, _ = model_output
                    else:
                        outputs, recons = model_output

                f_dim = -1 if self.args.features == 'MS' else 0
               
                outputs = torch.cat([outputs[j][:, -h:, :] for j, h in enumerate(self.args.horizon_lengths) if j in self.idx], dim=1)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                   
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
       
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                n_batches += 1  # Count batches for efficiency
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                   
                    if self.args.local_rank == 0:
                        if self.args.output_attention:
                            attn = attns[0].cpu().numpy()[0, 0, :, :]
                            attn_map(attn, os.path.join(folder_path, f'attn_{i}_{self.args.local_rank}.pdf'))
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
       

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        
       
        f = open("result_long_term_forecast.txt", 'a')
        
        # Determine task type
        if hasattr(self.args, 'is_zeroshot') and self.args.is_zeroshot:
            task_type = "zero-shot"
        elif hasattr(self.args, 'percent') and self.args.percent < 100:
            task_type = f"few-shot({self.args.percent}%)"
        else:
            task_type = "full"
        
        # Extract dataset name from data_path or use data argument
        dataset_name = getattr(self.args, 'data_path', '').replace('.csv', '') or self.args.data
        
        # Write formatted result
        f.write(f"Dataset: {dataset_name} | Task: {task_type} | PredLen: {self.args.pred_len}\n")
        f.write(f"Model: {self.args.model} | ModelID: {self.args.model_id}\n")
        f.write(f"Setting: {setting}\n")
        # Add checkpoint info if available
        if hasattr(self, 'loaded_checkpoint'):
            f.write(f"Checkpoint: \"{self.loaded_checkpoint}\"\n")
        f.write(f"MSE: {mse:.6f} | MAE: {mae:.6f}\n\n")
        f.close()

        return
