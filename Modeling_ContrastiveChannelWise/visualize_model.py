import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
from captum.attr import IntegratedGradients

def visualize_all_channels(model, eeg_batch, channel_names, sample_idx):
    fig, axes = plt.subplots(7, 2, figsize=(12, 14))
    axes = axes.flatten()

    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    for i, ch_name in enumerate(channel_names):
        enc = base_model.encoders[i] # give me encoder model for this channel

        x = eeg_batch[sample_idx, :, i].unsqueeze(0).unsqueeze(0)  # [1,1,L]
        activations = {}

        def hook_fn(module, input, output):
            activations['conv'] = output.detach().cpu()

        h = enc.conv.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = enc(x.cuda())
        h.remove()

        conv_out = activations['conv'].squeeze(0)  # [filters, L']
        mean_act = F.relu(conv_out).mean(dim=0)
        #mean_act = (mean_act - mean_act.min()) / (mean_act.max() - mean_act.min() + 1e-8)
        mean_act_up = F.interpolate(mean_act[None, None, :], size=x.shape[-1],
                                    mode='linear', align_corners=True).squeeze().numpy()

        axes[i].plot(x.squeeze().cpu().numpy(), color='black', lw=0.8)
        axes[i].fill_between(np.arange(len(mean_act_up)),
                             0, mean_act_up * x.std().item(),
                             color='dodgerblue', alpha=0.4)
        axes[i].set_title(ch_name)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.suptitle("EEG Activation Maps (Mean over Conv Filters)", y=0.93)
    plt.tight_layout()
    plt.savefig("ConvsVisualized.PDF")
    return fig


def visualize_integrated_gradients_all_channels(model, eeg_batch, channel_names, sample_idx, device="cuda"):
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    # Create subplots (14 channels → 7x2)
    fig, axes = plt.subplots(7, 2, figsize=(12, 14))
    axes = axes.flatten()

    # Disable gradient updates elsewhere
    base_model.eval()

    # iterate over each channel
    for i, ch_name in enumerate(channel_names):
        # build wrapper for single-channel IG
        class ModelWrapper(torch.nn.Module):
            def __init__(self, full_model, channel_idx):
                super().__init__()
                self.model = full_model
                self.channel_idx = channel_idx

            def forward(self, x):
                # x: [B, L]
                full_x = torch.zeros(x.size(0), x.size(1), len(channel_names), device=x.device)
                full_x[:, :, self.channel_idx] = x
                out, _, _ = self.model(full_x)
                return out

        wrapped = ModelWrapper(base_model, channel_idx=i).to(device)
        ig = IntegratedGradients(wrapped)

        # pick the EEG window
        x = eeg_batch[sample_idx, :, i].unsqueeze(0).to(device)  # [1, L]
        baseline = torch.zeros_like(x)

        was_training = base_model.training
        base_model.train()

        # compute IG attributions
        attributions, _ = ig.attribute(inputs=x, baselines=baseline, n_steps=128, return_convergence_delta=True)

        if not was_training:
            base_model.eval()

        attr = attributions.squeeze().detach().cpu().numpy()

        # normalize to [0,1] for visualization
        attr_norm = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        signal = x.squeeze().detach().cpu().numpy()
        signal_norm = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
        # plot
        ax = axes[i]
        ax.plot(signal_norm, color='black', lw=0.8)
        ax.fill_between(np.arange(len(attr_norm)),
                        0, attr_norm,
                        color='crimson', alpha=0.4)
        ax.set_title(ch_name)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Integrated Gradients Attributions (per channel)", y=0.93)
    plt.tight_layout()
    plt.savefig("IntegratedGradients.PDF")
    return fig
