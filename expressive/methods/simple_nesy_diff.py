from typing import Optional

import numpy as np

from expressive.methods.base_model import BaseNeSyDiffusion
from expressive.util import safe_sample_categorical
import torch
from torch import Tensor
from torch.nn.functional import one_hot

from torch.distributions import Categorical

from expressive.methods.logger import TrainingLog


class SimpleNeSyDiffusion(BaseNeSyDiffusion):
    """This model adds conditioning to y when generating w, essentially interleaving their computation. 
    """

    def rloo_loss(
        self,
        log_probs_SB: Tensor,
        reward_SBD: Tensor,
    ) -> Tensor:
        """
        Reinforce with leave-one-out (rloo) loss. 
        log_probs: predicted log probabilities for each action taken
        rewards: resulting rewards (negative loss)
        """

        # Compute loo baselines, but only for dimensions where at least two samples are different from w0 (otherwise the baseline is not defined)
        baseline_SBD = (torch.sum(reward_SBD, dim=0) - reward_SBD) / (reward_SBD.shape[0] - 1)

        # Compute RLOO gradient via rho
        rho_BD = torch.mean(
            (log_probs_SB)[..., None]
            * (reward_SBD - baseline_SBD).detach(),
            dim=0,
        )
        mu_BD = torch.mean(reward_SBD, dim=0)

        # Hacky way to get 1 over average as extra multiplicative term in the loss.
        #  I got this from https://github.com/ML-KULeuven/catlog/blob/main/ADDITION/addition.py#L6
        #  The idea here is that you want to get an estimate of the gradient of - log WMC, which (when differentiated)
        #  should be 1 / WMC * dWMC/dtheta. This is a way to get that 1 / WMC term (or rather an estimate of it)
        #  And also means that the loss will be an interpretable term (namely an estimate of semantic loss).
        epsilon = torch.tensor(1e-8, device=mu_BD.device)
        return torch.log((mu_BD - rho_BD).detach() + rho_BD + epsilon)

    def loss(
        self, x_BX: Tensor, y_0_BY: Tensor, log: TrainingLog, eval_w_0_BW: Optional[Tensor] = None
    ) -> Tensor:
        """
        Shapes legend:
        - B: Batch size
        - X: Input size
        - Y: number of RVs for y
        - W: number of RVs for w
        - d: number of values for y or w
        - D: number of values for y or w + 1 (mask dim)
        - S: number of samples of \tilde{w_0} (=S)
        """
        self.train()
        # initialize embedding of x
        encoding_BWE = self.p.encode_x(x_BX)

        # Create mask matrix bm (a matrix full of the number of values possible for w (with 0 indexing))
        s_w = y_0_BY.shape[:-1] + self.problem.shape_w()
        bm_BW = torch.ones(s_w[:-1], device=x_BX.device, dtype=torch.long) * s_w[-1]

        # initialize q(w_0|x, y_0)
        q_w_0_BWD = self.p.distribution(
            (bm_BW),
            encoding_BWE,
            torch.zeros_like(y_0_BY[..., 0], device=bm_BW.device),
        )

        # TODO: H (entropy)
        # Sample w_0
        w_1_BW = torch.full(
            (x_BX.shape[0],) + self.problem.shape_w()[:-1],
            self.mask_dim_w(),
            device=x_BX.device,
        )
        var_w_0_BW = self.sample(
            x_BX,
            w_1_BW,
            y_0_BY,
            self.args.variational_K,
            self.args.variational_T,
            self.args.variational_K,
            encoding_BWE,
            only_w=True,
        )[0]
        var_w_0_BWD = one_hot(var_w_0_BW, s_w[-1] + 1).float()

        # Sample timesteps
        t_B = torch.rand((x_BX.shape[0],), device=x_BX.device)

        # Compute q(w_t | w_0)
        q_w_t_BWD: Tensor = self.q_w.t_step(var_w_0_BWD, t_B)

        w_t_BW = safe_sample_categorical(Categorical(probs=q_w_t_BWD))

        # Compute p(\tilde{w}_0|y_t, w_t x)
        p_w_0_BWD = self.p.distribution(w_t_BW, encoding_BWE, t_B)[..., :-1]

        # Sample S_2 values for \tilde{w}_0
        tw_0 = Categorical(probs=p_w_0_BWD)
        tw_0_SBW = safe_sample_categorical(tw_0, (self.args.loss_S,))

        # Compute deterministic function for tw_0, with carry-over unmasking on y_t_BY
        ty_0_SBY = self.problem.y_from_w(tw_0_SBW)

        #####################
        # LOSS FUNCTIONS
        #####################
        #####################
        # w RLOO denoising loss
        #####################
        # Take mean over dimensions of w (for stability over hyperparameters)
        L_w_denoising = self.loss_weight(tw_0.log_prob(var_w_0_BW).mean(-1), t_B).mean()

        #####################
        # y RLOO denoising loss
        #####################
         # Compute log probs of tw_0
        orbit_M = getattr(self, "orbit_M", 0)
        if orbit_M > 0:
            # Orbit-averaged score: average log q over M value-relabelings sigma in S_V (the
            # label-symmetry group; y_from_w is relabeling-invariant so the reward is unchanged).
            # Replaces the raw score dlog q(w) with a q-weighted average of dlog q(sigma.w),
            # which projects out the reward-unidentifiable direction and aligns the gradient.
            V = p_w_0_BWD.shape[-1]
            dev = tw_0_SBW.device
            sig_MV = torch.stack([torch.randperm(V, device=dev) for _ in range(orbit_M)])
            sig_MV[0] = torch.arange(V, device=dev)                     # anchor identity
            w_sig_MSBW = sig_MV[:, tw_0_SBW]                            # [M,S,B,W]
            logq_MSB = tw_0.log_prob(w_sig_MSBW).sum(-1)               # [M,S,B]
            wts_MSB = torch.softmax(logq_MSB.detach(), dim=0)          # q-weighted (self-norm)
            log_probs_SB = (wts_MSB * logq_MSB).sum(0)                 # [S,B]
            # DIAGNOSTIC: effective # of orbit members (ESS). ~1 => averaging collapsed to
            # identity (carry-over one-hot positions kill relabeled worlds); ~M => full averaging.
            self.orbit_ess = float((1.0 / wts_MSB.pow(2).sum(0)).mean())
        else:
            log_probs_SB = tw_0.log_prob(tw_0_SBW).sum(-1)
        # Compute all constraints for y
        constraint_y0_SBY = (y_0_BY[None, :, :] == ty_0_SBY).float()
        reward_y_0_SBY = constraint_y0_SBY

        # Compute the RLOO loss for all constraints. Uses LOO with w0 injection
        L_denoising_BY = self.rloo_loss(
            log_probs_SB,
            reward_y_0_SBY,
        )

        # Take the mean over the dimensions of y (for stability over hyperparameters)
        L_y_denoising = self.loss_weight(
            L_denoising_BY.mean(-1), t_B
        ).mean()

        #####################
        # Entropy on variational distribution (will be negated in loss)
        #####################
        q_entropy: Tensor = self.entropy_loss(y_0_BY, q_w_0_BWD).mean()

        #####################
        # Optional negative entropy on unmasking model
        #####################
        entropy_denoising_B = self.entropy_loss(y_0_BY, p_w_0_BWD)
        L_entropy_denoising = self.loss_weight(entropy_denoising_B, t_B).mean() if self.args.denoising_entropy else 0.0

        var_y_0_BY = self.problem.y_from_w(var_w_0_BW)
        var_violations_y_0_BY = var_y_0_BY != y_0_BY

        if var_w_0_BW is not None:
            log.var_accuracy_w += (var_w_0_BW == eval_w_0_BW).float().mean().item()
            log.w_preds = np.concatenate([log.w_preds, var_w_0_BW.flatten().detach().cpu().int().numpy()])
            log.w_targets = np.concatenate([log.w_targets, eval_w_0_BW.flatten().detach().cpu().int().numpy()])

        log.var_entropy += q_entropy.item()
        log.unmasking_entropy += entropy_denoising_B.mean().item()
        log.w_denoise += L_w_denoising.item()
        log.y_denoise += L_y_denoising.item()
        log.avg_constraints += constraint_y0_SBY.float().mean().item()
        log.avg_var_violations += var_violations_y_0_BY.float().mean().item()
        log.var_accuracy_y += torch.min(~var_violations_y_0_BY, dim=-1)[0].float().mean().item()
        
        return (
            L_y_denoising
            + self.args.w_denoise_weight * L_w_denoising
            - self.args.entropy_weight * q_entropy
            # Note: Already gets negated through self.loss_weight
            + self.args.entropy_weight * L_entropy_denoising
        )