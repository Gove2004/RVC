import torch
from torch.nn import functional as F


def feature_loss(fmap_r, fmap_g):
    loss = 0.0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl.float().detach() - gl.float()))
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0.0
    real_losses = []
    generated_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        real_losses.append(r_loss.item())
        generated_losses.append(g_loss.item())
    return loss, real_losses, generated_losses


def generator_loss(disc_outputs):
    loss = 0.0
    generated_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        gen_loss = torch.mean((1 - dg) ** 2)
        generated_losses.append(gen_loss)
        loss += gen_loss
    return loss, generated_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    return kl / torch.sum(z_mask)
