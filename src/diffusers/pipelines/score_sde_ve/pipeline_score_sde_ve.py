#!/usr/bin/env python3
import torch

from diffusers import DiffusionPipeline
from tqdm.auto import tqdm


# TODO(Patrick, Anton, Suraj) - rename `x` to better variable names
class ScoreSdeVePipeline(DiffusionPipeline):
    def __init__(self, model, scheduler):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)

    def old_call(self, num_inference_steps=2000, generator=None):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        img_size = self.model.config.image_size
        channels = self.model.config.num_channels
        shape = (1, channels, img_size, img_size)

        model = self.model.to(device)

        # TODO(Patrick) move to scheduler config
        n_steps = 1

        x = torch.randn(*shape) * self.scheduler.config.sigma_max
        x = x.to(device)

        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.set_sigmas(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            sigma_t = self.scheduler.sigmas[i] * torch.ones(shape[0], device=device)
            for _ in range(n_steps):
                with torch.no_grad():
                    result = self.model(x, sigma_t)
                x = self.scheduler.step_correct(result, x)

            with torch.no_grad():
                result = model(x, sigma_t)
            print(i, sigma_t.item(), result.min().item(), result.max().item())

            x, x_mean = self.scheduler.step_pred(result, x, t)


        return x_mean

    def determ_call(self, num_inference_steps=2000, generator=None):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        img_size = self.model.config.image_size
        channels = self.model.config.num_channels
        shape = (1, channels, img_size, img_size)

        model = self.model.to(device)
        self.scheduler.set_timesteps(num_inference_steps)

        sample = torch.randn(*shape, generator=generator) * self.scheduler.schedule[0]
        sample = sample.to(device)

        for i in range(num_inference_steps):
            t = self.scheduler.timesteps[i] * torch.ones(shape[0], device=device)
            t_next = self.scheduler.timesteps[i + 1] * torch.ones(shape[0], device=device)
            sigma = self.scheduler.schedule[i] * torch.ones(shape[0], device=device)
            sigma_deriv = self.scheduler.schedule_deriv[i] * torch.ones(shape[0], device=device)

            with torch.no_grad():
                D = sample + sigma * model(sample, torch.log(0.5 * sigma))

            d_curr = (sigma_deriv / sigma) * sample - (sigma_deriv / sigma) * D
            next_sample = sample + (t_next - t) * d_curr

            sigma_next = self.scheduler.schedule[i + 1] * torch.ones(shape[0], device=device)
            sigma_deriv_next = self.scheduler.schedule_deriv[i + 1] * torch.ones(shape[0], device=device)
            if sigma_next != 0:
                # apply 2nd order correction
                with torch.no_grad():
                    D_next = next_sample + sigma_next * model(next_sample, torch.log(0.5 * sigma_next))

                d_next = (sigma_deriv_next / sigma_next) * next_sample - (sigma_deriv_next / sigma_next) * D_next
                next_sample = sample + (t_next - t) * (0.5 * d_curr + 0.5 * d_next)

            sample = next_sample
            print(i, 0.5 * sigma, sample.min().item(), sample.max().item())

        return sample


    def stochastic_call(self, batch_size=1, num_inference_steps=2000, generator=None, s_noise=1.007):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        img_size = self.model.config.image_size
        channels = self.model.config.num_channels
        shape = (1, channels, img_size, img_size)

        model = self.model.to(device)
        self.scheduler.set_timesteps(num_inference_steps)

        x = torch.randn(*shape, generator=generator) * self.scheduler.timesteps[0]
        x = x.to(device)

        for i in range(num_inference_steps):
            t = self.scheduler.timesteps[i] * torch.ones(shape[0], device=device)
            t_next = self.scheduler.timesteps[i + 1] * torch.ones(shape[0], device=device)
            gamma = self.scheduler.gammas[i] * torch.ones(shape[0], device=device)

            eps = torch.randn(*shape, generator=generator) * s_noise
            eps = eps.to(device)
            t_hat = t + gamma * t

            x_hat = x + ((t_hat**2 - t**2)**0.5 * eps)

            with torch.no_grad():
                D = x_hat + t_hat * model(x_hat, torch.log(0.5 * t_hat))

            d = (x_hat - D) / t_hat
            x_next = x_hat + (t_next - t_hat) * d
            if t_next != 0:
                with torch.no_grad():
                    D_next = x_next + t_next * model(x_next, torch.log(0.5 * t_next))

                d_next = (x_next - D_next)/t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d + 0.5 * d_next)
            x = x_next
            print(i, x.min().item(), x.max().item())
        return x

