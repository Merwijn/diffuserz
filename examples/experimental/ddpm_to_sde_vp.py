import torch

from diffusers import DDPMPipeline, DDPMScheduler, DiffusionPipeline, ScoreSdeVeScheduler, UNetModel
from tqdm.auto import tqdm


class CustomSdeVePipeline(DiffusionPipeline):
    def __init__(self, model, scheduler):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)

    def __call__(self, num_inference_steps=2000, generator=None):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        img_size = self.model.config.resolution
        channels = self.model.config.in_channels
        shape = (1, channels, img_size, img_size)

        model = self.model.to(device)

        # TODO(Patrick) move to scheduler config
        n_steps = 1

        x = torch.randn(*shape) * self.scheduler.config.sigma_max
        x = x.to(device)

        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.set_sigmas(num_inference_steps)

        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            sigma_t = self.scheduler.sigmas[i] * torch.ones(shape[0], device=device)

            for _ in range(n_steps):
                with torch.no_grad():
                    result = self.model(x, sigma_t)
                x = self.scheduler.step_correct(result, x)

            with torch.no_grad():
                result = model(x, sigma_t)

            x, x_mean = self.scheduler.step_pred(result, x, t)

        return x_mean


model_id = "fusing/ddpm-lsun-church"
model = UNetModel.from_pretrained(model_id)

scheduler = ScoreSdeVeScheduler.from_config("fusing/church_256-ncsnpp-ve")
# scheduler = ScoreSdeVeScheduler(beta_min=0.0001, beta_max=0.02, tensor_format="pt")
sde_ve = CustomSdeVePipeline(model=model, scheduler=scheduler)

torch.manual_seed(0)
image = sde_ve(num_inference_steps=2000)

image = (image[0].cpu() * 255).round().type(torch.uint8).permute(1, 2, 0).numpy()
print(image)


noise_scheduler = DDPMScheduler.from_config(model_id)
noise_scheduler = noise_scheduler.set_format("pt")

ddpm = DDPMPipeline(unet=model, noise_scheduler=noise_scheduler)

generator = torch.manual_seed(0)
image = ddpm(generator=generator)
image = ((image[0].cpu() + 1) * 127.5).round().type(torch.uint8).permute(1, 2, 0).numpy()
print(image)
