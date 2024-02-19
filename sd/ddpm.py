import torch
import numpy as np

class DDPMSampler:

    def __init__(self, generator: torch.Generator,
                 num_training_steps=1000, 
                 beta_start: float = 0.00085,
                 beta_end: float = 0.0120):
        # the scaled Linear scheduler
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, 
                                    num_training_steps, dtype=torch.float32) ** 2
        
        self.alphas = 1.0 - self.betas
        # [alpha_0, alpha_0*alpha_1, alpha_0*alpha_1*alpha_2,...]
        self.alpha_cumprod = torch.cumprod(self.alphas, 0) 
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps

        # initial timesteps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_training_steps = num_inference_steps
        # 999, 998, 997, 996, ...,0 -> 1000 steps
        # 999, 999-20, 999-40,...,0 -> 50 steps

        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    # 一度前のタイムステップを計算する関数
    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - (self.num_training_steps // self.num_training_steps)
        return prev_t
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        prev_t = self._get_previous_timestep(timestep)

        # α_cumprod[t] = α_0*α_1*,...,α_t
        alpha_prod_t = self.alpha_cumprod[timestep]
        # α_cumprod[t-1] = 1*α_0*α_1*,...,α_t-1
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # (7)式参照
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        # Zero division error 回避のため，var=0のときは非常に小さな値を与えることにする
        variance = torch.clamp(variance, min=1e-20)

        return variance



    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # α_cumprod[t] = α_0*α_1*,...,α_t
        alpha_prod_t = self.alpha_cumprod[timestep]
        # α_cumprod[t-1] = 1*α_0*α_1*,...,α_t-1
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 = alpha_prod_t_prev
        # α[t] = (α_0*α_1*,...,α_t-1*α_t) / (α_0*α_1*,...,α_t-1)
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t


        # ここから(6)式のコーディングを行う（Reverse Processを1回分進める操作）

        # (15)式からx_0の近似を行う
        # x_0 ≈ (x_t - √beta_prod_t * ε_θ(x_t)) / √alpha_prod_t
        pred_original_samples = (latents - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5

        # (7)式からμ_t(x_t, x_0)を計算する
        pred_original_sample_coeff = (alpha_prod_t_prev**0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = (alpha_prod_t**0.5 * beta_prod_t_prev) / beta_prod_t
        pred_prev_sample = pred_original_sample_coeff * pred_prev_sample + current_sample_coeff * latents

        # 何回か使用するため，分散は関数で計算する((7)式参照)
        variance = 0
        if t > 0:
            device = model_output.device
            # ノイズをサンプルする
            # 動画で話しているdeviceは多分誤植で，以下の値は原文ママです(ここの引数わかる方誰か教えて)
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            variance = (self._get_variance(t)**0.5) * noise

        # N(0, 1) --> N(mu, sigma)
        # X = mu + sigma * Z where Z ~ N(0, 1)
        # 分散を加える((6)式の値の完成)
        pred_prev_sample = pred_prev_sample + variance

    def set_strength(self, strength=1):
        """
        image to imageでは オリジナル画像 -> encoder -> add noise to latentという過程が最初にあるが，
        どれだけ最初にノイズを加えるかを決定する．1で純粋なノイズ，0でノイズなしとなる．
        コードにもあるように，実質ステップをスキップすることに相当する．

        pipelineにかかれてた内容
        strength: float
        """
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step




    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        
        #  論文等にかかれてた式を用いて一回で指定回数分のノイズを加える
        alpha_cumprod = self.alpha_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        
        sqrt_alpha_prod = (alpha_cumprod[timesteps] ** 0.5).flatten()

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # 標準偏差
        sqrt_one_minus_alpha_prod = ((1 - alpha_cumprod[timesteps]) ** 0.5).flatten() 
        
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # DDPMの(4)式より
        # N(mean, std**2) = mean + std * N(0, 1)
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = (sqrt_alpha_prod * original_samples) + (sqrt_one_minus_alpha_prod) * noise

        return noisy_samples
    


