"""Asynchronous data generation via MCMC."""

import threading

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from src.sampling.mcmc import run_mcmc
from src.trainer.buffer import Buffer


class MCMCSampler:
    def __init__(
        self,
        model_apply_fn,
        init_params,  # callable: returns current model parameters
        sample_fn,  # callable: returns batch of graph samples
        atomic_number,
        step_size=1e-3,
        num_mcmc_steps=10,
        kernel_method="mala",
        batch_size=4,
        basis_name="sto-3g",
        xc_method="lda",
        grid_level=3,
        buffer_max_size=None,
        device=None,
        seed=42,
    ):
        self.model_apply_fn = model_apply_fn
        self.params = init_params
        self.sample_fn = sample_fn

        # Molecular System parameters
        self.atomic_number = atomic_number
        self.basis_name = basis_name
        self.xc_method = xc_method
        self.grid_level = grid_level

        # MCMC parameters
        self.step_size = step_size
        self.init_step_size = step_size  # used to reset step size if it goes below zero
        self.num_mcmc_steps = num_mcmc_steps
        self.kernel_method = kernel_method
        self.batch_size = batch_size
        self.acceptance_rate = 1.0  # Initial acceptance rate

        # Buffer parameters
        self.buffer_max_size = buffer_max_size
        if buffer_max_size is None:
            # set buffer to batch size will cause
            # sync every time batch is generated
            self.buffer_max_size = batch_size

        self.buffer = Buffer(max_size=buffer_max_size)

        self.device = device if device is not None else jax.devices()[0]
        self.key = jax.random.PRNGKey(seed)

        # Threading and synchronization
        self.stop_event = threading.Event()
        self.sync_event = threading.Event()
        self.sync_completed_event = threading.Event()
        self.thread = threading.Thread(target=self.worker, daemon=True)

    def update_params(self, params):
        # Synchronize the data generation model with updated parameters
        # of the training model
        with jax.default_device(self.device):
            updated_sampler_params = jax.device_put(params, device=self.device)
            jax.block_until_ready(updated_sampler_params)

            self.params = updated_sampler_params

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def check_sync(self):
        return self.sync_event.is_set()

    def signal_completed_sync(self):
        self.sync_event.clear()
        self.sync_completed_event.set()

    def sample(self, num_samples: int = None, verbose: bool = False):
        num_samples = self.batch_size if num_samples is None else num_samples

        pos = []
        pbar = range(0, num_samples, self.batch_size)
        if verbose:
            pbar = tqdm(pbar)

        for _ in pbar:
            self.key, subkey = jax.random.split(self.key)

            atomic_number = np.array(
                self.atomic_number[np.newaxis, ...].repeat(self.batch_size, axis=0)
            )

            sample_graphs = self.sample_fn(self.batch_size)
            x0 = sample_graphs.position

            graphs, step_size, acceptance_rate = run_mcmc(
                key=subkey,
                params=self.params,
                apply_fn=self.model_apply_fn,
                atomic_number=atomic_number,
                init_pos=x0,
                step_size=self.step_size,
                n_iter=self.num_mcmc_steps,
                basis_name=self.basis_name,
                xc_method=self.xc_method,
                grid_level=self.grid_level,
                kernel=self.kernel_method,
            )

            batch_pos = [
                np.array(pos)
                for pos, energy in zip(graphs.position, graphs.energy)
                if not jnp.isnan(energy)
            ]

            if step_size <= 0:
                step_size = self.init_step_size

            self.step_size = step_size
            self.acceptance_rate = acceptance_rate

            pos.extend(batch_pos)

        return pos

    def worker(self):
        with jax.default_device(self.device):
            while not self.stop_event.is_set():
                if len(self.buffer) >= self.buffer_max_size:
                    self.sync_event.set()
                    self.sync_completed_event.wait()
                    self.sync_completed_event.clear()
                    continue

                pos = self.sample()
                self.buffer.add(pos)
