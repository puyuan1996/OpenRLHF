import os

import ray
import vllm
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM

from openrlhf.trainer.ray.utils import ray_noset_visible_devices
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


@ray.remote
def get_all_env_variables():

    return os.environ


@ray.remote
class VllmLLMRayActor:
    def __init__(self, *args, **kwargs):
        noset_visible_devices = kwargs.pop("noset_visible_devices", False)
        self.use_gpu_executor = kwargs["tensor_parallel_size"] == 1 and not noset_visible_devices
        self.__version__ = vllm.__version__
        assert self.__version__ >= "0.4.2", "OpenRLHF only supports vLLM >= 0.4.2"

        # See https://github.com/vllm-project/vllm/blob/main/vllm/executor/gpu_executor.py
        if self.use_gpu_executor:
            from openrlhf.trainer.ray.vllm_worker_wrap import WorkerWrap

            vllm.worker.worker.Worker = WorkerWrap
        else:
            # RayGPUExecutor
            # See the patch https://github.com/vllm-project/vllm/commit/479d69fad0538f04cb22bf13e76ff91cfeb8a4e5
            #! worker_use_ray is a vllm only parameter
            kwargs["worker_use_ray"] = True
            print("kwargs['worker_use_ray']", kwargs["worker_use_ray"])
            if vllm.__version__ > "0.6.4.post1":
                # https://github.com/vllm-project/vllm/pull/10555
                kwargs["worker_cls"] = "openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap"
            else:
                responses = []

<<<<<<< HEAD:openrlhf/trainer/ray/vllm_engine.py
            offset = 0
            self.responses = {}
            for actor_rank, num in num_requests:
                self.responses[actor_rank] = responses[offset : offset + num]
                offset += num
=======
                RayWorkerWrapperPath.RayWorkerWrapper = RayWorkerWrapper
        kwargs.pop("backend")
        self.llm = vllm.LLM(*args, **kwargs)

    def generate(self, sampling_params, prompt_token_ids, stop_token_ids):
        sampling_params = vllm.SamplingParams(**sampling_params)
        outputs = self.llm.generate(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
        return outputs

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.init_process_group(
                master_address, master_port, rank_offset, world_size, group_name, backend
            )
        else:
            return self.llm.llm_engine.model_executor._run_workers(
                "init_process_group", master_address, master_port, rank_offset, world_size, group_name, backend
            )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        self.stop_remote_worker_execution_loop()
>>>>>>> 06ce7866d66799af8bab597df52b00f4fa577aee:openrlhf/trainer/ray/vllm_llm_ray_actor.py

            self.actor_counter = 0
            self.requests = {}

<<<<<<< HEAD:openrlhf/trainer/ray/vllm_engine.py
    def get_responses(self, actor_rank):
        """
        Return the responses for the actor with the given rank
        """
        return self.responses.pop(actor_rank)
=======
    def stop_remote_worker_execution_loop(self):
        # Fix error for using 2 communication group
        # https://github.com/vllm-project/vllm/commit/eb6d3c264d0cd8e44dec16bca7947fbe96415ce9#diff-e1ad69e38e033accddfa5480ec808c4740eb39244d1ef51cc3407e20dde8cfd4
        if self.__version__ > "0.4.2":
            print("self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop")
            self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop()
>>>>>>> 06ce7866d66799af8bab597df52b00f4fa577aee:openrlhf/trainer/ray/vllm_llm_ray_actor.py


def create_llm_ray_actor_vllm(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    num_total_actors: int,
    shared_pg=None,
    gpu_memory_utilization=None,
    vllm_enable_sleep=False,
):
<<<<<<< HEAD:openrlhf/trainer/ray/vllm_engine.py
    import vllm

    assert vllm.__version__ >= "0.7.2", "OpenRLHF only supports vllm >= 0.7.2"

    vllm_engines = []
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    use_hybrid_engine = shared_pg is not None
    num_gpus = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1:
        # every worker will use 0.2 GPU, so that we can schedule
        # 2 instances on the same GPUs.
        num_gpus = 0.2

    if not use_hybrid_engine:
        # Create a big placement group to ensure that all engines are packed
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_engines * tensor_parallel_size)]
        shared_pg = placement_group(bundles, strategy="PACK")
        ray.get(shared_pg.ready())

    for i in range(num_engines):
        bundle_indices = None
        if tensor_parallel_size > 1:
            bundle_indices = list(range(i * tensor_parallel_size, (i + 1) * tensor_parallel_size))

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=shared_pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=i * tensor_parallel_size
        )

        if num_engines >= num_total_actors:
            num_actors = 1
        else:
            num_actors = num_total_actors // num_engines + int(i < num_total_actors % num_engines)

        vllm_engines.append(
            LLMRayActor.options(
                num_cpus=num_gpus,
=======
    inference_engines = []
    # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES will always be set in current context,
    # So we need to get env variables from ray process to check if it is set.
    noset_visible_devices = ray_noset_visible_devices(ray.get(get_all_env_variables.remote()))
    for i in range(num_engines):
        # When tensor_parallel_size=1 and RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is not set
        # (vLLM mp backend will work smoothly only when *_VISIBLE_DEVICES is modified),
        # vLLM init model in LLMEngine directly, assign 1 GPU for it.
        num_gpus = int(tensor_parallel_size == 1 and not noset_visible_devices)
        scheduling_strategy = None
        if tensor_parallel_size > 1 or noset_visible_devices:
            bundles = [{"GPU": 1, "CPU": 1}] * tensor_parallel_size
            pg = placement_group(bundles)
            ray.get(pg.ready())
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0
            )

        actor_cls = VllmLLMRayActor

        inference_engines.append(
            actor_cls.options(
                num_cpus=1,
>>>>>>> 06ce7866d66799af8bab597df52b00f4fa577aee:openrlhf/trainer/ray/vllm_llm_ray_actor.py
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                model=pretrain,
                enforce_eager=enforce_eager,
                worker_cls="openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap",
                tensor_parallel_size=tensor_parallel_size,
                seed=seed + i,
                distributed_executor_backend=distributed_executor_backend,
                max_model_len=max_model_len,
<<<<<<< HEAD:openrlhf/trainer/ray/vllm_engine.py
                enable_prefix_caching=enable_prefix_caching,
                dtype="bfloat16",
                trust_remote_code=True,
                num_actors=num_actors,
                gpu_memory_utilization=gpu_memory_utilization,
                bundle_indices=bundle_indices,
                num_gpus=0.2 if use_hybrid_engine else 1,
                enable_sleep_mode=vllm_enable_sleep,
                noset_visible_devices=ray_noset_visible_devices(),
            )
        )
    
    if vllm_enable_sleep:
        batch_vllm_engine_call(vllm_engines, "sleep", rank_0_only=False)

    return vllm_engines


def batch_vllm_engine_call(engines: List[Any], method_name: str, *args, rank_0_only: bool = True, **kwargs):
    """
    Batch call a method on multiple vLLM engines.
    Args:
        engines: List of vLLM engine instances
        method_name: Name of the method to call
        rank_0_only: Only execute on rank 0 if True
        *args: Positional arguments to pass to the method
        **kwargs: Keyword arguments to pass to the method
    Returns:
        List of results from ray.get() if on rank 0, None otherwise
    """
    import torch

    if rank_0_only and torch.distributed.get_rank() != 0:
        return None

    refs = []
    for engine in engines:
        method = getattr(engine, method_name)
        refs.append(method.remote(*args, **kwargs))

    return ray.get(refs)
=======
                backend="vllm",
            )
        )
    return inference_engines
>>>>>>> 06ce7866d66799af8bab597df52b00f4fa577aee:openrlhf/trainer/ray/vllm_llm_ray_actor.py
