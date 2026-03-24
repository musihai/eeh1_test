from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module


VALID_LEGACY_WORKER_IMPLS = {"auto", "enable", "disable"}


@dataclass(frozen=True)
class ImportSpec:
    module_path: str
    attribute_name: str

    def load(self):
        module = import_module(self.module_path)
        return getattr(module, self.attribute_name)


@dataclass(frozen=True)
class ActorRolloutSpec:
    worker: ImportSpec
    ray_worker_group: ImportSpec
    role_name: str


@dataclass(frozen=True)
class RegisteredWorkerSpec:
    worker: ImportSpec
    pool_name: str = "global_pool"


def resolve_legacy_worker_impl(config) -> str:
    mode = config.trainer.get("use_legacy_worker_impl", "auto")
    if mode not in VALID_LEGACY_WORKER_IMPLS:
        raise ValueError(f"Invalid use_legacy_worker_impl: {mode}")
    return mode


def build_actor_rollout_spec(config) -> ActorRolloutSpec:
    legacy_mode = resolve_legacy_worker_impl(config)
    if legacy_mode == "disable":
        role_name = (
            "ActorRolloutRef"
            if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss
            else "ActorRollout"
        )
        return ActorRolloutSpec(
            worker=ImportSpec("verl.workers.engine_workers", "ActorRolloutRefWorker"),
            ray_worker_group=ImportSpec("verl.single_controller.ray", "RayWorkerGroup"),
            role_name=role_name,
        )

    rollout_mode = config.actor_rollout_ref.rollout.mode
    if rollout_mode == "sync":
        raise ValueError(
            "Rollout mode 'sync' has been removed. Please set "
            "`actor_rollout_ref.rollout.mode=async` to use the native server rollout."
        )

    strategy = config.actor_rollout_ref.actor.strategy
    if strategy in {"fsdp", "fsdp2"}:
        module_path = "verl.workers.fsdp_workers"
    elif strategy == "megatron":
        module_path = "verl.workers.megatron_workers"
    else:
        raise NotImplementedError

    class_name = "AsyncActorRolloutRefWorker" if rollout_mode == "async" else "ActorRolloutRefWorker"
    return ActorRolloutSpec(
        worker=ImportSpec(module_path, class_name),
        ray_worker_group=ImportSpec("verl.single_controller.ray", "RayWorkerGroup"),
        role_name="ActorRollout",
    )


def build_critic_worker_spec(config) -> RegisteredWorkerSpec:
    strategy = config.critic.strategy
    if strategy in {"fsdp", "fsdp2"}:
        legacy_mode = resolve_legacy_worker_impl(config)
        if legacy_mode in {"auto", "enable"}:
            module_path = "verl.workers.fsdp_workers"
        else:
            module_path = "verl.workers.engine_workers"
    elif strategy == "megatron":
        module_path = "verl.workers.megatron_workers"
    else:
        raise NotImplementedError

    return RegisteredWorkerSpec(worker=ImportSpec(module_path, "CriticWorker"))


def build_reward_model_worker_spec(config) -> RegisteredWorkerSpec | None:
    if not config.reward_model.enable:
        return None

    resolve_legacy_worker_impl(config)
    strategy = config.reward_model.strategy
    if strategy in {"fsdp", "fsdp2"}:
        module_path = "verl.workers.fsdp_workers"
    elif strategy == "megatron":
        module_path = "verl.workers.megatron_workers"
    else:
        raise NotImplementedError

    pool_name = "reward_pool" if config.reward_model.enable_resource_pool else "global_pool"
    return RegisteredWorkerSpec(worker=ImportSpec(module_path, "RewardModelWorker"), pool_name=pool_name)


def should_register_ref_policy(config) -> bool:
    return resolve_legacy_worker_impl(config) != "disable" and (
        config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss
    )


def build_resource_pool_spec(config) -> dict[str, list[int]]:
    resource_pool_spec = {
        "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }

    if config.reward_model.enable_resource_pool:
        if config.reward_model.n_gpus_per_node <= 0:
            raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
        if config.reward_model.nnodes <= 0:
            raise ValueError("config.reward_model.nnodes must be greater than 0")

        resource_pool_spec["reward_pool"] = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes

    return resource_pool_spec
