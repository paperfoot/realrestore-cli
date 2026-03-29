"""Scheduler optimization module for inference step reduction.

Reducing inference steps from the default 28 to 4-8 with optimized
schedulers provides the highest wall-clock speedup (3.5-7x) without
requiring hardware changes. This module provides:

- Scheduler selection (DPM++, DDIM, Euler, UniPC) with quality presets
- Adaptive step counts based on degradation complexity
- Step-aware guidance decay (reduce CFG scale as steps progress)
- Scheduler benchmarking for empirical comparison

Research findings (2026-03-29):
- DPM++ 2M SDE Karras gives best quality at low step counts (4-8)
- UniPC is competitive and slightly faster per-step
- DDIM is the most stable fallback but needs more steps for quality
- Euler ancestral adds beneficial stochasticity for texture restoration
- CFG guidance decay prevents over-saturation at low step counts
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QualityPreset(Enum):
    """Quality presets controlling step count and scheduler choice."""

    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"


class SchedulerType(Enum):
    """Supported scheduler types from diffusers."""

    DPMPLUSPLUS_2M_SDE_KARRAS = "dpmpp_2m_sde_karras"
    DPMPLUSPLUS_2M_KARRAS = "dpmpp_2m_karras"
    DDIM = "ddim"
    EULER = "euler"
    EULER_ANCESTRAL = "euler_ancestral"
    UNIPC = "unipc"


@dataclass
class SchedulerConfig:
    """Configuration for a scheduler with step count and guidance settings."""

    scheduler_type: SchedulerType
    num_steps: int
    guidance_scale: float = 3.0
    guidance_decay: bool = True
    guidance_min_scale: float = 1.0

    def get_guidance_scale_at_step(self, step: int) -> float:
        """Compute guidance scale with linear decay over steps.

        Linearly reduces CFG scale from guidance_scale to guidance_min_scale
        as steps progress. This prevents over-saturation at low step counts
        where each step has outsized influence.
        """
        if not self.guidance_decay or self.num_steps <= 1:
            return self.guidance_scale

        progress = step / (self.num_steps - 1)
        scale_range = self.guidance_scale - self.guidance_min_scale
        return self.guidance_scale - (progress * scale_range)


# Preset configurations: maps each quality preset to its scheduler and step count
PRESET_CONFIGS: dict[QualityPreset, SchedulerConfig] = {
    QualityPreset.FAST: SchedulerConfig(
        scheduler_type=SchedulerType.DPMPLUSPLUS_2M_SDE_KARRAS,
        num_steps=8,
        guidance_scale=3.0,
        guidance_decay=True,
        guidance_min_scale=1.2,
    ),
    QualityPreset.BALANCED: SchedulerConfig(
        scheduler_type=SchedulerType.DPMPLUSPLUS_2M_SDE_KARRAS,
        num_steps=14,
        guidance_scale=3.0,
        guidance_decay=True,
        guidance_min_scale=1.5,
    ),
    QualityPreset.HIGH: SchedulerConfig(
        scheduler_type=SchedulerType.DPMPLUSPLUS_2M_KARRAS,
        num_steps=28,
        guidance_scale=3.0,
        guidance_decay=False,
    ),
}

# Adaptive step hints: simpler degradation types need fewer steps
TASK_COMPLEXITY: dict[str, int] = {
    "denoise": -2,       # Simple — noise removal converges fast
    "compression": -2,   # Simple — artifact removal is well-defined
    "deblur": 0,         # Medium — sharpening needs moderate steps
    "dehaze": 0,         # Medium — global tone correction
    "low_light": 0,      # Medium — brightness + detail recovery
    "derain": 2,         # Hard — structure under rain streaks
    "moire": 2,          # Hard — periodic pattern removal
    "lens_flare": 2,     # Hard — localized bright artifact removal
    "reflection": 4,     # Hardest — separating layers
    "auto": 0,           # Default — no adjustment
}


def get_scheduler_config(
    preset: str | QualityPreset = "balanced",
    task: str = "auto",
    adaptive: bool = True,
) -> SchedulerConfig:
    """Get scheduler configuration for a given preset and task.

    Args:
        preset: Quality preset name or enum value.
        task: Degradation task type for adaptive step adjustment.
        adaptive: Whether to adjust steps based on task complexity.

    Returns:
        SchedulerConfig with appropriate steps and scheduler.
    """
    if isinstance(preset, str):
        preset = QualityPreset(preset)

    config = PRESET_CONFIGS[preset]

    if not adaptive:
        return config

    # Adjust steps based on task complexity
    step_delta = TASK_COMPLEXITY.get(task, 0)
    adjusted_steps = max(4, min(config.num_steps + step_delta, 28))

    return SchedulerConfig(
        scheduler_type=config.scheduler_type,
        num_steps=adjusted_steps,
        guidance_scale=config.guidance_scale,
        guidance_decay=config.guidance_decay,
        guidance_min_scale=config.guidance_min_scale,
    )


def create_scheduler(
    scheduler_type: SchedulerType | str,
    pipeline: Any,
) -> Any:
    """Create and set a diffusers scheduler on the pipeline.

    Swaps the pipeline's scheduler to the requested type, preserving
    compatible config values from the existing scheduler.

    Args:
        scheduler_type: Which scheduler to use.
        pipeline: A diffusers pipeline instance.

    Returns:
        The pipeline with the new scheduler set.
    """
    if isinstance(scheduler_type, str):
        scheduler_type = SchedulerType(scheduler_type)

    existing_config = pipeline.scheduler.config

    scheduler_cls = _get_scheduler_class(scheduler_type)
    pipeline.scheduler = scheduler_cls.from_config(existing_config)

    return pipeline


def _get_scheduler_class(scheduler_type: SchedulerType) -> type:
    """Import and return the diffusers scheduler class."""
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        UniPCMultistepScheduler,
    )

    mapping: dict[SchedulerType, tuple[type, dict[str, Any]]] = {
        SchedulerType.DPMPLUSPLUS_2M_SDE_KARRAS: (
            DPMSolverMultistepScheduler,
            {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True},
        ),
        SchedulerType.DPMPLUSPLUS_2M_KARRAS: (
            DPMSolverMultistepScheduler,
            {"algorithm_type": "dpmsolver++", "use_karras_sigmas": True},
        ),
        SchedulerType.DDIM: (DDIMScheduler, {}),
        SchedulerType.EULER: (EulerDiscreteScheduler, {}),
        SchedulerType.EULER_ANCESTRAL: (EulerAncestralDiscreteScheduler, {}),
        SchedulerType.UNIPC: (UniPCMultistepScheduler, {}),
    }

    cls, extra_kwargs = mapping[scheduler_type]

    # Return a wrapper that applies extra kwargs during from_config
    if extra_kwargs:
        original_cls = cls

        class _ConfiguredScheduler(original_cls):
            @classmethod
            def from_config(cls, config, **kwargs):
                merged = {**extra_kwargs, **kwargs}
                return super().from_config(config, **merged)

        _ConfiguredScheduler.__name__ = original_cls.__name__
        _ConfiguredScheduler.__qualname__ = original_cls.__qualname__
        return _ConfiguredScheduler

    return cls


def apply_scheduler_preset(
    pipeline: Any,
    preset: str | QualityPreset = "balanced",
    task: str = "auto",
    adaptive: bool = True,
) -> tuple[Any, SchedulerConfig]:
    """Apply a quality preset to a pipeline, returning the config used.

    This is the main entry point for integrating scheduling optimization
    into the inference pipeline. It selects the scheduler and step count
    based on the preset and task complexity.

    Args:
        pipeline: A diffusers pipeline instance.
        preset: Quality preset (fast/balanced/high).
        task: Degradation task type for adaptive step adjustment.
        adaptive: Whether to adjust steps based on task complexity.

    Returns:
        Tuple of (modified pipeline, SchedulerConfig used).
    """
    config = get_scheduler_config(preset, task, adaptive)
    pipeline = create_scheduler(config.scheduler_type, pipeline)
    return pipeline, config


@dataclass
class SchedulerBenchmarkResult:
    """Result from benchmarking a single scheduler configuration."""

    scheduler_type: str
    num_steps: int
    elapsed_seconds: float
    seconds_per_step: float


def benchmark_schedulers(
    pipeline: Any,
    image: Any,
    prompt: str,
    negative_prompt: str = "",
    seed: int = 42,
    size_level: int = 512,
    schedulers: list[SchedulerType] | None = None,
    step_counts: list[int] | None = None,
) -> list[SchedulerBenchmarkResult]:
    """Benchmark multiple scheduler/step combinations on a single image.

    Runs inference with each scheduler+step combination and measures
    wall-clock time. Results are sorted by elapsed time (fastest first).

    Args:
        pipeline: A loaded diffusers pipeline (already on device).
        image: Input PIL Image.
        prompt: Text prompt for restoration.
        negative_prompt: Negative prompt.
        seed: Random seed for reproducibility.
        size_level: Size level for inference.
        schedulers: Schedulers to test (defaults to all).
        step_counts: Step counts to test (defaults to [4, 8, 14, 28]).

    Returns:
        List of SchedulerBenchmarkResult sorted by elapsed time.
    """
    import torch

    if schedulers is None:
        schedulers = list(SchedulerType)

    if step_counts is None:
        step_counts = [4, 8, 14, 28]

    # Save original scheduler to restore later
    original_scheduler = pipeline.scheduler

    results: list[SchedulerBenchmarkResult] = []

    for sched_type in schedulers:
        for steps in step_counts:
            try:
                create_scheduler(sched_type, pipeline)

                # Warmup synchronization
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()

                start = time.perf_counter()
                pipeline(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=3.0,
                    seed=seed,
                    size_level=size_level,
                )

                if torch.backends.mps.is_available():
                    torch.mps.synchronize()

                elapsed = time.perf_counter() - start

                results.append(SchedulerBenchmarkResult(
                    scheduler_type=sched_type.value,
                    num_steps=steps,
                    elapsed_seconds=round(elapsed, 3),
                    seconds_per_step=round(elapsed / steps, 3),
                ))
            except Exception:
                # Skip schedulers that fail (incompatible with pipeline)
                continue

    # Restore original scheduler
    pipeline.scheduler = original_scheduler

    results.sort(key=lambda r: r.elapsed_seconds)
    return results
