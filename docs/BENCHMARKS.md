# Historical GPU experiment

These values were recorded by an earlier Kaggle run, not remeasured during the publication audit. They are PyTorch allocator-reserved peaks in MiB, NOT total device VRAM. CPU RAM, driver memory, other allocators and device-wide peaks were not measured.

| Stage | Seconds | Recorded peak reserved MiB |
| --- | ---: | ---: |
| Panorama | 200.65 | 634 |
| Depth | 2.65 | 202 |
| Lift | 330.75 | 152 |
| Occlusion | 3.60 | 106 |

Reported device: Tesla T4; resolution: 768; sequential offload enabled. Reported splat count: 294,912 before versus 294,911 after the full pipeline.

These logs do not prove image-quality improvement, successful execution of every inpaint branch, or the exact number of fitting iterations. Historical fitting code swallowed exceptions. The publication audit removes that silent fallback; a new instrumented GPU run is still required. No offload-disabled footprint is established here. The pole-inpaint experiment produced artifacts and was disabled.
