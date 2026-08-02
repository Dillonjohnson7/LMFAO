import numpy as np

from lmfao import AugmentationPipeline


video = np.full((16, 64, 64, 3), 128, dtype=np.uint8)

pipeline = AugmentationPipeline.from_config(
    [
        # Add registered feature configs here, for example:
        # {"name": "lighting", "params": {"strength": 0.5}},
    ],
    seed=42,
)

augmented_video, metadata = pipeline(video, metadata={"source": "demo"})

print(augmented_video.shape)
print(metadata)
