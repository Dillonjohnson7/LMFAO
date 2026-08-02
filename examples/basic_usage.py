from lmfao import AugmentationPipeline


class ExampleRuntime:
    backend = "example"

    def load_video(self, path):
        return {"video": path}

    def execute(self, operation_name, video, params, metadata, stream=None):
        print(operation_name, video, params, stream)
        return video


runtime = ExampleRuntime()
video = runtime.load_video("path/to/episode")

pipeline = AugmentationPipeline.from_config(
    [
        # Add registered feature configs here after implementing features.
        # {"name": "occlusion.random_box", "params": {"area": 0.2}, "probability": 0.75},
    ],
    seed=42,
)

augmented_video, metadata = pipeline(video, runtime, metadata={"episode_id": "demo-001"})

print(augmented_video)
print(metadata)
