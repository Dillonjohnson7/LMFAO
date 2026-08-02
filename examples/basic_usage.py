from lmfao import KernelPipeline


class ExampleRuntime:
    def upload_video(self, path):
        return {"device_buffer": path}

    def launch_kernel(self, kernel_name, grid, block, args, stream=None):
        print(kernel_name, grid, block, args, stream)


runtime = ExampleRuntime()
video = runtime.upload_video("path/to/video.mp4")

pipeline = KernelPipeline.from_config(
    [
        # Add registered feature configs here, for example:
        # {"name": "lighting.shadow", "params": {"strength": 0.5}, "probability": 0.75},
    ],
    seed=42,
)

augmented_video, metadata = pipeline(video, runtime, metadata={"source": "demo"})

print(augmented_video)
print(metadata)
