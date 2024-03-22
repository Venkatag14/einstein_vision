>Integrated PyTorch implementation of MIDAS for relative depth estimation from single images.
>Utilized torch.hub.load to load the "intel-isl/MiDaS" model with DPT large architecture.
>Transformed relative depth outputs to a uniform 0-100 scale across all frames.


>Implemented object detection using MMdet from openmmlab package.
>Installed mmdet library via openmim package for seamless integration.
>Employed Trackeval to evaluate object detection performance, leveraging mmdet as a third-party package.