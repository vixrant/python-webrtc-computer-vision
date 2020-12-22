# WebRTC & OpenCV in Python

This repo demonstrates how to perform image processing on a live p2p video chat using WebRTC.
I originally considered using [Pion WebRTC](https://github.com/pion/webrtc) because of its simpler API with
Celery for calling Python workers, but decided to stick with Python and Aiortc because my team in
the hackathon was using Python everywhere and it would have been easier to work with one technology rather than multiple.

## What it does

This repo creates an Aiohttp server with one endpoint: `/offer`. Using this endpoint you form a peer connection to this server
and send video to or from it. You can definitely replace Aiohttp with anything else for WebRTC signalling.

If you have 2 peers connected in a video chat, you need to add this server as another peer to form a mesh network of 3 peers.

```
Initially for peers 0 and 1:
0 ----- 1

Make connection to server s like:
0 ----- 1
 \     /
  \   /
    s
```

## What it requires

You need *Python 3.7* or above, because [asyncio.create_task()](https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task)
is used to run concurrent fuction calls. You can use some other job queue if you like, but I wanted to stick with vanilla python.

## How it works

In `pipeline.py`, I copied over the code for `MediaRecorder` in AioRTC and overwrote the `__run_track` function to create a
concurrent task for each frame received. This can definitely be done better using Celery, but this code was built in a
hackathon I haven't gotten around improving it. 

Currently it does have some framerate synchronization issues which could possibly be solved with better design. However,
this repo is only a reference to show how to perform computer vision concurrently with video streaming. A better approach
will be to use a sliding window/ queue in `pipeline.py`, so only the most recent frames are evaluated.
