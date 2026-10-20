"""Helpers for development scripts: call a tool, parse the JSON block, save attached images."""
import base64, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.runtime import Context  # noqa

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
os.makedirs(OUT, exist_ok=True)


def parse(reply):
    """Return (meta dict, list of image bytes) from an MCP reply."""
    blocks = reply.get("content", [])
    meta = None
    images = []
    for b in blocks:
        if b.get("type") == "text" and meta is None:
            try:
                meta = json.loads(b["text"])
            except Exception:
                meta = {"_text": b["text"]}
        elif b.get("type") == "image":
            images.append(base64.b64decode(b["data"]))
    return meta, images


def call(ctx, tool, tag=None, **args):
    reply = ctx.call(tool, **args)
    meta, images = parse(reply)
    if tag:
        paths = []
        for i, data in enumerate(images):
            p = os.path.join(OUT, f"{tag}_{i}.png")
            with open(p, "wb") as f:
                f.write(data)
            paths.append(p)
        with open(os.path.join(OUT, f"{tag}.json"), "w") as f:
            json.dump(meta, f, indent=1, default=str)
        meta["_image_paths"] = paths
    return meta


def short(meta, drop=("camera_intrinsics", "camera_poses", "attachments")):
    """Compact view of a reply for printing."""
    if not isinstance(meta, dict):
        return meta
    return {k: v for k, v in meta.items() if k not in drop}
