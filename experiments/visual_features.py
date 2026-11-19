"""Fixed 56-column evidence vectors built only from two rendered frames and an action."""
import numpy as np

GRID = 4


def original(before, issued, after):
    """The original front-tile luminance, contrast and motion summaries."""
    tile = before[:, 40:48, 24:32].astype(np.float32).mean(3) / 255
    local = tile.reshape(-1, 4, 2, 4, 2).mean((2, 4)).reshape(-1, 16)
    bounds = local.max(1, keepdims=True) - local.min(1, keepdims=True)
    shape = (local - local.min(1, keepdims=True)) / np.maximum(bounds, .03)
    delta = np.abs(after.astype(np.int16) - before)
    moved = delta.max((1, 2, 3)) > 0
    still = (~moved).astype(np.float32)[:, None]
    top = (before[:, :8] != 0).any((1, 2, 3))
    dark = (before[:, 8:48] == 0).mean((1, 2, 3))
    return np.column_stack((
        issued == 2, moved, np.log1p(delta.mean((1, 2, 3))),
        top, dark, still * top[:, None], still * dark[:, None],
        still * local, still * shape, still * bounds,
        still * shape * (~top)[:, None],
    )).astype(np.float32)


def luma(frame):
    return frame.mean(axis=-1, dtype=np.float32) / np.float32(255)

def patches(image):
    """Return N x 4 x 4 x pixels on equally spaced cells of one crop."""
    n, height, width = image.shape
    cells = image.reshape(n, GRID, height // GRID, GRID, width // GRID)
    return cells.transpose(0, 1, 3, 2, 4).reshape(n, GRID, GRID, -1)

def spatial(before, issued, after, *, aligned):
    delta = np.abs(after.astype(np.int16) - before)
    moved = delta.max(axis=(1,2,3)) > 0
    still = (~moved)[:,None]
    top = (before[:,:8] != 0).any(axis=(1,2,3))
    dark = (before[:,8:48] == 0).mean(axis=(1,2,3))
    front = patches(luma(before[:,40:48,24:32])).mean(axis=-1).reshape(len(before),16)
    span = front.max(axis=1,keepdims=True) - front.min(axis=1,keepdims=True)
    front = (front - front.min(axis=1,keepdims=True)) / np.maximum(span,.03)
    low = patches(luma(before[:,48:56,12:44])).mean(axis=-1).reshape(len(before),16)
    a = patches(luma(before[:,24:56,12:44]))
    b = patches(luma(after[:,24:56,12:44]))
    if not aligned:
        # Compare the same per-cell samples at sorted intensity ranks. This
        # preserves coarse location and intensity distributions but removes
        # the pairing of a particular before pixel with its after coordinate.
        a,b = np.sort(a,axis=-1),np.sort(b,axis=-1)
    effect = np.abs(b - a).mean(axis=-1).reshape(len(before),16)
    out = np.column_stack((
        issued == 2, moved, np.log1p(delta.mean(axis=(1, 2, 3))),
        top, dark, still[:, 0] * top, still[:, 0] * dark, still[:, 0] * low.mean(axis=1),
        still * front, still * low, effect,
    )).astype(np.float32)
    assert out.shape == (len(before), 56)
    return out

def aligned(before, issued, after):
    return spatial(before, issued, after, aligned=True)

def unaligned(before, issued, after):
    return spatial(before, issued, after, aligned=False)
