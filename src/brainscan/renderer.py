"""GPU renderer: maps weight tensors to an 8K texture via wgpu.

The renderer takes flattened weight data (one float per pixel), uploads it to a
storage buffer, and runs a fragment shader that applies a colourmap to produce
the final image. All colourmap computation happens on the GPU.

The bottom strip of the canvas carries two text lanes (audience speech and
model output) rendered from an antialiased coverage atlas, with each
model-lane character coloured by its softmax probability.

Two renderer classes are provided:
- OffscreenRenderer: render to a texture, read back as numpy array
- LiveRenderer: render directly to a fullscreen window via rendercanvas
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import wgpu

from brainscan import tuning
from brainscan.font import (
    LANE_GLYPH_H,
    LANE_GLYPH_W,
    generate_font_atlas,
    generate_lane_font_atlas,
)
from brainscan.layout import TextOverlay

if TYPE_CHECKING:
    from rendercanvas.contexts import WgpuContext
    from rendercanvas.glfw import RenderCanvas

SHADER_SOURCE = """
const ATTR_PARTIAL: u32 = 1u;
const ATTR_SOURCE_TAG: u32 = 2u;
const LANE_CELL_W: u32 = 32u;
const LANE_CELL_H: u32 = 64u;

struct Uniforms {
    width: u32,
    height: u32,
    param_count: u32,
    colormap: u32,
    audience_y: u32,
    audience_height: u32,
    audience_count: u32,
    audience_offset_px: u32,
    model_y: u32,
    model_height: u32,
    model_count: u32,
    model_offset_px: u32,
    vmax: f32,
    model_caret_col: u32,
    audience_pulse: f32,
    audience_edge_pulse: f32,
    global_brightness: f32,
    overlay_run_count: u32,
    stretch_k: f32,
    shimmer: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> font_data: array<u32>;
@group(0) @binding(3) var<storage, read> audience_chars: array<u32>;
@group(0) @binding(4) var<storage, read> audience_attrs: array<u32>;
@group(0) @binding(5) var<storage, read> model_chars: array<u32>;
@group(0) @binding(6) var<storage, read> model_probs: array<f32>;
@group(0) @binding(7) var<storage, read> overlay_chars: array<u32>;
@group(0) @binding(8) var<storage, read> overlay_runs: array<vec4<u32>>;
@group(0) @binding(9) var<storage, read> lane_font: array<u32>;
@group(0) @binding(10) var<storage, read> deltas: array<f32>;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var out: VertexOutput;
    out.pos = vec4<f32>(positions[idx], 0.0, 1.0);
    out.uv = uvs[idx];
    return out;
}

fn diverging(v: f32) -> vec3<f32> {
    // Black at zero: sign carries hue (negative blue, positive orange) and
    // magnitude carries luminance, desaturating towards white at the extremes
    // so outliers read as hot. Two linear segments per side: black -> mid on
    // |t| in [0, 0.5], mid -> top on [0.5, 1].
    let t = clamp(v, -1.0, 1.0);
    let a = abs(t);
    let lo = min(a * 2.0, 1.0);
    let hi = max(a * 2.0 - 1.0, 0.0);
    var mid: vec3<f32>;
    var top: vec3<f32>;
    if t < 0.0 {
        mid = vec3<f32>(0.05, 0.30, 0.72);
        top = vec3<f32>(0.64, 0.90, 1.00);
    } else {
        mid = vec3<f32>(0.74, 0.26, 0.02);
        top = vec3<f32>(1.00, 0.80, 0.38);
    }
    return mix(mid * lo, top, hi);
}

fn thermal(v: f32) -> vec3<f32> {
    let t = clamp((v + 1.0) * 0.5, 0.0, 1.0);
    let r = clamp(t * 3.0 - 1.0, 0.0, 1.0);
    let g = clamp(t * 3.0 - 2.0, 0.0, 1.0);
    let b = clamp(min(t * 3.0, 2.0 - t * 3.0), 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

fn asinh_stretch(v: f32, k: f32) -> f32 {
    // Signed log-like contrast stretch: expands small magnitudes and
    // compresses the outlier tail. Identity at v = -1, 0, +1. k <= 0 disables.
    if k < 1e-6 {
        return v;
    }
    return asinh(v * k) / asinh(k);
}

fn font_pixel(char_idx: u32, gx: u32, gy: u32) -> bool {
    let byte_offset = char_idx * 16u + gy;
    let word_idx = byte_offset / 4u;
    let byte_in_word = byte_offset % 4u;
    let word = font_data[word_idx];
    let byte_val = (word >> (byte_in_word * 8u)) & 0xFFu;
    return (byte_val & (0x80u >> gx)) != 0u;
}

fn lane_coverage(char_idx: u32, gx: u32, gy: u32) -> f32 {
    // 8-bit antialiased coverage from the 32x64 lane atlas, sampled 1:1.
    let texel = char_idx * (LANE_CELL_W * LANE_CELL_H) + gy * LANE_CELL_W + gx;
    let word = lane_font[texel / 4u];
    let byte_val = (word >> ((texel % 4u) * 8u)) & 0xFFu;
    return f32(byte_val) / 255.0;
}

fn overlay_pixel(px: u32, py: u32) -> vec4<f32> {
    // Returns dim-grey overlay colour if this pixel lights up an overlay
    // glyph; otherwise returns vec4(-1, ...) so the caller falls through to
    // weight rendering.
    for (var i: u32 = 0u; i < uniforms.overlay_run_count; i = i + 1u) {
        let run = overlay_runs[i];
        let rx = run.x;
        let ry = run.y;
        let length = run.z;
        let char_offset = run.w;
        let rw = length * 8u;
        let rh = 16u;
        if px < rx || px >= rx + rw || py < ry || py >= ry + rh {
            continue;
        }
        let local_x = px - rx;
        let glyph_col = local_x / 8u;
        let gx = local_x % 8u;
        let gy = py - ry;
        let glyph = overlay_chars[char_offset + glyph_col];
        if font_pixel(glyph, gx, gy) {
            return vec4<f32>(0.55, 0.55, 0.60, 1.0);
        }
        return vec4<f32>(-1.0, 0.0, 0.0, 1.0);
    }
    return vec4<f32>(-1.0, 0.0, 0.0, 1.0);
}

fn audience_band(px: u32, py: u32) -> vec4<f32> {
    if py < uniforms.audience_y || py >= uniforms.audience_y + uniforms.audience_height {
        return vec4<f32>(-1.0, 0.0, 0.0, 1.0);
    }
    let lane_py = py - uniforms.audience_y;
    let glyph_top = (uniforms.audience_height - LANE_CELL_H) / 2u;
    let bg = vec4<f32>(0.04, 0.04, 0.06, 1.0);
    var final_rgb: vec3<f32>;
    if lane_py < glyph_top || lane_py >= glyph_top + LANE_CELL_H {
        final_rgb = bg.rgb;
    } else {
        let scroll_x = px + uniforms.audience_offset_px;
        let col = scroll_x / LANE_CELL_W;
        if col >= uniforms.audience_count {
            final_rgb = bg.rgb;
        } else {
            let glyph = audience_chars[col];
            let gx = scroll_x % LANE_CELL_W;
            let gy = lane_py - glyph_top;
            let cov = lane_coverage(glyph, gx, gy);
            let attrs = audience_attrs[col];
            var c: vec3<f32>;
            if (attrs & ATTR_PARTIAL) != 0u {
                c = vec3<f32>(0.50, 0.46, 0.38);
            } else if (attrs & ATTR_SOURCE_TAG) != 0u {
                c = vec3<f32>(0.62, 0.56, 0.42) * (1.0 + uniforms.audience_pulse * 0.6);
            } else {
                c = vec3<f32>(0.94, 0.88, 0.72);
            }
            final_rgb = mix(bg.rgb, c, cov);
        }
    }
    if uniforms.audience_edge_pulse > 0.0 && px >= uniforms.width - 24u {
        final_rgb = final_rgb + vec3<f32>(0.15, 0.10, 0.05) * uniforms.audience_edge_pulse;
        final_rgb = clamp(final_rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    return vec4<f32>(final_rgb, 1.0);
}

fn model_band(px: u32, py: u32) -> vec4<f32> {
    if py < uniforms.model_y || py >= uniforms.model_y + uniforms.model_height {
        return vec4<f32>(-1.0, 0.0, 0.0, 1.0);
    }
    let lane_py = py - uniforms.model_y;
    let glyph_top = (uniforms.model_height - LANE_CELL_H) / 2u;
    let bg = vec4<f32>(0.02, 0.02, 0.04, 1.0);
    if lane_py < glyph_top || lane_py >= glyph_top + LANE_CELL_H {
        return bg;
    }
    let scroll_x = px + uniforms.model_offset_px;
    let col = scroll_x / LANE_CELL_W;

    // Caret check FIRST (before count guard so caret_col == count is reachable)
    if uniforms.model_caret_col != 0xFFFFFFFFu && col == uniforms.model_caret_col {
        if (scroll_x % LANE_CELL_W) < 6u {
            return vec4<f32>(0.85, 0.81, 0.94, 1.0);
        }
        return bg;
    }

    if col >= uniforms.model_count {
        return bg;
    }
    let glyph = model_chars[col];
    let gx = scroll_x % LANE_CELL_W;
    let gy = lane_py - glyph_top;
    let cov = lane_coverage(glyph, gx, gy);
    let prob = model_probs[col];
    let brightness = 0.25 + prob * 0.75;
    let c = vec3<f32>(brightness, brightness * 0.95, brightness * 1.10);
    return vec4<f32>(mix(bg.rgb, c, cov), 1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let px = u32(in.uv.x * f32(uniforms.width));
    let py = u32(in.uv.y * f32(uniforms.height));

    let cap_a = audience_band(px, py);
    if cap_a.x >= 0.0 {
        return vec4<f32>(cap_a.rgb * uniforms.global_brightness, 1.0);
    }

    let cap_m = model_band(px, py);
    if cap_m.x >= 0.0 {
        return vec4<f32>(cap_m.rgb * uniforms.global_brightness, 1.0);
    }

    let cap_o = overlay_pixel(px, py);
    if cap_o.x >= 0.0 {
        return vec4<f32>(cap_o.rgb * uniforms.global_brightness, 1.0);
    }

    let idx = py * uniforms.width + px;
    var final_rgb: vec3<f32>;
    if idx >= uniforms.param_count {
        final_rgb = vec3<f32>(0.05, 0.05, 0.05);
    } else {
        let raw = weights[idx];
        let w_lin = select(raw / uniforms.vmax, 0.0, uniforms.vmax < 1e-10);
        let w = asinh_stretch(w_lin, uniforms.stretch_k);
        if uniforms.colormap == 1u {
            final_rgb = thermal(w);
        } else {
            final_rgb = diverging(w);
        }
        // Floor tint (tuning.WEIGHT_FLOOR_TINT): lift genuine weights off
        // true black so they survive crushed blacks on TV panels. Padding
        // (gutters, label bands, rect tails) is exactly 0.0 and stays black,
        // which keeps the matrix rects reading as separate panels.
        if raw != 0.0 {
            final_rgb = max(final_rgb, vec3<f32>(0.030, 0.032, 0.052));
        }
        // Learning shimmer: deltas holds pre-normalised |delta-w| in 0..1;
        // shimmer carries strength x decay, so each weight upload flashes
        // where training moved parameters and fades until the next one.
        if uniforms.shimmer > 0.0 {
            let flash = vec3<f32>(0.45, 1.0, 0.75) * (deltas[idx] * uniforms.shimmer);
            final_rgb = clamp(final_rgb + flash, vec3<f32>(0.0), vec3<f32>(1.0));
        }
    }
    return vec4<f32>(final_rgb * uniforms.global_brightness, 1.0);
}
"""

COLORMAP_DIVERGING = 0
COLORMAP_THERMAL = 1

# LANE_GLYPH_W / LANE_GLYPH_H (the 32x64 lane cell) come from font.py, which
# owns the atlas geometry; LANE_SCALE is their ratio to the 8x16 chrome glyph.
LANE_SCALE = 4

_UNIFORM_DTYPE = np.dtype([
    ("width", np.uint32),
    ("height", np.uint32),
    ("param_count", np.uint32),
    ("colormap", np.uint32),
    ("audience_y", np.uint32),
    ("audience_height", np.uint32),
    ("audience_count", np.uint32),
    ("audience_offset_px", np.uint32),
    ("model_y", np.uint32),
    ("model_height", np.uint32),
    ("model_count", np.uint32),
    ("model_offset_px", np.uint32),
    ("vmax", np.float32),
    ("model_caret_col", np.uint32),
    ("audience_pulse", np.float32),
    ("audience_edge_pulse", np.float32),
    ("global_brightness", np.float32),
    ("overlay_run_count", np.uint32),
    ("stretch_k", np.float32),
    ("shimmer", np.float32),
])


def _pick_surface_format(context: WgpuContext, device: wgpu.GPUDevice) -> str:
    preferred = context.get_preferred_format(device.adapter)
    if "srgb" not in preferred:
        return preferred
    linear = preferred.replace("-srgb", "")
    return linear


@dataclass(frozen=True)
class RenderConfig:
    width: int
    height: int
    colormap: int = COLORMAP_DIVERGING
    audience_height: int = 0
    model_height: int = 0

    @property
    def model_y(self) -> int:
        if self.model_height == 0:
            return 0
        return self.height - self.model_height

    @property
    def audience_y(self) -> int:
        if self.audience_height == 0:
            return 0
        return max(0, self.height - self.model_height - self.audience_height)

    @property
    def lane_capacity(self) -> int:
        return max(1, self.width // LANE_GLYPH_W)


@dataclass
class RenderResources:
    device: wgpu.GPUDevice
    config: RenderConfig
    uniform_data: np.ndarray
    uniform_buffer: wgpu.GPUBuffer
    weight_buffer: wgpu.GPUBuffer
    delta_buffer: wgpu.GPUBuffer
    font_buffer: wgpu.GPUBuffer
    lane_font_buffer: wgpu.GPUBuffer
    audience_chars_buffer: wgpu.GPUBuffer
    audience_attrs_buffer: wgpu.GPUBuffer
    model_chars_buffer: wgpu.GPUBuffer
    model_probs_buffer: wgpu.GPUBuffer
    overlay_chars_buffer: wgpu.GPUBuffer
    overlay_runs_buffer: wgpu.GPUBuffer
    bind_group: wgpu.GPUBindGroup
    pipeline: wgpu.GPURenderPipeline


def get_device() -> wgpu.GPUDevice:
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    return adapter.request_device_sync()


def flatten_weights(
    weights: dict[str, np.ndarray],
    layout_order: list[str] | None = None,
) -> tuple[np.ndarray, int]:
    """Flatten weight tensors into a single float32 array.

    Returns the flattened array and the total parameter count.
    """
    if layout_order is None:
        layout_order = list(weights.keys())
    parts = [weights[name].ravel() for name in layout_order if name in weights]
    flat = np.concatenate(parts).astype(np.float32)
    return flat, len(flat)


def create_render_pipeline(
    config: RenderConfig,
    device: wgpu.GPUDevice,
    target_format: str,
) -> RenderResources:
    uniform_data = np.zeros(1, dtype=_UNIFORM_DTYPE)
    uniform_data["width"] = config.width
    uniform_data["height"] = config.height
    uniform_data["colormap"] = config.colormap
    uniform_data["audience_y"] = config.audience_y
    uniform_data["audience_height"] = config.audience_height
    uniform_data["model_y"] = config.model_y
    uniform_data["model_height"] = config.model_height
    uniform_data["model_caret_col"] = np.uint32(0xFFFFFFFF)
    uniform_data["audience_pulse"] = np.float32(0.0)
    uniform_data["audience_edge_pulse"] = np.float32(0.0)
    uniform_data["global_brightness"] = np.float32(1.0)
    uniform_data["overlay_run_count"] = np.uint32(0)
    uniform_data["stretch_k"] = np.float32(tuning.WEIGHT_STRETCH_K)
    uniform_data["shimmer"] = np.float32(0.0)

    uniform_size = max(((_UNIFORM_DTYPE.itemsize + 15) // 16) * 16, 64)
    uniform_buffer = device.create_buffer(
        size=uniform_size,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    max_params = config.width * config.height
    weight_buffer = device.create_buffer(
        size=max_params * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    # |delta-w| between consecutive weight uploads; wgpu zero-initialises,
    # so shimmer is a no-op until the first delta upload.
    delta_buffer = device.create_buffer(
        size=max_params * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    font_size = 1024 * 4
    font_buffer = device.create_buffer(
        size=font_size,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    font_data = generate_font_atlas()
    device.queue.write_buffer(font_buffer, 0, font_data.tobytes())

    lane_font_data = generate_lane_font_atlas()
    lane_font_buffer = device.create_buffer(
        size=lane_font_data.nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    device.queue.write_buffer(lane_font_buffer, 0, lane_font_data.tobytes())

    cap = max(config.lane_capacity, 1)
    audience_chars_buffer = device.create_buffer(
        size=cap * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    audience_attrs_buffer = device.create_buffer(
        size=cap * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    model_chars_buffer = device.create_buffer(
        size=cap * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    model_probs_buffer = device.create_buffer(
        size=cap * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    OVERLAY_CHARS_CAP = 1024  # ample for ~50 overlays of short text
    OVERLAY_RUNS_CAP = 256    # at most ~50 runs in production
    overlay_chars_buffer = device.create_buffer(
        size=OVERLAY_CHARS_CAP * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    overlay_runs_buffer = device.create_buffer(
        size=OVERLAY_RUNS_CAP * 16,  # vec4<u32> = 16 bytes
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": i,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {
                    "type": wgpu.BufferBindingType.uniform
                    if i == 0
                    else wgpu.BufferBindingType.read_only_storage
                },
            }
            for i in range(11)
        ]
    )

    buffers = [
        uniform_buffer,
        weight_buffer,
        font_buffer,
        audience_chars_buffer,
        audience_attrs_buffer,
        model_chars_buffer,
        model_probs_buffer,
        overlay_chars_buffer,
        overlay_runs_buffer,
        lane_font_buffer,
        delta_buffer,
    ]
    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": i,
                "resource": {
                    "buffer": buf,
                    "offset": 0,
                    "size": buf.size,
                },
            }
            for i, buf in enumerate(buffers)
        ],
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    shader = device.create_shader_module(code=SHADER_SOURCE)
    pipeline = device.create_render_pipeline(
        layout=pipeline_layout,
        vertex={"module": shader, "entry_point": "vs_main"},
        fragment={
            "module": shader,
            "entry_point": "fs_main",
            "targets": [{"format": target_format}],
        },
        primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
    )

    return RenderResources(
        device=device,
        config=config,
        uniform_data=uniform_data,
        uniform_buffer=uniform_buffer,
        weight_buffer=weight_buffer,
        delta_buffer=delta_buffer,
        font_buffer=font_buffer,
        lane_font_buffer=lane_font_buffer,
        audience_chars_buffer=audience_chars_buffer,
        audience_attrs_buffer=audience_attrs_buffer,
        model_chars_buffer=model_chars_buffer,
        model_probs_buffer=model_probs_buffer,
        overlay_chars_buffer=overlay_chars_buffer,
        overlay_runs_buffer=overlay_runs_buffer,
        bind_group=bind_group,
        pipeline=pipeline,
    )


def upload_overlays(
    res: RenderResources, overlays: list[TextOverlay]
) -> None:
    """Pack overlays and upload to the static GPU buffers.

    Stores ``overlay_run_count`` in ``res.uniform_data`` so the next ``draw()``
    call sees the new count. Caller is responsible for triggering a draw.
    """
    chars: list[int] = []
    runs: list[tuple[int, int, int, int]] = []
    for ov in overlays:
        char_offset = len(chars)
        for ch in ov.text:
            chars.append(ord(ch) & 0xFF)
        runs.append((ov.x, ov.y, len(ov.text), char_offset))

    chars_arr = (
        np.array(chars, dtype=np.uint32) if chars else np.zeros(1, dtype=np.uint32)
    )
    runs_arr = (
        np.array(runs, dtype=np.uint32).reshape(-1, 4)
        if runs else np.zeros((1, 4), dtype=np.uint32)
    )

    res.device.queue.write_buffer(res.overlay_chars_buffer, 0, chars_arr.tobytes())
    res.device.queue.write_buffer(res.overlay_runs_buffer, 0, runs_arr.tobytes())
    res.uniform_data["overlay_run_count"] = np.uint32(len(runs))


@dataclass
class LaneFrame:
    chars: np.ndarray
    attrs_or_probs: np.ndarray
    count: int
    offset_px: int = 0
    caret_col: int = -1
    pulse: float = 0.0
    edge_pulse: float = 0.0


def draw(
    res: RenderResources,
    target_view: wgpu.GPUTextureView,
    flat_weights: np.ndarray,
    audience: LaneFrame | None = None,
    model: LaneFrame | None = None,
    global_brightness: float = 1.0,
    stretch_k: float = tuning.WEIGHT_STRETCH_K,
    vmax: float | None = None,
    upload_weights: bool = True,
    flat_deltas: np.ndarray | None = None,
    shimmer: float = 0.0,
) -> None:
    """Render one frame to ``target_view``.

    ``vmax`` overrides the normalisation max; when ``None`` it is computed
    from ``flat_weights``. With ``upload_weights=False`` the weight storage
    buffer is not re-written and whatever was last uploaded is reused; the
    live renderer sets this when only the text lanes changed, so an
    unchanged weight buffer is not re-sent to the GPU on every present.

    ``flat_deltas`` carries pre-normalised (0..1) |delta-w| per pixel and is
    uploaded under the same ``upload_weights`` gate; ``shimmer`` scales the
    additive flash it produces (0 disables, regardless of buffer contents).
    """
    device = res.device
    param_count = len(flat_weights)

    audience_count = 0
    audience_offset_px = 0
    if audience is not None:
        audience_count = audience.count
        audience_offset_px = audience.offset_px
        device.queue.write_buffer(
            res.audience_chars_buffer,
            0,
            audience.chars.astype(np.uint32).tobytes(),
        )
        device.queue.write_buffer(
            res.audience_attrs_buffer,
            0,
            audience.attrs_or_probs.astype(np.uint32).tobytes(),
        )

    model_count = 0
    model_offset_px = 0
    if model is not None:
        model_count = model.count
        model_offset_px = model.offset_px
        device.queue.write_buffer(
            res.model_chars_buffer,
            0,
            model.chars.astype(np.uint32).tobytes(),
        )
        device.queue.write_buffer(
            res.model_probs_buffer,
            0,
            model.attrs_or_probs.astype(np.float32).tobytes(),
        )

    if vmax is None:
        vmax = float(np.max(np.abs(flat_weights))) if param_count > 0 else 0.0

    model_caret_col_val = np.uint32(0xFFFFFFFF)
    audience_pulse_val = np.float32(0.0)
    audience_edge_pulse_val = np.float32(0.0)
    if model is not None:
        model_caret_col_val = (
            np.uint32(0xFFFFFFFF) if model.caret_col < 0
            else np.uint32(model.caret_col)
        )
    if audience is not None:
        audience_pulse_val = np.float32(audience.pulse)
        audience_edge_pulse_val = np.float32(audience.edge_pulse)

    res.uniform_data["param_count"] = param_count
    res.uniform_data["colormap"] = res.config.colormap
    res.uniform_data["audience_count"] = audience_count
    res.uniform_data["audience_offset_px"] = audience_offset_px
    res.uniform_data["model_count"] = model_count
    res.uniform_data["model_offset_px"] = model_offset_px
    res.uniform_data["vmax"] = vmax
    res.uniform_data["model_caret_col"] = model_caret_col_val
    res.uniform_data["audience_pulse"] = audience_pulse_val
    res.uniform_data["audience_edge_pulse"] = audience_edge_pulse_val
    res.uniform_data["global_brightness"] = np.float32(global_brightness)
    res.uniform_data["stretch_k"] = np.float32(stretch_k)
    res.uniform_data["shimmer"] = np.float32(shimmer)
    device.queue.write_buffer(
        res.uniform_buffer, 0, res.uniform_data.tobytes()
    )

    if upload_weights:
        device.queue.write_buffer(
            res.weight_buffer, 0, flat_weights.astype(np.float32).tobytes()
        )
        if flat_deltas is not None:
            device.queue.write_buffer(
                res.delta_buffer, 0, flat_deltas.astype(np.float32).tobytes()
            )

    command_encoder = device.create_command_encoder()
    render_pass = command_encoder.begin_render_pass(
        color_attachments=[
            {
                "view": target_view,
                "resolve_target": None,
                "clear_value": (0.05, 0.05, 0.05, 1.0),
                "load_op": "clear",
                "store_op": "store",
            }
        ]
    )
    render_pass.set_pipeline(res.pipeline)
    render_pass.set_bind_group(0, res.bind_group)
    render_pass.draw(3, 1, 0, 0)
    render_pass.end()
    device.queue.submit([command_encoder.finish()])


class OffscreenRenderer:
    """Render weight data to an offscreen texture and read back as numpy."""

    def __init__(
        self,
        width: int,
        height: int,
        device: wgpu.GPUDevice | None = None,
        colormap: int = COLORMAP_DIVERGING,
        audience_height: int = 0,
        model_height: int = 0,
    ):
        self.width = width
        self.height = height
        self.colormap = colormap
        self.device = device or get_device()

        self.config = RenderConfig(
            width, height, colormap, audience_height, model_height
        )
        self._res = create_render_pipeline(
            self.config, self.device, wgpu.TextureFormat.rgba8unorm
        )

        self._target_texture = self.device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )

    def render(
        self,
        flat_weights: np.ndarray,
        audience: LaneFrame | None = None,
        model: LaneFrame | None = None,
        global_brightness: float = 1.0,
        stretch_k: float = tuning.WEIGHT_STRETCH_K,
        vmax: float | None = None,
        deltas: np.ndarray | None = None,
        shimmer: float = 0.0,
    ) -> np.ndarray:
        """Render weight data and return RGBA image as numpy array."""
        target_view = self._target_texture.create_view()
        draw(
            self._res, target_view, flat_weights, audience, model,
            global_brightness, stretch_k, vmax=vmax,
            flat_deltas=deltas, shimmer=shimmer,
        )

        data = self.device.queue.read_texture(
            {"texture": self._target_texture, "mip_level": 0, "origin": (0, 0, 0)},
            {"bytes_per_row": self.width * 4},
            (self.width, self.height, 1),
        )
        return np.frombuffer(data, dtype=np.uint8).reshape(
            self.height, self.width, 4
        )

    def set_overlays(self, overlays: list[TextOverlay]) -> None:
        upload_overlays(self._res, overlays)


class LiveRenderer:
    """Render weight data directly to a fullscreen window.

    The training loop runs in a background thread and calls ``update()`` to
    push new weight data. The canvas event loop runs on the main thread via
    ``run()``.
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        device: wgpu.GPUDevice | None = None,
        colormap: int = COLORMAP_DIVERGING,
        audience_height: int = 0,
        model_height: int = 0,
        fullscreen: bool = True,
        max_fps: int = 30,
        canvas: RenderCanvas | None = None,
        display_size: tuple[int, int] | None = None,
    ):
        self.width = width
        self.height = height
        self.device = device or get_device()

        if fullscreen and display_size is None and canvas is None:
            # Size the GLFW window to the primary monitor up front so the wgpu
            # surface is configured at the physical screen size from the start.
            # Without this, the surface is born at the logical canvas size
            # (e.g. 7680x4320) and a later resize in _go_fullscreen can leave
            # the swapchain mis-sized, cropping the bottom of the render.
            import glfw

            glfw.init()
            mode = glfw.get_video_mode(glfw.get_primary_monitor())
            display_size = (mode.size.width, mode.size.height)

        window_w, window_h = display_size or (width, height)

        if canvas is None:
            from rendercanvas.glfw import RenderCanvas

            canvas = RenderCanvas(
                size=(window_w, window_h),
                title="LLM Brainscan",
                update_mode="continuous",
                max_fps=max_fps,
            )
        self._canvas: RenderCanvas = canvas

        self._context = self._canvas.get_wgpu_context()
        surface_format = _pick_surface_format(self._context, self.device)
        self._context.configure(device=self.device, format=surface_format)

        self.config = RenderConfig(
            width, height, colormap, audience_height, model_height
        )
        self._res = create_render_pipeline(
            self.config, self.device, surface_format
        )

        self._lock = threading.Lock()
        self._flat_weights: np.ndarray | None = None
        self._vmax: float = 0.0
        self._weights_dirty: bool = False
        self._deltas: np.ndarray | None = None
        self._weights_uploaded_at: float = 0.0
        self._audience: LaneFrame | None = None
        self._model: LaneFrame | None = None
        self._global_brightness: float = 1.0
        self._stretch_k: float = tuning.WEIGHT_STRETCH_K

        if fullscreen:
            self._go_fullscreen()

        self._canvas.request_draw(self._draw)

    def update(
        self,
        flat_weights: np.ndarray,
        audience: LaneFrame | None = None,
        model: LaneFrame | None = None,
        global_brightness: float = 1.0,
        stretch_k: float = tuning.WEIGHT_STRETCH_K,
        vmax: float | None = None,
        deltas: np.ndarray | None = None,
    ) -> None:
        """Thread-safe update of weight and lane data for the next frame.

        Caches the normalisation max and marks the weight buffer dirty so the
        next ``_draw`` re-uploads it. Use ``update_lanes()`` when the weights
        have not changed, to skip that upload. ``deltas`` (pre-normalised
        0..1 |delta-w|) triggers the learning shimmer, which flashes on the
        upload and decays with ``tuning.SHIMMER_HALF_LIFE_SECONDS``.
        """
        weights = flat_weights.astype(np.float32, copy=True)
        if vmax is None:
            vmax = float(np.max(np.abs(weights))) if weights.size else 0.0
        with self._lock:
            self._flat_weights = weights
            self._vmax = vmax
            self._weights_dirty = True
            # No defensive copy: the train loop hands over a freshly built
            # array each snapshot and never mutates it afterwards.
            self._deltas = (
                deltas.astype(np.float32, copy=False) if deltas is not None else None
            )
            self._weights_uploaded_at = time.monotonic()
            self._audience = audience
            self._model = model
            self._global_brightness = global_brightness
            self._stretch_k = stretch_k

    def update_lanes(
        self,
        audience: LaneFrame | None = None,
        model: LaneFrame | None = None,
        global_brightness: float = 1.0,
    ) -> None:
        """Thread-safe update of just the text lanes and global brightness.

        Leaves the cached weight buffer untouched, so the cheap per-loop
        lane snapshot can refresh the bottom text strip (and the rebirth
        fade) between the costly weight captures that drive ``update()``.
        """
        with self._lock:
            self._audience = audience
            self._model = model
            self._global_brightness = global_brightness

    def _draw(self) -> None:
        with self._lock:
            weights = self._flat_weights
            vmax = self._vmax
            upload_weights = self._weights_dirty
            self._weights_dirty = False
            deltas = self._deltas
            uploaded_at = self._weights_uploaded_at
            audience = self._audience
            model = self._model
            global_brightness = self._global_brightness
            stretch_k = self._stretch_k

        if weights is None:
            return

        shimmer = 0.0
        if deltas is not None and tuning.SHIMMER_STRENGTH > 0.0:
            elapsed = time.monotonic() - uploaded_at
            shimmer = tuning.SHIMMER_STRENGTH * (
                0.5 ** (elapsed / tuning.SHIMMER_HALF_LIFE_SECONDS)
            )

        # WgpuContext.get_current_texture is typed as `object` upstream;
        # the runtime value is always a wgpu.GPUTexture.
        texture = cast("wgpu.GPUTexture", self._context.get_current_texture())
        draw(
            self._res, texture.create_view(), weights, audience, model,
            global_brightness, stretch_k, vmax=vmax,
            upload_weights=upload_weights,
            flat_deltas=deltas, shimmer=shimmer,
        )

    def _go_fullscreen(self) -> None:
        # Strip decorations and size to the monitor; on GNOME / Mutter that
        # still leaves the top panel in place, so also send the EWMH
        # _NET_WM_STATE_FULLSCREEN hint via wmctrl. glfw.set_window_monitor
        # (exclusive fullscreen) doesn't behave reliably under Mutter.
        import glfw

        # _window is a private rendercanvas internal not in the public API
        # but stable across the versions we use.
        window = self._canvas._window
        if window is None:
            raise RuntimeError("LiveRenderer: GLFW window was not created")
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        w, h = mode.size.width, mode.size.height
        glfw.set_window_attrib(window, glfw.DECORATED, glfw.FALSE)
        glfw.set_window_attrib(window, glfw.AUTO_ICONIFY, glfw.FALSE)
        glfw.set_window_pos(window, 0, 0)
        glfw.set_window_size(window, w, h)

        if hasattr(glfw, "get_x11_window"):
            xid = glfw.get_x11_window(window)
            if xid:
                import subprocess

                subprocess.run(
                    [
                        "wmctrl", "-i", "-r", f"0x{xid:08x}",
                        "-b", "add,fullscreen",
                    ],
                    check=False, timeout=2,
                )

    def run(self) -> None:
        """Enter the canvas event loop (blocks until the window is closed)."""
        group = self._canvas._rc_canvas_group
        loop = group.get_loop()
        assert loop is not None, "RenderCanvas has no event loop"
        loop.run()

    def close(self) -> None:
        self._canvas.close()

    def set_overlays(self, overlays: list[TextOverlay]) -> None:
        upload_overlays(self._res, overlays)
