"""GPU renderer: maps weight tensors to an 8K texture via wgpu.

The renderer takes flattened weight data (one float per pixel), uploads it to a
storage buffer, and runs a fragment shader that applies a colourmap to produce
the final image. All colourmap computation happens on the GPU.

The bottom strip of the canvas can display generated text using a bitmap font,
with each character coloured by its softmax probability.

Two renderer classes are provided:
- OffscreenRenderer: render to a texture, read back as numpy array
- LiveRenderer: render directly to a fullscreen window via rendercanvas
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import wgpu

from brainscan.layout import TextOverlay

SHADER_SOURCE = """
const ATTR_PARTIAL: u32 = 1u;
const ATTR_SOURCE_TAG: u32 = 2u;
const LANE_SCALE: u32 = 4u;
const LANE_CELL_W: u32 = 32u;
const LANE_CELL_H: u32 = 64u;
const CAPTIONS_SCALE: u32 = 2u;
const CAPTIONS_CELL_W: u32 = 16u;
const CAPTIONS_CELL_H: u32 = 32u;

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
    captions_y: u32,
    captions_height: u32,
    captions_count: u32,
    vmax: f32,
    model_caret_col: u32,
    audience_pulse: f32,
    audience_edge_pulse: f32,
    global_brightness: f32,
    overlay_run_count: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> font_data: array<u32>;
@group(0) @binding(3) var<storage, read> audience_chars: array<u32>;
@group(0) @binding(4) var<storage, read> audience_attrs: array<u32>;
@group(0) @binding(5) var<storage, read> model_chars: array<u32>;
@group(0) @binding(6) var<storage, read> model_probs: array<f32>;
@group(0) @binding(7) var<storage, read> captions_chars: array<u32>;
@group(0) @binding(8) var<storage, read> overlay_chars: array<u32>;
@group(0) @binding(9) var<storage, read> overlay_runs: array<vec4<u32>>;

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
    let t = clamp(v, -1.0, 1.0);
    let r = clamp(0.5 + t * 0.5, 0.0, 1.0);
    let g = clamp(0.5 - abs(t) * 0.4, 0.0, 1.0);
    let b = clamp(0.5 - t * 0.5, 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

fn thermal(v: f32) -> vec3<f32> {
    let t = clamp((v + 1.0) * 0.5, 0.0, 1.0);
    let r = clamp(t * 3.0 - 1.0, 0.0, 1.0);
    let g = clamp(t * 3.0 - 2.0, 0.0, 1.0);
    let b = clamp(min(t * 3.0, 2.0 - t * 3.0), 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

fn font_pixel(char_idx: u32, gx: u32, gy: u32) -> bool {
    let byte_offset = char_idx * 16u + gy;
    let word_idx = byte_offset / 4u;
    let byte_in_word = byte_offset % 4u;
    let word = font_data[word_idx];
    let byte_val = (word >> (byte_in_word * 8u)) & 0xFFu;
    return (byte_val & (0x80u >> gx)) != 0u;
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
            let gx = (scroll_x % LANE_CELL_W) / LANE_SCALE;
            let gy = (lane_py - glyph_top) / LANE_SCALE;
            if font_pixel(glyph, gx, gy) {
                let attrs = audience_attrs[col];
                var c: vec3<f32>;
                if (attrs & ATTR_PARTIAL) != 0u {
                    c = vec3<f32>(0.50, 0.46, 0.38);
                } else if (attrs & ATTR_SOURCE_TAG) != 0u {
                    c = vec3<f32>(0.62, 0.56, 0.42) * (1.0 + uniforms.audience_pulse * 0.6);
                } else {
                    c = vec3<f32>(0.94, 0.88, 0.72);
                }
                final_rgb = c;
            } else {
                final_rgb = bg.rgb;
            }
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
    let gx = (scroll_x % LANE_CELL_W) / LANE_SCALE;
    let gy = (lane_py - glyph_top) / LANE_SCALE;
    if font_pixel(glyph, gx, gy) {
        let prob = model_probs[col];
        let brightness = 0.25 + prob * 0.75;
        return vec4<f32>(brightness, brightness * 0.95, brightness * 1.10, 1.0);
    }
    return bg;
}

fn captions_band(px: u32, py: u32) -> vec4<f32> {
    if py < uniforms.captions_y || py >= uniforms.captions_y + uniforms.captions_height {
        return vec4<f32>(-1.0, 0.0, 0.0, 1.0);
    }
    let cap_py = py - uniforms.captions_y;
    let col = px / CAPTIONS_CELL_W;
    let row = cap_py / CAPTIONS_CELL_H;
    let cols = uniforms.width / CAPTIONS_CELL_W;
    let char_pos = row * cols + col;
    let bg = vec4<f32>(0.01, 0.01, 0.01, 1.0);
    if char_pos >= uniforms.captions_count {
        return bg;
    }
    let glyph = captions_chars[char_pos];
    let gx = (px % CAPTIONS_CELL_W) / CAPTIONS_SCALE;
    let gy = (cap_py % CAPTIONS_CELL_H) / CAPTIONS_SCALE;
    if font_pixel(glyph, gx, gy) {
        return vec4<f32>(0.40, 0.40, 0.42, 1.0);
    }
    return bg;
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

    let cap_c = captions_band(px, py);
    if cap_c.x >= 0.0 {
        return vec4<f32>(cap_c.rgb * uniforms.global_brightness, 1.0);
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
        let w = select(raw / uniforms.vmax, 0.0, uniforms.vmax < 1e-10);
        if uniforms.colormap == 1u {
            final_rgb = thermal(w);
        } else {
            final_rgb = diverging(w);
        }
    }
    return vec4<f32>(final_rgb * uniforms.global_brightness, 1.0);
}
"""

COLORMAP_DIVERGING = 0
COLORMAP_THERMAL = 1

LANE_SCALE = 4
LANE_GLYPH_W = 8 * LANE_SCALE   # 32 px
LANE_GLYPH_H = 16 * LANE_SCALE  # 64 px
CAPTIONS_SCALE = 2
CAPTIONS_GLYPH_W = 8 * CAPTIONS_SCALE   # 16 px
CAPTIONS_GLYPH_H = 16 * CAPTIONS_SCALE  # 32 px

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
    ("captions_y", np.uint32),
    ("captions_height", np.uint32),
    ("captions_count", np.uint32),
    ("vmax", np.float32),
    ("model_caret_col", np.uint32),
    ("audience_pulse", np.float32),
    ("audience_edge_pulse", np.float32),
    ("global_brightness", np.float32),
    ("overlay_run_count", np.uint32),
])


def _pick_surface_format(context: wgpu.GPUCanvasContext, device: wgpu.GPUDevice) -> str:
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
    captions_height: int = 0

    @property
    def captions_y(self) -> int:
        return self.height - self.captions_height if self.captions_height > 0 else 0

    @property
    def model_y(self) -> int:
        if self.model_height == 0:
            return 0
        return self.height - self.captions_height - self.model_height

    @property
    def audience_y(self) -> int:
        if self.audience_height == 0:
            return 0
        return max(0, self.height - self.captions_height - self.model_height - self.audience_height)

    @property
    def lane_capacity(self) -> int:
        return max(1, self.width // LANE_GLYPH_W)

    @property
    def captions_capacity(self) -> int:
        return max(self.width // CAPTIONS_GLYPH_W, 1)


@dataclass
class RenderResources:
    device: wgpu.GPUDevice
    config: RenderConfig
    uniform_data: np.ndarray
    uniform_buffer: wgpu.GPUBuffer
    weight_buffer: wgpu.GPUBuffer
    font_buffer: wgpu.GPUBuffer
    audience_chars_buffer: wgpu.GPUBuffer
    audience_attrs_buffer: wgpu.GPUBuffer
    model_chars_buffer: wgpu.GPUBuffer
    model_probs_buffer: wgpu.GPUBuffer
    captions_chars_buffer: wgpu.GPUBuffer
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
    uniform_data["captions_y"] = config.captions_y
    uniform_data["captions_height"] = config.captions_height
    uniform_data["model_caret_col"] = np.uint32(0xFFFFFFFF)
    uniform_data["audience_pulse"] = np.float32(0.0)
    uniform_data["audience_edge_pulse"] = np.float32(0.0)
    uniform_data["global_brightness"] = np.float32(1.0)
    uniform_data["overlay_run_count"] = np.uint32(0)

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

    font_size = 1024 * 4
    font_buffer = device.create_buffer(
        size=font_size,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    from brainscan.font import generate_font_atlas
    font_data = generate_font_atlas()
    device.queue.write_buffer(font_buffer, 0, font_data.tobytes())

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

    captions_cap = max(config.captions_capacity, 1)
    captions_chars_buffer = device.create_buffer(
        size=captions_cap * 4,
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
            for i in range(10)
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
        captions_chars_buffer,
        overlay_chars_buffer,
        overlay_runs_buffer,
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
        font_buffer=font_buffer,
        audience_chars_buffer=audience_chars_buffer,
        audience_attrs_buffer=audience_attrs_buffer,
        model_chars_buffer=model_chars_buffer,
        model_probs_buffer=model_probs_buffer,
        captions_chars_buffer=captions_chars_buffer,
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


@dataclass
class CaptionsFrame:
    chars: np.ndarray
    count: int


def draw(
    res: RenderResources,
    target_view: wgpu.GPUTextureView,
    flat_weights: np.ndarray,
    audience: LaneFrame | None = None,
    model: LaneFrame | None = None,
    captions: CaptionsFrame | None = None,
    global_brightness: float = 1.0,
) -> None:
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

    captions_count = 0
    if captions is not None:
        captions_count = captions.count
        device.queue.write_buffer(
            res.captions_chars_buffer,
            0,
            captions.chars.astype(np.uint32).tobytes(),
        )

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
    res.uniform_data["captions_count"] = captions_count
    res.uniform_data["vmax"] = vmax
    res.uniform_data["model_caret_col"] = model_caret_col_val
    res.uniform_data["audience_pulse"] = audience_pulse_val
    res.uniform_data["audience_edge_pulse"] = audience_edge_pulse_val
    res.uniform_data["global_brightness"] = np.float32(global_brightness)
    device.queue.write_buffer(
        res.uniform_buffer, 0, res.uniform_data.tobytes()
    )

    device.queue.write_buffer(
        res.weight_buffer, 0, flat_weights.astype(np.float32).tobytes()
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
        captions_height: int = 0,
    ):
        self.width = width
        self.height = height
        self.colormap = colormap
        self.device = device or get_device()

        self.config = RenderConfig(
            width, height, colormap, audience_height, model_height, captions_height
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
        captions: CaptionsFrame | None = None,
        global_brightness: float = 1.0,
    ) -> np.ndarray:
        """Render weight data and return RGBA image as numpy array."""
        target_view = self._target_texture.create_view()
        draw(self._res, target_view, flat_weights, audience, model, captions, global_brightness)

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
        captions_height: int = 0,
        fullscreen: bool = True,
        max_fps: int = 30,
        canvas: object | None = None,
        display_size: tuple[int, int] | None = None,
    ):
        self.width = width
        self.height = height
        self.device = device or get_device()

        window_w, window_h = display_size or (width, height)

        if canvas is None:
            from rendercanvas.glfw import RenderCanvas

            canvas = RenderCanvas(
                size=(window_w, window_h),
                title="LLM Brainscan",
                update_mode="continuous",
                max_fps=max_fps,
            )
        self._canvas = canvas

        self._context = self._canvas.get_wgpu_context()  # type: ignore[union-attr]
        surface_format = _pick_surface_format(self._context, self.device)
        self._context.configure(device=self.device, format=surface_format)

        self.config = RenderConfig(
            width, height, colormap, audience_height, model_height, captions_height
        )
        self._res = create_render_pipeline(
            self.config, self.device, surface_format
        )

        self._lock = threading.Lock()
        self._flat_weights: np.ndarray | None = None
        self._audience: LaneFrame | None = None
        self._model: LaneFrame | None = None
        self._captions: CaptionsFrame | None = None
        self._global_brightness: float = 1.0

        if fullscreen:
            self._go_fullscreen()

        self._canvas.request_draw(self._draw)  # type: ignore[union-attr]

    def update(
        self,
        flat_weights: np.ndarray,
        audience: LaneFrame | None = None,
        model: LaneFrame | None = None,
        captions: CaptionsFrame | None = None,
        global_brightness: float = 1.0,
    ) -> None:
        """Thread-safe update of weight and lane data for the next frame."""
        with self._lock:
            self._flat_weights = flat_weights.astype(np.float32, copy=True)
            self._audience = audience
            self._model = model
            self._captions = captions
            self._global_brightness = global_brightness

    def _draw(self) -> None:
        with self._lock:
            weights = self._flat_weights
            audience = self._audience
            model = self._model
            captions = self._captions
            global_brightness = self._global_brightness

        if weights is None:
            return

        texture = self._context.get_current_texture()
        draw(self._res, texture.create_view(), weights, audience, model, captions, global_brightness)

    def _go_fullscreen(self) -> None:
        # Strip decorations and size to the monitor; on GNOME / Mutter that
        # still leaves the top panel in place, so also send the EWMH
        # _NET_WM_STATE_FULLSCREEN hint via wmctrl. glfw.set_window_monitor
        # (exclusive fullscreen) doesn't behave reliably under Mutter.
        import glfw  # type: ignore[import]

        window = self._canvas._window  # type: ignore[union-attr]
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
        group = self._canvas._rc_canvas_group  # type: ignore[union-attr]
        group.get_loop().run()

    def close(self) -> None:
        self._canvas.close()  # type: ignore[union-attr]

    def set_overlays(self, overlays: list[TextOverlay]) -> None:
        upload_overlays(self._res, overlays)
