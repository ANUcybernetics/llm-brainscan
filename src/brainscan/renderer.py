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

SHADER_SOURCE = """
struct Uniforms {
    width: u32,
    height: u32,
    param_count: u32,
    colormap: u32,
    audience_y: u32,
    audience_height: u32,
    model_y: u32,
    model_height: u32,
    captions_y: u32,
    captions_height: u32,
    audience_count: u32,
    model_count: u32,
    captions_count: u32,
    vmax: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> font_data: array<u32>;
@group(0) @binding(3) var<storage, read> audience_chars: array<u32>;
@group(0) @binding(4) var<storage, read> audience_attrs: array<f32>;
@group(0) @binding(5) var<storage, read> model_chars: array<u32>;
@group(0) @binding(6) var<storage, read> model_probs: array<f32>;
@group(0) @binding(7) var<storage, read> captions_chars: array<u32>;

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

fn audience_band(px: u32, py: u32) -> vec4<f32> {
    if py < uniforms.audience_y || py >= uniforms.audience_y + uniforms.audience_height {
        return vec4<f32>(-1.0, 0.0, 0.0, 1.0);
    }
    let lane_py = py - uniforms.audience_y;
    let cell_w = 24u;
    let cell_h = 48u;
    let col = px / cell_w;
    let row = lane_py / cell_h;
    let cols = uniforms.width / cell_w;
    let char_pos = row * cols + col;
    let bg = vec4<f32>(0.04, 0.04, 0.06, 1.0);
    if char_pos >= uniforms.audience_count {
        return bg;
    }
    let glyph = audience_chars[char_pos];
    let gx = (px % cell_w) / 3u;
    let gy = (lane_py % cell_h) / 3u;
    if font_pixel(glyph, gx, gy) {
        let attr = audience_attrs[char_pos];
        let c = vec3<f32>(0.94, 0.88, 0.72) * attr;
        return vec4<f32>(c, 1.0);
    }
    return bg;
}

fn model_band(px: u32, py: u32) -> vec4<f32> {
    if py < uniforms.model_y || py >= uniforms.model_y + uniforms.model_height {
        return vec4<f32>(-1.0, 0.0, 0.0, 1.0);
    }
    let lane_py = py - uniforms.model_y;
    let cell_w = 24u;
    let cell_h = 48u;
    let col = px / cell_w;
    let row = lane_py / cell_h;
    let cols = uniforms.width / cell_w;
    let char_pos = row * cols + col;
    let bg = vec4<f32>(0.02, 0.02, 0.04, 1.0);
    if char_pos >= uniforms.model_count {
        return bg;
    }
    let glyph = model_chars[char_pos];
    let gx = (px % cell_w) / 3u;
    let gy = (lane_py % cell_h) / 3u;
    if font_pixel(glyph, gx, gy) {
        let prob = model_probs[char_pos];
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
    let cell_w = 8u;
    let cell_h = 16u;
    let col = px / cell_w;
    let row = cap_py / cell_h;
    let cols = uniforms.width / cell_w;
    let char_pos = row * cols + col;
    let bg = vec4<f32>(0.01, 0.01, 0.01, 1.0);
    if char_pos >= uniforms.captions_count {
        return bg;
    }
    let glyph = captions_chars[char_pos];
    let gx = px % cell_w;
    let gy = cap_py % cell_h;
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
        return cap_a;
    }

    let cap_m = model_band(px, py);
    if cap_m.x >= 0.0 {
        return cap_m;
    }

    let cap_c = captions_band(px, py);
    if cap_c.x >= 0.0 {
        return cap_c;
    }

    let idx = py * uniforms.width + px;
    if idx >= uniforms.param_count {
        return vec4<f32>(0.05, 0.05, 0.05, 1.0);
    }
    let raw = weights[idx];
    let w = select(raw / uniforms.vmax, 0.0, uniforms.vmax < 1e-10);
    var color: vec3<f32>;
    if uniforms.colormap == 1u {
        color = thermal(w);
    } else {
        color = diverging(w);
    }
    return vec4<f32>(color, 1.0);
}
"""

COLORMAP_DIVERGING = 0
COLORMAP_THERMAL = 1

LANE_SCALE = 3
LANE_GLYPH_W = 8 * LANE_SCALE   # 24 px
LANE_GLYPH_H = 16 * LANE_SCALE  # 48 px
CAPTIONS_GLYPH_W = 8            # 1× scale

_UNIFORM_DTYPE = np.dtype([
    ("width", np.uint32),
    ("height", np.uint32),
    ("param_count", np.uint32),
    ("colormap", np.uint32),
    ("audience_y", np.uint32),
    ("audience_height", np.uint32),
    ("model_y", np.uint32),
    ("model_height", np.uint32),
    ("captions_y", np.uint32),
    ("captions_height", np.uint32),
    ("audience_count", np.uint32),
    ("model_count", np.uint32),
    ("captions_count", np.uint32),
    ("vmax", np.float32),
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
    audience_height: int = 90
    model_height: int = 90
    captions_height: int = 12

    @property
    def audience_y(self) -> int:
        strip_total = self.audience_height + self.model_height + self.captions_height
        return self.height - strip_total

    @property
    def model_y(self) -> int:
        return self.audience_y + self.audience_height

    @property
    def captions_y(self) -> int:
        return self.model_y + self.model_height

    @property
    def lane_capacity(self) -> int:
        cols = self.width // LANE_GLYPH_W
        rows = self.audience_height // LANE_GLYPH_H
        return max(cols * rows, 1)

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

    uniform_size = max(_UNIFORM_DTYPE.itemsize, 64)
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
            for i in range(8)
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
        bind_group=bind_group,
        pipeline=pipeline,
    )


@dataclass
class LaneFrame:
    chars: np.ndarray
    attrs_or_probs: np.ndarray
    count: int


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
) -> None:
    device = res.device
    param_count = len(flat_weights)

    audience_count = 0
    if audience is not None:
        audience_count = audience.count
        device.queue.write_buffer(
            res.audience_chars_buffer,
            0,
            audience.chars.astype(np.uint32).tobytes(),
        )
        device.queue.write_buffer(
            res.audience_attrs_buffer,
            0,
            audience.attrs_or_probs.astype(np.float32).tobytes(),
        )

    model_count = 0
    if model is not None:
        model_count = model.count
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
    res.uniform_data["param_count"] = param_count
    res.uniform_data["colormap"] = res.config.colormap
    res.uniform_data["audience_count"] = audience_count
    res.uniform_data["model_count"] = model_count
    res.uniform_data["captions_count"] = captions_count
    res.uniform_data["vmax"] = vmax
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
        audience_height: int = 90,
        model_height: int = 90,
        captions_height: int = 12,
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
    ) -> np.ndarray:
        """Render weight data and return RGBA image as numpy array."""
        target_view = self._target_texture.create_view()
        draw(self._res, target_view, flat_weights, audience, model, captions)

        data = self.device.queue.read_texture(
            {"texture": self._target_texture, "mip_level": 0, "origin": (0, 0, 0)},
            {"bytes_per_row": self.width * 4},
            (self.width, self.height, 1),
        )
        return np.frombuffer(data, dtype=np.uint8).reshape(
            self.height, self.width, 4
        )


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
        audience_height: int = 90,
        model_height: int = 90,
        captions_height: int = 12,
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

        if fullscreen:
            self._go_fullscreen()

        self._canvas.request_draw(self._draw)  # type: ignore[union-attr]

    def update(
        self,
        flat_weights: np.ndarray,
        audience: LaneFrame | None = None,
        model: LaneFrame | None = None,
        captions: CaptionsFrame | None = None,
    ) -> None:
        """Thread-safe update of weight and lane data for the next frame."""
        with self._lock:
            self._flat_weights = flat_weights.astype(np.float32, copy=True)
            self._audience = audience
            self._model = model
            self._captions = captions

    def _draw(self) -> None:
        with self._lock:
            weights = self._flat_weights
            audience = self._audience
            model = self._model
            captions = self._captions

        if weights is None:
            return

        texture = self._context.get_current_texture()
        draw(self._res, texture.create_view(), weights, audience, model, captions)

    def _go_fullscreen(self) -> None:
        try:
            import glfw  # type: ignore[import]

            window = self._canvas._window  # type: ignore[union-attr]
            if window is None:
                return
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(
                window,
                monitor,
                0,
                0,
                mode.size.width,
                mode.size.height,
                mode.refresh_rate,
            )
        except (ImportError, AttributeError):
            pass

    def run(self) -> None:
        """Enter the canvas event loop (blocks until the window is closed)."""
        group = self._canvas._rc_canvas_group  # type: ignore[union-attr]
        group.get_loop().run()

    def close(self) -> None:
        self._canvas.close()  # type: ignore[union-attr]
