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

import numpy as np
import wgpu

SHADER_SOURCE = """
struct Uniforms {
    width: u32,
    height: u32,
    param_count: u32,
    colormap: u32,
    text_y: u32,
    text_scale: u32,
    text_cols: u32,
    text_count: u32,
    vmax: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> font_data: array<u32>;
@group(0) @binding(3) var<storage, read> text_chars: array<u32>;
@group(0) @binding(4) var<storage, read> text_probs: array<f32>;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    // Fullscreen triangle (oversized, clipped to screen)
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
    // Blue (negative) -> grey (zero) -> red (positive)
    let t = clamp(v, -1.0, 1.0);
    let r = clamp(0.5 + t * 0.5, 0.0, 1.0);
    let g = clamp(0.5 - abs(t) * 0.4, 0.0, 1.0);
    let b = clamp(0.5 - t * 0.5, 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

fn thermal(v: f32) -> vec3<f32> {
    // Black -> blue -> red -> yellow -> white
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let px = u32(in.uv.x * f32(uniforms.width));
    let py = u32(in.uv.y * f32(uniforms.height));

    if uniforms.text_y > 0u && py >= uniforms.text_y {
        let text_py = py - uniforms.text_y;
        let scale = uniforms.text_scale;
        let cell_w = 8u * scale;
        let cell_h = 16u * scale;
        let col = px / cell_w;
        let row = text_py / cell_h;
        let char_pos = row * uniforms.text_cols + col;

        if char_pos < uniforms.text_count {
            let glyph = text_chars[char_pos];
            let gx = (px % cell_w) / scale;
            let gy = (text_py % cell_h) / scale;

            if font_pixel(glyph, gx, gy) {
                let prob = text_probs[char_pos];
                let brightness = 0.25 + prob * 0.75;
                return vec4<f32>(
                    brightness,
                    brightness * 0.95,
                    brightness * 0.85,
                    1.0,
                );
            }
        }
        return vec4<f32>(0.02, 0.02, 0.02, 1.0);
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

TEXT_SCALE_DEFAULT = 3

_UNIFORM_DTYPE = np.dtype([
    ("width", np.uint32),
    ("height", np.uint32),
    ("param_count", np.uint32),
    ("colormap", np.uint32),
    ("text_y", np.uint32),
    ("text_scale", np.uint32),
    ("text_cols", np.uint32),
    ("text_count", np.uint32),
    ("vmax", np.float32),
])


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


class _RenderPipeline:
    """Shared GPU pipeline for weight visualisation.

    Holds buffers, bind group, and pipeline. Both OffscreenRenderer and
    LiveRenderer delegate to this for setup and draw submission.
    """

    def __init__(
        self,
        device: wgpu.GPUDevice,
        width: int,
        height: int,
        colormap: int,
        text_strip_height: int,
        text_scale: int,
        target_format: str,
    ):
        self.device = device
        self.width = width
        self.height = height
        self.colormap = colormap
        self.text_strip_height = text_strip_height
        self.text_scale = text_scale
        self.text_y = height - text_strip_height if text_strip_height > 0 else 0
        self.text_cols = width // (8 * text_scale) if text_strip_height > 0 else 0

        self._uniform_data = np.zeros(1, dtype=_UNIFORM_DTYPE)
        self._uniform_data["width"] = width
        self._uniform_data["height"] = height
        self._uniform_data["colormap"] = colormap
        self._uniform_data["text_y"] = self.text_y
        self._uniform_data["text_scale"] = text_scale
        self._uniform_data["text_cols"] = self.text_cols

        uniform_size = max(_UNIFORM_DTYPE.itemsize, 36)
        self._uniform_buffer = device.create_buffer(
            size=uniform_size,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        max_params = width * height
        self._weight_buffer = device.create_buffer(
            size=max_params * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        font_size = max(1024 * 4, 4)
        self._font_buffer = device.create_buffer(
            size=font_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        self.max_text = max(
            self.text_cols * (text_strip_height // (16 * text_scale)), 1
        )
        self._text_chars_buffer = device.create_buffer(
            size=self.max_text * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self._text_probs_buffer = device.create_buffer(
            size=self.max_text * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        if text_strip_height > 0:
            from brainscan.font import generate_font_atlas

            font_data = generate_font_atlas()
            device.queue.write_buffer(self._font_buffer, 0, font_data.tobytes())

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
                for i in range(5)
            ]
        )

        buffers = [
            self._uniform_buffer,
            self._weight_buffer,
            self._font_buffer,
            self._text_chars_buffer,
            self._text_probs_buffer,
        ]
        self._bind_group = device.create_bind_group(
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
        self._pipeline = device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={"module": shader, "entry_point": "vs_main"},
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [{"format": target_format}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        )

    def draw(
        self,
        target_view: wgpu.GPUTextureView,
        flat_weights: np.ndarray,
        text_chars: np.ndarray | None = None,
        text_probs: np.ndarray | None = None,
    ) -> None:
        device = self.device
        param_count = len(flat_weights)

        text_count = 0
        if text_chars is not None and text_probs is not None:
            text_count = min(len(text_chars), self.max_text)
            device.queue.write_buffer(
                self._text_chars_buffer,
                0,
                text_chars[:text_count].astype(np.uint32).tobytes(),
            )
            device.queue.write_buffer(
                self._text_probs_buffer,
                0,
                text_probs[:text_count].astype(np.float32).tobytes(),
            )

        vmax = float(np.max(np.abs(flat_weights))) if param_count > 0 else 0.0
        self._uniform_data["param_count"] = param_count
        self._uniform_data["colormap"] = self.colormap
        self._uniform_data["text_count"] = text_count
        self._uniform_data["vmax"] = vmax
        device.queue.write_buffer(
            self._uniform_buffer, 0, self._uniform_data.tobytes()
        )

        device.queue.write_buffer(
            self._weight_buffer, 0, flat_weights.astype(np.float32).tobytes()
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
        render_pass.set_pipeline(self._pipeline)
        render_pass.set_bind_group(0, self._bind_group)
        render_pass.draw(3, 1, 0, 0)
        render_pass.end()
        device.queue.submit([command_encoder.finish()])


class OffscreenRenderer:
    """Render weight data to an offscreen texture and read back as numpy.

    Optionally renders a text strip at the bottom of the canvas showing
    generated text coloured by token probability.
    """

    def __init__(
        self,
        width: int,
        height: int,
        device: wgpu.GPUDevice | None = None,
        colormap: int = COLORMAP_DIVERGING,
        text_strip_height: int = 0,
        text_scale: int = TEXT_SCALE_DEFAULT,
    ):
        self.width = width
        self.height = height
        self.colormap = colormap
        self.text_strip_height = text_strip_height
        self.text_scale = text_scale
        self.device = device or get_device()

        self._pipe = _RenderPipeline(
            self.device,
            width,
            height,
            colormap,
            text_strip_height,
            text_scale,
            wgpu.TextureFormat.rgba8unorm,
        )
        self.text_y = self._pipe.text_y
        self.text_cols = self._pipe.text_cols

        self._target_texture = self.device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )

    def render(
        self,
        flat_weights: np.ndarray,
        text_chars: np.ndarray | None = None,
        text_probs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render weight data and return RGBA image as numpy array.

        Normalisation is done on the GPU --- pass raw (unnormalised) weights.
        Pre-normalised weights in [-1, 1] also work fine (vmax will be ~1.0).

        Args:
            flat_weights: float32 array of weight values.
            text_chars: uint32 array of character indices (byte values).
            text_probs: float32 array of per-character probabilities.

        Returns:
            RGBA uint8 numpy array of shape (height, width, 4).
        """
        target_view = self._target_texture.create_view()
        self._pipe.draw(target_view, flat_weights, text_chars, text_probs)

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
    ``run()``. Rendering goes straight to the display surface --- no GPU
    readback.
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        device: wgpu.GPUDevice | None = None,
        colormap: int = COLORMAP_DIVERGING,
        text_strip_height: int = 0,
        text_scale: int = TEXT_SCALE_DEFAULT,
        fullscreen: bool = True,
        max_fps: int = 30,
        canvas: object | None = None,
    ):
        self.width = width
        self.height = height
        self.device = device or get_device()

        if canvas is None:
            from rendercanvas.glfw import RenderCanvas

            canvas = RenderCanvas(
                size=(width, height),
                title="LLM Brainscan",
                update_mode="continuous",
                max_fps=max_fps,
            )
        self._canvas = canvas

        self._context = self._canvas.get_wgpu_context()  # type: ignore[union-attr]
        self._context.configure(
            device=self.device, format=wgpu.TextureFormat.rgba8unorm
        )

        self._pipe = _RenderPipeline(
            self.device,
            width,
            height,
            colormap,
            text_strip_height,
            text_scale,
            wgpu.TextureFormat.rgba8unorm,
        )
        self.text_y = self._pipe.text_y
        self.text_cols = self._pipe.text_cols

        self._lock = threading.Lock()
        self._flat_weights: np.ndarray | None = None
        self._text_chars: np.ndarray | None = None
        self._text_probs: np.ndarray | None = None

        if fullscreen:
            self._go_fullscreen()

        self._canvas.request_draw(self._draw)  # type: ignore[union-attr]

    def update(
        self,
        flat_weights: np.ndarray,
        text_chars: np.ndarray | None = None,
        text_probs: np.ndarray | None = None,
    ) -> None:
        """Thread-safe update of weight and text data for the next frame."""
        with self._lock:
            self._flat_weights = flat_weights.astype(np.float32, copy=True)
            self._text_chars = (
                text_chars.astype(np.uint32, copy=True)
                if text_chars is not None
                else None
            )
            self._text_probs = (
                text_probs.astype(np.float32, copy=True)
                if text_probs is not None
                else None
            )

    def _draw(self) -> None:
        with self._lock:
            weights = self._flat_weights
            chars = self._text_chars
            probs = self._text_probs

        if weights is None:
            return

        texture = self._context.get_current_texture()
        self._pipe.draw(texture.create_view(), weights, chars, probs)

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
