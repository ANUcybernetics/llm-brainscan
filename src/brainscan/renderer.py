"""GPU renderer: maps weight tensors to an 8K texture via wgpu.

The renderer takes flattened weight data (one float per pixel), uploads it to a
storage buffer, and runs a fragment shader that applies a colourmap to produce
the final image. All colourmap computation happens on the GPU.

The bottom strip of the canvas can display generated text using a bitmap font,
with each character coloured by its softmax probability.

Can operate in two modes:
- offscreen: render to a texture, read back as numpy array
- windowed: render to a canvas via rendercanvas (for live display)
"""

from __future__ import annotations

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

    let w = weights[idx];
    // Normalisation is done CPU-side before upload
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


def normalise_weights(flat: np.ndarray) -> np.ndarray:
    """Normalise to [-1, 1] range for colourmap input."""
    vmax = np.max(np.abs(flat))
    if vmax < 1e-10:
        return np.zeros_like(flat)
    return flat / vmax


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
        self.text_y = height - text_strip_height if text_strip_height > 0 else 0
        self.text_cols = width // (8 * text_scale) if text_strip_height > 0 else 0
        self.device = device or get_device()
        self._setup_pipeline()

    def _setup_pipeline(self) -> None:
        device = self.device

        self._shader = device.create_shader_module(code=SHADER_SOURCE)

        self._uniform_data = np.array(
            [
                self.width,
                self.height,
                0,
                self.colormap,
                self.text_y,
                self.text_scale,
                self.text_cols,
                0,
            ],
            dtype=np.uint32,
        )
        self._uniform_buffer = device.create_buffer(
            size=self._uniform_data.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        max_params = self.width * self.height
        self._weight_buffer = device.create_buffer(
            size=max_params * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        font_size = max(1024 * 4, 4)
        self._font_buffer = device.create_buffer(
            size=font_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        max_text = max(self.text_cols * (self.text_strip_height // (16 * self.text_scale)), 1)
        self._text_chars_buffer = device.create_buffer(
            size=max_text * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self._text_probs_buffer = device.create_buffer(
            size=max_text * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self._max_text = max_text

        if self.text_strip_height > 0:
            from brainscan.font import generate_font_atlas

            font_data = generate_font_atlas()
            device.queue.write_buffer(self._font_buffer, 0, font_data.tobytes())

        bind_group_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
            ]
        )

        self._bind_group = device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self._uniform_buffer,
                        "offset": 0,
                        "size": self._uniform_data.nbytes,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self._weight_buffer,
                        "offset": 0,
                        "size": self._weight_buffer.size,
                    },
                },
                {
                    "binding": 2,
                    "resource": {
                        "buffer": self._font_buffer,
                        "offset": 0,
                        "size": self._font_buffer.size,
                    },
                },
                {
                    "binding": 3,
                    "resource": {
                        "buffer": self._text_chars_buffer,
                        "offset": 0,
                        "size": self._text_chars_buffer.size,
                    },
                },
                {
                    "binding": 4,
                    "resource": {
                        "buffer": self._text_probs_buffer,
                        "offset": 0,
                        "size": self._text_probs_buffer.size,
                    },
                },
            ],
        )

        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )

        self._render_texture_format = wgpu.TextureFormat.rgba8unorm

        self._pipeline = device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={"module": self._shader, "entry_point": "vs_main"},
            fragment={
                "module": self._shader,
                "entry_point": "fs_main",
                "targets": [{"format": self._render_texture_format}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        )

        self._target_texture = device.create_texture(
            size=(self.width, self.height, 1),
            format=self._render_texture_format,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )

    def render(
        self,
        flat_weights: np.ndarray,
        text_chars: np.ndarray | None = None,
        text_probs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render normalised weight data and return RGBA image as numpy array.

        Args:
            flat_weights: float32 array of normalised weights in [-1, 1].
            text_chars: uint32 array of character indices (byte values).
            text_probs: float32 array of per-character probabilities.

        Returns:
            RGBA uint8 numpy array of shape (height, width, 4).
        """
        device = self.device
        param_count = len(flat_weights)

        text_count = 0
        if text_chars is not None and text_probs is not None:
            text_count = min(len(text_chars), self._max_text)
            chars = text_chars[:text_count].astype(np.uint32)
            probs = text_probs[:text_count].astype(np.float32)
            device.queue.write_buffer(self._text_chars_buffer, 0, chars.tobytes())
            device.queue.write_buffer(self._text_probs_buffer, 0, probs.tobytes())

        self._uniform_data[2] = param_count
        self._uniform_data[3] = self.colormap
        self._uniform_data[7] = text_count
        device.queue.write_buffer(self._uniform_buffer, 0, self._uniform_data.tobytes())

        weight_bytes = flat_weights.astype(np.float32).tobytes()
        device.queue.write_buffer(self._weight_buffer, 0, weight_bytes)

        command_encoder = device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self._target_texture.create_view(),
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

        data = device.queue.read_texture(
            {"texture": self._target_texture, "mip_level": 0, "origin": (0, 0, 0)},
            {"bytes_per_row": self.width * 4},
            (self.width, self.height, 1),
        )
        return np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 4)
