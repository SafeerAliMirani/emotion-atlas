// render-core.js - WebGPU renderer for the embedding cloud: instanced points
// (storage-buffer vertex pulling), a highlighted query star, and neighbour
// lines, with a depth buffer and 4x MSAA. Hand-written pipelines; no libraries.
import { POINT_SHADER, LINE_SHADER } from "./shaders.js";

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.sampleCount = 4;
    this.count = 0;
    this.overlayCount = 0;
    this.lineCount = 0;
    this.lost = false;
    this.bg = { r: 0.02, g: 0.03, b: 0.06, a: 1.0 };
  }

  static async supported() { return !!navigator.gpu; }

  async init() {
    if (!navigator.gpu) throw new Error("WebGPU not available in this browser.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No compatible GPU adapter found.");
    this.device = await adapter.requestDevice();
    this.device.lost.then((info) => { if (info.reason !== "destroyed") { this.lost = true; console.error("device lost:", info.message); } });
    this.context = this.canvas.getContext("webgpu");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
    this._build();
    this.resize();
  }

  _build() {
    const dev = this.device;
    this.uniformBuffer = dev.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.uniformData = new Float32Array(24);

    const ptsL = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ]});
    const lineL = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
    ]});
    this.ptsLayout = ptsL;

    const blend = {
      color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
      alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
    };
    const ms = { count: this.sampleCount };
    const pm = dev.createShaderModule({ code: POINT_SHADER });
    const ptsPL = dev.createPipelineLayout({ bindGroupLayouts: [ptsL] });

    this.pointPipe = dev.createRenderPipeline({
      layout: ptsPL,
      vertex: { module: pm, entryPoint: "vs" },
      fragment: { module: pm, entryPoint: "fs", targets: [{ format: this.format, blend }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
      multisample: ms,
    });
    this.overlayPipe = dev.createRenderPipeline({
      layout: ptsPL,
      vertex: { module: pm, entryPoint: "vs" },
      fragment: { module: pm, entryPoint: "fs", targets: [{ format: this.format, blend }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "always" },
      multisample: ms,
    });

    const lm = dev.createShaderModule({ code: LINE_SHADER });
    this.linePipe = dev.createRenderPipeline({
      layout: dev.createPipelineLayout({ bindGroupLayouts: [lineL] }),
      vertex: { module: lm, entryPoint: "vs", buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] }] },
      fragment: { module: lm, entryPoint: "fs", targets: [{ format: this.format, blend }] },
      primitive: { topology: "line-list" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "less" },
      multisample: ms,
    });
    this.bgLine = dev.createBindGroup({ layout: lineL, entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }] });
  }

  _pointBuf(arr) {
    const buf = this.device.createBuffer({ size: arr.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, arr);
    const bg = this.device.createBindGroup({ layout: this.ptsLayout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 1, resource: { buffer: buf } },
    ]});
    return { buf, bg };
  }

  setPoints(arr, count) {
    if (this.ptsBuf) this.ptsBuf.buf.destroy();
    this.ptsBuf = this._pointBuf(arr);
    this.count = count;
  }
  setOverlay(arr, count) {
    if (this.ovBuf) this.ovBuf.buf.destroy();
    if (!count) { this.overlayCount = 0; return; }
    this.ovBuf = this._pointBuf(arr);
    this.overlayCount = count;
  }
  setLines(arr) {
    if (this.lineBuf) this.lineBuf.destroy();
    if (!arr || !arr.length) { this.lineCount = 0; return; }
    this.lineBuf = this.device.createBuffer({ size: arr.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.lineBuf, 0, arr);
    this.lineCount = arr.length / 3;
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width === w && this.canvas.height === h && this.msaa) return;
    this.canvas.width = w; this.canvas.height = h;
    for (const t of [this.msaa, this.depth]) if (t) t.destroy();
    this.msaa = this.device.createTexture({ size: [w, h], sampleCount: this.sampleCount, format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.depth = this.device.createTexture({ size: [w, h], sampleCount: this.sampleCount, format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.msaaView = this.msaa.createView();
    this.depthView = this.depth.createView();
  }

  render(viewProj, state) {
    if (this.lost) return;
    const w = this.canvas.width, h = this.canvas.height;
    const dpr = w / Math.max(1, this.canvas.clientWidth);
    this.uniformData.set(viewProj, 0);
    this.uniformData[16] = state.pointSize ?? 5.5;
    this.uniformData[17] = dpr;
    this.uniformData[18] = state.starSize ?? 15.0;
    this.uniformData[19] = state.time ?? 0;
    this.uniformData[20] = w;
    this.uniformData[21] = h;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData);

    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: this.msaaView, resolveTarget: this.context.getCurrentTexture().createView(), clearValue: this.bg, loadOp: "clear", storeOp: "store" }],
      depthStencilAttachment: { view: this.depthView, depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store" },
    });
    if (this.ptsBuf && this.count) {
      pass.setPipeline(this.pointPipe);
      pass.setBindGroup(0, this.ptsBuf.bg);
      pass.draw(6, this.count);
    }
    if (this.lineCount) {
      pass.setPipeline(this.linePipe);
      pass.setBindGroup(0, this.bgLine);
      pass.setVertexBuffer(0, this.lineBuf);
      pass.draw(this.lineCount);
    }
    if (this.ovBuf && this.overlayCount) {
      pass.setPipeline(this.overlayPipe);
      pass.setBindGroup(0, this.ovBuf.bg);
      pass.draw(6, this.overlayCount);
    }
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
}
