// shaders.js - WGSL for the embedding point cloud (instanced screen-space
// discs, coloured by emotion) and the neighbour lines. One shared uniform
// block. Struct is named Uniforms (never U) to avoid colliding with the var U.

const UNIFORMS = `
struct Uniforms {
  viewProj : mat4x4<f32>,
  params   : vec4<f32>,   // x: pointSizePx  y: dpr  z: starSizePx  w: time
  screen   : vec4<f32>,   // x: width(devpx)  y: height(devpx)
};
@group(0) @binding(0) var<uniform> U : Uniforms;
`;

export const POINT_SHADER = UNIFORMS + `
@group(0) @binding(1) var<storage, read> pts : array<vec4<f32>>;  // xyz + emotion

struct VO {
  @builtin(position) pos : vec4<f32>,
  @location(0) color : vec3<f32>,
  @location(1) uv : vec2<f32>,
};

fn emoColor(e : f32) -> vec3<f32> {
  if (e < 0.5) { return vec3<f32>(0.22, 0.53, 0.90); }   // sadness  blue
  if (e < 1.5) { return vec3<f32>(0.95, 0.70, 0.17); }   // joy      amber
  if (e < 2.5) { return vec3<f32>(0.91, 0.48, 0.64); }   // love     pink
  if (e < 3.5) { return vec3<f32>(0.89, 0.29, 0.28); }   // anger    red
  if (e < 4.5) { return vec3<f32>(0.56, 0.52, 0.91); }   // fear     violet
  if (e < 5.5) { return vec3<f32>(0.11, 0.69, 0.48); }   // surprise teal
  return vec3<f32>(1.0, 1.0, 1.0);                        // query star
}

@vertex fn vs(@builtin(instance_index) inst : u32, @builtin(vertex_index) vid : u32) -> VO {
  var o : VO;
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0,-1.0), vec2<f32>(1.0,-1.0), vec2<f32>(1.0,1.0),
    vec2<f32>(-1.0,-1.0), vec2<f32>(1.0,1.0), vec2<f32>(-1.0,1.0));
  let c = corners[vid];
  let p = pts[inst];
  let clip = U.viewProj * vec4<f32>(p.xyz, 1.0);
  if (clip.w <= 0.0) { o.pos = vec4<f32>(2.0,2.0,2.0,1.0); o.color = vec3<f32>(0.0); o.uv = vec2<f32>(2.0,2.0); return o; }
  let isStar = p.w > 5.5;
  let sizePx = select(U.params.x, U.params.z, isStar);
  let off = c * (sizePx * U.params.y) / U.screen.xy * 2.0;
  let ndc = clip.xy / clip.w;
  o.pos = vec4<f32>((ndc + off) * clip.w, clip.z, clip.w);
  o.color = emoColor(p.w);
  o.uv = c;
  return o;
}

@fragment fn fs(i : VO) -> @location(0) vec4<f32> {
  let r = length(i.uv);
  if (r > 1.0) { discard; }
  let a = smoothstep(1.0, 0.55, r);
  return vec4<f32>(i.color, a);
}
`;

export const LINE_SHADER = UNIFORMS + `
@vertex fn vs(@location(0) p : vec3<f32>) -> @builtin(position) vec4<f32> {
  return U.viewProj * vec4<f32>(p, 1.0);
}
@fragment fn fs() -> @location(0) vec4<f32> {
  return vec4<f32>(1.0, 0.95, 0.72, 0.6);
}
`;
