use std::{
    error::Error,
    mem::size_of,
    num::NonZero, time::Instant
};

use glm::{Mat4, Vec3};
use glow::HasContext;
use glutin::{
    config::{Api, Config, ConfigTemplateBuilder, GlConfig},
    context::{ContextApi, ContextAttributesBuilder, GlProfile, NotCurrentGlContext, PossiblyCurrentContext, Version},
    display::{GetGlDisplay, GlDisplay},
    surface::{GlSurface, Surface, SwapInterval, WindowSurface}
};
use glutin_winit::{DisplayBuilder, GlWindow};
use tinyrand_std::ThreadLocalRand;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{Key, NamedKey},
    raw_window_handle::HasWindowHandle,
    window::{Fullscreen, Window, WindowId}
};

use crate::util;

const IMG_SCALE_X: f64 = 0.24;
const DVD_SPEED: f64 = 0.19;

#[derive(Default)]
pub struct App {
    window: Option<Window>,
    gl_context: Option<glow::Context>,
    gl_surface: Option<Surface<WindowSurface>>,
    glutin_context: Option<PossiblyCurrentContext>,
    gl_objects: Option<OpenGLObjects<glow::Context>>
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let (window, gl_config) = Self::create_window(event_loop).unwrap();
        let (gl_surface, glutin_context, gl_context) = Self::create_gl_context(&window, gl_config).unwrap();

        let PhysicalSize { width, height } = window.inner_size();
        let (dvd_texture, dvd_scale) = load_texture(&gl_context, width as f64, height as f64).unwrap();
        let (rectangle_vao, shader_program) = unsafe { setup_rectangle(&gl_context, dvd_scale).unwrap() };

        self.window = Some(window);
        self.gl_context = Some(gl_context);
        self.gl_surface = Some(gl_surface);
        self.glutin_context = Some(glutin_context);
        self.gl_objects = Some(OpenGLObjects {
            dvd_texture,
            rectangle_vao,
            shader_program,
            dvd_scale,
            dvd_coords: (0.35, -0.3),
            dvd_color: Vec3::new(1.0, 1.0, 1.0),
            dvd_direction: (1, -1),
            last_time: Instant::now(),
            rng: tinyrand_std::thread_rand()
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                unsafe {
                    let PhysicalSize { width, height } = self.window.as_ref().unwrap().inner_size();

                    let screen_ratio = width as f64 / height as f64;            
                    
                    let gl_objects = self.gl_objects.as_mut().unwrap();
                    let ctx = self.gl_context.as_ref().unwrap();

                    ctx.clear(glow::COLOR_BUFFER_BIT);

                    ctx.use_program(Some(gl_objects.shader_program));

                    let model_loc = ctx.get_uniform_location(gl_objects.shader_program, "model").unwrap();
                    
                    let mut model = Mat4::identity();

                    let act_time = Instant::now();
                    let delta_time = act_time.duration_since(gl_objects.last_time);
                    gl_objects.last_time = act_time;

                    gl_objects.dvd_coords.0 += delta_time.as_secs_f64() * DVD_SPEED * gl_objects.dvd_direction.0 as f64;
                    gl_objects.dvd_coords.1 += delta_time.as_secs_f64() * DVD_SPEED * gl_objects.dvd_direction.1 as f64 * screen_ratio;

                    match gl_objects.dvd_coords.0 {
                        x if x + IMG_SCALE_X > 2.0 => {
                            gl_objects.dvd_direction.0 = -1;
                            util::change_color(gl_objects);
                        }
                        x if x < 0.0 => {
                            gl_objects.dvd_direction.0 = 1;
                            util::change_color(gl_objects);
                        }
                        _ => ()
                    }

                    match gl_objects.dvd_coords.1 {
                        y if y - gl_objects.dvd_scale < -2.0 => {
                            gl_objects.dvd_direction.1 = 1;
                            util::change_color(gl_objects);
                        }
                        y if y > 0.0 => {
                            gl_objects.dvd_direction.1 = -1;
                            util::change_color(gl_objects);
                        }
                        _ => ()
                    }

                    model.append_translation_mut(&Vec3::new(gl_objects.dvd_coords.0 as f32, gl_objects.dvd_coords.1 as f32, 0.0));

                    ctx.uniform_matrix_4_f32_slice(Some(&model_loc), false, model.as_slice());

                    let color_loc = ctx.get_uniform_location(gl_objects.shader_program, "color").unwrap();

                    ctx.uniform_3_f32_slice(Some(&color_loc), gl_objects.dvd_color.as_slice());

                    ctx.bind_texture(glow::TEXTURE_2D, Some(gl_objects.dvd_texture));

                    ctx.bind_vertex_array(Some(gl_objects.rectangle_vao));
                    ctx.draw_elements(glow::TRIANGLE_FAN, 6, glow::UNSIGNED_BYTE, 0);
                    ctx.bind_vertex_array(None);

                    ctx.bind_texture(glow::TEXTURE_2D, None);

                }

                self.gl_surface.as_ref().unwrap().swap_buffers(&self.glutin_context.as_ref().unwrap()).unwrap();
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed && event.logical_key == Key::Named(NamedKey::Escape) {
                    event_loop.exit();
                }
            }
            _ => ()
        }
    }
}

impl App {
    fn create_window(event_loop: &ActiveEventLoop) -> Result<(Window, Config), Box<dyn Error>> {
        let window_attributes = Window::default_attributes()
            .with_fullscreen(Some(Fullscreen::Borderless(None)))
            .with_title("DVD Screen Saver");

        let config_template = ConfigTemplateBuilder::new().with_api(Api::OPENGL);

        let (wnd, gl_config) = DisplayBuilder::new().with_window_attributes(Some(window_attributes)).build(event_loop, config_template, |configs| {configs.reduce(|acc, config| {
            if config.num_samples() > acc.num_samples() {
                return config
            }
            acc
        }).unwrap()})?;

        let wnd = wnd.unwrap();

        wnd.set_cursor_visible(false);

        Ok((wnd, gl_config))
    }

    fn create_gl_context(window: &Window, gl_config: Config) -> Result<(Surface<WindowSurface>, PossiblyCurrentContext, glow::Context), Box<dyn Error>> {
        let raw_wnd = window.window_handle().ok().map(|el| { el.as_raw() });

        let gl_display = gl_config.display();

        let context_attributes = ContextAttributesBuilder::new().with_context_api(ContextApi::OpenGl(Some(Version::new(3, 3)))).with_profile(GlProfile::Core).build(raw_wnd);

        let not_current_context = unsafe {gl_display.create_context(&gl_config, &context_attributes)?};

        let surface_attributes = window.build_surface_attributes(<_>::default())?;
        let gl_surface = unsafe { gl_display.create_window_surface(&gl_config, &surface_attributes)? };

        let glutin_ctx = not_current_context.make_current(&gl_surface)?;

        let ctx = unsafe {glow::Context::from_loader_function_cstr(|func_name| {gl_display.get_proc_address(func_name)})};

        let gl_version = ctx.version();
        println!("[GLOW] Running on OpenGL {}.{} (Rev.{})", gl_version.major, gl_version.minor, gl_version.revision.unwrap_or(1));
        println!("[GLOW] Driver: {}", gl_version.vendor_info);

        if let Err(msg) = gl_surface.set_swap_interval(&glutin_ctx, SwapInterval::Wait(NonZero::new(1).unwrap())) {
            eprintln!("Failed to enable VSync: {}", msg);
        }

        let PhysicalSize { width, height } = window.inner_size();

        unsafe {
            ctx.clear_color(0.0, 0.0, 0.0, 1.0);

            ctx.enable(glow::BLEND);
            ctx.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

            ctx.viewport(0, 0, width as i32, height as i32);
        }

        Ok((gl_surface, glutin_ctx, ctx))
    }
}

unsafe fn setup_rectangle<CTX: HasContext>(ctx: &CTX, img_scale_y: f64) -> Result<(CTX::VertexArray, CTX::Program), Box<dyn Error>> {
    let triangle_data = [
        -1.0,               1.0,               0.0, 1.0,
        -1.0 + IMG_SCALE_X, 1.0,               1.0, 1.0,
        -1.0 + IMG_SCALE_X, 1.0 - img_scale_y, 1.0, 0.0,
        -1.0,               1.0 - img_scale_y, 0.0, 0.0
    ];
    let triangle_faces = [
        0, 1, 2, 2, 3_u8

        // Triangle fan mode generates:
        // [0, 1, 2]
        // [0, 2, 3]
    ];

    let vao = ctx.create_vertex_array()?;
    ctx.bind_vertex_array(Some(vao));

    let vbo = ctx.create_buffer()?;
    ctx.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
    ctx.buffer_data_u8_slice(glow::ARRAY_BUFFER, util::to_raw(&triangle_data), glow::STATIC_DRAW);

    let ebo = ctx.create_buffer()?;
    ctx.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ebo));
    ctx.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, util::to_raw(&triangle_faces), glow::STATIC_DRAW);

    ctx.vertex_attrib_pointer_f32(0, 2, glow::DOUBLE, false, 4 * size_of::<f64>() as i32, 0);
    ctx.enable_vertex_attrib_array(0);

    ctx.vertex_attrib_pointer_f32(1, 2, glow::DOUBLE, false, 4 * size_of::<f64>() as i32, 2 * size_of::<f64>() as i32);
    ctx.enable_vertex_attrib_array(1);

    let shader = basic_shaders(ctx)?;

    Ok((vao, shader))
}

const IMG_BYTES: &[u8] = include_bytes!("DVD_Image.png");

fn load_texture<CTX: HasContext>(ctx: &CTX, width: f64, height: f64) -> Result<(CTX::Texture, f64), Box<dyn Error>> {
    unsafe {
        let dvd_image = {
            let bitmap: imagine::Bitmap = imagine::try_bitmap_rgba(IMG_BYTES, false).map_err(|err_msg| {Box::<dyn Error>::from(format!("Imagine error: {:?}", err_msg))})?;

            let imagine::Bitmap { width, height, pixels } = bitmap;

            let next_pixels = pixels.iter().flat_map(|pixel| {[pixel.r, pixel.g, pixel.b, pixel.a]}).collect::<Vec<_>>();

            (width, height, next_pixels)
        };

        let tex = ctx.create_texture()?;
        ctx.bind_texture(glow::TEXTURE_2D, Some(tex));

        ctx.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_BORDER as i32);
        ctx.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_BORDER as i32);
        ctx.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR_MIPMAP_LINEAR as i32);
        ctx.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);

        ctx.tex_image_2d(glow::TEXTURE_2D, 0, glow::SRGB8_ALPHA8 as i32, dvd_image.0 as i32, dvd_image.1 as i32, 0, glow::RGBA, glow::UNSIGNED_BYTE, Some(dvd_image.2.as_slice()));
        ctx.generate_mipmap(glow::TEXTURE_2D);

        let scale = (IMG_SCALE_X * (width / height)) / (dvd_image.0 as f64 / dvd_image.1 as f64);

        Ok((tex, scale))
    }
}

const VERT_SHADER: &str = "
#version 330 core

layout (location = 0) in vec2 pos;
layout (location = 1) in vec2 tex_coords;

uniform mat4 model;

out vec2 bTex_coords;

void main() {
    bTex_coords = tex_coords;
    gl_Position = model * vec4(pos, 0.0, 1.0);
}
";

const FRAG_SHADER: &str = "
#version 330 core

in vec2 bTex_coords;

uniform sampler2D tex;
uniform vec3 color;

out vec4 FragColor;

void main() {
    FragColor = vec4(color, texture(tex, bTex_coords).a);
}
";

unsafe fn basic_shaders<CTX: HasContext>(ctx: &CTX) -> Result<CTX::Program, Box<dyn Error>> {
    let vertex_shader = ctx.create_shader(glow::VERTEX_SHADER)?;
    ctx.shader_source(vertex_shader, VERT_SHADER);
    ctx.compile_shader(vertex_shader);
    if !ctx.get_shader_compile_status(vertex_shader) {
        let error_message = format!("[Vertex] Erreur de compilation: {}", ctx.get_shader_info_log(vertex_shader));
        return Err(Box::<dyn Error>::from(error_message.as_str()))
    }

    let fragment_shader = ctx.create_shader(glow::FRAGMENT_SHADER)?;
    ctx.shader_source(fragment_shader, FRAG_SHADER);
    ctx.compile_shader(fragment_shader);
    if !ctx.get_shader_compile_status(fragment_shader) {
        let error_message = format!("[Fragment] Erreur de compilation: {}", ctx.get_shader_info_log(fragment_shader));
        return Err(Box::<dyn Error>::from(error_message.as_str()))
    }

    let shader_program = ctx.create_program()?;
    ctx.attach_shader(shader_program, vertex_shader);
    ctx.attach_shader(shader_program, fragment_shader);
    ctx.link_program(shader_program);
    if !ctx.get_program_link_status(shader_program) {
        let error_message = format!("Erreur de linkage: {}", ctx.get_program_info_log(shader_program));
        return Err(Box::<dyn Error>::from(error_message.as_str()))
    }

    ctx.delete_shader(vertex_shader);
    ctx.delete_shader(fragment_shader);

    Ok(shader_program)
}

pub struct OpenGLObjects<CTX: HasContext> {
    dvd_texture: CTX::Texture,
    rectangle_vao: CTX::VertexArray,
    shader_program: CTX::Program,
    dvd_coords: (f64, f64),
    dvd_direction: (i8, i8),
    dvd_scale: f64,
    pub dvd_color: Vec3,
    last_time: Instant,
    pub rng: ThreadLocalRand
}
