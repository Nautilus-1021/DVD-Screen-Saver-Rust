use std::{error::Error, num::NonZeroU32, mem::size_of, time::Instant};

use glow::HasContext;
use glutin::{config::{ConfigTemplateBuilder, Api, GlConfig, Config}, context::{ContextAttributesBuilder, ContextApi, Version, GlProfile, NotCurrentContext, NotCurrentGlContext}, display::{GetGlDisplay, GlDisplay, Display}, surface::{GlSurface, SwapInterval}};
use glutin_winit::{DisplayBuilder, GlWindow};
use tinyrand_std::thread_rand;
use winit::{window::{WindowBuilder, Fullscreen, Window}, event_loop::{EventLoop, ControlFlow}, event::{Event, WindowEvent, ElementState}, keyboard::{Key, NamedKey}, dpi::PhysicalSize};
use raw_window_handle::HasRawWindowHandle;
use glm::{Mat4, Vec3};

mod util;
use util::change_color;

const IMG_SCALE_X: f64 = 0.24;
const DVD_SPEED: f64 = 0.19;

fn main() {
    let (event_loop, wnd, gl_display, not_current_context, gl_config) = create_window().unwrap();

    let attrs = wnd.build_surface_attributes(<_>::default());
    let gl_surface = unsafe {gl_display.create_window_surface(&gl_config, &attrs).unwrap()};

    let glutin_ctx = not_current_context.make_current(&gl_surface).unwrap();

    let ctx = unsafe {glow::Context::from_loader_function_cstr(|func_name| {gl_display.get_proc_address(func_name)})};
    
    let PhysicalSize { width, height } = wnd.inner_size();

    let screen_ratio = width as f64 / height as f64;

    let (dvd_tex, scale) = load_texture(&ctx, width as f64, height as f64).unwrap();

    let triangle = unsafe {
        gl_surface.set_swap_interval(&glutin_ctx, SwapInterval::Wait(NonZeroU32::new_unchecked(1))).unwrap();

        ctx.clear_color(0.0, 0.0, 0.0, 1.0);

        ctx.enable(glow::BLEND);
        ctx.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

        ctx.viewport(0, 0, width as i32, height as i32);

        setup_rectangle(&ctx, scale).unwrap()
    };

    let mut last_time = Instant::now();
    let mut dvd_x = 0.35;
    let mut dvd_y = -0.3;
    let mut direction = (1.0, -1.0);
    let mut color = Vec3::new(1.0, 1.0, 1.0);

    let mut rng = thread_rand();

    event_loop.run(|event, window_target| {
        match event {
            Event::WindowEvent { event: window_event, .. } => {
                match window_event {
                    WindowEvent::CloseRequested => {
                        window_target.exit();
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.state == ElementState::Pressed && event.logical_key == Key::Named(NamedKey::Escape) {
                            window_target.exit();
                        }
                    }
                    _window_event => ()
                }
            }
            Event::AboutToWait => {
                unsafe {
                    ctx.clear(glow::COLOR_BUFFER_BIT);

                    ctx.use_program(Some(triangle.1));

                    let model_loc = ctx.get_uniform_location(triangle.1, "model").unwrap();
                    
                    let mut model = Mat4::identity();

                    let act_time = Instant::now();
                    let delta_time = act_time.duration_since(last_time);
                    last_time = act_time;

                    dvd_x += delta_time.as_secs_f64() * DVD_SPEED * direction.0;
                    dvd_y += delta_time.as_secs_f64() * DVD_SPEED * direction.1 * screen_ratio;

                    match dvd_x {
                        x if x + IMG_SCALE_X > 2.0 => {
                            direction.0 = -1.0;
                            change_color(&mut rng, &mut color);
                        }
                        x if x < 0.0 => {
                            direction.0 = 1.0;
                            change_color(&mut rng, &mut color);
                        }
                        _ => ()
                    }

                    match dvd_y {
                        y if y - scale < -2.0 => {
                            direction.1 = 1.0;
                            change_color(&mut rng, &mut color);
                        }
                        y if y > 0.0 => {
                            direction.1 = -1.0;
                            change_color(&mut rng, &mut color);
                        }
                        _ => ()
                    }

                    model.append_translation_mut(&Vec3::new(dvd_x as f32, dvd_y as f32, 0.0));

                    ctx.uniform_matrix_4_f32_slice(Some(&model_loc), false, model.as_slice());

                    let color_loc = ctx.get_uniform_location(triangle.1, "color").unwrap();

                    ctx.uniform_3_f32_slice(Some(&color_loc), color.as_slice());

                    ctx.bind_texture(glow::TEXTURE_2D, Some(dvd_tex));

                    ctx.bind_vertex_array(Some(triangle.0));
                    ctx.draw_elements(glow::TRIANGLE_FAN, 6, glow::UNSIGNED_BYTE, 0);
                    ctx.bind_vertex_array(None);

                    ctx.bind_texture(glow::TEXTURE_2D, None);

                    gl_surface.swap_buffers(&glutin_ctx).unwrap();
                    wnd.request_redraw();
                }
            }
            Event::LoopExiting => {
                wnd.set_cursor_visible(true);
            }
            _event => ()
        }
    }).unwrap();
}

fn create_window() -> Result<(EventLoop<()>, Window, Display, NotCurrentContext, Config), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let window_builder = WindowBuilder::new().with_fullscreen(Some(Fullscreen::Borderless(None))).with_title("DVD Screen Saver");

    let config_template = ConfigTemplateBuilder::new().with_api(Api::OPENGL);

    let display_builder = DisplayBuilder::new().with_window_builder(Some(window_builder));

    let (wnd, gl_config) = display_builder.build(&event_loop, config_template, |configs| {configs.reduce(|acc, config| {
            if config.num_samples() == 0 {
                return config
            }
            acc
        }).unwrap()})?;

    let wnd = wnd.unwrap();

    wnd.set_cursor_visible(false);

    let gl_display = gl_config.display();

    let raw_wnd = wnd.raw_window_handle();

    let context_attributes = ContextAttributesBuilder::new().with_context_api(ContextApi::OpenGl(Some(Version::new(3, 3)))).with_profile(GlProfile::Core).build(Some(raw_wnd));

    let not_current_context = unsafe {gl_display.create_context(&gl_config, &context_attributes)?};

    Ok((event_loop, wnd, gl_display, not_current_context, gl_config))
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
