mod util;
use app::App;

use winit::event_loop::EventLoop;

pub mod app;

fn main() {
    let event_loop = EventLoop::new().unwrap();

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
