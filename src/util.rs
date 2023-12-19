use std::{mem::size_of, f64::consts::PI};

use glm::Vec3;
use tinyrand::Rand;
use tinyrand_std::ThreadLocalRand;

const COLORS: [Vec3; 4] = [
    Vec3::new(0.92156862745, 0.25098039215, 0.20392156862),
    Vec3::new(0.85098039215, 0.79607843137, 0.05882352941),
    Vec3::new(0.05882352941, 0.25882352941, 0.85098039215),
    Vec3::new(0.05882352941, 0.85098039215, 0.25882352941)
];

pub fn to_raw<T>(data: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * size_of::<T>())
    }
}

#[allow(dead_code)]
pub fn to_radians(degrees: f64) -> f64 {
    degrees * PI / 180.0
}

pub fn change_color(rng: &mut ThreadLocalRand, color: &mut Vec3) {
    let next = {
        let mut next_color;
        loop {
            next_color = COLORS[rng.next_lim_usize(4)];
            if next_color != *color {
                break;
            }
        }
        next_color
    };

    let _ = std::mem::replace(color, next);
}
