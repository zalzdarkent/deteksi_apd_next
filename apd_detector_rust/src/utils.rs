use opencv::core::{Point, Scalar, Mat, Size, CV_32F};
use opencv::prelude::*;
use opencv::imgproc;
use std::fmt;

#[derive(Clone, Debug)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class_id: usize,
    pub class_name: String,
}

impl Detection {
    pub fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    pub fn iou(&self, other: &Detection) -> f32 {
        let x1_inter = self.x1.max(other.x1);
        let y1_inter = self.y1.max(other.y1);
        let x2_inter = self.x2.min(other.x2);
        let y2_inter = self.y2.min(other.y2);

        if x2_inter <= x1_inter || y2_inter <= y1_inter {
            return 0.0;
        }

        let inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter);
        let self_area = self.area();
        let other_area = other.area();
        let union_area = self_area + other_area - inter_area;

        if union_area > 0.0 {
            inter_area / union_area
        } else {
            0.0
        }
    }
}

impl fmt::Display for Detection {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[{}] ({:.2}, {:.2}) -> ({:.2}, {:.2}) conf: {:.2}",
            self.class_name, self.x1, self.y1, self.x2, self.y2, self.confidence
        )
    }
}

/// Non-Maximum Suppression untuk menghilangkan overlapping detections
pub struct NonMaxSuppression;

impl NonMaxSuppression {
    pub fn apply(detections: &[Detection], iou_threshold: f32) -> Vec<Detection> {
        if detections.is_empty() {
            return vec![];
        }

        // Sort by confidence descending
        let mut sorted = detections.to_vec();
        sorted.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut keep = Vec::new();
        let mut used = vec![false; sorted.len()];

        for i in 0..sorted.len() {
            if used[i] {
                continue;
            }

            keep.push(sorted[i].clone());
            used[i] = true;

            // Suppress detections dengan IOU tinggi
            for j in (i + 1)..sorted.len() {
                if !used[j] && sorted[i].iou(&sorted[j]) > iou_threshold {
                    used[j] = true;
                }
            }
        }

        keep
    }
}

/// Color palette untuk box dan text
pub struct ColorPalette;

impl ColorPalette {
    /// Get warna RGB untuk class_id tertentu
    pub fn get_color(class_id: usize) -> (u8, u8, u8) {
        let colors = vec![
            (0, 255, 0),     // helmet - green
            (255, 0, 0),     // person - blue
            (0, 0, 255),     // fire-extinguisher - red
        ];
        colors[class_id % colors.len()]
    }
}

/// Draw detections pada frame
pub fn draw_detections(
    frame: &mut Mat,
    detections: &[Detection],
) -> opencv::Result<()> {
    for det in detections {
        let color = ColorPalette::get_color(det.class_id);
        let scalar = Scalar::new(color.0 as f64, color.1 as f64, color.2 as f64, 0.0);

        // Hitung pixel coordinates
        let x1 = det.x1 as i32;
        let y1 = det.y1 as i32;
        let x2 = det.x2 as i32;
        let y2 = det.y2 as i32;

        // Draw bounding box
        imgproc::rectangle(
            frame,
            opencv::core::Rect::new(x1, y1, x2 - x1, y2 - y1),
            scalar,
            2,
            imgproc::LINE_8,
            0,
        )?;

        // Draw label
        let label = format!("{} {:.2}%", det.class_name, det.confidence * 100.0);
        let font = imgproc::FONT_HERSHEY_SIMPLEX;
        let font_scale = 0.7;
        let thickness = 2;
        let mut baseline = 0;

        let text_size = imgproc::get_text_size(&label, font, font_scale, thickness, &mut baseline)?;
        let label_y = (y1 - 5).max(text_size.height);

        // Draw background untuk text
        imgproc::rectangle(
            frame,
            opencv::core::Rect::new(x1, label_y - text_size.height - 5, text_size.width + 5, text_size.height + 10),
            scalar,
            -1,
            imgproc::LINE_8,
            0,
        )?;

        // Draw text (tidak terganggu oleh model's internal labeling)
        imgproc::put_text(
            frame,
            &label,
            Point::new(x1 + 2, label_y - 3),
            font,
            font_scale,
            Scalar::new(255.0, 255.0, 255.0, 0.0),
            thickness,
            imgproc::LINE_8,
            false,
        )?;
    }

    Ok(())
}

/// Normalisasi frame ke format model (channel-first, normalized)
pub fn preprocess_frame(
    frame: &Mat,
    input_height: usize,
    input_width: usize,
) -> opencv::Result<Vec<f32>> {
    use opencv::core::{Size, CV_32F};
    use opencv::imgproc;

    // Resize frame
    let mut resized = Mat::default();
    imgproc::resize(
        frame,
        &mut resized,
        Size::new(input_width as i32, input_height as i32),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    // Convert BGR to RGB dan normalize
    let mut rgb = Mat::default();
    imgproc::cvt_color(&resized, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;

    // Convert to float32
    let mut float_mat = Mat::default();
    rgb.convert_to(&mut float_mat, CV_32F, 1.0, 0.0)?;

    // Extract data dan convert to channel-first format
    let data = float_mat.data_typed::<f32>()?;
    let total_size = input_height * input_width * 3;
    let mut result = vec![0.0f32; total_size];

    // Planar format (NCHW) - Normalize to [0, 1]
    for i in 0..input_height * input_width {
        for c in 0..3 {
            result[c * input_height * input_width + i] = 
                data[i * 3 + c] / 255.0; // RGB order from imgproc::COLOR_BGR2RGB
        }
    }

    Ok(result)
}

/// Save detections ke JSON
pub fn save_detections_json(detections: &[Detection], output_path: &str) -> std::io::Result<()> {
    use serde_json::json;

    let data = json!({
        "detections": detections.iter().map(|d| json!({
            "class": d.class_name,
            "confidence": d.confidence,
            "bbox": {
                "x1": d.x1,
                "y1": d.y1,
                "x2": d.x2,
                "y2": d.y2,
            }
        })).collect::<Vec<_>>()
    });

    std::fs::write(output_path, serde_json::to_string_pretty(&data)?)?;
    Ok(())
}
