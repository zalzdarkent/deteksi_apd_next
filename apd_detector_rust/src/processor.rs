use crate::detector::APDDetector;
use crate::error::APDDetectorError;
use crate::utils::{draw_detections, preprocess_frame};
use anyhow::Result;
use opencv::core::Mat;
use opencv::prelude::*;
use opencv::videoio::{VideoCapture, VideoWriter, CAP_PROP_FPS, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_COUNT};
use std::path::Path;

pub struct VideoProcessor<'a> {
    detector: &'a APDDetector,
    output_dir: String,
}

impl<'a> VideoProcessor<'a> {
    pub fn new(detector: &'a APDDetector, output_dir: &str) -> Self {
        VideoProcessor {
            detector,
            output_dir: output_dir.to_string(),
        }
    }

    /// Process semua video di direktori
    pub fn process_videos_in_directory(&self, video_dir: &str) -> Result<()> {
        let path = Path::new(video_dir);
        
        if !path.is_dir() {
            return Err(APDDetectorError::VideoProcessingError(
                format!("Directory not found: {}", video_dir),
            )
            .into());
        }

        let mut video_count = 0;

        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Some(ext) = path.extension() {
                    let ext_str = ext.to_string_lossy().to_lowercase();
                    
                    // Cek jika file adalah video
                    if matches!(ext_str.as_str(), "mp4" | "avi" | "mov" | "mkv" | "flv" | "wmv") {
                        match self.process_video(&path) {
                            Ok(_) => {
                                video_count += 1;
                                println!(
                                    "✓ Berhasil memproses: {:?}",
                                    path.file_name().unwrap_or_default()
                                );
                            }
                            Err(e) => {
                                eprintln!(
                                    "✗ Error memproses {:?}: {}",
                                    path.file_name().unwrap_or_default(),
                                    e
                                );
                            }
                        }
                    }
                }
            }
        }

        println!(
            "\nTotal video diproses: {}",
            video_count
        );

        Ok(())
    }

    /// Process single video
    pub fn process_video(&self, video_path: &Path) -> Result<()> {
        let video_name = video_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("video")
            .to_string();

        println!("\nMemproses: {}", video_name);

        // Open video file
        let mut cap = VideoCapture::from_file(
            &video_path.to_string_lossy(),
            opencv::videoio::CAP_ANY,
        )
        .map_err(|e| {
            APDDetectorError::VideoProcessingError(format!(
                "Gagal membuka video {}: {}",
                video_name, e
            ))
        })?;

        if !cap.is_opened()? {
            return Err(APDDetectorError::VideoProcessingError(
                format!("Tidak dapat membuka video: {}", video_name),
            )
            .into());
        }

        // Get video properties
        let frame_width = cap.get(CAP_PROP_FRAME_WIDTH)? as i32;
        let frame_height = cap.get(CAP_PROP_FRAME_HEIGHT)? as i32;
        let fps = cap.get(CAP_PROP_FPS)? as f64;
        let total_frames = cap.get(CAP_PROP_FRAME_COUNT)? as i32;

        if fps <= 0.0 {
            return Err(APDDetectorError::VideoProcessingError(
                "FPS tidak valid".to_string(),
            )
            .into());
        }

        println!(
            "  Resolusi: {}x{} | FPS: {:.1} | Total frames: {}",
            frame_width, frame_height, fps, total_frames
        );

        // Prepare output video
        let output_path = format!(
            "{}/detected_{}",
            self.output_dir,
            video_name
        );

        let fourcc = opencv::videoio::VideoWriter::fourcc('m', 'p', '4', 'v')?;
        let mut writer = VideoWriter::new(
            &output_path,
            fourcc,
            fps,
            opencv::core::Size::new(frame_width, frame_height),
            true,
        )
        .map_err(|e| {
            APDDetectorError::VideoProcessingError(format!(
                "Gagal membuat output video: {}",
                e
            ))
        })?;

        if !writer.is_opened()? {
            return Err(APDDetectorError::VideoProcessingError(
                "Output video writer tidak siap".to_string(),
            )
            .into());
        }

        // Process frames
        let input_h = self.detector.get_input_shape().0;
        let input_w = self.detector.get_input_shape().1;
        let mut frame = Mat::default();
        let mut frame_count = 0;

        loop {
            if !cap.read(&mut frame)? {
                break;
            }

            frame_count += 1;

            // Show progress
            if frame_count % 30 == 0 {
                print!("\r  Processing frame: {}/{}", frame_count, total_frames);
                use std::io::Write;
                std::io::stdout().flush().ok();
            }

            // Preprocess frame
            let processed_data = preprocess_frame(&frame, input_h as usize, input_w as usize)?;

            // Run detection
            match self.detector.detect(&processed_data, frame_height as i32, frame_width as i32) {
                Ok(detections) => {
                    // Draw detections - LABEL TETAP SESUAI MODEL TANPA KACAU
                    let mut output_frame = frame.clone();
                    draw_detections(&mut output_frame, &detections)
                        .map_err(|e| {
                            APDDetectorError::VideoProcessingError(format!(
                                "Error drawing detections: {}",
                                e
                            ))
                        })?;

                    // Write frame
                    writer.write(&output_frame).map_err(|e| {
                        APDDetectorError::VideoProcessingError(format!(
                            "Error writing frame: {}",
                            e
                        ))
                    })?;

                    // Print detections for first frame
                    if frame_count == 1 && !detections.is_empty() {
                        println!(
                            "\n  Detections pada frame 1: {} objects",
                            detections.len()
                        );
                        for (i, det) in detections.iter().enumerate() {
                            println!("    [{}] {}", i + 1, det);
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "\n  Error detection pada frame {}: {}",
                        frame_count, e
                    );
                    // Tetap tulis frame original tanpa detections jika ada error
                    writer.write(&frame).map_err(|e| {
                        APDDetectorError::VideoProcessingError(format!(
                            "Error writing frame: {}",
                            e
                        ))
                    })?;
                }
            }
        }

        println!(
            "\n  ✓ Selesai! Output: {}\n",
            output_path
        );

        drop(cap);
        drop(writer);

        Ok(())
    }

    /// Process single frame (utility)
    pub fn process_frame(&self, frame: &Mat) -> Result<()> {
        let input_h = self.detector.get_input_shape().0;
        let input_w = self.detector.get_input_shape().1;

        let rows = frame.rows() as usize;
        let cols = frame.cols() as usize;

        // Preprocess
        let processed_data = preprocess_frame(frame, input_h as usize, input_w as usize)?;

        // Detect
        let detections = self.detector.detect(&processed_data, rows as i32, cols as i32)?;

        // Print hasil
        println!("Detections found: {}", detections.len());
        for (i, det) in detections.iter().enumerate() {
            println!("  [{}] {}", i + 1, det);
        }

        Ok(())
    }
}
