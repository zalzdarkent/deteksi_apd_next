// Advanced Examples for APD Detector
// Uncomment sections sesuai use case

use anyhow::Result;
use std::path::Path;

// Import modules dari main project
// Pastikan compile dengan: cargo build --example advanced_examples

mod detector;
mod processor;
mod utils;
mod error;

use detector::APDDetector;
use processor::VideoProcessor;
use utils::Detection;

/// Example 1: Process single frame dari image file
#[allow(dead_code)]
fn example_single_frame() -> Result<()> {
    use opencv::imgcodecs;
    use opencv::core::Mat;

    let image_path = "../samples/sample_frame.jpg";
    let model_path = "../model/yolov26_retrain/best.onnx";
    let classes = vec!["helmet", "person", "fire-extinguisher"];

    // Load image
    let mut frame = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)?;

    // Create detector
    let detector = APDDetector::new(model_path, &classes)?;

    // Get video processor
    let processor = VideoProcessor::new(&detector, "../output_frames");
    
    // Process frame
    processor.process_frame(&frame)?;

    Ok(())
}

/// Example 2: Custom confidence threshold
#[allow(dead_code)]
fn example_custom_threshold() -> Result<()> {
    let model_path = "../model/yolov26_retrain/best.onnx";
    let classes = vec!["helmet", "person", "fire-extinguisher"];

    let mut detector = APDDetector::new(model_path, &classes)?;

    // Set custom thresholds
    detector.set_conf_threshold(0.7);  // Only high confidence detections
    detector.set_iou_threshold(0.3);   // Stricter NMS

    let video_processor = VideoProcessor::new(&detector, "../output_videos");
    video_processor.process_videos_in_directory("../samples")?;

    Ok(())
}

/// Example 3: Batch process dengan progress callback
#[allow(dead_code)]
fn example_batch_with_progress() -> Result<()> {
    let model_path = "../model/yolov26_retrain/best.onnx";
    let classes = vec!["helmet", "person", "fire-extinguisher"];
    let detector = APDDetector::new(model_path, &classes)?;

    let videos_dir = "../samples";
    let path = Path::new(videos_dir);

    let mut total = 0;
    let mut success = 0;

    // List dan count videos
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if matches!(ext_str.as_str(), "mp4" | "avi" | "mov" | "mkv") {
                    total += 1;
                }
            }
        }
    }

    println!("Total videos: {}", total);

    // Process dengan progress
    let processor = VideoProcessor::new(&detector, "../output_videos");
    for (idx, entry) in std::fs::read_dir(path)?.enumerate() {
        let entry = entry?;
        let video_path = entry.path();

        if video_path.is_file() {
            if let Some(ext) = video_path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if matches!(ext_str.as_str(), "mp4" | "avi" | "mov" | "mkv") {
                    print!("\rProcessing [{}/{}] ", idx + 1, total);
                    use std::io::Write;
                    std::io::stdout().flush().ok();

                    if processor.process_video(&video_path).is_ok() {
                        success += 1;
                    }
                }
            }
        }
    }

    println!("\n\nResults: {}/{} videos processed successfully", success, total);
    Ok(())
}

/// Example 4: Filter detections by class
#[allow(dead_code)]
fn example_filter_by_class(detections: Vec<Detection>) -> Vec<Detection> {
    // Only keep helmet and person detections
    let allowed_classes = vec!["helmet", "person"];
    
    detections
        .into_iter()
        .filter(|det| allowed_classes.contains(&det.class_name.as_str()))
        .collect()
}

/// Example 5: Statistics dari detections
#[allow(dead_code)]
fn example_statistics(detections: &[Detection]) {
    if detections.is_empty() {
        println!("No detections");
        return;
    }

    // Count per class
    let mut counts = std::collections::HashMap::new();
    let mut total_conf = 0.0;
    let mut max_conf = 0.0;
    let mut min_conf = 1.0;

    for det in detections {
        *counts.entry(&det.class_name).or_insert(0) += 1;
        total_conf += det.confidence;
        max_conf = max_conf.max(det.confidence);
        min_conf = min_conf.min(det.confidence);
    }

    println!("Detection Statistics:");
    println!("  Total detections: {}", detections.len());
    
    for (class, count) in counts {
        println!("  - {}: {}", class, count);
    }
    
    println!("  Confidence - Min: {:.2}, Max: {:.2}, Avg: {:.2}",
        min_conf,
        max_conf,
        total_conf / detections.len() as f32
    );
}

/// Example 6: Performance benchmark
#[allow(dead_code)]
fn example_benchmark(video_path: &str) -> Result<()> {
    use std::time::Instant;
    
    let model_path = "../model/yolov26_retrain/best.onnx";
    let classes = vec!["helmet", "person", "fire-extinguisher"];

    let start = Instant::now();
    
    let detector = APDDetector::new(model_path, &classes)?;
    println!("Model load time: {:.2}s", start.elapsed().as_secs_f32());

    let processor = VideoProcessor::new(&detector, "../output_videos");
    
    let process_start = Instant::now();
    processor.process_video(Path::new(video_path))?;
    
    println!("Processing time: {:.2}s", process_start.elapsed().as_secs_f32());

    Ok(())
}

fn main() -> Result<()> {
    println!("APD Detector - Advanced Examples");
    println!("=================================");
    println!();
    println!("Uncomment contoh yang ingin dijalankan di main()");
    println!();
    
    // Jalankan default: process semua video
    let model_path = "../model/yolov26_retrain/best.onnx";
    let classes = vec!["helmet", "person", "fire-extinguisher"];
    
    let detector = APDDetector::new(model_path, &classes)?;
    let processor = VideoProcessor::new(&detector, "../output_videos");
    
    processor.process_videos_in_directory("../samples")?;

    // Uncomment contoh lain sesuai kebutuhan:
    // example_custom_threshold()?;
    // example_batch_with_progress()?;
    // example_benchmark("../samples/kaizen_casting.mp4")?;

    Ok(())
}
