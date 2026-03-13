use anyhow::Result;
use std::path::PathBuf;

mod detector;
mod processor;
mod utils;
mod error;

use detector::APDDetector;
use processor::VideoProcessor;

#[tokio::main]
async fn main() -> Result<()> {
    // Konfigurasi
    let model_path = "../model/yolov26_retrain/best.onnx";
    let video_dir = "../samples";
    let output_dir = "../output_videos";
    
    // Classes sesuai dengan model
    let classes = vec!["helmet", "person", "fire-extinguisher"];
    
    // Inisialisasi detector
    println!("Memuat model ONNX: {}", model_path);
    let detector = APDDetector::new(model_path, &classes)?;
    
    // Buat direktori output jika belum ada
    std::fs::create_dir_all(output_dir).ok();
    
    // Proses semua video di folder samples
    let video_processor = VideoProcessor::new(&detector, output_dir);
    
    println!("Memproses video dari folder: {}", video_dir);
    video_processor.process_videos_in_directory(video_dir)?;
    
    println!("Selesai! Output video tersimpan di: {}", output_dir);
    Ok(())
}
