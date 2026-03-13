use thiserror::Error;

#[derive(Error, Debug)]
pub enum APDDetectorError {
    #[error("Model load error: {0}")]
    ModelLoadError(String),
    
    #[error("Inference error: {0}")]
    InferenceError(String),
    
    #[error("Video processing error: {0}")]
    VideoProcessingError(String),
    
    #[error("Invalid model output: {0}")]
    InvalidModelOutput(String),
    
    #[error("ONNX Runtime error: {0}")]
    OnnxRuntimeError(String),
    
    #[error("OpenCV error: {0}")]
    OpenCVError(String),
    
    #[error("File error: {0}")]
    FileError(String),
}
