use crate::error::APDDetectorError;
use crate::utils::{Detection, NonMaxSuppression};
use anyhow::Result;
use ndarray::ArrayViewD;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

pub struct APDDetector {
    session: Session,
    classes: Vec<String>,
    conf_threshold: f32,
    iou_threshold: f32,
}

impl APDDetector {
    pub fn new(model_path: &str, classes: &[&str]) -> Result<Self> {
        let _ = ort::init().commit();
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create SessionBuilder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load model from {}: {}", model_path, e))?;

        Ok(Self {
            session,
            classes: classes.iter().map(|s| s.to_string()).collect(),
            conf_threshold: 0.25,
            iou_threshold: 0.45,
        })
    }

    pub fn get_input_shape(&self) -> (i32, i32) {
        (640, 640)
    }

    pub fn detect(&self, frame_data: &[f32], frame_height: i32, frame_width: i32) -> Result<Vec<Detection>> {
        let h = 640;
        let w = 640;

        let input_name = self.session.inputs()[0].name().to_string();

        // Create input value using (shape, Vec) to avoid ndarray version conflicts
        let input_shape = vec![1i64, 3, h as i64, w as i64];
        let input_value = ort::value::Value::from_array((input_shape, frame_data.to_vec()))
            .map_err(|e| anyhow::anyhow!("Failed to create input value: {}", e))?;

        // Run inference
        let outputs = self
            .session
            .run(vec![(input_name, input_value)])
            .map_err(|e| {
                anyhow::anyhow!("Inference failed: {}", e)
            })?;

        // Extract output manually
        let (output_shape, output_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract tensor: {}", e))?;

        // Convert shape to Vec<usize> for ndarray
        let shape_vec: Vec<usize> = output_shape.iter().map(|&x| x as usize).collect();
        
        // Create ArrayViewD
        let output_tensor = ArrayViewD::from_shape(shape_vec, output_data)
            .map_err(|e| anyhow::anyhow!("Failed to create ArrayView: {}", e))?;

        // Parse detections
        let mut detections = self.parse_detections(output_tensor, frame_height, frame_width)?;

        // Apply NMS
        detections = NonMaxSuppression::apply(&detections, self.iou_threshold);

        Ok(detections)
    }

    fn parse_detections(
        &self,
        output_tensor: ArrayViewD<f32>,
        frame_height: i32,
        frame_width: i32,
    ) -> Result<Vec<Detection>> {
        let mut detections = Vec::new();
        let shape = output_tensor.shape();
        
        if shape.len() != 3 || shape[0] != 1 {
            return Err(APDDetectorError::InvalidModelOutput(format!(
                "Unexpected output shape: {:?}",
                shape
            ))
            .into());
        }

        let num_elements = shape[1]; 
        let num_anchors = shape[2];  

        for i in 0..num_anchors {
            let mut max_prob = 0.0;
            let mut class_id = 0;

            for j in 4..num_elements {
                let prob = output_tensor[[0, j, i]];
                if prob > max_prob {
                    max_prob = prob;
                    class_id = j - 4;
                }
            }

            if max_prob > self.conf_threshold {
                let cx = output_tensor[[0, 0, i]];
                let cy = output_tensor[[0, 1, i]];
                let w = output_tensor[[0, 2, i]];
                let h = output_tensor[[0, 3, i]];

                let x1 = (cx - w / 2.0) * (frame_width as f32 / 640.0);
                let y1 = (cy - h / 2.0) * (frame_height as f32 / 640.0);
                let x2 = (cx + w / 2.0) * (frame_width as f32 / 640.0);
                let y2 = (cy + h / 2.0) * (frame_height as f32 / 640.0);

                if class_id < self.classes.len() {
                    detections.push(Detection {
                        x1,
                        y1,
                        x2,
                        y2,
                        confidence: max_prob,
                        class_id,
                        class_name: self.classes[class_id].clone(),
                    });
                }
            }
        }

        Ok(detections)
    }
}
