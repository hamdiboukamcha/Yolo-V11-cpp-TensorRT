/**
 * @file YOLOv11.h
 * @brief Header file for the YOLOv11 object detection model using TensorRT and OpenCV.
 *
 * This class encapsulates the preprocessing, inference, and postprocessing steps required to
 * perform object detection using a YOLOv11 model with TensorRT.
 */

#pragma once

#include "NvInfer.h"
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace std;
using namespace cv;

/**
 * @struct Detection
 * @brief A structure representing a detected object.
 *
 * Contains the confidence score, class ID, and bounding box for a detected object.
 */
struct Detection
{
    float conf;      //!< Confidence score of the detection.
    int class_id;    //!< Class ID of the detected object.
    Rect bbox;       //!< Bounding box of the detected object.
};

/**
 * @class YOLOv11
 * @brief A class for running YOLOv11 object detection using TensorRT and OpenCV.
 *
 * This class handles model initialization, inference, and postprocessing to detect objects
 * in images.
 */
class YOLOv11
{
public:

    /**
     * @brief Constructor to initialize the YOLOv11 object.
     *
     * Loads the model and initializes TensorRT objects.
     *
     * @param model_path Path to the model engine or ONNX file.
     * @param logger Reference to a TensorRT logger for error reporting.
     */
    YOLOv11(string model_path, nvinfer1::ILogger& logger);

    /**
     * @brief Destructor to clean up resources.
     *
     * Frees the allocated memory and TensorRT resources.
     */
    ~YOLOv11();

    /**
     * @brief Preprocess the input image.
     *
     * Prepares the image for inference by resizing and normalizing it.
     *
     * @param image The input image to be preprocessed.
     */
    void preprocess(Mat& image);

    /**
     * @brief Run inference on the preprocessed image.
     *
     * Executes the TensorRT engine for object detection.
     */
    void infer();

    /**
     * @brief Postprocess the output from the model.
     *
     * Filters and decodes the raw output from the TensorRT engine into detection results.
     *
     * @param output A vector to store the detected objects.
     */
    void postprocess(vector<Detection>& output);

    /**
     * @brief Draw the detected objects on the image.
     *
     * Overlays bounding boxes and class labels on the image for visualization.
     *
     * @param image The input image where the detections will be drawn.
     * @param output A vector of detections to be visualized.
     */
    void draw(Mat& image, const vector<Detection>& output);

private:
    /**
     * @brief Initialize TensorRT components from the given engine file.
     *
     * @param engine_path Path to the serialized TensorRT engine file.
     * @param logger Reference to a TensorRT logger for error reporting.
     */
    void init(std::string engine_path, nvinfer1::ILogger& logger);

    float* gpu_buffers[2]; //!< The vector of device buffers needed for engine execution.
    float* cpu_output_buffer; //!< Pointer to the output buffer on the host.

    cudaStream_t stream; //!< CUDA stream for asynchronous execution.
    IRuntime* runtime; //!< The TensorRT runtime used to deserialize the engine.
    ICudaEngine* engine; //!< The TensorRT engine used to run the network.
    IExecutionContext* context; //!< The context for executing inference using an ICudaEngine.

    // Model parameters
    int input_w; //!< Width of the input image.
    int input_h; //!< Height of the input image.
    int num_detections; //!< Number of detections output by the model.
    int detection_attribute_size; //!< Size of each detection attribute.
    int num_classes = 80; //!< Number of object classes that can be detected.
    const int MAX_IMAGE_SIZE = 4096 * 4096; //!< Maximum allowed input image size.
    float conf_threshold = 0.3f; //!< Confidence threshold for filtering detections.
    float nms_threshold = 0.4f; //!< Non-Maximum Suppression (NMS) threshold for filtering overlapping boxes.

    vector<Scalar> colors; //!< A vector of colors for drawing bounding boxes.

    /**
     * @brief Build the TensorRT engine from the ONNX model.
     *
     * @param onnxPath Path to the ONNX file.
     * @param logger Reference to a TensorRT logger for error reporting.
     */
    void build(std::string onnxPath, nvinfer1::ILogger& logger);

    /**
     * @brief Save the TensorRT engine to a file.
     *
     * @param filename Path to save the serialized engine.
     * @return True if the engine was saved successfully, false otherwise.
     */
    bool saveEngine(const std::string& filename);
};
