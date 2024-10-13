#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <iostream>
#include <string>
#include "yolov11.h"


/**
 * @brief Setting up Tensorrt logger
*/
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;


int main(int argc, char* argv[]) {

    // Define color codes for terminal output
    const std::string RED_COLOR = "\033[31m";
    const std::string GREEN_COLOR = "\033[32m";
    const std::string YELLOW_COLOR = "\033[33m";
    const std::string RESET_COLOR = "\033[0m";

    // Check for valid number of arguments
    if (argc < 4 || argc > 5) {
        std::cerr << RED_COLOR << "Usage: " << RESET_COLOR << argv[0]
            << " <mode> <input_path> <engine_path> [onnx_path]" << std::endl;
        std::cerr << YELLOW_COLOR << "  <mode> - Mode of operation: 'convert', 'infer_video', or 'infer_image'" << RESET_COLOR << std::endl;
        std::cerr << YELLOW_COLOR << "  <input_path> - Path to the input video/image or ONNX model" << RESET_COLOR << std::endl;
        std::cerr << YELLOW_COLOR << "  <engine_path> - Path to the TensorRT engine file" << RESET_COLOR << std::endl;
        std::cerr << YELLOW_COLOR << "  [onnx_path] - Path to the ONNX model (only for 'convert' mode)" << RESET_COLOR << std::endl;
        return 1;
    }

    // Parse command-line arguments
    std::string mode = argv[1];
    std::string inputPath = argv[2];
    std::string enginePath = argv[3];
    std::string onnxPath;

    // Validate mode and arguments
    if (mode == "convert") {
        if (argc != 5) {  // 'convert' requires onnx_path
            std::cerr << RED_COLOR << "Usage for conversion: " << RESET_COLOR << argv[0]
                << " convert <onnx_path> <engine_path>" << std::endl;
            return 1;
        }
        onnxPath = inputPath;  // In 'convert' mode, inputPath is actually onnx_path
    }
    else if (mode == "infer_video" || mode == "infer_image") {
        if (argc != 4) {
            std::cerr << RED_COLOR << "Usage for " << mode << ": " << RESET_COLOR << argv[0]
                << " " << mode << " <input_path> <engine_path>" << std::endl;
            return 1;
        }
    }
    else {
        std::cerr << RED_COLOR << "Invalid mode. Use 'convert', 'infer_video', or 'infer_image'." << RESET_COLOR << std::endl;
        return 1;
    }

    // Initialize the Logger
    Logger logger;

    // Handle 'convert' mode
    if (mode == "convert") {
        try {
            // Initialize YOLOv11 with the ONNX model path
            YOLOv11 yolov11(onnxPath, logger);
            std::cout << GREEN_COLOR << "Model conversion successful. Engine saved." << RESET_COLOR << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << RED_COLOR << "Error during model conversion: " << e.what() << RESET_COLOR << std::endl;
            return 1;
        }
    }
    // Handle inference modes
    else if (mode == "infer_video" || mode == "infer_image") {
        try {
            // Initialize YOLOv11 with the TensorRT engine path
            YOLOv11 yolov11(enginePath, logger);

            if (mode == "infer_video") {
                // Open the video file
                cv::VideoCapture cap(inputPath);
                if (!cap.isOpened()) {
                    std::cerr << RED_COLOR << "Failed to open video file: " << inputPath << RESET_COLOR << std::endl;
                    return 1;
                }

                // Prepare video writer to save the output (optional)
                std::string outputVideoPath = "output_video.avi";
                int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
                int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
                cv::VideoWriter video(outputVideoPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30,
                    cv::Size(frame_width, frame_height));

                cv::Mat frame;
                while (cap.read(frame)) {
                    // Preprocess the frame
                    yolov11.preprocess(frame);

                    // Perform inference
                    yolov11.infer();

                    // Postprocess to get detections
                    std::vector<Detection> detections;
                    yolov11.postprocess(detections);

                    // Draw detections on the frame
                    yolov11.draw(frame, detections);

                    // Display the frame (optional)
                    cv::imshow("Inference", frame);
                    if (cv::waitKey(1) == 27) { // Exit on 'ESC' key
                        break;
                    }

                    // Write the frame to the output video
                    video.write(frame);
                }

                cap.release();
                video.release();
                cv::destroyAllWindows();
                std::cout << GREEN_COLOR << "Video inference completed. Output saved to "
                    << outputVideoPath << RESET_COLOR << std::endl;
            }
            else if (mode == "infer_image") {
                // Read the image
                cv::Mat image = cv::imread(inputPath);
                if (image.empty()) {
                    std::cerr << RED_COLOR << "Failed to read image: " << inputPath << RESET_COLOR << std::endl;
                    return 1;
                }

                // Preprocess the image
                yolov11.preprocess(image);

                // Perform inference
                yolov11.infer();

                // Postprocess to get detections
                std::vector<Detection> detections;
                yolov11.postprocess(detections);

                // Draw detections on the image
                yolov11.draw(image, detections);

                // Display the image (optional)
                cv::imshow("Inference", image);
                cv::waitKey(0); // Wait indefinitely until a key is pressed

                // Save the output image
                std::string outputImagePath = "output_image.jpg";
                cv::imwrite(outputImagePath, image);
                std::cout << GREEN_COLOR << "Image inference completed. Output saved to "
                    << outputImagePath << RESET_COLOR << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cerr << RED_COLOR << "Error during inference: " << e.what() << RESET_COLOR << std::endl;
            return 1;
        }
    }

    return 0;
}