#include <Ultrasonic.h>
#include <esp_camera.h>
#include <Arduino.h>
#include <smartbin2_inferencing.h>
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"

#define EI_CLASSIFIER_OBJECT_DETECTION 0

#define EI_CLASSIFIER_HAS_ANOMALY 0

// Ultrasonic sensor pins
Ultrasonic ultrasonic(12, 13);

// Initialize the camera
camera_config_t camera_config = {
    .pin_pwdn = -1,      // power down is not used
    .pin_reset = -1,     // software reset will be performed
    .pin_xclk = 4,      // XCLK is connected to GPIO4
    .pin_sscb_sda = 18, // SDA is connected to GPIO18
    .pin_sscb_scl = 23, // SCL is connected to GPIO23
    .pin_d7 = 36,       // D7 is connected to GPIO36
    .pin_d6 = 39,       // D6 is connected to GPIO39
    .pin_d5 = 34,       // D5 is connected to GPIO34
    .pin_d4 = 35,       // D4 is connected to GPIO35
    .pin_d3 = 32,       // D3 is connected to GPIO32
    .pin_d2 = 33,       // D2 is connected to GPIO33
    .pin_d1 = 1,        // D1 is connected to GPIO1
    .pin_d0 = 3,        // D0 is connected to GPIO3
    .pin_vsync = 5,     // VSYNC is not connected
    .pin_href = 27,     // HREF is not connected
    .pin_pclk = 25,     // PCLK is not connected

    // XCLK 20MHz or 10MHz for OV2640 double FPS (Experimental)
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,

    .pixel_format = PIXFORMAT_JPEG, // Use JPEG format
    .frame_size = FRAMESIZE_QVGA,   // Set frame size to QVGA (320x240)
    .jpeg_quality = 12,             // Set JPEG quality to 12
    .fb_count = 1                    // Only one frame buffer in use
};

#define EI_CAMERA_FRAME_BUFFER_SIZE (2 * 1024 * 1024)
static uint8_t frame_buffer[EI_CAMERA_FRAME_BUFFER_SIZE] __attribute__((aligned(32)));
static uint8_t *frame_buffer_cur = frame_buffer;

static void *realloc_frame_buffer(void) {
    size_t fb_len = esp_camera_fb_get_allocated_size(&camera_config);
    if (fb_len > EI_CAMERA_FRAME_BUFFER_SIZE) {
        if (frame_buffer_cur != frame_buffer) {
            free(frame_buffer_cur);
            frame_buffer_cur = frame_buffer;
        }
        frame_buffer_cur = (uint8_t *)malloc(fb_len);
        if (frame_buffer_cur == NULL) {
            ei_printf("ERR: Failed to allocate camera frame buffer of size %lu\n", fb_len);
            return NULL;
        }
    }
    return frame_buffer_cur;
}

static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr) {
    if (offset + length > esp_camera_fb_get_len()) {
        ei_printf("ERR: Camera frame buffer overflow\n");
        return -1;
    }
    memcpy(out_ptr, esp_camera_fb_get(), length);
    return 0;
}

bool captureImage() {
    frame_buffer_cur = (uint8_t *)realloc_frame_buffer();
    if (frame_buffer_cur == NULL) {
        return false;
    }

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ei_printf("Camera capture failed\n");
        return false;
    }

    memcpy(frame_buffer_cur, fb->buf, fb->len);
    esp_camera_fb_return(fb);

    return true;
}

bool classifyImage() {
    ei_impulse_result_t result;
    signal_t signal;
    signal.total_length = 224*224*3; // * EI_CLASSIFIER_INPUT_HEIGHT * EI_CLASSIFIER_INPUT_CHANNELS;
    signal.get_data = &ei_camera_get_data;

    EI_IMPULSE_ERROR res = run_classifier(&signal, &result, true);
    if (res != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", res);
        return false;
    }

    // Print the predictions
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        ei_printf("    %s: %.5f\n", result.classification[ix].label, result.classification[ix].value);
    }

    // Determine the predicted class
    // This logic assumes the labels are arranged in alphabetical order
    int predictedClass = 0;
    float maxProbability = 0.0;
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        if (result.classification[ix].value > maxProbability) {
            predictedClass = ix;
            maxProbability = result.classification[ix].value;
        }
    }

    // Trigger motors based on predicted class
    triggerMotors(predictedClass);

    return true;
}

void triggerMotors(int predictedClass) {
    // Add your motor control logic here
    // This function will be called after classifying the image
    // and you can implement logic to trigger motors based on the predicted class
    // For example:
    // if (predictedClass == 0) {
    //     // Move motors for Inorganic waste
    // } else if (predictedClass == 1) {
    //     // Move motors for Organic waste
    // } else if (predictedClass == 2) {
    //     // Move motors for Metal waste
    // }
}

void setup() {
    Serial.begin(115200);
    // Initialize camera
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x\n", err);
        return;
    }
    // Initialize ultrasonic sensor pins
}

void loop() {
    // Read distance from ultrasonic sensor
    float distance = ultrasonic.read();
    
    // If object detected (distance within threshold)
    if (distance < 30/* threshold distance */) {
        // Capture image
        if (captureImage()) {
            // Classify captured image
            if (classifyImage()) {
                // Object processed, wait for next trigger
                delay(5000); // Delay to avoid continuous triggers from the same object
            }
        }
    }
}
