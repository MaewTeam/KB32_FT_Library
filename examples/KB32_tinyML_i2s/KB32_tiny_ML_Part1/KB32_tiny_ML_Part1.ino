
// #define debug // uncomment for debug

#define INTERVAL_Hz 2285.0f // 16000/7
#define FFT_N 128 // Must be a power of 2
#define TOTAL_TIME ((float)FFT_N/INTERVAL_Hz) //The time in which data was captured. This is equal to FFT_N/sampling_freq
#define SOUND_THRESHOLD 0.05f


#include <arduino.h>
#include "FFT.h" // include the library
#include <TFT_eSPI_KB32.h> // Hardware-specific library
#include "colormap.h" // include the library

#include <driver/i2s.h> // Hardware i2s module
#define bck_io_pin   17     //Servo2
#define ws_io_pin    4     //SDA1
#define data_in_pin  5     //SCL1


const i2s_port_t I2S_PORT = I2S_NUM_0;
volatile float adc_reading;


TFT_eSPI_KB32 tft = TFT_eSPI_KB32();

float fft_input[FFT_N];
float fft_output[FFT_N];

float max_magnitude = 0;
float fundamental_freq = 0;
float backgroundSound = 0;

uint32_t time_interval = 0;
volatile uint16_t fft_input_index = 0;
uint16_t x, y;
volatile uint8_t display_mode = 0;

char print_buf[300];

fft_config_t* real_fft_plan;

void i2sInit()
{
    Serial.println("Configuring I2S...");
    esp_err_t err;

    // The I2S config as per the example
    const i2s_config_t i2s_config = {
        .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX), // Receive, not transfer
        .sample_rate = 16000,                         // 16KHz
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT, // could only get it to work with 32bits
        .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT, // although the SEL config should be left, it seems to transmit on right
        .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,     // Interrupt level 1
        .dma_buf_count = 4,                           // number of buffers
        .dma_buf_len = 8                              // 8 samples per buffer (minimum)
    };

    // The pin config as per the setup
    const i2s_pin_config_t pin_config = {
        .bck_io_num = bck_io_pin,   // BCKL
        .ws_io_num = ws_io_pin,    // LRCL
        .data_out_num = -1, // not used (only for speakers)
        .data_in_num = data_in_pin   // Data
    };

    // Configuring the I2S driver and pins.
    // This function must be called before any I2S driver read/write operations.
    err = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("Failed installing driver: %d\n", err);
        while (true);
    }
    err = i2s_set_pin(I2S_PORT, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("Failed setting pin: %d\n", err);
        while (true);
    }
    Serial.println("I2S driver installed.");
}

void reader(void* pvParameters) {
    uint8_t sampling_devider = 0;
    while (1)
    {
        int32_t sample = 0;
        int bytes_read = i2s_pop_sample(I2S_PORT, (char*)&sample, portMAX_DELAY); // no timeout
        if (bytes_read > 0) {
            sampling_devider++;
            adc_reading = (float)(sample >> 16) / 32767.0 * 10; // normalize to 10
            if (fft_input_index < FFT_N && sampling_devider == 7) {
                sampling_devider = 0;
                real_fft_plan->input[fft_input_index] = adc_reading;
                fft_input_index++;
            }
        }
    }
}

float  readMic() {
    return adc_reading;
}

void calibrate() {
    int i = 0;
    backgroundSound = 0;
    while (i < 200) {
        if (micros() - time_interval >= (1000000L / INTERVAL_Hz)) {
            time_interval = micros();
            backgroundSound += abs(readMic());
            i++;
        }
    }
    backgroundSound /= 200;

#ifdef debug
    Serial.print("Background sound level is ");
    Serial.println(backgroundSound);
#endif
}

void IRAM_ATTR S1_pressed() {
    display_mode = 0;
}

void IRAM_ATTR S2_pressed() {
    display_mode = 1;
}

void setup() {

    Serial.begin(115200);

    pinMode(16, INPUT_PULLUP); //S1
    pinMode(14, INPUT_PULLUP); //s2

    attachInterrupt(16, S1_pressed, FALLING);
    attachInterrupt(14, S2_pressed, FALLING);

    real_fft_plan = fft_init(FFT_N, FFT_REAL, FFT_FORWARD, fft_input, fft_output);

    Serial.println("Initialized LCD");
    tft.init();   // initialize a ST7735S chip
    tft.fillScreen(TFT_BLACK);
    tft.Set_brightness(16); // 0-16 level
    tft.setTextColor(TFT_ORANGE);

    // Initialize the I2S peripheral
    i2sInit();
    // Create a task that will read the data
    xTaskCreatePinnedToCore(reader, "i2s_reader", 2048, NULL, 1, NULL, 1);
    delay(100);
}

void loop() {

    calibrate();

    float sound_mag = 0;

    while (sound_mag < SOUND_THRESHOLD) {
        float mic_mag = abs(readMic());
        sound_mag = (mic_mag - backgroundSound);
        tft.fillRect(158, 0, 2, 80 - (mic_mag * 100), TFT_BLACK);
        tft.fillRect(158, 80 - (mic_mag * 100) + 1, 2, (mic_mag * 100), mag_2_color(mic_mag * 10 * 15));
    }

    delay(20);
    fft_input_index = 0;
    while (fft_input_index != FFT_N) { //wait for fill buffer
        delay(1);
    }

    if (fft_input_index == FFT_N) {
        

        long int t1 = micros();

        // Execute transformation
        fft_execute(real_fft_plan);
        fft_input_index = 0;
       
        max_magnitude = 0;
        x = 0;
        for (int k = 1; k < real_fft_plan->size / 2; k++)
        {
            /*The real part of a magnitude at a frequency is followed by the corresponding imaginary part in the output*/
            float mag = 10 * sqrt(pow(real_fft_plan->output[2 * k], 2) + pow(real_fft_plan->output[2 * k + 1], 2)) / 1;
            float freq = k * 1.0 / TOTAL_TIME;



            // Print the output
            sprintf(print_buf, "%f", mag);
            Serial.print(print_buf);

            if (k + 1 < real_fft_plan->size / 2)
                Serial.print(",");

#ifdef debug
            sprintf(print_buf, "%f Hz : %f", freq, mag);
            Serial.println(print_buf);
#endif
            if (mag > max_magnitude)
            {
                max_magnitude = mag;
                fundamental_freq = freq;
            }

            if (display_mode == 1) {
                if (y + 1 == 80)
                    tft.drawFastHLine(0, 0, 159, TFT_WHITE);
                else
                    tft.drawFastHLine(0, y + 1, 159, TFT_WHITE);

                if (x < 160) {
                    tft.drawPixel(2 * x, y, mag_2_color(mag * 5));
                    tft.drawPixel(2 * x + 1, y, mag_2_color(mag * 5));
                    x++;
                }
            }
            else {
                y = 0;
                if (x * 2 < 160) {
                    tft.fillRect(x * 2, 0, 2, 80 - (mag * 2), TFT_BLACK);
                    tft.fillRect(x * 2, 80 - (mag * 2) + 1, 2, (mag * 2), mag_2_color(mag * 5));
                    x++;
                }
            }
        }
        y++;
        if (y >= 80)
            y = 0;

        if (display_mode == 0) {
            sprintf(print_buf, "%0.2f Hz : %0.2f Vpp", fundamental_freq, max_magnitude);
            tft.setCursor(1, 1);
            tft.print(print_buf);
        }


        long int t2 = micros();

        Serial.println();

#ifdef debug
        /*Multiply the magnitude of the DC component with (1/FFT_N) to obtain the DC component*/
        sprintf(print_buf, "DC component : %f V\n", (real_fft_plan->output[0]) / FFT_N);  // DC is at [0]
        Serial.println(print_buf);

        /*Multiply the magnitude at all other frequencies with (2/FFT_N) to obtain the amplitude at that frequency*/
        sprintf(print_buf, "Fundamental Freq : %f Hz\t Mag: %f V\n", fundamental_freq, (max_magnitude) * 2 / FFT_N);
        Serial.println(print_buf);

        Serial.print("Time taken: ");
        Serial.print((t2 - t1) * 1.0 / 1000);
        Serial.println(" milliseconds!\n\n\n");
#endif
        delay(600);
    }

}


