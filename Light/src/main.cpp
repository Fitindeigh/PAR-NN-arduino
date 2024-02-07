#include <Arduino.h>
#include "Watchdog\src\Watchdog.h"
#include <Wire.h>
#include "BH1750-master\src\BH1750.h"
#include "APDS-9960_Sensor\src\SparkFun_APDS9960.h"
#include "LiquidCrystal/src/LiquidCrystal.h"

LiquidCrystal lcd(8, 9, 4, 5, 6, 7);  // uno
#define period 1000

Watchdog watchdog;

SparkFun_APDS9960 apds = SparkFun_APDS9960();

uint16_t ambient_light = 0;
uint16_t red_light = 0;
uint16_t green_light = 0;
uint16_t blue_light = 0;

BH1750 lightMeter;


// значения для нормализации данных
float max_lux = 25201.0;
float min_lux = 31.0;
float max_rl = 887.0;
float min_rl = 0.0;
float max_gl = 859.0;
float min_gl = 1.0;
float max_bl = 666.0;
float min_bl = 1.0;
float max_par = 407;
float min_par = 0.41;

// создание матриц
float ** setMassive(int n, int m){
    float ** mas = new float *[n];
    for (int i = 0; i < n; i++) *(mas+i) =  new float [m];

    for (int i = 0; i < n; i++){
        for (int j = 0; j < m; j++){
            *(*(mas+i)+j) = rand() % 20 + 2;
        }
    }
    return mas;
}
// удаление матриц
void deleteMatrix( float ** mass, int n, int k){
    for (int i = k; i < n; i++){
        delete [] mass[i];
    }
    delete [] mass;
}


void clearLine(int line){
  switch (line){
    case 1:
    lcd.setCursor(0, line); lcd.print(F("                ")); break;
    case 2:   
    lcd.setCursor(0, line); lcd.print(F("                ")); break;
    case 3:
      lcd.setCursor(0, 0);
      lcd.print(F("                "));
      lcd.setCursor(0, 1);
      lcd.print(F("                "));
      break;

  }
}

// размер массива весов
const uint8_t n_w01 = 4;
const uint8_t m_w01 = 8;

float weights_w01 [n_w01][m_w01] = {{-0.15967719,  0.36839018, -0.99977125, -0.39533481, -0.70648822, -0.79091725,
  -0.62747958,-0.32310005},
 {-0.19727816 , 0.02075321, -0.16161097,  0.37043907, -0.5910955,   0.77348804,
  -0.94522481,  0.33040741},
 {-0.15347735,  0.04839765, -0.71922612, -0.60379702 , 0.60148914  ,0.95761408,
  -0.37315164,  0.37201914},
 { 0.76500396 , 0.72336166 ,-0.82991158 ,-0.92189043 ,-0.66033916 , 0.77610101,
  -0.80330633 ,-0.16969541}};

const uint8_t n_w12 = 8;
const uint8_t m_w12 = 8;

float weights_w12 [n_w12][m_w12] = {{ 0.90984974,  0.06519271,  0.38375423, -0.36905551 , 0.37629943 , 0.66925134,
  -0.97099154 , 0.50307602},
 { 0.9395265 ,  0.49037624 ,-0.43911202 , 0.57736694, -0.77782596, -0.10421295
   ,0.77995208 ,-0.39978631},
 {-0.42444932 ,-0.73994286 ,-0.96126608  ,0.35767107 ,-0.57674377 ,-0.46890668,
  -0.01685368, -0.89327491},
 { 0.14823522 ,-0.70654285 , 0.17861107 , 0.39951672 ,-0.79533114 ,-0.17188802
  , 0.38880033 ,-0.17164146},
 {-0.90009308 , 0.07179281,  0.32758929  ,0.02977822  ,0.88918951 , 0.17311008,
   0.80680383 ,-0.72505059},
 {-0.76164298 , 0.60757169 ,-0.20464633, -0.67033574 , 0.87462573, -0.30446828,
   0.45621988,  0.46819767},
 { 0.76661218  ,0.24734441 , 0.50188487, -0.30220332 ,-0.46014422 , 0.79177244,
  -0.14381762 , 0.92968009},
 { 0.32290387,  0.2424731 , -0.77050805 , 0.89886796 ,-0.0977084  , 0.15677923,
  -0.18932248 ,-0.52394008}};

const uint8_t n_w23 = 8;
const uint8_t m_w23 = 1;

float weights_w23 [n_w23][m_w23] = { 0.79695434,
  0.09911479
 ,-0.99425935
 , 0.23428985
 ,-0.3467102 
 ,-0.00470652
 , 0.7718842 
 ,-0.29271039};

// функция активации

void relu(float ** mass, int n, int m){
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (mass[i][j] < 0) mass[i][j] = 0;
    }
  }

}

// функция перемножения матриц

float ** dot(float ** mass1, int n1, int m1, float ** mass2, int n2, int m2){
    if (m1 != n2) return 0;
    else{

        float ** massdot = setMassive(n1, m2);

        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < m2; j++) {
                massdot[i][j] = 0;
                for (int k = 0; k < m1; k++) {
                    massdot[i][j] += mass1[i][k] * mass2[k][j];}
            }
         }
        return massdot;
    }


}

// прямой проход
float forward_pass(float ** input_weights, int n, int m){
  float PAR = 0.0;
  // первый слой
  float ** weights_w01_m = setMassive(n_w01, m_w01);
    for (int i = 0; i < n_w01; i++) {
      for (int j = 0; j < m_w01; j++) {
      weights_w01_m[i][j] = weights_w01[i][j];
    }
  }
  float ** layer1 = dot(input_weights, n, m, weights_w01_m, n_w01, m_w01);
  relu(layer1, n, m_w01);
  deleteMatrix(weights_w01_m, n_w01, m_w01);
  // второй слой
  float ** weights_w12_m = setMassive(n_w12, m_w12);
  for (int i = 0; i < n_w12; i++) {
    for (int j = 0; j < m_w12; j++) {
      weights_w12_m[i][j] = weights_w12[i][j];
    }
  }
  float ** layer2 = dot(layer1, n, m_w01, weights_w12_m, n_w12, m_w12);
  relu(layer2, n, m_w12);
  deleteMatrix(weights_w12_m, n_w12, m_w12);
  deleteMatrix(layer1, n, m_w01);
  // третий слой
  float ** weights_w23_m = setMassive(n_w23, m_w23);
  for (int i = 0; i < n_w23; i++) {
    for (int j = 0; j < m_w23; j++) {
      weights_w23_m[i][j] = weights_w23[i][j];
    }
  }
  float ** layer3 = dot(layer2, n, m_w12, weights_w23_m, n_w23, m_w23);
  deleteMatrix(layer2, n, m_w12);
  deleteMatrix(weights_w23_m, n_w23, m_w23);
  PAR = layer3[0][0];
  deleteMatrix(layer2, n, m_w23);

  return PAR;
}

void setup() {
  lcd.begin(16, 2);
  Serial.begin(9600);
  Wire.begin();
  watchdog.enable(Watchdog::TIMEOUT_2S);

    if ( apds.init() ) {
    Serial.println(F("APDS-9960 initialization complete"));
  } else {
    Serial.println(F("Something went wrong during APDS-9960 init!"));
  }
  
  // Start running the APDS-9960 light sensor (no interrupts)
  if ( apds.enableLightSensor(false) ) {
    Serial.println(F("Light sensor is now running"));
  } else {
    Serial.println(F("Something went wrong during light sensor init!"));
  }
  lightMeter.begin();
  // Wait for initialization and calibration to finish
  delay(500);
}


void loop() {
  static uint32_t timer1;
  if (millis() - timer1 >= period){
    timer1 = millis();

    float lux = 0;
    for (int i = 0; i < 10; i++) {lux += lightMeter.readLightLevel(); delay(50);}
    lux /= 10;
    Serial.print(int(lux));
    Serial.print(" ");
      // Read the light levels (ambient, red, green, blue)
    int bl = 0;
    int rl = 0;
    int gl = 0;
    for (int i = 0; i < 10; i++) {
    if (  !apds.readRedLight(red_light) ||
          !apds.readGreenLight(green_light) ||
          !apds.readBlueLight(blue_light) ) {
      Serial.println("Error reading light values");
    } else {
      bl += blue_light;
      gl += green_light;
      rl += red_light;
    }
    }
    rl /= 10;
    gl /= 10;
    bl /= 10;
    Serial.print(" "); //r
    Serial.print(rl);
    Serial.print(" ");//g
    Serial.print(gl);
    Serial.print(" ");//b
    Serial.println(bl);


    // создание входного вектора
    int n1 = 1;
    int m1 = 4;

    float ** input_weights = setMassive(n1, m1);
    // нормализация данных
    input_weights[0][0] = (lux - min_lux) / (max_lux - min_lux);
    input_weights[0][1] = (rl - min_rl) / (max_rl - min_rl);
    input_weights[0][2] = (gl - min_gl) / (max_gl - min_gl);
    input_weights[0][3] = (bl - min_bl) / (max_bl - min_bl);
    float PAR = forward_pass(input_weights, n1, m1);
    deleteMatrix(input_weights, n1, m1);
    Serial.print("PAR = ");
    float fin_PAR = PAR*(max_par - min_par) + min_par;
    Serial.println(fin_PAR);
    clearLine(3);
    lcd.setCursor(0, 0);
    lcd.print(F("PAR:"));
    lcd.setCursor(6, 0);
    lcd.print(fin_PAR);
  }
  watchdog.reset();
}

