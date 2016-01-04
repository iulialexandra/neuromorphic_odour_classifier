/* Sensor shield version v0.1
   fieldnames = ['Elapsed_time', 'Temperature', 'Humidity', 'TGS2600', 'TGS2602', 'TGS2610']

  */
#include <Wire.h>
#include <Adafruit_ADS1015.h>
//#include "Adafruit_HTU21DF.h"
#include <Adafruit_HTU21DF.h>

Adafruit_ADS1015 ads;  
Adafruit_HTU21DF htu;
int v_per_bit;

int16_t adc0, adc1, adc2, adc3, adc4, adc5, aadc0;
float temp, hum;
int time_ms;

void setup() {
  Serial.begin(9600);
  htu.begin();
  ads.begin();
  ads.setGain(GAIN_ONE); // 1bit = 2mV, +/- 4.096 V
  v_per_bit=2;
  Serial.println("Ready");
}

void loop() {
  if (Serial.available() > 0){
    temp = htu.readTemperature();
    hum = htu.readHumidity();
    adc0 = ads.readADC_SingleEnded(0);
    adc1 = ads.readADC_SingleEnded(1);
    //adc2 = ads.readADC_SingleEnded(2);
    adc3 = ads.readADC_SingleEnded(3);
    //adc4 = ads.readADC_SingleEnded(4);
    //adc5 = ads.readADC_SingleEnded(5);
   // aadc0 = analogRead(0);
    
    Serial.print(millis()); Serial.print('\t');
    Serial.print(temp); Serial.print('\t');
    Serial.print(hum); Serial.print('\t');
    Serial.print(adc0*v_per_bit); Serial.print('\t');
    Serial.print(adc1*v_per_bit); Serial.print('\t');
    //Serial.print(adc2*v_per_bit); Serial.print('\t');
    Serial.print(adc3*v_per_bit); Serial.print('\t');
    //Serial.print(adc4*v_per_bit); Serial.print('\t');
    //Serial.print(adc5*v_per_bit); Serial.print('\t');
    //Serial.print(map(aadc0, 0, 1023, 0, 5000));
    Serial.println();
    Serial.read();
    }
}
