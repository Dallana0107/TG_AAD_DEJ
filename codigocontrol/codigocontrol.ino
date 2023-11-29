#include <OneWire.h>

int DS18S20_Pin = 2;    // DS18S20 Signal pin on digital 2
int alarma = 13;        // Conexión del buzzer
int calentador = 12;    // Conexión de la resistencia calentadora
int bomba = 11;         // Conexión de la bomba
float maxima = 50;      // Máximo valor de temperatura permisible para que suene el buzzer
float minima = 20;      // Mínimo valor de temperatura permisible para activar la alarma
float setpoint = 35;    // Set point
float tolerancia = 0.5; // Tolerancia del sensor
int R = 0;              // Variable que indica si se encuentra prendida o apagada la resistencia
int B = 0;              // Variable que indica si se encuentra encendida o apagada la bomba
float tiempo;
int dsp1;             // Variable del primer contador para el set point
int dsp2 = -1000;     // Variable del segundo contador para el set point

// Temperature chip i/o
OneWire ds(DS18S20_Pin); // On digital pin 2

void setup(void) {
  Serial.begin(9600);
  Serial.println("CLEARDATA");                                      // Borra los datos en Excel.
  Serial.println("LABEL,TIEMPO (s), TEMPERATURA (C), RESISTENCIA, BOMBA, SETPOINT"); // Titulos de los ejes en la grafica de Excel.
  pinMode(alarma, OUTPUT);
  pinMode(calentador, OUTPUT);
  pinMode(bomba, OUTPUT);
  pinMode(7, OUTPUT);
}

void loop(void) {
  dsp1 = Serial.read(); // Lee datos ingresados al setpoint del Excel
  if (dsp2 == -1000) {
    dsp2 = dsp1;
  }
  if (dsp1 != dsp2) {
    setpoint = (dsp1 - 48) * 10;
    dsp1 = Serial.read();
    setpoint = setpoint + (dsp1 - 48);
    dsp1 = Serial.read();
    dsp1 = Serial.read();
    dsp2 = dsp1;
  }
  digitalWrite(7, LOW);
  float temperature = getTemp(); // Tomará alrededor de 750 ms para ejecutarse

  // Si la temperatura es igual o mayor que setpoint
  if (temperature >= minima && temperature <= setpoint - tolerancia && temperature < maxima && B != 1) {
    // Calentador encendido y bomba apagada
    digitalWrite(calentador, HIGH);
    R = 1;
    digitalWrite(bomba, LOW);
    B = 0;
  }
  else if (temperature > setpoint + tolerancia && temperature < maxima && R != 1) {
    // Calentador apagado y bomba encendida
    digitalWrite(calentador, LOW);
    R = 0;
    digitalWrite(bomba, HIGH);
    B = 1;
  } 
  else {
    // Calentador apagado y bomba apagada
    digitalWrite(calentador, LOW);
    R = 0;
    digitalWrite(bomba, LOW);
    B = 0;
  }

  if (temperature >= maxima || temperature <= minima) {
    digitalWrite(alarma, HIGH);
  } else {
    digitalWrite(alarma, LOW);
  }

  tiempo = millis() / 1000; // Define la variable "tiempo" como el tiempo en ejecución del programa en [s].

  if (temperature >= -55) {
    Serial.print("DATA,"); // Envía los valores a Excel.
    Serial.print(tiempo);  // Salida del valor de tiempo en [s].
    Serial.print(", ");
    Serial.print(temperature);
    Serial.print(", ");
    Serial.print(R);
    Serial.print(", ");
    Serial.print(B);
    Serial.print(", ");
    Serial.println(setpoint);
    delay(1000);
  }
}

float getTemp() {
  // Retorna la temperatura desde un DS18S20 en grados Celsius
  byte data[12];
  byte addr[8];
  if (!ds.search(addr)) {
    // No hay más sensores en la cadena, reinicia la búsqueda
    ds.reset_search();
    return -1000;
  }
  if (OneWire::crc8(addr, 7) != addr[7]) {
    Serial.println("CRC no es válido!");
    return -1000;
  }
  if (addr[0] != 0x10 && addr[0] != 0x28) {
    Serial.print("El dispositivo no está reconocido");
    return -1000;
  }
  ds.reset();
  ds.select(addr);
  ds.write(0x44, 1); // Inicia la conversión, con alimentación parásita al final

  delay(750); // Espera a que la conversión de temperatura se complete
  byte present = ds.reset();
  ds.select(addr);
  ds.write(0xBE); // Lee el Scratchpad

  for (int i = 0; i < 9; i++) { // Necesitamos 9 bytes
    data[i] = ds.read();
  }

  ds.reset_search();

  byte MSB = data[1];
  byte LSB = data[0];
  float tempRead = ((MSB << 8) | LSB); // Usando complemento a dos
  float TemperatureSum = tempRead / 16;

  return TemperatureSum;
}