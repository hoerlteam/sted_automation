#include <Time.h>

/*
  TimeLogger based on AnalogReadSerial from Arduino Examples
*/

#define TIME_MSG_LEN  11   // time sync to PC is HEADER and unix time_t as ten ascii digits
#define TIME_HEADER  'T'   // Header tag for serial time sync message

bool timeWasSet;
int sampleTime; // time between samples (ms)
int baudRate;
// the setup routine runs once when you press reset:
void setup() {
  baudRate = 9600;
  timeWasSet = false;
  sampleTime = 1000;
  analogReference(INTERNAL1V1);
  // initialize serial communication at 9600 bits per second:
  Serial.begin(baudRate);
}

void getPCtime() {
  // if time available from serial port, sync the DateTime library
  while(Serial.available() >=  TIME_MSG_LEN ){  // time message
    if( Serial.read() == TIME_HEADER ) {        
      time_t pctime = 0;
      for(int i=0; i < TIME_MSG_LEN -1; i++){   
        char c= Serial.read();          
        if( c >= '0' && c <= '9')   
          pctime = (10 * pctime) + (c - '0') ; // convert digits to a number            
      }   
      setTime(pctime);   // Sync DateTime clock to the time received on the serial port
      timeWasSet = true;
    }  
  }
}


// the loop routine runs over and over again forever:
void loop() {

  getPCtime();

  if (timeWasSet)
  {
    // read the input on analog pin 0:
    int sensorValue = analogRead(A0);
    // print out the value you read:
  
    double vIn = (1100 / 1024.0 ) * sensorValue;
    double temp = (vIn - 500.0 ) / 10;
    Serial.print(now());
    Serial.print(";");
    Serial.println(temp);
    delay(sampleTime);        // delay in between reads for stability
  }
}

