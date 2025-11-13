#include <Arduino.h>

// put function declarations here:

//AudioResponse() //Signal recieved from Raspberry Pi
//turnLeft()
//turnRight()
//goForward()
//goBackward()

//D10, D11, D12, D13 pins are for SPI

void setup() {
  // put your setup code here, to run once:
  pinMode(SCK, OUTPUT);

  
}

void loop() {
  digitalWrite(SCK, HIGH);
  delay(1000);
  digitalWrite(SCK, LOW);
  delay(1000);
}

// put function definitions here:
int myFunction(int x, int y) {
  return x + y;
}
