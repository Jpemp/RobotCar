#include <Arduino.h>
#include <SPI.h>

// put function declarations here:

//AudioResponse() //Signal recieved from Raspberry Pi
//turnLeft()
//turnRight()
//goForward()
//goBackward()


//variable declaration for SPI communication
const int CS = D10; //D10 = CS -- Chip select. Choosing which peripheral to interact with
const int COPI = D11; //D11 = COPI -- SPI outputs data to peripheral
const int CIPO = D12; //D12 = CIPO -- SPI inputs data from peripheral
const int SCLK = D13;  //D13 = SCK -- SPI Serial Clock

//variable declaration for H-bridge switches
const int SW1 = D2; 
const int SW2 = D3;
const int SW3 = D4;
const int SW4 = D5;


void setup() {
  //pin setup as either input or output
  pinMode(CS, OUTPUT);
  pinMode(COPI, OUTPUT);
  pinMode(CIPO, INPUT);
  pinMode(SCLK, OUTPUT);

  pinMode(SW1, OUTPUT);
  pinMode(SW2, OUTPUT);
  pinMode(SW3, OUTPUT);
  pinMode(SW4, OUTPUT);

  SPI.begin(); //starts SPI communication
  //Set CS LOW to select peripheral to communicate with and HIGH to de-select and stop communicating
  
}

void loop() {

}

// put function definitions here:
int myFunction(int x, int y) {
  return x + y;
}
