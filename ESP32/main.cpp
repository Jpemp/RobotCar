#include <Arduino.h>
#include <SPI.h>

#include <cstdint>
#include <iostream>

// put function declarations here:
char* readSPI(void);
void audioResponse(char*); //Signal recieved from Raspberry Pi
//turnLeft()
//turnRight()
//goForward()
//goBackward()
//carStop()


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
  if(digitalRead(CIPO) == HIGH){ //If a message is recieved from Raspberry Pi peripheral in SPI
    char* input_message;
    input_message = readSPI();
    audioResponse(input_message);
    delete[] input_message;
  }
  
  delay(100);
}

// put function definitions here:
void audioResponse(char* RPmessage){
  int message_length = sizeof(RPmessage)/sizeof(*RPmessage);

}

char* readSPI(void){
  digitalWrite(CS, LOW); //Selects peripheral to communicate with, which is the Raspberry Pi

  //SPI.beginTransaction(SPISettings(());

  byte byteNum = SPI.transfer(0x00); //Raspberry Pi sends a message that tells us how long the message will be

  int message_length =  (int)byteNum;
  char message[message_length];

  for(int i=0; i<message_length; i++){
    std::cout << "Hello World!" << std::endl;
  }

  //byte readByte = SPI.transfer(0x00); //Storing the message from the Raspberry Pi in this readByte variable
  digitalWrite(CS, HIGH); //Deselects Raspberry Pi for SPI communication
  return message;
}
