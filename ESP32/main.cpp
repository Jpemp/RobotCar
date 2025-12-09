#include <Arduino.h>
#include <SPI.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>

#include <cstdint>
#include <iostream>



// put function declarations here:
char* readSPI(void);
void audioResponse(char*); //Signal recieved from Raspberry Pi
void OnDataRecv(const uint8_t*, const uint8_t*, int);
void car_control(void);

//variable declaration for SPI communication
const int CS = D10; //D10 = CS -- Chip select. Choosing which peripheral to interact with
const int COPI = D11; //D11 = COPI -- SPI outputs data to peripheral
const int CIPO = D12; //D12 = CIPO -- SPI inputs data from peripheral
const int SCLK = D13;  //D13 = SCK -- SPI Serial Clock

//variable declaration for H-bridge switches
const int SW1 = A0; 
const int SW2 = A1;
const int SW3 = A2;
const int SW4 = A3;

const int SERVO = A6; //variable declaration for steering motor

uint8_t sender_address[] = {0x40, 0x22, 0xd8, 0x7b, 0x83, 0x64};

typedef struct message {
  int speedPot;
  int turnPot;
  bool controlSwitch;
} message;

message control_message;

esp_now_peer_info_t peerInfo; //information about remote (sender) ESP32

void setup() { //runs once
  //GPIO pin setup
  pinMode(CS, OUTPUT);
  pinMode(COPI, OUTPUT);
  pinMode(CIPO, INPUT);
  pinMode(SCLK, OUTPUT);

  pinMode(SW1, OUTPUT);
  pinMode(SW2, OUTPUT);
  pinMode(SW3, OUTPUT);
  pinMode(SW4, OUTPUT);

  pinMode(SERVO, OUTPUT);

  SPI.begin(); //starts SPI communication
  Serial.begin(115200);


  WiFi.mode(WIFI_STA);
  WiFi.begin(); //only need this to get MAC address of ESP32 Nano
  uint8_t mac[6];
  esp_err_t ret = esp_wifi_get_mac(WIFI_IF_STA, mac);
  Serial.printf("%02x:%02x:%02x:%02x:%02x:%02x\n", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

  if(esp_now_init() != ESP_OK){ //initializing ESP-NOW communication
    Serial.println("Error initializing ESP-NOW");
    return;
  }
  else{
    Serial.println("ESP-NOW successfully initialized!");
  }

  esp_now_register_recv_cb(esp_now_recv_cb_t(OnDataRecv)); //when data is recieved from the sender, this callback function will be ran
  
  memcpy(peerInfo.peer_addr, sender_address, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK){ //connects to sender peer
    Serial.println("Failed to pair with peer");
    return;
  }
  else{
    Serial.println("Paired with peer!");
  }
  //Set CS LOW to select peripheral to communicate with and HIGH to de-select and stop communicating

}

void loop() {

  if(control_message.controlSwitch == true){
    car_control();
  }

  else{
    if(digitalRead(CIPO) == HIGH){ //If a message is recieved from Raspberry Pi peripheral in SPI
      char* input_message;
      input_message = readSPI();
      audioResponse(input_message);
      delete[] input_message;
    }
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

void OnDataRecv(const uint8_t *mac, const uint8_t *incomingData, int len){
  memcpy(&control_message, incomingData, sizeof(control_message)); //stores data from sender into a struct
}

void car_control(void){
  //Pot goes from 0-1023 in analog value
  //Analog write goes from 0-255 in value
  int speedVal = control_message.speedPot; 
  int turnVal = control_message.turnPot;

  if(speedVal/4 == 255){ //doesnt move
    analogWrite(SW1, 0);
    analogWrite(SW2, 0);
    analogWrite(SW3, 0);
    analogWrite(SW4, 0);
  }
  else if(speedVal/4 > 255){ //goes forward
    analogWrite(SW1, 255-speedVal); //translating the pot value to be used for analogWrite (0-255)
    analogWrite(SW4, 255-speedVal);
  }
  else{ //goes backward
    analogWrite(SW2, speedVal-255); //translating the pot value to be used for analogWrite (0-255)
    analogWrite(SW3, speedVal-255);
  }

  analogWrite(SERVO, turnVal/4);

}
