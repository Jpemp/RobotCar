#include <Arduino.h>
#include <SPI.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>

#include <cstdint>
#include <iostream>



// put function declarations here:
char* readSPI(void); //message recieved from Raspberry Pi
void audioResponse(char*); //car moves based on Raspberry Pi message
void OnDataRecv(const uint8_t*, const uint8_t*, int); //callback function that occurs when data is recieved from sender on ESP-NOW
void manual_control(void); //car moves based on manual control from the remote

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

uint8_t sender_address[] = {0x40, 0x22, 0xd8, 0x7b, 0x83, 0x64}; //mac address of remote ESP32


typedef struct message { //struct that contains car control data
  int speedPot;
  int turnPot;
  bool controlSwitch;
} message;

message recieved_message; //where recieved data goes
message control_message; //contains current speed and turn angle


esp_now_peer_info_t peerInfo; //information about remote (sender) ESP32

void setup() { //runs once
  //GPIO pin setup
  pinMode(CS, OUTPUT); //SPI-related pins
  pinMode(COPI, OUTPUT);
  pinMode(CIPO, INPUT);
  pinMode(SCLK, OUTPUT);

  pinMode(SW1, OUTPUT); //H-bridge drive pins
  pinMode(SW2, OUTPUT);
  pinMode(SW3, OUTPUT);
  pinMode(SW4, OUTPUT);

  pinMode(SERVO, OUTPUT); //Steering pin

  SPI.begin(); //starts SPI communication

  Serial.begin(115200); //Allows to print things to a console


  WiFi.mode(WIFI_STA); //turns esp32 into station mode
  WiFi.begin(); //only need this to get MAC address of ESP32 Nano
  uint8_t mac[6]; //retrieved mac address is stored here
  esp_err_t ret = esp_wifi_get_mac(WIFI_IF_STA, mac); //gets the mac address
  Serial.printf("%02x:%02x:%02x:%02x:%02x:%02x\n", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

  if(esp_now_init() != ESP_OK){ //initializing ESP-NOW communication
    Serial.println("Error initializing ESP-NOW");
    return;
  }
  else{
    Serial.println("ESP-NOW successfully initialized!");
  }

  esp_now_register_recv_cb(esp_now_recv_cb_t(OnDataRecv)); //when data is recieved from the sender, this callback function will be ran
  
  memcpy(peerInfo.peer_addr, sender_address, 6); //sets up remote ESP32 information for ESP-NOW communication
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

void loop() { //loops infinitely
  if(control_message.controlSwitch == true){
    manual_control();
  }

  else{
    if(digitalRead(CIPO) == HIGH){ //If a message is recieved from Raspberry Pi peripheral in SPI
      char* rasp_message; //data recieved from Raspberry Pi will be stored in this variable
      rasp_message = readSPI();
      audioResponse(rasp_message); //interprets Raspberry Pi message for controlling the car
      delete[] rasp_message; //gets rid of message and frees up memory
    }

  }
  
  delay(100);
}

// put function definitions here:
void audioResponse(char* RPmessage){
  int msg_length = sizeof(RPmessage)/sizeof(RPmessage[0]); //message is read through and car makes moves based on each character in the message
  char command;
  for (int i = 0; i < msg_length; i++){
    command = RPmessage[i];
    switch(command){
      case 's': //stop car
        analogWrite(SW1, 0);
        analogWrite(SW2, 0);
        analogWrite(SW3, 0);
        analogWrite(SW4, 0);
        control_message.speedPot = 127;
        break;
      case 'u': //car goes foward
        analogWrite(SW1, 127);
        analogWrite(SW2, 0);
        analogWrite(SW3, 0);
        analogWrite(SW4, 127);
        control_message.speedPot = 255;
        break;
      case 'd': //car goes backward
        analogWrite(SW1, 0);
        analogWrite(SW2, 127);
        analogWrite(SW3, 127);
        analogWrite(SW4, 0);
        control_message.speedPot = 0;
        break;
      case 'l': //car turn left
        analogWrite(SERVO, 63);
        control_message.turnPot = 63;
        break;
      case 'r': //car turn right
        analogWrite(SERVO, 190);
        control_message.turnPot = 190;
        break;
      case 'f': //car turn straight
        analogWrite(SERVO, 128);
        control_message.turnPot = 128;
        break;
      case 'h': //increase speed
        control_message.speedPot = control_message.speedPot + 10;
        if(control_message.speedPot > 255){
          control_message.speedPot = 255;
        }
        
        if((analogRead(SW1) == 0) && (analogRead(SW2) == 0)){
          analogWrite(SW1, 0);
          analogWrite(SW2, 0);
          analogWrite(SW3, 0);
          analogWrite(SW4, 0);
        }
        else if((analogRead(SW1) == 0) && (analogRead(SW2) != 0)){
          analogWrite(SW2, control_message.speedPot);
          analogWrite(SW3, control_message.speedPot);
        }
        else{
          analogWrite(SW1, control_message.speedPot);
          analogWrite(SW4, control_message.speedPot);
        }
        break;
      case 'j': //decrese speed
        control_message.speedPot = control_message.speedPot - 10;
        if(control_message.speedPot < 0){
          control_message.speedPot = 0;
        }

        if((analogRead(SW1) == 0) && (analogRead(SW2) == 0)){
          analogWrite(SW1, 0);
          analogWrite(SW2, 0);
          analogWrite(SW3, 0);
          analogWrite(SW4, 0);
        }
        else if((analogRead(SW1) == 0) && (analogRead(SW2) != 0)){
          analogWrite(SW2, control_message.speedPot);
          analogWrite(SW3, control_message.speedPot);
        }
        else{
          analogWrite(SW1, control_message.speedPot);
          analogWrite(SW4, control_message.speedPot);
        }
        break;
      case 'c': //increase left
        control_message.turnPot = control_message.turnPot - 10;
        if(control_message.turnPot < 0){
          control_message.turnPot = 0;
        }
        analogWrite(SERVO, control_message.turnPot);
        break;
      case 'v': //increase right
        control_message.turnPot = control_message.turnPot + 10;
        if(control_message.turnPot > 255){
          control_message.turnPot = 255;
        }
        analogWrite(SERVO, control_message.turnPot);
        break;
      default:
        Serial.println("Unrecognized character passed through audioResponse function. Nothing happens");
        break;
    }
  }
  

}

char* readSPI(void){
  digitalWrite(CS, LOW); //Selects peripheral to communicate with, which is the Raspberry Pi

  byte byteNum = SPI.transfer(0x01); //Raspberry Pi sends a message that tells us how long the message will be. We send a confirmation message "0x01"

  int message_length =  (int)byteNum;
  char message[message_length];

  for(int i=0; i<message_length; i++){ //stores message recieved from RaspPi. This loop runs based on message length number that was previously recieved from RaspPi
    message[i] = SPI.transfer(0x01);
  }
  digitalWrite(CS, HIGH); //Deselects Raspberry Pi for SPI communication
  return message;
}

void OnDataRecv(const uint8_t *mac, const uint8_t *incomingData, int len){
  memcpy(&recieved_message, incomingData, sizeof(recieved_message)); //stores data from sender into a struct
   
}

void manual_control(void){
  //Pot goes from 0-4095 in analog value
  //Analog write goes from 0-255 in value
  control_message.speedPot = recieved_message.speedPot/8; //translating the pot value to be used for analogWrite (0-255)
  control_message.turnPot = recieved_message.turnPot/16; //translating the pot value to be used for analogWrite (0-255)

  if((control_message.speedPot) == 255){ //doesn't drive
    analogWrite(SW1, 0);
    analogWrite(SW2, 0);
    analogWrite(SW3, 0);
    analogWrite(SW4, 0);
  }
  else if((control_message.speedPot) > 255){ //drives forward
    analogWrite(SW1, control_message.speedPot-256); 
    analogWrite(SW2, 0);
    analogWrite(SW3, 0);
    analogWrite(SW4, control_message.speedPot-256);
  }
  else{ //drives backward
    analogWrite(SW1, 0);
    analogWrite(SW2, 255-control_message.speedPot); 
    analogWrite(SW3, 255-control_message.speedPot);
    analogWrite(SW4, 0);
  }

  analogWrite(SERVO, control_message.turnPot); //turning

}
