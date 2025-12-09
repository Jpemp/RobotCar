#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>

#include <iostream>

//Button Control GPIO pins
const int DRIVE = 26;
const int STEER = 25;

//Switch between voice control and button control
const int CSWITCH = 14;

typedef struct message {
  int speedPot;
  int turnPot;
  bool controlSwitch;
} message;

message control_message;

bool micFlag = false;
bool sendFlag = false;
uint8_t recieverAddress[]; //macAddress for the Car ESP32 (reciever)
esp_now_peer_info_t peerInfo; //information about the car (reciever) ESP32

// put function declarations here:
void movement(void);
void OnDataSent(const uint8_t*, esp_now_send_status_t);



void setup() { //runs once
  //GPIO pin setup
  pinMode(DRIVE, INPUT); 
  pinMode(STEER, INPUT);
  pinMode(CSWITCH, INPUT);
  
  

  Serial.begin(115200);
  WiFi.mode(WIFI_STA); //setting the ESP32 as a Wi-Fi station

  /*uint8_t mac[6]; //only need this part to get the MAC address of the ESP32s
  esp_err_t ret = esp_wifi_get_mac(WIFI_IF_STA, mac); 
  Serial.printf("%02x:%02x:%02x:%02x:%02x:%02x\n", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]); */
  //ESP32 Feather MAC--40:22:d8:7b:83:64
  
  if(esp_now_init() != ESP_OK){ //initializing ESP-NOW communication
    Serial.println("Error initializing ESP-NOW");
    return;
  }
  else{
    Serial.println("ESP-NOW successfully initialized!");
  }

  esp_now_register_send_cb(esp_now_send_cb_t(OnDataSent)); //a callback function to indicate when data is sent to Car ESP32

  //setting up peer/reciever info
  memcpy(peerInfo.peer_addr, recieverAddress, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  if(esp_now_add_peer(&peerInfo) != ESP_OK){ //connects to reciever peer
    Serial.println("Failed to pair with peer");
    return;
  }
  else{
    Serial.println("Paired with peer!");
  }

}

void loop() {
  // put your main code here, to run repeatedly:


  //Data from potentiometers and switch

  control_message.speedPot = analogRead(DRIVE); //value from the drive potentiometer is stored here
  control_message.turnPot = analogRead (STEER); //value from the steering potentiometer is stored here
  control_message.controlSwitch = digitalRead(CSWITCH); //tells if the car esp32 should listen to the remote or audio processor
  
  esp_err_t result = esp_now_send(recieverAddress, (uint8_t*) &control_message, sizeof(message)); //data sent to reciever
     
  if(result != ESP_OK){
    Serial.println("Error sending data");
  }
  else{
    Serial.println("Data successfully sent!");
  }

  delay(1000);
}

// put function definitions here:
void OnDataSent(const uint8_t *mac, esp_now_send_status_t status){
  Serial.print("Packet Sent Status: ");
  if(status == ESP_NOW_SEND_SUCCESS){
    Serial.println("Success");
  }
  else{
    Serial.println("Failure");
  }
}


