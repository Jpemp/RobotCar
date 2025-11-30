#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>

#include <iostream>

//Button Control GPIO pins
int UP = 13;
int DOWN = 12;
int LEFT = 27;
int RIGHT = 33;

//Switch between voice control and button control
int CSWITCH = 21;

//Microphone for voice control
int MIC = 25; //A1

//Speed Changer for the Car? Extra feature if there is time
int SPEEDPWM = 34; //A2

uint8_t recieverAddress[]; //macAddress for the Car ESP32 (reciever)
esp_now_peer_info_t peerInfo; //information about the reciever ESP32

// put function declarations here:
int movement();
void OnDataSent(esp_now_send_status_t);

void setup() {
  // put your setup code here, to run once:
  pinMode(UP, INPUT); //establishing control pins
  pinMode(DOWN, INPUT); 
  pinMode(LEFT, INPUT);
  pinMode(RIGHT, INPUT); 
  pinMode(CSWITCH, INPUT);
  pinMode(MIC, INPUT);
  pinMode(SPEEDPWM, INPUT);
  
  

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

  esp_now_register_send_cb(OnDataSent); //a callback function to indicate when data is sent to Car ESP32

  //setting up peer/reciever info
  memcpy(peerInfo.peer_addr, recieverAddress, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  if(esp_now_add_peer(&peerInfo) != ESP_OK){
    Serial.println("Failed to pair with peer");
    return;
  }
  else{
    Serial.println("Peer has been paired!");
  }


}

void loop() {
  // put your main code here, to run repeatedly:
  if(digitalRead(CSWITCH) == LOW){
    int speedVal;
    speedVal = analogRead(SpeedPWM);
  }
  else{
    
  }
  

  
  
  
  char* data = "Hello World!";
  esp_err_t result = esp_now_send(recieverAddress, (uint8_t*) &data, sizeof(data));

  if(result != ESP_OK){
    Serial.println("Error sending data");
  }
  else{
    Serial.println("Data successfully sent!");
  }
  delay(5000);
}

// put function definitions here:
void OnDataSent(esp_now_send_status_t status){
  Serial.print("Packet Sent Status: ");
  if(status == ESP_NOW_SEND_SUCCESS){
    Serial.println("Success");
  }
  else{
    Serial.println("Failure");
  }
}
