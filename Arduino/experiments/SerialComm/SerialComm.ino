const char DEV_BEGIN = 'B';
// Steering wheel and speed feedback.
const char DEV_STATUS = 'S';
// Debug. Send message to computer to display.
const char DEV_DEBUG = 'D';

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
}

void loop() {
  // This code does not work consistently.
//  char buf;
//  String pos = "";
//  while(buf != ';') {
//    buf = Serial.read();
//    pos += buf;
//  }

//  buf = Serial.read();
//  pos += buf;
//  if (buf == ';') {
//    pos += "z";
//  }

  // Multiple characters
  // Very slow
//  while (Serial.available() == 0);
//  char cmd = Serial.read();
//  Serial.print(cmd);
//  if (cmd == 's') {
//    while (Serial.available() < 5);
//    char pos[4];
//    byte num = Serial.readBytesUntil(';', pos, 5);
//    Serial.print(pos);
//    Serial.print('\n');
//  }

  // Try with single characters.
  while (Serial.available() == 0);
  char cmd = Serial.read();
  Serial.print(cmd);
  if (cmd == 's') {
    for (int i=0; i<5; i++) {
      while (Serial.available() == 0);
      char pos = Serial.read();
      Serial.print(pos);
    }
    Serial.print('\n');
  }
//  Serial.print(Serial.available());
}
