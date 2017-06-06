int p1 = 3;
int p2 = 5;
int p3 = 6;
int p4 = 9;
int p5 = 10;
int p6 = 11;
int p7 = 2;
int p8 = 4;
int p9 = 7;
int p10 = 8;
int p11 = 12;
int p12 = 13;

void setup() {
  // put your setup code here, to run once:
  pinMode(p1, OUTPUT);
  pinMode(p2, OUTPUT);
  pinMode(p3, OUTPUT);
  pinMode(p4, OUTPUT);
  pinMode(p5, OUTPUT);
  pinMode(p6, OUTPUT);
  pinMode(p7, OUTPUT);
  pinMode(p8, OUTPUT);
  pinMode(p9, OUTPUT);
  pinMode(p10, OUTPUT);
  pinMode(p11, OUTPUT);
  pinMode(p12, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  analogWrite(p1, 200);
  analogWrite(p2, 0);
  analogWrite(p3, 255);
  analogWrite(p4, 0);
  analogWrite(p5, 0);
  analogWrite(p6, 0);
  digitalWrite(p7, LOW);
  digitalWrite(p8, LOW);
  digitalWrite(p9, LOW);
  digitalWrite(p10, LOW);
  digitalWrite(p11, LOW);
  digitalWrite(p12, LOW);
  delay(2000);
  analogWrite(p1, 0);
  analogWrite(p2, 200);
  analogWrite(p3, 0);
  analogWrite(p4, 0);
  analogWrite(p5, 0);
  analogWrite(p6, 0);
  digitalWrite(p7, LOW);
  digitalWrite(p8, LOW);
  digitalWrite(p9, LOW);
  digitalWrite(p10, LOW);
  digitalWrite(p11, LOW);
  digitalWrite(p12, LOW);
  delay(2000);  
}
