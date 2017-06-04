void setup() {
  // put your setup code here, to run once:
  pinMode(1, INPUT);
  pinMode(2, INPUT);
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  Serial.print(digitalRead(2));
  Serial.print(" ");
  Serial.println(digitalRead(1));
}
