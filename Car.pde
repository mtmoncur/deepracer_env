class Car {
  int x, y;
  int v = 0;
  int maxV = 90;
  int drag = 3;
  float viewAngle, turnAngle;
  float direction;
  
 Car(int ix, int iy, float iviewAngle) {
   x = ix;
   y = iy;
   v = 0;
   viewAngle = iviewAngle;
   turnAngle = 0;
 }
  
 void move(int throttle) {
   v += throttle;
   v = max(min(v, maxV), -maxV);
 }
   
void turn(int iturnAngle) {
   turnAngle = iturnAngle/180.0;
 }
 
 void update() {
   direction += turnAngle*v/maxV;
   x += cos(direction)*v;
   y += sin(direction)*v;
   v = min(v+drag, 0) + max(v-drag, 0);
   turnAngle = 0;
 }
}
