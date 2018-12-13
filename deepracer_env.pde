PImage track;
int x,y;
Car car;
boolean isUp, isDown, isLeft, isRight;
boolean recording=false;
int w = 20000;
int h = 10000;

void setup() {
  frameRate(180);
  x = 0;
  y = 0;
  size(1000, 600, P3D);
  track = loadImage("aws-track.png");
  car = new Car(0, 0, PI/3);
}

void keyPressed() {
  if (key == 'r' || key == 'R') {
    recording = !recording;
  } else {
    setMove(keyCode, true);
  }
}
 
void keyReleased() {
  setMove(keyCode, false);
}
 
boolean setMove(int k, boolean b) {
  switch (k) {
  case UP:
    return isUp = b;
  case DOWN:
    return isDown = b;
  case LEFT:
    return isLeft = b;
  case RIGHT:
    return isRight = b;
  default:
    return b;
  }
}

void moveCar() {
  if (isLeft) {
    car.turn(7);
  }
  if (isRight) {
    car.turn(-7);
  }
  if (isUp) {
    car.move(7);
  }
  if (isDown) {
    car.move(-5);
  }
}

void draw() {
  //color green = color(34, 177, 76);
  color green = color(49, 169, 141);
  background(green);

  translate(width/2, height);  
  rotateX(car.viewAngle);

  if (keyPressed) moveCar();
  car.update();
  
  rotateZ(car.direction);
  image(track, car.y-w/2, car.x-h/2, w, h);
  
  translate(-width/2, -height);
  
  //if (recording) {
  //  saveFrame("frames/###.png");
  //  fill(255,0,0);
  //} else {
  //  fill(0,255,0);
  //}
  if (frameCount%100==0) {
    println(frameRate);
  }
  //ellipse(width/2, height, 30, 30);
}
