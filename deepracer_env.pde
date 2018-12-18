import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.ClassNotFoundException;
import java.net.ServerSocket;
import java.net.Socket;

import processing.net.*;

int port = 10002;
boolean myServerRunning = true;
Server s;
//SocketServer s;
Client c;

PImage track;
Car car;
boolean isUp, isDown, isLeft, isRight;
boolean recording=false;
int w = 20000;
int h = 10000;
PGraphics pg;

void setup(){
  //frameRate(30);
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
  color green = color(49, 169, 141);
  background(green);

  translate(width/2, height);  
  rotateX(car.viewAngle);

  if (keyPressed) moveCar();
  car.update();
  
  rotateZ(car.direction);
  
  image(track, car.y-w/2, car.x-h/2, w, h);
  rotateZ(-car.direction);
  translate(-width/2, -height, 1);

  if (recording) {
    saveFrame("frames/###.png");
    fill(255,0,0);
  } else {
    fill(0,255,0);
  }
  if (frameCount%100==0) {
    println(frameRate);
  }
  ellipse(width/2, height, 30, 30);
}
