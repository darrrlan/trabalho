import pygame
from pygame.locals import *  # noqa
import sys
import random
from MLP import MLP
import numpy as np


class FlappyBird:
    
    def __init__(self):
       
        self.bird = pygame.Rect(65, 50, 50, 50)
        self.distance = 0
        self.gap = 145
        self.wallx = 400
        self.birdY = 350
        self.jump = 0
        self.jumpSpeed = 15
        self.gravity = 10
        self.dead = False
        self.counter = 0
        self.offset = random.randint(-200, 200)
        self.mlp = MLP(5,8,1)
        
    def calculateInput(self):
        dist_X_to_The_Wall = self.wallx+80
        dist_Y_to_The_Wall_UP = self.birdY-(0 - self.gap - self.offset+500)
        dist_Y_to_The_Wall_DOWN = self.birdY-(360 + self.gap - self.offset)
        dist_Y_TOP = self.birdY
        dist_Y_BOTTOM = 720-self.birdY
        res = [dist_X_to_The_Wall,dist_Y_to_The_Wall_UP,dist_Y_to_The_Wall_DOWN,dist_Y_TOP,dist_Y_BOTTOM]
        return res
    
    def centerWalls(self):
        return 0 - self.gap - self.offset+572.5
    
    def downWall(self):
        return 360 + self.gap - self.offset
    
    def posBird(self):
        return self.birdY
    
    def isDead(self):
        return self.dead
    
    def TotalDistance(self):
        return self.distance
    
    def Totalcounter(self):
        return self.distance


    
    def updateWalls(self):
        self.wallx -= 4
        if self.wallx < -80:
            self.wallx = 400
            self.counter += 1
            self.offset = random.randint(-200, 200)

    def birdUpdate(self):
        self.distance =  self.distance + 1 
        if self.jump:
            self.jumpSpeed -= 1
            self.birdY -= self.jumpSpeed
            self.jump -= 1
        else:
            self.birdY += self.gravity
            self.gravity += 0.2
        self.bird[1] = self.birdY
        upRect = pygame.Rect(self.wallx,
                             360 + self.gap - self.offset + 10,
                             88,
                             500)
        downRect = pygame.Rect(self.wallx,
                               0 - self.gap - self.offset - 10,
                               88,
                               500)
        if upRect.colliderect(self.bird):
            self.dead = True
        if downRect.colliderect(self.bird):
            self.dead = True
        if not 0 < self.bird[1] < 720:
            self.dead=True
            
        

    def tick(self):
        
        mlp_input = self.calculateInput()

        mlp_input = [(x - 0) / (400 - 0) for x in mlp_input]

        prediction = self.mlp.feedForward(mlp_input)
        
        print(prediction)
                
        if (prediction > 0.5) and not self.dead:
            self.jump = 17
            self.gravity = 10
            self.jumpSpeed = 15
            
        self.updateWalls()
        self.birdUpdate()
        
class FlappyBird_GAME:
    
    def __init__(self):
        self.screen = pygame.display.set_mode((400, 700))
        self.bird = pygame.Rect(65, 50, 50, 50)
        self.background = pygame.image.load("assets/background.png").convert()
        self.birdSprites = [pygame.image.load("assets/1.png").convert_alpha(),
                            pygame.image.load("assets/2.png").convert_alpha(),
                            pygame.image.load("assets/dead.png")]
        self.wallUp = pygame.image.load("assets/bottom.png").convert_alpha()
        self.wallDown = pygame.image.load("assets/top.png").convert_alpha()
        self.distance = 0
        self.gap = 145
        self.wallx = 400
        self.birdY = 350
        self.jump = 0
        self.jumpSpeed = 15
        self.gravity = 10
        self.dead = False
        self.counter = 0
        self.offset = random.randint(-200, 200)
        self.sprite = 0
        self.mlp = MLP(5,8,1)
        self.clock = pygame.time.Clock()
        
    def calculateInput(self):
        dist_X_to_The_Wall = self.wallx+80
        dist_Y_to_The_Wall_UP = self.birdY-(0 - self.gap - self.offset+500)
        dist_Y_to_The_Wall_DOWN = self.birdY-(360 + self.gap - self.offset)
        dist_Y_TOP = self.birdY
        dist_Y_BOTTOM = 720-self.birdY
        res = [dist_X_to_The_Wall,dist_Y_to_The_Wall_UP,dist_Y_to_The_Wall_DOWN,dist_Y_TOP,dist_Y_BOTTOM]
        return res
    
    def isDead(self):
        return self.dead
    
    def TotalDistance(self):
        return self.distance

    def centerWalls(self):
        return 0 - self.gap - self.offset+572.5
    
    def downWall(self):
        return 360 + self.gap - self.offset
    
    def posBird(self):
        return self.birdY
    
    def updateWalls(self):
        self.wallx -= 4
        if self.wallx < -80:
            self.wallx = 400
            self.counter += 1
            self.offset = random.randint(-200, 200)

    def birdUpdate(self):
        self.distance =  self.distance + 1 
        if self.jump:
            self.jumpSpeed -= 1
            self.birdY -= self.jumpSpeed
            self.jump -= 1
        else:
            self.birdY += self.gravity
            self.gravity += 0.2
        self.bird[1] = self.birdY
        upRect = pygame.Rect(self.wallx,
                             360 + self.gap - self.offset + 10,
                             88,
                             500)
        downRect = pygame.Rect(self.wallx,
                               0 - self.gap - self.offset - 10,
                               88,
                               500)
        if upRect.colliderect(self.bird):
            self.dead = True
        if downRect.colliderect(self.bird):
            self.dead = True
        if not 0 < self.bird[1] < 720:
            self.dead=True
            
        

    def tick(self):
        self.clock.tick(60)
        
        mlp_input = self.calculateInput()

        mean = np.mean(mlp_input)
        std_dev = np.std(mlp_input)
        mlp_input_zscore = [(x - mean) / std_dev for x in mlp_input]
        prediction = self.mlp.feedForward(mlp_input)
                
        if (prediction > 0.5) and not self.dead:
            self.jump = 17
            self.gravity = 10
            self.jumpSpeed = 15

        pygame.font.init()  # Inicialize o módulo de fonte do pygame
        font = pygame.font.Font(None, 36)  # Escolha a fonte e o tamanho desejados
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.wallUp,
                             (self.wallx, 360 + self.gap - self.offset))
    
        self.screen.blit(self.wallDown,
                             (self.wallx, 0 - self.gap - self.offset))
        self.screen.blit(font.render(str(self.counter),-1,(255, 255, 255)),(200, 50))
        if self.dead:
            self.sprite = 2
        elif self.jump:
            self.sprite = 1
        self.screen.blit(self.birdSprites[self.sprite], (70, self.birdY))
        if not self.dead:
            self.sprite = 0
        self.updateWalls()
        self.birdUpdate()
        pygame.display.update()
              
        