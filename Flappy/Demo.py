import pygame
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from MLP import MLP

class FlappyBird_Human:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 700))
        pygame.display.set_caption("Flappy Bird")
        self.bird = pygame.Rect(65, 50, 50, 50)
        self.background = pygame.image.load("assets/background.png").convert()
        self.birdSprites = [pygame.image.load("assets/1.png").convert_alpha(),
                            pygame.image.load("assets/2.png").convert_alpha(),
                            pygame.image.load("assets/dead.png")]
        self.wallUp = pygame.image.load("assets/bottom.png").convert_alpha()
        self.wallDown = pygame.image.load("assets/top.png").convert_alpha()
        self.gap = 145
        self.wallx = 400
        self.birdY = 350
        self.jump = 0
        self.jumpSpeed = 15
        self.gravity = 10
        self.dead = False
        self.sprite = 0
        self.counter = 0
        self.offset = random.randint(-200, 200)

        self.mlp = MLP(5,8,1)
        self.generation_counter = 0
        self.generation_history = []
        self.pipes_passed_history = []

    def shouldJump(self):
        hole_center = 390 + self.gap / 2 - self.offset
        proximity_threshold = 10
        distance_to_hole_center = abs(self.birdY - hole_center)

        if distance_to_hole_center < proximity_threshold or self.birdY > hole_center:
            return 1
        else:
            return 0

    def updateWalls(self):
        self.wallx -= 4
        if self.wallx < -80:
            self.wallx = 400
            self.counter += 1
            self.offset = random.randint(-200, 200)

    def birdUpdate(self):
        if self.jump:
            self.jumpSpeed -= 1
            self.birdY -= self.jumpSpeed
            self.jump -= 1
        else:
            self.birdY += self.gravity
            self.gravity += 0.2
        self.bird[1] = int(self.birdY)
        upRect = pygame.Rect(self.wallx,
                             360 + self.gap - self.offset + 10,
                             self.wallUp.get_width() - 10,
                             self.wallUp.get_height())
        downRect = pygame.Rect(self.wallx,
                               0 - self.gap - self.offset - 10,
                               self.wallDown.get_width() - 10,
                               self.wallDown.get_height())
        if upRect.colliderect(self.bird):
            self.dead = True
            self.checkJumpValidity()
            self.pipes_passed_history.append(self.counter)
            self.resetGame()
        if downRect.colliderect(self.bird):
            self.dead = True
            self.checkJumpValidity()
            self.pipes_passed_history.append(self.counter)
            self.resetGame()
        if not 0 < self.bird[1] < 720:
            self.pipes_passed_history.append(self.counter)
            self.resetGame()

    def checkJumpValidity(self):
        if self.jump and self.dead:
            print("O pássaro morreu devido a um pulo inválido!")

    def resetGame(self):
        self.birdY = 350
        self.jump = 0
        self.gravity = 10
        self.dead = False
        self.counter = 0
        self.wallx = 400
        self.offset = random.randint(-110, 110)
        self.generation_counter += 1
        self.generation_history.append(self.generation_counter)

    def run(self):
        clock = pygame.time.Clock()
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 50)
        while True:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    with open("generation_pipes_data.txt", "w") as data_file:
                        for generation, pipes_passed in zip(self.generation_history, self.pipes_passed_history):
                            data_file.write(f"{generation}\t{pipes_passed}\n")

                    plt.plot(self.generation_history, self.pipes_passed_history)
                    plt.xlabel("Geração")
                    plt.ylabel("Número de Canos Passados")
                    plt.title("Geração vs. Número de Canos Passados")
                    plt.grid(True)
                    plt.savefig("generation_pipes_plot.png")
                    plt.show()

                    pygame.quit()
                    sys.exit()
                if (event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN) and not self.dead:
                    self.jump = 17
                    self.gravity = 10
                    self.jumpSpeed = 15

            self.screen.fill((255, 255, 255))
            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.wallUp,
                            (self.wallx, 360 + self.gap - self.offset))
            self.screen.blit(self.wallDown,
                            (self.wallx, 0 - self.gap - self.offset - 10))

            generation_text = font.render(f"Geração: {self.generation_counter}", -1, (255, 255, 255))
            pipes_passed_text = font.render(f"Scorre: {self.counter}", -1, (255, 255, 255))
            self.screen.blit(generation_text, (10, 10))
            self.screen.blit(pipes_passed_text, (10, 60))

            mlp_input = [
                390 + self.gap / 2 - self.offset,
                abs(self.birdY - 390 + self.gap / 2 - self.offset),
                abs(self.birdY - 390 + self.gap / 2 - self.offset - self.gap),
                self.birdY,
                700 - self.birdY
            ]

            mlp_input = [(x - 0) / (400 - 0) for x in mlp_input]

            prediction = self.mlp.backpropagation(mlp_input, self.shouldJump())
            print(prediction)

            if prediction > 0.5 and not self.dead:
                self.jump = 17
                self.gravity = 10
                self.jumpSpeed = 15
            if self.dead:
                self.sprite = 2
            elif self.jump:
                self.sprite = 1
            self.screen.blit(self.birdSprites[self.sprite], (70, int(self.birdY)))

            if not self.dead:
                self.sprite = 0
            self.updateWalls()
            self.birdUpdate()
            pygame.display.update()
            
    def run_game_with_mlp(self, mlp):
        clock = pygame.time.Clock()
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 50)
        
        while True:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if (event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN) and not self.dead:
                    self.jump = 17
                    self.gravity = 10
                    self.jumpSpeed = 15

            self.screen.fill((255, 255, 255))
            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.wallUp,
                            (self.wallx, 360 + self.gap - self.offset))
            self.screen.blit(self.wallDown,
                            (self.wallx, 0 - self.gap - self.offset - 10))

            generation_text = font.render(f"Geração: {self.generation_counter}", -1, (255, 255, 255))
            pipes_passed_text = font.render(f"Scorre: {self.counter}", -1, (255, 255, 255))
            self.screen.blit(generation_text, (10, 10))
            self.screen.blit(pipes_passed_text, (10, 60))

            mlp_input = [
                390 + self.gap / 2 - self.offset,
                abs(self.birdY - 390 + self.gap / 2 - self.offset),
                abs(self.birdY - 390 + self.gap / 2 - self.offset - self.gap),
                self.birdY,
                700 - self.birdY
            ]

            mlp_input = [(x - 0) / (400 - 0) for x in mlp_input]

            prediction = mlp.feedForward(mlp_input)
            print(prediction)

            if prediction > 0.5 and not self.dead:
                self.jump = 17
                self.gravity = 10
                self.jumpSpeed = 15
            if self.dead:
                self.sprite = 2
            elif self.jump:
                self.sprite = 1
            self.screen.blit(self.birdSprites[self.sprite], (70, int(self.birdY)))

            if not self.dead:
                self.sprite = 0
            self.updateWalls()
            self.birdUpdate()
            pygame.display.update()

if __name__ == "__main__":
    flappy_bird = FlappyBird_Human().run()
   