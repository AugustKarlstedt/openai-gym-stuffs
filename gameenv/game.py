# Reinforcement learning agents playing the game of "it"
# where whomever is "it" has to chase the other
# and once they tag them (by colliding with them)
# the other player becomes it!

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import numpy as np
import pygame
import pygame.gfxdraw
import random
import sys

pygame.init()

size = width, height = (800, 600)
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()

class Player():

    def __init__(self, screen, position, velocity=pygame.Vector2(0.5, 0.5), radius=32, thickness=8, id=0):
        self.screen = screen
        self.position = position
        self.velocity = velocity
        self.rect = pygame.draw.circle(screen, pygame.Color('limegreen'), (int(position.x), int(position.y)), radius, thickness)
        self.forward = pygame.draw.line(screen, pygame.Color('white'), (int(position.x), int(position.y)), (int(position.x+velocity.x*64), int(position.y+velocity.y*64)), 4)
        self.is_it = False

    def update(self, delta):
        seconds = pygame.time.get_ticks() / 1000

        self.velocity.x = np.math.cos(seconds) * 0.25
        self.velocity.y = np.math.sin(seconds) * 0.25
        self.velocity *= delta

        self.position += self.velocity

        if self.position.x < 0:
            self.position.x = width
        elif self.position.x > width:
            self.position.x = 0

        if self.position.y < 0:
            self.position.y = height
        elif self.position.y > height:
            self.position.y = 0

    def draw(self):
        self.rect = pygame.gfxdraw.filled_circle(self.screen, int(self.position.x), int(self.position.y), 32, pygame.Color('limegreen'))
        self.forward = pygame.gfxdraw.line(screen, int(self.position.x), int(self.position.y), int(self.position.x+self.velocity.x*64), int(self.position.y+self.velocity.y*64), pygame.Color('white'))

    def get_reward(self, other):
        if self.is_it:
            if self.rect.colliderect(other):
                self.is_it = False
                other.is_it = True
                return 10 # we tagged the other player, big reward!
            return -1 # we're it we aren't tagging the other player, boooooooo
        else:        
            return 1 # we're not it, but we're not being tagged. great!

player_1 = Player(screen, pygame.Vector2(0.0, 0.0))
player_2 = Player(screen, pygame.Vector2(400, 400))


while True:
    delta = clock.tick()
    #print(f"{clock.get_fps():.2f}")

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    player_1.update(delta)
    player_2.update(delta)

    screen.fill(pygame.Color('black'))
    player_1.draw()
    player_2.draw()
    pygame.display.flip()