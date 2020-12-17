"""Important Notes
1. One pixel is Five centimeters

"""
import sys
import pygame
from sim_utils.model import Car
from sim_utils.define import *

# game settings
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

# create new car object and load
car = Car()
car.load_car()
car.states.set_velocity(3)


def update_screen():
    screen.fill([192, 153, 144])
    screen.blit(car.get_surface(), car.get_rect())
    pygame.display.update()


# restart simulation
pygame.time.set_timer(pygame.USEREVENT, 8000)

# simulatiob loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        # elif event.type == pygame.KEYDOWN:
        #     if event.key == pygame.K_UP:
        #         car_rect.centery -= 10
        #     if event.key == pygame.K_DOWN:
        #         car_rect.centery += 10

        elif event.type == pygame.USEREVENT:
            print('Simulation Reset')
            car.set_car_to_origin()


    update_screen()
    clock.tick(FRAME_RATE)
