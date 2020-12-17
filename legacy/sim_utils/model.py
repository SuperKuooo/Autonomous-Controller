import pygame
import math
from sim_utils.define import *


class Car:
    """
    Translates actual units to pixels and correctly displays it on the screen
    """

    def __init__(self):
        # load car
        self.surface = None
        self.rect = None
        self.states = States()

    def load_car(self):
        surface = pygame.image.load('./assets/toro.png')
        self.surface = pygame.transform.scale(surface, CAR_SIZE)
        self.rect = self.surface.get_rect(center=ORIGIN)

    def set_car_to_origin(self):
        self.states.set_coordinates(ORIGIN)
        self.rect.centerx, self.rect.centery = self.states.get_coordinates()

    def get_surface(self):
        return self.surface

    def get_rect(self):
        return self.rect


class States:
    """
    States are in real world units. Center of the screen is (0,0) with some LEGIT coordinates lol
    """

    def __init__(self):
        self.velocity = 0
        self.heading_angle = 0
        self.coordinates = (0, 0)

    def set_velocity(self, velocity):
        """sets the velocity of the car

        :param velocity:
        :return: None
        """
        self.velocity = velocity

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates

    def get_coordinates(self):
        return self.coordinates

    def update(self):
        x = self.coordinates[0]
        y = self.coordinates[1]

        x += self.velocity * T_DELTA


