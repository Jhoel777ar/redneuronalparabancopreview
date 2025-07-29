import pygame

def play_alert():
    try:
        pygame.mixer.init()
        sound = pygame.mixer.Sound("sounds/alert.mp3")
        sound.play()
    except Exception as e:
        print(f"Error al reproducir el sonido: {e}")