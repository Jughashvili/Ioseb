import pygame
import numpy as np
import random
import sys
import math

# --- 시뮬레이션 설정 ---
WIDTH, HEIGHT = 1000, 750
FPS = 30
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
YELLOW = (255, 255, 0)
LIGHT_GRAY = (200, 200, 200)

# 크리처 기본 설정
INITIAL_CRITTERS = 20
MAX_CRITTERS = 200 # 과도한 개체수 증가 방지
CRITTER_MIN_SIZE = 5
CRITTER_MAX_SIZE = 15
CRITTER_MIN_SPEED = 0.5
CRITTER_MAX_SPEED = 3.0
CRITTER_SENSE_RADIUS_FACTOR = 5 # 크기 대비 감각 범위 비율
INITIAL_ENERGY = 100
ENERGY_DECAY_RATE = 0.1 # 매 프레임 소모 에너지
ENERGY_FROM_FOOD = 50
REPRODUCTION_ENERGY_THRESHOLD = 150 # 번식에 필요한 최소 에너지
REPRODUCTION_COST = 70 # 번식 시 소모 에너지
MUTATION_RATE = 0.1 # 돌연변이 확률 (0.0 ~ 1.0)
MUTATION_STRENGTH = 0.2 # 돌연변이 강도 (변화량의 최대 비율)

# 먹이 설정
FOOD_COUNT = 30
FOOD_RADIUS = 4
FOOD_SPAWN_INTERVAL = 2 * FPS # 초당 프레임 기준 (예: 2초마다)

# --- 유틸리티 함수 ---
def random_color():
    return (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))

def limit_value(value, min_val, max_val):
    return max(min_val, min(value, max_val))

# --- 크리처 클래스 ---
class Critter:
    def __init__(self, x, y, dna=None):
        self.x = x
        self.y = y
        self.energy = INITIAL_ENERGY
        self.age = 0

        if dna is None:
            self.dna = {
                "size": random.uniform(CRITTER_MIN_SIZE * 0.8, CRITTER_MAX_SIZE * 1.2), # 초기 다양성 조금 더
                "speed": random.uniform(CRITTER_MIN_SPEED * 0.8, CRITTER_MAX_SPEED * 1.2),
                "color_r": random.randint(0, 255),
                "color_g": random.randint(0, 255),
                "color_b": random.randint(0, 255),
                "sense_radius_factor": random.uniform(3, 7) # 감각 범위 계수
            }
        else:
            self.dna = dna.copy() # 부모 DNA 복사

        # 표현형 적용 (DNA 값 제한)
        self.size = limit_value(self.dna["size"], CRITTER_MIN_SIZE, CRITTER_MAX_SIZE)
        self.speed = limit_value(self.dna["speed"], CRITTER_MIN_SPEED, CRITTER_MAX_SPEED)
        self.color = (
            limit_value(int(self.dna["color_r"]), 0, 255),
            limit_value(int(self.dna["color_g"]), 0, 255),
            limit_value(int(self.dna["color_b"]), 0, 255)
        )
        self.sense_radius = self.size * limit_value(self.dna["sense_radius_factor"], 2, 10)

        # 이동 관련
        self.angle = random.uniform(0, 2 * math.pi) # 이동 방향 (라디안)
        self.target_food = None

    def mutate(self):
        mutated_dna = self.dna.copy()
        for key, value in mutated_dna.items():
            if random.random() < MUTATION_RATE:
                if isinstance(value, (int, float)):
                    change = value * MUTATION_STRENGTH * random.uniform(-1, 1)
                    mutated_dna[key] = value + change
                # 색상 같은 경우 다르게 처리할 수도 있음 (예: +- 정수값)
                elif key.startswith("color_"):
                     mutated_dna[key] = limit_value(value + random.randint(-30, 30), 0, 255)
        return mutated_dna

    def reproduce(self):
        if self.energy >= REPRODUCTION_ENERGY_THRESHOLD:
            self.energy -= REPRODUCTION_COST
            child_dna = self.mutate() # 돌연변이된 DNA를 자손에게
            # 자식은 부모 근처에 생성
            offset_x = random.uniform(-self.size*2, self.size*2)
            offset_y = random.uniform(-self.size*2, self.size*2)
            child_x = limit_value(self.x + offset_x, 0, WIDTH)
            child_y = limit_value(self.y + offset_y, 0, HEIGHT)
            return Critter(child_x, child_y, child_dna)
        return None

    def find_closest_food(self, foods):
        closest_food = None
        min_dist = float('inf')
        for food in foods:
            dist = math.hypot(self.x - food.x, self.y - food.y)
            if dist < self.sense_radius and dist < min_dist:
                min_dist = dist
                closest_food = food
        return closest_food

    def update(self, foods):
        self.age += 1
        self.energy -= ENERGY_DECAY_RATE
        self.energy -= self.size * 0.005 # 크기가 클수록 에너지 소모 증가
        self.energy -= self.speed * 0.01  # 속도가 빠를수록 에너지 소모 증가

        if self.energy <= 0:
            return False # 죽음

        # 먹이 찾기 및 이동
        self.target_food = self.find_closest_food(foods)

        if self.target_food:
            # 먹이 방향으로 이동
            dx = self.target_food.x - self.x
            dy = self.target_food.y - self.y
            self.angle = math.atan2(dy, dx)
        else:
            # 무작위 방황 (방향 약간씩 변경)
            self.angle += random.uniform(-0.2, 0.2)

        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        # 화면 경계 처리 (튕기기)
        if self.x < self.size or self.x > WIDTH - self.size:
            self.angle = math.pi - self.angle # X축 반사
            self.x = limit_value(self.x, self.size, WIDTH - self.size)
        if self.y < self.size or self.y > HEIGHT - self.size:
            self.angle = -self.angle # Y축 반사
            self.y = limit_value(self.y, self.size, HEIGHT - self.size)
            
        # 먹이 섭취
        eaten_food_indices = []
        for i, food in enumerate(foods):
            if math.hypot(self.x - food.x, self.y - food.y) < self.size + food.radius:
                self.energy += ENERGY_FROM_FOOD
                eaten_food_indices.append(i)
        
        # 먹힌 음식 제거 (역순으로 제거해야 인덱스 문제 없음)
        for i in sorted(eaten_food_indices, reverse=True):
            del foods[i]

        return True # 생존

    def draw(self, screen):
        # 몸통
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), int(self.size))
        # 테두리 (에너지 시각화 - 선택적)
        # energy_ratio = max(0, min(1, self.energy / REPRODUCTION_ENERGY_THRESHOLD))
        # border_color = (int(255 * (1-energy_ratio)), int(255 * energy_ratio), 0)
        # pygame.draw.circle(screen, border_color, (int(self.x), int(self.y)), int(self.size), 2)

        # 감각 범위 (디버그용)
        # pygame.draw.circle(screen, LIGHT_GRAY, (int(self.x), int(self.y)), int(self.sense_radius), 1)
        
        # 이동 방향 표시선 (선택적)
        # end_x = self.x + math.cos(self.angle) * self.size * 1.5
        # end_y = self.y + math.sin(self.angle) * self.size * 1.5
        # pygame.draw.line(screen, BLACK, (self.x, self.y), (end_x, end_y), 1)

# --- 먹이 클래스 ---
class Food:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = FOOD_RADIUS

    def draw(self, screen):
        pygame.draw.circle(screen, GREEN, (self.x, self.y), self.radius)

# --- 메인 함수 ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("에코에보: 디지털 생명의 진화")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)
    small_font = pygame.font.SysFont(None, 20)

    critters = [Critter(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(INITIAL_CRITTERS)]
    foods = [Food(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(FOOD_COUNT)]

    running = True
    paused = False
    show_debug_info = False
    simulation_time_steps = 0
    food_spawn_timer = 0

    selected_critter = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_d:
                    show_debug_info = not show_debug_info
                if event.key == pygame.K_r: # 리셋
                    critters = [Critter(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(INITIAL_CRITTERS)]
                    foods = [Food(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(FOOD_COUNT)]
                    simulation_time_steps = 0
                    selected_critter = None
                if event.key == pygame.K_ESCAPE:
                    running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # 왼쪽 클릭
                    mouse_x, mouse_y = event.pos
                    selected_critter = None
                    for critter in critters:
                        if math.hypot(critter.x - mouse_x, critter.y - mouse_y) < critter.size:
                            selected_critter = critter
                            break


        if not paused:
            simulation_time_steps += 1
            food_spawn_timer += 1

            # 먹이 자동 생성
            if food_spawn_timer >= FOOD_SPAWN_INTERVAL:
                for _ in range(FOOD_COUNT - len(foods)): # 부족한 만큼만 생성
                    if len(foods) < FOOD_COUNT * 1.5 : # 최대 먹이 개수 제한
                         foods.append(Food(random.randint(0, WIDTH), random.randint(0, HEIGHT)))
                food_spawn_timer = 0

            # 크리처 업데이트
            new_critters = []
            survivors = []
            for critter in critters:
                if critter.update(foods): # 생존했으면
                    survivors.append(critter)
                    if len(critters) + len(new_critters) < MAX_CRITTERS: # 개체수 제한
                        child = critter.reproduce()
                        if child:
                            new_critters.append(child)
            
            critters = survivors + new_critters

        # 그리기
        screen.fill(BLACK)

        for food in foods:
            food.draw(screen)

        for critter in critters:
            critter.draw(screen)
            if show_debug_info: # 디버그 정보 (예: 에너지)
                energy_text = small_font.render(f"{critter.energy:.0f}", True, LIGHT_GRAY)
                screen.blit(energy_text, (critter.x + critter.size, critter.y - critter.size))

        # 정보 표시
        info_text_time = font.render(f"Time Steps: {simulation_time_steps}", True, WHITE)
        info_text_critters = font.render(f"Critters: {len(critters)}", True, WHITE)
        info_text_food = font.render(f"Food: {len(foods)}", True, WHITE)
        screen.blit(info_text_time, (10, 10))
        screen.blit(info_text_critters, (10, 40))
        screen.blit(info_text_food, (10, 70))

        if paused:
            pause_text = font.render("PAUSED", True, YELLOW)
            text_rect = pause_text.get_rect(center=(WIDTH // 2, 30))
            screen.blit(pause_text, text_rect)

        # 선택된 크리처 정보 표시
        if selected_critter:
            if selected_critter.energy <= 0: # 선택된 크리처가 죽으면 선택 해제
                selected_critter = None
            else:
                pygame.draw.circle(screen, YELLOW, (int(selected_critter.x), int(selected_critter.y)), int(selected_critter.size) + 3, 2) # 선택 표시
                info_y_offset = 100
                sel_title = font.render("Selected Critter:", True, YELLOW)
                screen.blit(sel_title, (10, info_y_offset))
                
                dna_info = [
                    f"Energy: {selected_critter.energy:.1f}",
                    f"Age: {selected_critter.age}",
                    f"Size: {selected_critter.dna['size']:.2f} (P: {selected_critter.size:.2f})",
                    f"Speed: {selected_critter.dna['speed']:.2f} (P: {selected_critter.speed:.2f})",
                    f"Sense Factor: {selected_critter.dna['sense_radius_factor']:.2f} (P: {selected_critter.sense_radius:.1f})",
                    f"Color DNA: ({int(selected_critter.dna['color_r'])},{int(selected_critter.dna['color_g'])},{int(selected_critter.dna['color_b'])})"
                ]
                for i, line in enumerate(dna_info):
                    dna_text = small_font.render(line, True, WHITE)
                    screen.blit(dna_text, (10, info_y_offset + 30 + i * 20))


        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()