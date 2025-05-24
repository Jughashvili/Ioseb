import pygame
import numpy as np
import sys

# --- 시뮬레이션 설정 ---
WIDTH, HEIGHT = 1000, 750
FPS = 60
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# 로렌츠 어트랙터 파라미터 (기본값)
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0
DT = 0.01  # 시간 간격 (작을수록 정확하지만 느림)

# 시각화 설정
SCALE = 10 # 화면 크기에 맞게 궤적 크기 조절
OFFSET_X = WIDTH // 2
OFFSET_Y = HEIGHT // 2
MAX_POINTS = 1500 # 각 궤적이 저장할 최대 점 개수

# --- 로렌츠 방정식 ---
def lorenz(x, y, z, sigma=SIGMA, rho=RHO, beta=BETA):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz

# --- 3D 좌표를 2D 화면 좌표로 변환 (단순 투영 및 회전) ---
def project_3d_to_2d(x, y, z, angle_x, angle_y, angle_z):
    # Z축 회전
    x_rot_z = x * np.cos(angle_z) - y * np.sin(angle_z)
    y_rot_z = x * np.sin(angle_z) + y * np.cos(angle_z)
    z_rot_z = z
    x, y, z = x_rot_z, y_rot_z, z_rot_z

    # Y축 회전
    x_rot_y = x * np.cos(angle_y) + z * np.sin(angle_y)
    y_rot_y = y
    z_rot_y = -x * np.sin(angle_y) + z * np.cos(angle_y)
    x, y, z = x_rot_y, y_rot_y, z_rot_y

    # X축 회전
    x_rot_x = x
    y_rot_x = y * np.cos(angle_x) - z * np.sin(angle_x)
    z_rot_x = y * np.sin(angle_x) + z * np.cos(angle_x)
    x, y, z = x_rot_x, y_rot_x, z_rot_x
    
    # 간단한 원근 투영 (z값에 따라 크기 조절 - 선택 사항)
    # perspective = 600 / (600 + z_rot_x) # z가 카메라에 가까울수록 크게
    # screen_x = int(x_rot_x * perspective * SCALE + OFFSET_X)
    # screen_y = int(y_rot_x * perspective * SCALE + OFFSET_Y)
    
    # 직교 투영 (Orthographic projection)
    screen_x = int(x * SCALE + OFFSET_X)
    screen_y = int(y * SCALE + OFFSET_Y)
    return screen_x, screen_y


# --- 메인 함수 ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("혼돈 이론: 나비 효과 시뮬레이터 (로렌츠 어트랙터)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)

    # --- 초기 조건 ---
    # 궤적 1 (기준)
    x1, y1, z1 = 0.1, 0.0, 0.0
    points1 = []
    
    # 궤적 2 (미세 변화)
    delta = 0.00001 # 초기 조건의 미세한 차이
    x2, y2, z2 = x1 + delta, y1, z1 
    points2 = []

    # 시뮬레이션 제어 변수
    running = True
    paused = False
    show_help = True

    # 뷰 회전 각도
    angle_x, angle_y, angle_z = np.radians(30), np.radians(0), np.radians(0) # 초기 각도
    rotation_speed = np.radians(1) # 회전 속도

    sim_time = 0.0 # 시뮬레이션 시간

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r: # 초기화
                    x1, y1, z1 = 0.1, 0.0, 0.0
                    points1 = []
                    x2, y2, z2 = x1 + delta, y1, z1
                    points2 = []
                    sim_time = 0.0
                if event.key == pygame.K_h:
                    show_help = not show_help
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # 키 입력에 따른 회전 (누르고 있을 때)
        keys = pygame.key.get_pressed()
        if not paused: # 일시정지 중에는 회전 방지 (선택)
            if keys[pygame.K_LEFT]:
                angle_y -= rotation_speed
            if keys[pygame.K_RIGHT]:
                angle_y += rotation_speed
            if keys[pygame.K_UP]:
                angle_x -= rotation_speed
            if keys[pygame.K_DOWN]:
                angle_x += rotation_speed
            if keys[pygame.K_q]:
                angle_z -= rotation_speed
            if keys[pygame.K_e]:
                angle_z += rotation_speed


        if not paused:
            # 궤적 1 계산
            dx1, dy1, dz1 = lorenz(x1, y1, z1)
            x1 += dx1 * DT
            y1 += dy1 * DT
            z1 += dz1 * DT
            points1.append((x1, y1, z1))
            if len(points1) > MAX_POINTS:
                points1.pop(0)

            # 궤적 2 계산
            dx2, dy2, dz2 = lorenz(x2, y2, z2)
            x2 += dx2 * DT
            y2 += dy2 * DT
            z2 += dz2 * DT
            points2.append((x2, y2, z2))
            if len(points2) > MAX_POINTS:
                points2.pop(0)
            
            sim_time += DT

        # 그리기
        screen.fill(BLACK)

        # 궤적 1 그리기
        if len(points1) > 1:
            projected_points1 = [project_3d_to_2d(p[0], p[1], p[2], angle_x, angle_y, angle_z) for p in points1]
            pygame.draw.aalines(screen, RED, False, projected_points1) # Anti-aliased lines

        # 궤적 2 그리기
        if len(points2) > 1:
            projected_points2 = [project_3d_to_2d(p[0], p[1], p[2], angle_x, angle_y, angle_z) for p in points2]
            pygame.draw.aalines(screen, BLUE, False, projected_points2)

        # 두 궤적 간의 현재 거리 계산 및 표시
        if points1 and points2:
            p1_curr = np.array(points1[-1])
            p2_curr = np.array(points2[-1])
            distance = np.linalg.norm(p1_curr - p2_curr)
            dist_text = font.render(f"Distance: {distance:.4f}", True, YELLOW)
            screen.blit(dist_text, (10, 10))
        
        # 시뮬레이션 시간 표시
        time_text = font.render(f"Time: {sim_time:.2f}", True, WHITE)
        screen.blit(time_text, (10, 40))

        # 파라미터 표시
        param_text_sigma = font.render(f"Sigma (σ): {SIGMA:.1f}", True, WHITE)
        param_text_rho = font.render(f"Rho (ρ): {RHO:.1f}", True, WHITE)
        param_text_beta = font.render(f"Beta (β): {BETA:.2f}", True, WHITE)
        screen.blit(param_text_sigma, (WIDTH - 200, 10))
        screen.blit(param_text_rho, (WIDTH - 200, 40))
        screen.blit(param_text_beta, (WIDTH - 200, 70))
        
        # 초기 조건 차이 표시
        delta_text = font.render(f"Initial Diff (Δx): {delta}", True, WHITE)
        screen.blit(delta_text, (WIDTH - 200, 100))

        # 도움말
        if show_help:
            help_texts = [
                "Controls:",
                "SPACE: Pause/Resume",
                "R: Reset Simulation",
                "H: Toggle Help",
                "Arrow Keys: Rotate View (X, Y axes)",
                "Q/E: Rotate View (Z axis)",
                "ESC: Quit"
            ]
            for i, text in enumerate(help_texts):
                help_surface = font.render(text, True, GREEN)
                screen.blit(help_surface, (10, HEIGHT - 30 * (len(help_texts) - i)))
        
        if paused:
            pause_text = font.render("PAUSED", True, WHITE)
            text_rect = pause_text.get_rect(center=(WIDTH//2, HEIGHT//2))
            screen.blit(pause_text, text_rect)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()