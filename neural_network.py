import pygame
import numpy as np
import sys
import time

# --- 시뮬레이션 설정 ---
WIDTH, HEIGHT = 1000, 750
FPS = 10  # 학습 과정을 천천히 보기 위해 FPS를 낮춤
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 100, 100)
BLUE = (100, 100, 255)
GREEN = (100, 255, 100)
GRAY = (150, 150, 150)
LIGHT_GRAY = (200, 200, 200)
YELLOW = (255, 255, 0)

NODE_RADIUS = 20
LAYER_SPACING = 200 # 레이어 간 간격
NODE_SPACING = 100  # 뉴런 간 간격

# --- 신경망 클래스 ---
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # 가중치와 편향 초기화 (Xavier/Glorot 초기화 유사하게)
        self.weights = []
        self.biases = []
        for i in range(self.num_layers - 1):
            # He 또는 Xavier 초기화와 유사하게 표준편차 조정
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            
        self.activations = [np.zeros((1, size)) for size in layer_sizes] # 각 층의 활성화 값 저장

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500))) # np.exp 오버플로우 방지

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.activations[0] = X
        a = X
        for i in range(self.num_layers - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            self.activations[i+1] = a
        return a

    def backward(self, X, y, output, learning_rate):
        errors = []
        deltas = []

        # 출력층 오차 및 델타
        error_output = y - output
        delta_output = error_output * self.sigmoid_derivative(output)
        errors.append(error_output)
        deltas.append(delta_output)

        # 은닉층 오차 및 델타 (역순으로 계산)
        for i in range(self.num_layers - 2, 0, -1):
            error_hidden = deltas[-1].dot(self.weights[i].T)
            delta_hidden = error_hidden * self.sigmoid_derivative(self.activations[i])
            errors.append(error_hidden)
            deltas.append(delta_hidden)
        
        deltas.reverse() # 정방향 순서로 맞춤

        # 가중치 및 편향 업데이트
        for i in range(self.num_layers - 1):
            input_to_layer = self.activations[i]
            delta = deltas[i]
            
            self.weights[i] += input_to_layer.T.dot(delta) * learning_rate
            self.biases[i] += np.sum(delta, axis=0, keepdims=True) * learning_rate
        
        return np.mean(np.abs(error_output)) # 평균 절대 오차 반환

    def train_step(self, X_sample, y_sample, learning_rate):
        output = self.forward(X_sample)
        loss = self.backward(X_sample, y_sample, output, learning_rate)
        return loss, output

# --- 시각화 함수 ---
def draw_network(screen, nn, current_input=None, current_output=None, current_target=None):
    font = pygame.font.SysFont(None, 20)
    layer_x_start = (WIDTH - (nn.num_layers - 1) * LAYER_SPACING) // 2

    node_positions = [] # 각 뉴런의 화면 좌표 저장

    # 뉴런 그리기 및 활성화 값 시각화
    for i, layer_size in enumerate(nn.layer_sizes):
        layer_positions = []
        x_pos = layer_x_start + i * LAYER_SPACING
        total_height = (layer_size - 1) * NODE_SPACING
        y_start = (HEIGHT - total_height) // 2
        
        for j in range(layer_size):
            y_pos = y_start + j * NODE_SPACING
            layer_positions.append((x_pos, y_pos))

            # 뉴런 활성화 값에 따른 색상 (0~1 사이 값으로 가정)
            activation_value = nn.activations[i][0, j] if nn.activations[i].shape[1] > j else 0.5
            color_intensity = int(activation_value * 200) + 55 # 55 ~ 255 범위
            node_color = (color_intensity, color_intensity, color_intensity)
            
            pygame.draw.circle(screen, node_color, (x_pos, y_pos), NODE_RADIUS)
            pygame.draw.circle(screen, BLACK, (x_pos, y_pos), NODE_RADIUS, 1) # 테두리

            # 활성화 값 텍스트 (선택적)
            # act_text = font.render(f"{activation_value:.2f}", True, BLACK)
            # screen.blit(act_text, (x_pos - NODE_RADIUS//2, y_pos - NODE_RADIUS*1.5))

        node_positions.append(layer_positions)

    # 시냅스(연결선) 그리기 및 가중치 시각화
    for i in range(nn.num_layers - 1):
        for j in range(nn.layer_sizes[i]):
            for k in range(nn.layer_sizes[i+1]):
                start_pos = node_positions[i][j]
                end_pos = node_positions[i+1][k]
                
                weight = nn.weights[i][j, k]
                
                # 가중치 크기에 따른 선 굵기 (0~5 범위로 정규화 시도)
                # 실제 가중치 범위는 다양하므로, 적절한 스케일링 필요
                line_thickness = min(5, max(1, int(abs(weight) * 2))) 
                
                # 가중치 부호에 따른 색상
                if weight > 0:
                    line_color = BLUE
                elif weight < 0:
                    line_color = RED
                else:
                    line_color = GRAY
                
                pygame.draw.line(screen, line_color, start_pos, end_pos, line_thickness)
    
    # 현재 입력, 예측, 목표값 표시
    info_font = pygame.font.SysFont(None, 28)
    if current_input is not None:
        input_text = info_font.render(f"Input: {current_input[0].tolist()}", True, WHITE)
        screen.blit(input_text, (10, HEIGHT - 90))
    if current_output is not None:
        output_text = info_font.render(f"Predicted: {current_output[0][0]:.3f}", True, YELLOW)
        screen.blit(output_text, (10, HEIGHT - 60))
    if current_target is not None:
        target_text = info_font.render(f"Target: {current_target[0][0]}", True, GREEN)
        screen.blit(target_text, (10, HEIGHT - 30))


def draw_info(screen, epoch, loss, learning_rate, paused):
    font = pygame.font.SysFont(None, 28)
    epoch_text = font.render(f"Epoch: {epoch}", True, WHITE)
    loss_text = font.render(f"Loss (MAE): {loss:.4f}", True, WHITE)
    lr_text = font.render(f"Learning Rate: {learning_rate:.3f}", True, WHITE)
    
    screen.blit(epoch_text, (10, 10))
    screen.blit(loss_text, (10, 40))
    screen.blit(lr_text, (10, 70))

    if paused:
        pause_text = font.render("PAUSED (Space to resume, S for step)", True, YELLOW)
        text_rect = pause_text.get_rect(center=(WIDTH // 2, 30))
        screen.blit(pause_text, text_rect)


# --- 메인 함수 ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("신경망 학습 시각화기 (XOR Problem)")
    clock = pygame.time.Clock()

    # 신경망 구조: 입력 2, 은닉 3, 출력 1
    layer_config = [2, 3, 1] 
    nn = NeuralNetwork(layer_config)

    # XOR 데이터셋
    X_train = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_train = np.array([[0], [1], [1], [0]])
    
    current_data_idx = 0

    # 학습 파라미터
    learning_rate = 0.1
    current_epoch = 0
    total_epochs_run = 0 # 실제 학습 진행된 epoch 수
    current_loss = 0.0
    
    # 시각화용 변수
    current_input_sample = None
    current_output_value = None
    current_target_value = None

    running = True
    paused = True 
    single_step_mode = False # S키로 한 스텝씩 실행

    # 초기 순전파 (시작 시 네트워크 상태 보기 위함)
    initial_X = np.array([[0,0]]) # 임의의 입력으로 초기 활성화 보기
    nn.forward(initial_X)


    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    single_step_mode = False # Space 누르면 연속 실행 모드
                if event.key == pygame.K_s: # 한 스텝 실행
                    paused = False
                    single_step_mode = True
                if event.key == pygame.K_r: # 리셋
                    nn = NeuralNetwork(layer_config)
                    current_epoch = 0
                    total_epochs_run = 0
                    current_loss = 0.0
                    current_data_idx = 0
                    paused = True
                    nn.forward(initial_X) # 리셋 후 초기 상태
                if event.key == pygame.K_UP:
                    learning_rate = min(1.0, learning_rate + 0.01)
                if event.key == pygame.K_DOWN:
                    learning_rate = max(0.001, learning_rate - 0.01)
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        if not paused:
            # 현재 데이터 샘플 가져오기
            X_sample = X_train[current_data_idx:current_data_idx+1]
            y_sample = y_train[current_data_idx:current_data_idx+1]
            
            current_input_sample = X_sample
            current_target_value = y_sample

            # 학습 단계 실행 (순전파 및 역전파)
            loss, output = nn.train_step(X_sample, y_sample, learning_rate)
            current_loss = loss 
            current_output_value = output
            
            current_data_idx += 1
            if current_data_idx >= len(X_train):
                current_data_idx = 0
                current_epoch += 1 # 모든 데이터를 한 번 다 보면 1 epoch
                total_epochs_run += 1

            if single_step_mode:
                paused = True # 한 스텝 실행 후 다시 일시정지
                single_step_mode = False


        # 그리기
        screen.fill(BLACK)
        draw_network(screen, nn, current_input_sample, current_output_value, current_target_value)
        draw_info(screen, total_epochs_run, current_loss, learning_rate, paused and not single_step_mode)
        
        # 도움말
        font_help = pygame.font.SysFont(None, 22)
        help_texts = [
            "Controls: SPACE (Pause/Resume), S (Single Step), R (Reset)",
            "UP/DOWN Arrows (Change Learning Rate), ESC (Quit)"
        ]
        for i, text in enumerate(help_texts):
            help_surface = font_help.render(text, True, LIGHT_GRAY)
            screen.blit(help_surface, (WIDTH - help_surface.get_width() - 10, 10 + i * 25))


        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()