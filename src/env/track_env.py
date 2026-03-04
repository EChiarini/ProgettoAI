import gymnasium as gym
import numpy as np
import pygame
import os
import random
import math
# Assicurati che crea_matrice_centro sia importata
from .track_utils import build_track, crea_matrice_distanze, argwhere, count_numpy_list, crea_matrice_centro
from .track_costants import *
import pygame.gfxdraw

class TrackEnv(gym.Env):

    def __init__(self, render_mode=None, is_testing=False): 
        self.metadata = {"render_fps": RENDER_FPS, "render_modes": ["human", "rgb_array"]}
        self.is_testing = is_testing
        
        self.matrix = build_track()
        self.distance_matrix = crea_matrice_distanze(get_default_track_path(), "destra")
        self._max_distance = self.distance_matrix.max()
        
        # Inizializziamo sempre la matrice del centro per evitare errori di attributo mancante
        self.center_matrix = crea_matrice_centro(self.matrix)

        self.render_mode = render_mode
        self.window_size = WINDOW_SIZE_PX
        self.window = None
        self.clock = None
        self.background_surface = None

        # Decoration cache
        self.decorations = [] 

        self.view_size = VIEW_SIZE
        self.road_width = ROAD_WIDTH
        self.trajectory = list()
        self.trajectory_heat_map = dict()

        coordinates = argwhere(self.matrix, TRACK_FINISH_VALUE)
        self._target_location = np.array(coordinates, dtype=np.int32)
        print(f"{coordinates}")

        self._agent_location = np.array(coordinates[self.road_width // 2], dtype=np.int32)

        # LETTURA DELLA COSTANTE (Deve essere "simple", "simple_aggiornato" o "velocity")
        self.movement_mode = MOVEMENT_MODE

        # --- SETUP ACTION & OBSERVATION SPACES ---
        if self.movement_mode == "velocity":
            self.action_space = gym.spaces.Discrete(VELOCITY_ACTION_SPACE_SIZE)
            self.speed = np.array([0, 0], dtype=np.int32)
            # Mappa azione -> (accelerazione_riga, accelerazione_colonna)
            self._action_to_acceleration = {
                0: (-1, -1),
                1: (-1,  0),
                2: (-1,  1),
                3: ( 0, -1),
                4: ( 0,  0),
                5: ( 0,  1),
                6: ( 1, -1),
                7: ( 1,  0),
                8: ( 1,  1),
            }
            self.observation_space = gym.spaces.Dict({
                "agent_view": gym.spaces.Box(
                    low=TRACK_UNKNOWN_VALUE,
                    high=TRACK_FINISH_VALUE,
                    shape=(self.view_size, self.view_size),
                    dtype=np.float32
                ),
                "speed": gym.spaces.Box(
                    low=-VELOCITY_MAX_SPEED,
                    high=VELOCITY_MAX_SPEED,
                    shape=(2,),
                    dtype=np.int32
                )
            })
        else:
            # Per "simple" e "simple_aggiornato"
            self.action_space = gym.spaces.Discrete(4)
            self._action_to_direction = {
                0: np.array([0, 1]),
                1: np.array([-1, 0]),
                2: np.array([0, -1]),
                3: np.array([1, 0]),
            }
            self.observation_space = gym.spaces.Dict({
                "agent_view": gym.spaces.Box(
                    low=TRACK_UNKNOWN_VALUE,
                    high=TRACK_FINISH_VALUE,
                    shape=(self.view_size, self.view_size),
                    dtype=np.float32
                )
            })

        self._last_action = 0

        self._checkpoints = dict()
        self.numero_checkpoints=NUM_CHECKPOINTS

        valore_checkpoint=self.distance_matrix.max()//self.numero_checkpoints

        for i in range(self.numero_checkpoints-1):
            coordinate_checkpoint=argwhere(self.distance_matrix, valore_checkpoint*(i+1))
            self._checkpoints[f"checkpoint_{i+1}"] = coordinate_checkpoint

        self._progresso = 0
        self._tempo_passato = 0
        self._global_time = 0 # Cumulative time
        self.trajectory_heat_map_single_episode = dict()
        self._drift_frames = 0 # Counter for drift effect

    def _get_obs(self):
        view_padding = self.view_size // 2
        max_x,max_y = self.matrix.shape

        view_matrix = np.ones(shape=(self.view_size, self.view_size))
        for i in range(self.view_size):
            view_matrix[i]=view_matrix[i]* TRACK_UNKNOWN_VALUE

        tl_x = self._agent_location[0] - view_padding
        tl_y = self._agent_location[1] - view_padding

        view_matrix_x = range(tl_x, tl_x+self.view_size,1)
        view_matrix_y = range(tl_y, tl_y+self.view_size,1)

        clipped_view_matrix_x = np.clip(view_matrix_x,0, max_x - 1)
        clipped_view_matrix_y = np.clip(view_matrix_y,0, max_y - 1)
        
        clipped_view_matrix_x = list(set(clipped_view_matrix_x))
        clipped_view_matrix_y = list(set(clipped_view_matrix_y))

        for x in clipped_view_matrix_x:
            for y in  clipped_view_matrix_y:
                view_matrix[x-tl_x , y-tl_y]=self.matrix[y,x]

        view_matrix = view_matrix.T
        obs = { "agent_view": view_matrix }
        if self.movement_mode == "velocity":
            obs["speed"] = np.array(self.speed, dtype=np.int32)
        return obs

    def _get_info(self):
        return { "agent_location":self._agent_location }

    def reset(self, seed = None, options = None):
        self.trajectory_heat_map_single_episode = dict()
        super().reset(seed=seed)

        self._tempo_passato = 0
        self._global_time = 0 
        
        self._progresso = 0
        self._last_action = 0

        if self.movement_mode == "velocity":
            self.speed = np.array([0, 0], dtype=np.int32)

        coordinates = np.argwhere(self.matrix == TRACK_FINISH_VALUE)

        # Gestione unificata della direzione (dal File 2)
        try:
            direzione = DEFAULT_DISTANCE_DIRECTION
        except NameError:
            direzione = "destra" # fallback sicuro se non definito in track_costants
            
        if options and "direzione" in options:
            direzione = options["direzione"]

        match direzione:
            case "destra": slider=[0,1]
            case "sinistra": slider=[0,-1]
            case "basso": slider=[1,0]
            case "alto": slider=[-1,0]
            case _: slider=[0,0]

        self._agent_location = np.array(coordinates[self.road_width // 2], dtype = np.int32)
        self._agent_location[0] = self._agent_location[0] + slider[0]
        self._agent_location[1] = self._agent_location[1] + slider[1]

        observation = self._get_obs()
        info = self._get_info()
      
        return observation, info

    def _check_out_track(self, new_position):
        """Controlla se il percorso verso new_position esce dalla pista (modalità velocity)."""
        row, col = new_position
        H, W = self.matrix.shape

        if row < 0 or row >= H or col < 0 or col >= W:
            return True

        if self.matrix[row, col] == TRACK_OFFROAD_VALUE:
            return True

        cur_row, cur_col = self._agent_location
        row_step = -1 if row < cur_row else 1
        col_step = -1 if col < cur_col else 1

        for r in range(cur_row, row, row_step):
            if self.matrix[r, cur_col] == TRACK_OFFROAD_VALUE:
                return True

        for c in range(cur_col, col, col_step):
            if self.matrix[row, c] == TRACK_OFFROAD_VALUE:
                return True

        return False

    def step(self, action):
        reward = 0
        size = self.matrix.shape[0]

        terminated = False
        truncated = False

        if action != self._last_action:
            self._drift_frames = 2
        else:
            if self._drift_frames > 0:
                self._drift_frames -= 1

        self._last_action = action

        old_agent_distance = self.distance_matrix[self._agent_location[0], self._agent_location[1]]
        
        # Salvataggio dati specifici per modalità
        old_position = self._agent_location.copy()
        if self.movement_mode == "simple_aggiornato":
            old_center_distance = self.center_matrix[self._agent_location[0], self._agent_location[1]]

        # ---- LOGICA DI MOVIMENTO ----
        if self.movement_mode == "velocity":
            row_acc, col_acc = self._action_to_acceleration[action]
            new_speed_row = self.speed[0] + row_acc
            new_speed_col = self.speed[1] + col_acc

            new_speed_row = int(np.clip(new_speed_row, -VELOCITY_MAX_SPEED, VELOCITY_MAX_SPEED))
            new_speed_col = int(np.clip(new_speed_col, -VELOCITY_MAX_SPEED, VELOCITY_MAX_SPEED))

            new_position = np.array([
                self._agent_location[0] + new_speed_row,
                self._agent_location[1] + new_speed_col
            ], dtype=np.int32)

            if self._check_out_track(new_position):
                reward = OFFROAD_REWARD
                terminated = True
                return self._get_obs(), reward, terminated, truncated, self._get_info()

            self._agent_location = new_position
            self.speed = np.array([new_speed_row, new_speed_col], dtype=np.int32)
            
        else: # "simple" e "simple_aggiornato"
            direction = self._action_to_direction[action]
            self._agent_location = np.clip(self._agent_location + direction, 0, size - 1)

        self._tempo_passato += 1
        self._global_time += 1 

        pos_tuple = tuple(self._agent_location)

        # ---- HEATMAP ----
        if pos_tuple not in self.trajectory_heat_map:  
            self.trajectory_heat_map[pos_tuple] = 1
        else:
            self.trajectory_heat_map[pos_tuple] += 1

        if pos_tuple not in self.trajectory_heat_map_single_episode:  
            self.trajectory_heat_map_single_episode[pos_tuple] = 1
            reward += 0.1
        else:
            self.trajectory_heat_map_single_episode[pos_tuple] += 1
            reward += REPEAT_PENALTY * self.trajectory_heat_map_single_episode[pos_tuple]
            if self.trajectory_heat_map_single_episode[pos_tuple] > 4: 
                truncated = True
                return self._get_obs(), reward, terminated, truncated, self._get_info()
            
        reward += STEP_PENALTY
        self.trajectory.append(self._agent_location)

        # ---- CONTROLLO CHECKPOINT E TRAGUARDO ----
        path_cells = [self._agent_location]
        if self.is_testing and self.movement_mode == "velocity" and 'old_position' in locals():
            path_cells = []
            cur_row, cur_col = old_position
            row, col = self._agent_location
            row_step = -1 if row < cur_row else 1
            col_step = -1 if col < cur_col else 1
            for r in range(cur_row, row, row_step): path_cells.append(np.array([r, cur_col]))
            for c in range(cur_col, col, col_step): path_cells.append(np.array([row, c]))
            path_cells.append(self._agent_location.copy())

        for cell in path_cells:
            for x in self._target_location:
                if np.array_equal(x, cell):
                    if self._progresso == self.numero_checkpoints - 1:
                        reward += FINISH_REWARD
                        self._progresso = self.numero_checkpoints
                    else:
                        reward = -FINISH_REWARD
                    terminated = True
                    return self._get_obs(), reward, terminated, truncated, self._get_info()

            checkpoint_key = f"checkpoint_{self._progresso+1}"
            checkpoint_list = self._checkpoints.get(checkpoint_key, [])

            for x in checkpoint_list:
                if np.array_equal(cell, x):
                    reward += CHECKPOINT_REWARD
                    self._progresso = self._progresso + 1
                    self._tempo_passato = 0
                    return self._get_obs(), reward, terminated, truncated, self._get_info()

        if self.matrix[self._agent_location[0],self._agent_location[1]] == TRACK_OFFROAD_VALUE:
            reward = OFFROAD_REWARD
            terminated = True
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        # ---- CALCOLO REWARD FINALE DISTANZE ----
        if not terminated:
            new_agent_distance = self.distance_matrix[self._agent_location[0], self._agent_location[1]]
            delta_distance = new_agent_distance - old_agent_distance

            # Wrap-around detection (utile sia per simple che velocity)
            if abs(delta_distance) > self._max_distance / 2:
                if delta_distance > 0:
                    delta_distance = delta_distance - self._max_distance 
                else:
                    delta_distance = delta_distance + self._max_distance

            # Diversificazione calcolo basata sul movimento
            if self.movement_mode == "simple_aggiornato":
                new_center_distance = self.center_matrix[self._agent_location[0], self._agent_location[1]]
                delta_centro = old_center_distance - new_center_distance
                
                if delta_distance > 0:
                    reward += delta_distance + (delta_centro * CENTER_WEIGHT)
                else:
                    reward += (delta_distance * BACKWARD_PENALTY) + (delta_centro * CENTER_WEIGHT)
            else:
                # Per "simple" classico e "velocity"
                if delta_distance > 0:
                    reward += delta_distance
                else:
                    reward += delta_distance * BACKWARD_PENALTY

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            return self._render_frame()

    # --- DRAWING HELPERS ---
    def _draw_star(self, surface, x, y, size, color):
        points = []
        for i in range(10):
            angle = i * (math.pi / 5) - (math.pi / 2)
            radius = size if i % 2 == 0 else size / 2.5
            points.append((x + math.cos(angle) * radius, y + math.sin(angle) * radius))
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _draw_banana(self, surface, x, y, size):
        rect = pygame.Rect(x - size, y - size/2, size*2, size)
        pygame.draw.arc(surface, COLOR_BANANA, rect, 0, 3.14, int(size/3))
        pygame.draw.line(surface, COLOR_BANANA_SPOT, (x - size, y), (x - size - 2, y - 2), 2)

    def _draw_mushroom(self, surface, x, y, size):
        cap_rect = pygame.Rect(x - size, y - size, size*2, size)
        pygame.draw.ellipse(surface, COLOR_MUSHROOM_CAP, cap_rect)
        pygame.draw.circle(surface, COLOR_MUSHROOM_WHITE, (x, y - size/2), size/3)
        stem_rect = pygame.Rect(x - size/2, y, size, size)
        pygame.draw.rect(surface, COLOR_MUSHROOM_STEM, stem_rect)

    def _draw_pipe(self, surface, x, y, size):
        w, h = size, size * 1.5
        pygame.draw.rect(surface, COLOR_PIPE_GREEN, (x - w/2 - 2, y - h, w + 4, 10))
        pygame.draw.rect(surface, COLOR_PIPE_GREEN, (x - w/2, y - h + 10, w, h - 10))
        pygame.draw.line(surface, COLOR_PIPE_L, (x - w/2 + 4, y - h + 12), (x - w/2 + 4, y-2), 2)

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.font.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            
            try:
                pygame.mixer.init()
                music_path = os.path.join("data", "music", DEFAULT_TRACK_MUSIC)
                if os.path.exists(music_path):
                    pygame.mixer.music.load(music_path)
                    pygame.mixer.music.play(-1) 
                    pygame.mixer.music.set_volume(0.5)
                else:
                    print(f"Music file not found at: {music_path}")
            except Exception as e:
                print(f"Error loading music: {e}")
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        max_dim = max(self.matrix.shape)
        SCALE = 2 
        hi_res_size = int(self.window_size * SCALE)

        canvas_hi = pygame.Surface((hi_res_size, hi_res_size))
        pix_square_size = int(self.window_size // max_dim)

        if self.background_surface is None:
            self.background_surface = pygame.Surface((self.window_size, self.window_size))
            self.background_surface.fill(COLOR_GRASS_MARIO)

            rows, cols = self.matrix.shape
            for r in range(rows):
                for c in range(cols):
                    val = self.matrix[r, c]
                    
                    rect = pygame.Rect(
                        int(c * pix_square_size),
                        int(r * pix_square_size),
                        pix_square_size + 1,
                        pix_square_size + 1
                    )

                    if val != TRACK_ROAD_VALUE and val != TRACK_CURB_VALUE and val != TRACK_FINISH_VALUE:
                         if random.random() < 0.05: 
                            decoration_type = random.choice(["pipe", "mushroom"])
                            center_x = rect.centerx + random.randint(-5, 5)
                            center_y = rect.centery + random.randint(-5, 5)
                            self.decorations.append({"type": decoration_type, "pos": (center_x, center_y), "size": pix_square_size/2})

                    if val == TRACK_ROAD_VALUE:
                        pygame.draw.rect(self.background_surface, COLOR_ROAD_MARIO, rect)
                        
                        if random.random() < 0.02: 
                            self.decorations.append({"type": "banana", "pos": rect.center, "size": pix_square_size/3})

                    elif val == TRACK_CURB_VALUE:
                        pygame.draw.rect(self.background_surface, COLOR_KERB_1, rect)
                        pygame.draw.polygon(self.background_surface, COLOR_KERB_2, [
                            (rect.left + 8, rect.top), 
                            (rect.right, rect.top), 
                            (rect.right, rect.bottom - 8),
                            (rect.left, rect.bottom - 8)
                        ])

                    elif val == TRACK_FINISH_VALUE:
                        pygame.draw.rect(self.background_surface, COLOR_FINISH_CHECKER_1, rect)
                        half = pix_square_size / 2
                        pygame.draw.rect(self.background_surface, COLOR_FINISH_CHECKER_2, (rect.left, rect.top, half, half))
                        pygame.draw.rect(self.background_surface, COLOR_FINISH_CHECKER_2, (rect.left + half, rect.top + half, half, half))


            for decor in self.decorations:
                if decor["type"] in ["pipe", "mushroom"]:
                    x, y = decor["pos"]
                    if decor["type"] == "pipe":
                        self._draw_pipe(self.background_surface, x, y, decor["size"])
                    elif decor["type"] == "mushroom":
                        self._draw_mushroom(self.background_surface, x, y, decor["size"])
            
        canvas = pygame.transform.smoothscale(canvas_hi, (self.window_size, self.window_size))
        canvas_hi.blit(self.background_surface, (0, 0))

        for decor in self.decorations:
            if decor["type"] == "banana":
                 self._draw_banana(canvas_hi, decor["pos"][0], decor["pos"][1], decor["size"])

        current_time = pygame.time.get_ticks() / 100
        for key, value in self._checkpoints.items():
            try:
                cp_idx = int(key.split("_")[1])
            except:
                cp_idx = 999
            
            if len(value) > 0 and cp_idx > self._progresso:
                coords = np.array(value)
                center_row = np.mean(coords[:, 0])
                center_col = np.mean(coords[:, 1])
                
                cx = (center_col + 0.5) * pix_square_size
                cy = (center_row + 0.5) * pix_square_size
                
                offset_y = math.sin(current_time * 0.5) * 5
                self._draw_star(canvas_hi, cx, cy + offset_y, pix_square_size/2.5, COLOR_STAR)

        cx = (self._agent_location[1] + 0.5) * pix_square_size
        cy = (self._agent_location[0] + 0.5) * pix_square_size
        
        angle = 0
        if self._last_action == 0: angle = 0   
        if self._last_action == 1: angle = 90  
        if self._last_action == 2: angle = 180 
        if self._last_action == 3: angle = 270 

        shadow_rect = pygame.Rect(0, 0, pix_square_size * 0.8, pix_square_size * 0.8)
        shadow_surf = pygame.Surface((pix_square_size, pix_square_size), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, 100), shadow_rect)
        shadow_rotated = pygame.transform.rotate(shadow_surf, angle)
        shadow_dest = shadow_rotated.get_rect(center=(cx + 5, cy + 5))
        canvas_hi.blit(shadow_rotated, shadow_dest)
        
        if self._drift_frames > 0:
             offset_base_x = 0
             offset_base_y = 0
             
             if self._last_action == 0: 
                 offset_base_x = -pix_square_size * 0.4
             elif self._last_action == 2: 
                 offset_base_x = pix_square_size * 0.4
             elif self._last_action == 1: 
                 offset_base_y = pix_square_size * 0.4
             elif self._last_action == 3: 
                 offset_base_y = -pix_square_size * 0.4

             for _ in range(2): 
                 rnd_x = random.randint(-5, 5)
                 rnd_y = random.randint(-5, 5)
                 
                 smoke_pos = (cx + offset_base_x + rnd_x, cy + offset_base_y + rnd_y)
                 smoke_size = random.randint(3, 7) 
                 
                 smoke_surf = pygame.Surface((smoke_size*2, smoke_size*2), pygame.SRCALPHA)
                 pygame.draw.circle(smoke_surf, (150, 150, 150, 120), (smoke_size, smoke_size), smoke_size) 
                 canvas_hi.blit(smoke_surf, (smoke_pos[0]-smoke_size, smoke_pos[1]-smoke_size))

        kart_surf = pygame.Surface((pix_square_size, pix_square_size), pygame.SRCALPHA)
        w, h = pix_square_size * 0.9, pix_square_size * 0.6
        kart_rect = pygame.Rect(pix_square_size/2 - w/2, pix_square_size/2 - h/2, w, h)
        
        pygame.draw.rect(kart_surf, COLOR_KART_EXHAUST, (kart_rect.left - 4, kart_rect.centery - 4, 6, 8))
        
        tire_w, tire_h = w * 0.25, h * 0.5
        pygame.draw.rect(kart_surf, (0,0,0), (kart_rect.left, kart_rect.top - 2, tire_w, tire_h))
        pygame.draw.rect(kart_surf, (0,0,0), (kart_rect.right - tire_w, kart_rect.top - 2, tire_w, tire_h))
        pygame.draw.rect(kart_surf, (0,0,0), (kart_rect.left, kart_rect.bottom - tire_h + 2, tire_w, tire_h))
        pygame.draw.rect(kart_surf, (0,0,0), (kart_rect.right - tire_w, kart_rect.bottom - tire_h + 2, tire_w, tire_h))

        pygame.draw.rect(kart_surf, COLOR_KART_BODY, kart_rect, border_radius=5)
        pygame.draw.circle(kart_surf, COLOR_DRIVER_HELMET, kart_rect.center, h * 0.45) 
        pygame.draw.circle(kart_surf, "white", (kart_rect.centerx + 2, kart_rect.centery - 2), 3)

        rotated_kart = pygame.transform.rotate(kart_surf, angle)
        rect_rotated = rotated_kart.get_rect(center=(cx, cy))
        canvas_hi.blit(rotated_kart, rect_rotated)

        if self.render_mode == "human":
            try:
                font = pygame.font.Font(None, 36) 
            except:
                font = pygame.font.SysFont(None, 36)

            def draw_text_outlined(text, x, y, color):
                base = font.render(text, True, color)
                outline = font.render(text, True, COLOR_HUD_STROKE)
                canvas_hi.blit(outline, (x-2, y))
                canvas_hi.blit(outline, (x+2, y))
                canvas_hi.blit(outline, (x, y-2))
                canvas_hi.blit(outline, (x, y+2))
                canvas_hi.blit(base, (x, y))

            draw_text_outlined(f"CHECKPOINT: {self._progresso}/{self.numero_checkpoints}", 20, 20, COLOR_HUD_TEXT)
            draw_text_outlined(f"TIME: {self._global_time}", 20, 60, (255, 255, 255))

        if self.render_mode == "human":
            self.window.blit(canvas_hi, canvas_hi.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas_hi)), axes=(1, 0, 2)
            )