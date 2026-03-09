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

    # --- DRAWING HELPERS (Realistic) ---

    def _draw_checkpoint_marker(self, surface, cx, cy, size):
        """Draw a futuristic neon checkpoint marker with pulsing glow."""
        glow_surf = pygame.Surface((int(size * 4), int(size * 4)), pygame.SRCALPHA)
        glow_center = int(size * 2)
        # Neon outer glow rings (cyan/electric blue)
        for i in range(5, 0, -1):
            alpha = 15 * i
            r = int(size * (0.5 + i * 0.3))
            pygame.draw.circle(glow_surf, (*COLOR_CHECKPOINT_GLOW, alpha), (glow_center, glow_center), r)
        surface.blit(glow_surf, (int(cx - size * 2), int(cy - size * 2)))

        # Diamond shape with neon color
        half = size * 0.45
        points = [
            (cx, cy - half),
            (cx + half, cy),
            (cx, cy + half),
            (cx - half, cy),
        ]
        int_points = [(int(p[0]), int(p[1])) for p in points]
        pygame.gfxdraw.aapolygon(surface, int_points, COLOR_CHECKPOINT)
        pygame.gfxdraw.filled_polygon(surface, int_points, COLOR_CHECKPOINT)
        # Inner bright core
        inner = size * 0.18
        inner_pts = [
            (int(cx), int(cy - inner)),
            (int(cx + inner), int(cy)),
            (int(cx), int(cy + inner)),
            (int(cx - inner), int(cy)),
        ]
        pygame.gfxdraw.filled_polygon(surface, inner_pts, (255, 255, 255))

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.font.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            
            # --- MUSIC INIT ---
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
        # Use exact floating point sizes to avoid integer rounding desync
        rows, cols = self.matrix.shape
        cell_w = self.window_size / cols
        cell_h = self.window_size / rows
        pix_square_size = min(cell_w, cell_h)

        canvas_hi = pygame.Surface((self.window_size, self.window_size))

        # --- BACKGROUND GENERATION (Smooth Mask Compositing) ---
        if self.background_surface is None:
            import scipy.ndimage
            
            # Create discrete masks for each surface type
            mask_road = (self.matrix == TRACK_ROAD_VALUE).astype(np.float32)
            mask_curb = (self.matrix == TRACK_CURB_VALUE).astype(np.float32)
            mask_finish = (self.matrix == TRACK_FINISH_VALUE).astype(np.float32)
            
            # Smooth upscale masks (bicubic interpolation creates beautiful curves)
            smooth_road = scipy.ndimage.zoom(mask_road, (self.window_size/rows, self.window_size/cols), order=3)
            smooth_curb = scipy.ndimage.zoom(mask_curb, (self.window_size/rows, self.window_size/cols), order=3)
            smooth_finish = scipy.ndimage.zoom(mask_finish, (self.window_size/rows, self.window_size/cols), order=3)
            
            # Base color arrays (H, W, 3)
            H, W = int(self.window_size), int(self.window_size)
            
            # --- Smooth radial purple gradient background ---
            y_coords_bg, x_coords_bg = np.indices((H, W))
            center_y, center_x = H / 2, W / 2
            dist_from_center = np.sqrt((x_coords_bg - center_x)**2 + (y_coords_bg - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            gradient_t = (dist_from_center / max_dist)[..., np.newaxis]  # 0 at center, 1 at corners
            
            # Blend from lighter purple (center) to darker purple (edges)
            color_center = np.array(COLOR_GRASS_LIGHT, dtype=np.float32)
            color_edge = np.array(COLOR_GRASS_DARK, dtype=np.float32)
            rgb_map = color_center * (1 - gradient_t) + color_edge * gradient_t

            # --- Neon fuchsia grid on background only ---
            grid_spacing = 40  # pixels between grid lines
            grid_thickness = 1  # 1px thin lines
            grid_on_x = (x_coords_bg % grid_spacing) < grid_thickness
            grid_on_y = (y_coords_bg % grid_spacing) < grid_thickness
            grid_mask = (grid_on_x | grid_on_y).astype(np.float32)
            # Exclude road and curb areas from the grid
            bg_only = np.clip(1.0 - smooth_road * 3.0 - smooth_curb * 3.0, 0, 1)
            grid_mask = grid_mask * bg_only
            grid_mask = grid_mask[..., np.newaxis]
            neon_fuchsia = np.array([255, 0, 180], dtype=np.float32)
            # Blend grid with a soft glow effect (semi-transparent)
            rgb_map = rgb_map * (1 - grid_mask * 0.45) + neon_fuchsia * (grid_mask * 0.45)

            # 2. Darker halo near road (subtle transition)
            alpha_mowed = np.clip(1.0 - np.abs(smooth_road - 0.25) / 0.15, 0, 1)
            alpha_mowed = alpha_mowed[..., np.newaxis]
            rgb_map = rgb_map * (1 - alpha_mowed) + np.array(COLOR_GRASS_DARK) * alpha_mowed

            # 3. Road blend (clean flat asphalt, subtle noise only)
            alpha_road = np.clip((smooth_road - 0.48) / 0.04, 0, 1)
            alpha_road = alpha_road[..., np.newaxis]
            noise_hf = (np.random.rand(H, W, 3).astype(np.float32) - 0.5) * 8.0
            road_texture = np.array(COLOR_ROAD) + noise_hf
            rgb_map = rgb_map * (1 - alpha_road) + road_texture * alpha_road

            # 4. Curb blend (solid uniform neon color)
            alpha_curb = np.clip((smooth_curb - 0.3) / 0.1, 0, 1)
            alpha_curb = alpha_curb[..., np.newaxis]
            rgb_map = rgb_map * (1 - alpha_curb) + np.array(COLOR_KERB) * alpha_curb

            # 4b. Neon glow edge line (road boundary)
            # Thin bright neon line right at the road-to-curb transition
            alpha_edge = np.clip(1.0 - np.abs(smooth_road - 0.45) / 0.03, 0, 1)
            alpha_edge = alpha_edge[..., np.newaxis]
            rgb_map = rgb_map * (1 - alpha_edge) + np.array(COLOR_ROAD_LINE) * alpha_edge

            # 5. Finish blend
            alpha_finish = np.clip((smooth_finish - 0.3) / 0.1, 0, 1)
            alpha_finish = alpha_finish[..., np.newaxis]
            y_coords, x_coords = np.indices((H, W))
            finish_pattern = ((((x_coords // 10) % 2) == ((y_coords // 10) % 2)))[..., np.newaxis]
            finish_color = np.where(finish_pattern, COLOR_FINISH_CHECKER_1, COLOR_FINISH_CHECKER_2)
            rgb_map = rgb_map * (1 - alpha_finish) + finish_color * alpha_finish

            # Finalize and convert to Pygame surface
            rgb_map = np.clip(rgb_map, 0, 255).astype(np.uint8)
            # Pygame surfarray expects (W, H, 3)
            self.background_surface = pygame.surfarray.make_surface(np.transpose(rgb_map, (1, 0, 2)))

        # Blit cached background
        canvas_hi.blit(self.background_surface, (0, 0))

        # --- CHECKPOINT MARKERS (diamond, only uncollected) ---
        for key, value in self._checkpoints.items():
            try:
                cp_idx = int(key.split("_")[1])
            except:
                cp_idx = 999
            
            if len(value) > 0 and cp_idx > self._progresso:
                coords = np.array(value)
                center_row = np.mean(coords[:, 0])
                center_col = np.mean(coords[:, 1])
                
                cx = (center_col + 0.5) * cell_w
                cy = (center_row + 0.5) * cell_h
                
                self._draw_checkpoint_marker(canvas_hi, cx, cy, pix_square_size * 0.6)


        # --- KART ---
        cx = (self._agent_location[1] + 0.5) * cell_w
        cy = (self._agent_location[0] + 0.5) * cell_h
        
        angle = 0
        if self._last_action == 0: angle = 0   
        if self._last_action == 1: angle = 90  
        if self._last_action == 2: angle = 180 
        if self._last_action == 3: angle = 270 

        # Shadow (SSAA: draw at hi-res, rotate, smoothscale down)
        # Shadow (SSAA: draw at hi-res, rotate, smoothscale down)
        k = RENDER_SCALE
        shadow_size = int(pix_square_size * 1.4 * k)
        shadow_surf = pygame.Surface((shadow_size, shadow_size), pygame.SRCALPHA)
        for i in range(4):
            alpha = 40 - i * 10
            shrink = i * 2 * k
            sr = pygame.Rect(shrink, shrink,
                             shadow_size - shrink * 2,
                             int(shadow_size * 0.6) - shrink)
            if sr.width > 0 and sr.height > 0:
                pygame.draw.ellipse(shadow_surf, (0, 0, 0, max(0, alpha)), sr)
        shadow_rot = pygame.transform.rotate(shadow_surf, angle)
        sw, sh = shadow_rot.get_width(), shadow_rot.get_height()
        shadow_final = pygame.transform.smoothscale(
            shadow_rot, (max(1, sw // k), max(1, sh // k)))
        shadow_dest = shadow_final.get_rect(center=(cx + 2, cy + 2))
        canvas_hi.blit(shadow_final, shadow_dest)

        # Drift smoke (subtle, darker)
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

             for _ in range(3): 
                 rnd_x = random.randint(-4, 4)
                 rnd_y = random.randint(-4, 4)
                 smoke_pos = (cx + offset_base_x + rnd_x, cy + offset_base_y + rnd_y)
                 smoke_size = random.randint(2, 5)
                 smoke_surf = pygame.Surface((smoke_size * 2, smoke_size * 2), pygame.SRCALPHA)
                 pygame.draw.circle(smoke_surf, (80, 80, 80, 70), (smoke_size, smoke_size), smoke_size) 
                 canvas_hi.blit(smoke_surf, (smoke_pos[0] - smoke_size, smoke_pos[1] - smoke_size))


        # Kart body (SSAA: draw at hi-res, rotate, smoothscale down)
        # SCALE INCREASE: Multiplier increased for larger kart visibility
        size_multiplier = 2.8 
        k = RENDER_SCALE
        kart_size = int(pix_square_size * size_multiplier * k)
        
        # Load and cache the kart image
        if self.kart_surface is None:
            image_path = os.path.join("data", "car.png")
            
            # CRITICAL: We must ensure a display is set before convert_alpha()
            if pygame.display.get_surface() is None:
                 # Create a dummy surface if running headlessly
                 pygame.display.set_mode((1,1), pygame.HIDDEN)
            
            if os.path.exists(image_path):
                # Load the image and retain transparency
                original_img = pygame.image.load(image_path).convert_alpha()
                # Assuming the image is facing North, rotate it -90 to face East (angle 0)
                original_img = pygame.transform.rotate(original_img, -90)
                
                # Scale up to our multi-sampled resolution 
                # (keeping aspect ratio based on original image dimensions)
                iw, ih = original_img.get_width(), original_img.get_height()
                ratio = ih / iw
                self.kart_surface = pygame.transform.smoothscale(
                    original_img, (kart_size, int(kart_size * ratio))
                )
            else:
                # Fallback if image doesn't exist
                self.kart_surface = pygame.Surface((kart_size, kart_size), pygame.SRCALPHA)
                pygame.draw.rect(self.kart_surface, COLOR_KART_BODY, (0, 0, kart_size, kart_size), border_radius=10)

        # Rotate at hi-res, then smoothscale down for clean AA
        rotated = pygame.transform.rotozoom(self.kart_surface, angle, 1.0)
        rw, rh = rotated.get_width(), rotated.get_height()
        rotated_kart = pygame.transform.smoothscale(rotated, (max(1, rw // k), max(1, rh // k)))
        rect_rotated = rotated_kart.get_rect(center=(cx, cy))
        canvas_hi.blit(rotated_kart, rect_rotated)


        # --- HUD (Futuristic Neon) ---
        if self.render_mode == "human":
            # Load Orbitron futuristic font
            try:
                font_path = os.path.join("data", "Orbitron.ttf")
                font = pygame.font.Font(font_path, 22)
                font_large = pygame.font.Font(font_path, 30)
            except:
                font = pygame.font.SysFont(None, 22)
                font_large = pygame.font.SysFont(None, 30)

            # Glassmorphism HUD panel (dark with neon border)
            hud_panel = pygame.Surface((280, 75), pygame.SRCALPHA)
            pygame.draw.rect(hud_panel, COLOR_HUD_BG, (0, 0, 280, 75), border_radius=6)
            # Accent border glow
            pygame.draw.rect(hud_panel, (*COLOR_HUD_ACCENT, 120), (0, 0, 280, 75), 1, border_radius=6)
            canvas_hi.blit(hud_panel, (12, 12))

            def draw_text_neon(surf, fnt, text, x, y, color, glow_color=None):
                """Draw text with neon glow effect."""
                if glow_color is None:
                    glow_color = color
                # Glow layers
                glow_surf = fnt.render(text, True, glow_color)
                glow_alpha = pygame.Surface(glow_surf.get_size(), pygame.SRCALPHA)
                glow_alpha.blit(glow_surf, (0, 0))
                glow_alpha.set_alpha(40)
                for dx, dy in [(-2,0),(2,0),(0,-2),(0,2),(-1,-1),(1,1),(-1,1),(1,-1)]:
                    surf.blit(glow_alpha, (x + dx, y + dy))
                # Main text
                base = fnt.render(text, True, color)
                surf.blit(base, (x, y))

            draw_text_neon(canvas_hi, font_large,
                           f"CP {self._progresso}/{self.numero_checkpoints}",
                           24, 18, COLOR_HUD_TEXT, COLOR_CHECKPOINT_GLOW)

            # Progress bar with neon fill
            bar_x, bar_y, bar_w, bar_h = 24, 56, 248, 8
            pygame.draw.rect(canvas_hi, (30, 30, 45), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
            progress_frac = self._progresso / max(1, self.numero_checkpoints)
            fill_w = int(bar_w * progress_frac)
            if fill_w > 0:
                pygame.draw.rect(canvas_hi, COLOR_HUD_ACCENT, (bar_x, bar_y, fill_w, bar_h), border_radius=4)
                # Glow on the fill
                glow_bar = pygame.Surface((fill_w, bar_h + 6), pygame.SRCALPHA)
                pygame.draw.rect(glow_bar, (*COLOR_HUD_ACCENT, 50), (0, 0, fill_w, bar_h + 6), border_radius=4)
                canvas_hi.blit(glow_bar, (bar_x, bar_y - 3))
            pygame.draw.rect(canvas_hi, (60, 60, 80), (bar_x, bar_y, bar_w, bar_h), 1, border_radius=4)

            # Time (top-right) with neon styling
            time_text = f"T {self._global_time}"
            time_surf = font.render(time_text, True, COLOR_HUD_TEXT)
            tw = time_surf.get_width()
            time_panel = pygame.Surface((tw + 24, 36), pygame.SRCALPHA)
            pygame.draw.rect(time_panel, COLOR_HUD_BG, (0, 0, tw + 24, 36), border_radius=6)
            pygame.draw.rect(time_panel, (*COLOR_HUD_ACCENT, 120), (0, 0, tw + 24, 36), 1, border_radius=6)
            canvas_hi.blit(time_panel, (self.window_size - tw - 36, 12))
            draw_text_neon(canvas_hi, font, time_text,
                           self.window_size - tw - 24, 18, COLOR_HUD_TEXT)


        if self.render_mode == "human":
            self.window.blit(canvas_hi, canvas_hi.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas_hi)), axes=(1, 0, 2)
            )
