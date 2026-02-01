import gymnasium as gym
import numpy as np
import pygame
import os
from .track_utils import build_track, crea_matrice_distanze, argwhere, count_numpy_list

class TrackEnv(gym.Env):

    def __init__(self, render_mode=None): 
      
        # Metadati per il rendering (es. 10 FPS per vederlo con calma)
        self.metadata = {"render_fps": 10, "render_modes": ["human", "rgb_array"]}
        
        self.matrix = build_track()
        self.distance_matrix = crea_matrice_distanze(os.getcwd() + "/data/tracks/track_imola.csv", "destra")
        
        # --- VARIABILI PER IL RENDERING ---
        self.render_mode = render_mode
        self.window_size = 800  # Dimensione della finestra in pixel
        self.window = None      # Finestra Pygame
        self.clock = None       # Clock per gestire gli FPS
        # ----------------------------------

        self.view_size = 7
        self.road_width = 5
        self.trajectory = list()

        coordinates = np.argwhere(self.matrix == 2)
        self._target_location = np.array(coordinates, dtype=np.int32)
        self._agent_location = np.array(coordinates[self.road_width // 2], dtype=np.int32)
        #self._agent_velocity = 1

        self.action_space = gym.spaces.Discrete(4)
        
        self._action_to_direction = {
            0: np.array([0, 1]),
            1: np.array([-1, 0]),
            2: np.array([0, -1]),
            3: np.array([1, 0]),
        }

        self.observation_space = gym.spaces.Dict({  #i dict servono per spazi eterogenei
                # 1. La sottomatrice locale (come prima)
                "agent_view": gym.spaces.Box( #i box servono per spazi omogenei
                    low=-2,
                    high=2,
                    shape=(self.view_size, self.view_size),
                    dtype=np.int8
                )
                #,

                # 2. La direzione (angolo in radianti o gradi)
                # Definiamo un array di shape=(1,) contenente un float
                # Esempio: da 0 a 2*PI (circa 6.28)
                #"direction": gym.spaces.Box(
                #    low=0,
                #    high=2 * math.pi,
                #    shape=(1,),
                #    dtype=np.float32
                #),

                #aggiungere velocità
                #"velocity": gym.spaces.Box(
                #    low=0,
                #   high=self.matrix.shape[0],
                #   shape=(1,),
                #   dtype=np.float32
                #)
            })

        self._checkpoints = dict()
        self.numero_checkpoints=7

        valore_checkpoint=self.distance_matrix.max()//self.numero_checkpoints
        for i in range(self.numero_checkpoints-1):
           
            #print(f"valore checkpoint n.{i} = {valore_checkpoint*(i+1)}")
            coordinate_checkpoint=argwhere(self.distance_matrix, valore_checkpoint*(i+1))
            self._checkpoints[f"checkpoint_{i+1}"] = coordinate_checkpoint

        #print(self._checkpoints)

        self._progresso = 0


    def _get_obs(self):
        view_padding = self.view_size // 2
        max_x,max_y = self.matrix.shape

        # HO CAMBIAMO IL LOW DELLO SPAZIO DELLE OSSERVAZIONI DA -1 A -2 PERCHè SERVE
        # UN MODO PER INDICARE UNA CELLA CHE NON ESISTE

        # creo la matrice che vede il pilota
        view_matrix = np.ones(shape=(self.view_size, self.view_size))
        # inizializzo tutte le celle viste dal pilota come non esistenti
        for i in range(self.view_size):
            view_matrix[i]=view_matrix[i]*-2
        #indice della prima cella (top-left) della matrice della vista pilota nella matrice grande
        tl_x = self._agent_location[0] - view_padding
        tl_y = self._agent_location[1] - view_padding

        view_matrix_x = range(tl_x, tl_x+self.view_size,1)
        view_matrix_y = range(tl_y, tl_y+self.view_size,1)

        #tengo soltanto gli indici interni alla matrice grande
        clipped_view_matrix_x = np.clip(view_matrix_x,0, max_x - 1)
        clipped_view_matrix_y = np.clip(view_matrix_y,0, max_y - 1)
        #rimuovo i duplicati creati da np.clip
        clipped_view_matrix_x = list(set(clipped_view_matrix_x))
        clipped_view_matrix_y = list(set(clipped_view_matrix_y))

        #copio i valori dalla matrice grande in quella della visione del pilota
        for x in clipped_view_matrix_x:
            for y in  clipped_view_matrix_y:
                view_matrix[x-tl_x , y-tl_y]=self.matrix[y,x]

        view_matrix = view_matrix.T


        return { "agent_view": view_matrix 
            #"direction": self._agent_direction,
            # "velocity":self._agent_velocity
            }


    #come info restituiamo la posizione e direzione del pilota
    def _get_info(self):
        return { "agent_location":self._agent_location
                #,"agent_direction":self._agent_direction,
                #"agent_velocity":self._agent_velocity
                }


    #con reset, rimettiamo il pilora nella posizione iniziale dal traguard
    # e ristabiliamo la sua direzione
    def reset(self, seed = None, options = None):

        super().reset(seed=seed)

        # self._agent_direction = math.pi

        #self._agent_velocity = 1

        self._progresso = 0

        coordinates = np.argwhere(self.matrix == 2)

        #da cambiare
        self._agent_location = np.array(coordinates[self.road_width // 2], dtype = np.int32)


        observation = self._get_obs()
        info = self._get_info()
        return observation, info


    def step(self, action):

        size = self.matrix.shape[0]
        direction = self._action_to_direction[action]

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        self._agent_location = np.clip( self._agent_location + direction, 0, size - 1 )


        self.trajectory.append(self._agent_location)
        # Check if agent reached the target
        terminated=False
        #for x in self._target_location:
        #   if np.array_equal(x, self._agent_location):
        #       print("--------------------")
        #       terminated = True

            # We don't use truncation in this simple environment
            # (could add a step limit here if desired)
        truncated = False


    # traguardo si attiva quando tutti i checkpoint sono stati traversati
    # se il tragurdo è traversato prima gli diamo un reward negativo assurdo


        # da aggingere logica reward negativo fuori strada
        '''stavo pensando se invece non é meglio terminare la simulazione se va fuori. Altrimenti se la continuiamo lui raccoglie dati su come guidare
        fuori dal tracciato e nella migliore delle ipotesi al massimo imparerebbe come rientrare in pista. Solo che il nostro obiettivo é che non esca proprio
        quindi aggiungerei un
        '''

        if self.matrix[self._agent_location[0],self._agent_location[1]] == -1:
            reward = -1000
            truncated = True
    
        else:
            # Default small negative step penalty
            reward = -1

            # Safely get the next checkpoint list (may be empty or missing)
            checkpoint_key = f"checkpoint_{self._progresso+1}"
            checkpoint_list = self._checkpoints.get(checkpoint_key, [])

            for x in checkpoint_list:
                if np.array_equal(self._agent_location, x):
                    reward = 1000
                    self._progresso = self._progresso + 1
                    break


        '''
        #1. reward = -1 costante + bonus checkpoint

        checkpoint pensavo di rappresentarlo con un dizionario {id_numerico: [stato, [riga1, colonna1], [riga2, colonna2], [riga3, colonna3]...]}
        dove stato indica se il checkpoint é 0 = da prendere, 1 = giá preso e 2 = attivo (se li vogliamo attivare sequenzialmente come avevamo detto)

        if checkpoint_taken(self._agent_location, checkpoint_list) == 2: < metodo da scrivere, prende la lista di checkpoint e controlla se l'agente si trova
                                                                    sopra checkpoint attivo, ritorna 2 se tutto OK, 1 se passa su checkpoint giá preso, 0 altrimenti
        reward = 10
        else:
        reward = -1
        _______________________________________________________________________________________________

        #2. reward = -(distanza_massima - distanza) costante checkpoint 2 volte -> reward negativo

            reward = -(self.distance_matrix.max() - self.distance_matrix[self._agent_location[0],self._agent_location[1]])
            if checkpoint_taken(self._agent_location, checkpoint_list) == 1:
            reward = -5






        # reward = -( numero_reward_traguardo - distnza) costante + logica checkpoint
        # reward =
        '''
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            return self._render_frame()


    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0)) # Sfondo nero di base

        # Calcolo della dimensione di ogni cella
        # Prendiamo il lato più lungo della matrice per scalare tutto nella finestra quadrata
        max_dim = max(self.matrix.shape)
        pix_square_size = self.window_size / max_dim

        # --- DISEGNO DELLA PISTA ---
        rows, cols = self.matrix.shape
        for r in range(rows):
            for c in range(cols):
                val = self.matrix[r, c]
                
                # Determina il colore in base al valore nella matrice
                color = None
                if val == 2:
                    color = (255, 0, 0)      # Rosso per il traguardo
                elif val == 0:                
                    color= (255,255,255)      #Bianco per i cordoli
                elif val == 1:
                    color = (128, 128, 128)  # Grigio per la strada
                else:
                    color = (34, 139, 34)    # Verde scuro per fuori strada (-1, -2)

                if color:
                    pygame.draw.rect(
                        canvas,
                        color,
                        pygame.Rect(
                            c * pix_square_size, # x (colonna)
                            r * pix_square_size, # y (riga)
                            pix_square_size,
                            pix_square_size,
                        ),
                    )

        # --- DISEGNO DELL'AGENTE ---
        # L'agente è un cerchio blu
        pygame.draw.circle(
            canvas,
            (0, 0, 255), # Blu
            (int((self._agent_location[1] + 0.5) * pix_square_size), # X = colonna
            int((self._agent_location[0] + 0.5) * pix_square_size)), # Y = riga
            int(pix_square_size / 3),
        )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )