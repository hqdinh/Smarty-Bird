""" Flappy Bird game in python
Real gravity physics. 
All parameters in Config can be changed without breaking the game
Simplified graphics (no background nor bird sprites)
Quit by pressing ESCAPE or click the x-mark
Play by clicking on the window or pressing any other key
"""
 

import sys
import numpy as np
import pygame

QUIT = pygame.QUIT
MOUSECLICK = pygame.MOUSEBUTTONDOWN
KEYDOWN = pygame.KEYDOWN
K_ESCAPE = pygame.K_ESCAPE
   
np.random.seed(1)

class FlappyBird:
  """Game class"""    
  def __init__(self):
    # Game configuration
    self.bird_x = 0  # Starting x-coor of bird. Stays the same throughout  
    self.bird_yo = 200  # Starting y-coor of bird
    self.gravity = 0.7  # How fast bird accel. downward
    self.jump_v = 8  # Each jump will set the bird's velocity to jump_v
    self.gap = 175  # Gap between each set of upper and lower pipes
    self.pipe_speed = 3.5  # How fast the pipes move to the left
    self.pipe_dist = 285  # How far each set of pipes are (the smaller the harder)
    self.offs = 190  # how far each subsequent gap may vary vertically 
    self.screen_W = 400  # screen width
    self.screen_H = 700  # screen height
    self.frame_speed = 120  # how fast the game runs
    self.limitless = True  # if True, game will run as fast as it can
        
    # Evolution configurations
    self.selected_pop = 10
    self.population = combination(self.selected_pop, 2)
    self.mutation_rate = 0.5
    self.gene_size = 26
    self.fund_sel_rate = 0.25
    self.ga = GA(self.gene_size, 
                 self.population, 
                 self.selected_pop, 
                 self.fund_sel_rate,
                 self.mutation_rate)
    
    # Game info
    self.score = 0
    self.high_score = 0
    self.generation = 0
    self.dead_birds = 0
    self.distance_traveled = 0  # this is used to measure fitness
    self.score_hist = []  # to be plotted at end game
        
    # Image loading  
    self.screen_extra = 300  # the white space to the right
    self.screen = pygame.display.set_mode((self.screen_W + self.screen_extra, self.screen_H))
    self.background = pygame.image.load("assets/white.png").convert_alpha()
    self.bird_img = pygame.image.load("assets/pusheen_50.png").convert_alpha()
    self.pipe_bot = pygame.image.load("assets/bottom.png").convert_alpha()
    self.pipe_top = pygame.image.load("assets/top.png").convert_alpha()
    self.pipe_W = self.pipe_top.get_width()
    self.pipe_H = self.pipe_top.get_height()
    self.pipe_num = self.screen_W//self.pipe_W
            
    # Bird settings
    print('Generation - Score')
    self.new_generation()
    
    
  def new_generation(self):
    """ Game reset """
    # When there's a new high score, do these to preserve good genes: 
    # - Increase fundamental selection rate
    # - Decrease mutation rate
    if self.score > self.high_score:
      self.high_score = self.score
      self.ga.mutation_rate *= 0.9
      self.ga.C *= 1.05
     
    self.distance_traveled = 0
    self.score = 0
    self.generation += 1
      
    # Pipe reset
    self.pipe_x = [self.screen_W + n*self.pipe_dist 
                   for n in range(self.pipe_num)]
    self.pipe_bot_y = [self.screen_H/2 + self.gap/2 + rand_offset(self.offs) 
                       for n in range(self.pipe_num)]
    self.pipe_top_y = [d - self.gap - self.pipe_H 
                       for d in self.pipe_bot_y]
    self.pipe_pos = [self.pipe_x, self.pipe_bot_y, self.pipe_top_y]
    self.pipe_it = 0  # Iterator
    
    # Bird reset
    self.dead_birds = 0
    self.birds = []

    if self.generation == 1:
      self.gene_pool = self.ga.generate_pool(1, -1)
    else:
      self.gene_pool = self.ga.evolve(self.gene_pool[:, -1])
    
    for i in range(self.population):
      gene = self.gene_pool[i, :]
      self.birds.append(Bird(i, self.bird_x, self.bird_yo, gene))
        
      
  def update_pipes(self):
    """Update pipes' positions. Note that
    pipe_pos[0] == pipe_x 
    pipe_pos[1] == pipe_bot_y 
    pipe_pos[2] == pipe_top_y""" 
    self.pipe_pos[0] = [x - self.pipe_speed for x in self.pipe_pos[0]]
    
    # replace left-most pipe with a pipe after curr pipe passed left edge of screen
    if self.pipe_pos[0][self.pipe_it] < -self.pipe_W:
      # x coordinates of each pair
      self.pipe_pos[0][self.pipe_it] = self.pipe_pos[0][self.pipe_it] + \
                                       self.pipe_num*self.pipe_dist
      # y coordinate of bottom pipe
      self.pipe_pos[1][self.pipe_it] = self.screen_H/2 + self.gap/2 + \
                                       rand_offset(self.offs)
      # y coordinate of top pipe
      self.pipe_pos[2][self.pipe_it] = self.pipe_pos[1][self.pipe_it] - \
                                       self.gap - self.pipe_H 
      self.pipe_it += 1
      self.score = int(self.score) + 1
      self.pipe_it = self.pipe_it % self.pipe_num
         
      
  def update_bird(self):
    """Update each bird's position"""
    # Decision step with 3 inputs
    for bird in self.birds:
      dx = self.pipe_pos[0][self.pipe_it] + self.pipe_W/2
      dy1 = self.pipe_pos[1][self.pipe_it] - self.gap - bird.y  # y-dist to top corner
      dy2 = self.pipe_pos[1][self.pipe_it] - bird.y  # y-dist to bot corner
      inputs = np.array([dx, dy1, dy2])
      bird.decide(inputs)
      
      if bird.jump:
        bird.v = - self.jump_v  # because y points down to the ground
        bird.jump = False
      else:
        bird.v = bird.v + self.gravity
      
      bird.y = bird.y - bird.v
      bird.rect[1] = bird.y  # Update bird rectangle's y-coor
        
      
  def collision_check(self):
    """ Checks for bird-pipe and bird-ground collisions """      
    # pipe_rects contains 2 pipe rectangles for collision check
    pipe_rects = []
    pipe_rects = [pygame.Rect(self.pipe_pos[0][self.pipe_it], 
                             self.pipe_pos[1][self.pipe_it],
                             self.pipe_W - 10,
                             self.pipe_H)]
    pipe_rects += [pygame.Rect(self.pipe_pos[0][self.pipe_it],
                             self.pipe_pos[2][self.pipe_it],
                             self.pipe_W - 10,
                             self.pipe_H)]
                
    for bird in self.birds:
      if not bird.dead:
        # checks for collision: top pipe, bottom pipe, ground, roof
        if(pipe_rects[0].colliderect(bird.rect) or
          pipe_rects[1].colliderect(bird.rect) or 
          bird.y > self.screen_H - 35 or 
          bird.y < -30):
            bird.dead = True
            Dy = self.pipe_pos[1][self.pipe_it] - self.gap - bird.y
            Dx = self.distance_traveled
            fitness = Dx - abs(Dy)
            self.gene_pool[bird.id, -1] = fitness
            self.dead_birds += 1
                    
            
  def draw_objects(self):
    self.screen.fill((0, 0, 0))
    for i in range(self.pipe_num):
      if self.pipe_pos[0][i] < self.screen_W:
        self.screen.blit(self.pipe_bot, (self.pipe_pos[0][i], self.pipe_pos[1][i]))
        self.screen.blit(self.pipe_top, (self.pipe_pos[0][i], self.pipe_pos[2][i]))
    for bird in self.birds:
      if not bird.dead:
        self.screen.blit(self.bird_img, (bird.x, bird.y))
    self.screen.blit(self.background, (self.screen_W, 0))
    
    txt_color = (100, 0, 100)
    str1 = '{:14}'.format('Generation: ') + str(self.generation)
    str2 = '{:14}'.format('Score: ') + str(self.score)
    str3 = '{:14}'.format('Birds alive: ') + str(self.population - self.dead_birds) + '/' + str(self.population)
    str4 = '{:14}'.format('High score: ') + str(self.high_score)
    str5 = '{:15}'.format('Mutation rate:') + '{:5.2f}'.format(self.ga.mutation_rate)
    str6 = '{:15}'.format('Selection rate:') + '{:5.2f}'.format(self.ga.C)
    self.screen.blit(self.font.render(str1, -1, txt_color), (self.screen_W + 5,50))
    self.screen.blit(self.font.render(str2, -1, txt_color), (self.screen_W + 5,75))
    self.screen.blit(self.font.render(str3, -1, txt_color), (self.screen_W + 5,100))
    self.screen.blit(self.font.render(str4, -1, txt_color), (self.screen_W + 5,125))
    self.screen.blit(self.font.render(str5, -1, txt_color), (self.screen_W + 5,175))
    self.screen.blit(self.font.render(str6, -1, txt_color), (self.screen_W + 5,200))
  
  
  def end_game(self):
    self.score_hist.append(self.score)
    print('{:10}'.format(self.generation), '-', self.score)
    print('Game over!')
    pygame.quit()
    sys.exit()
    
  def run_instance(self):
    """Main game"""            
    pygame.font.init()        
    self.font = pygame.font.SysFont("Courier_New", 20)
    
    while True:
      if not self.limitless:
        pygame.time.Clock().tick(self.frame_speed)
      
      events = pygame.event.get()
      for event in events:
        # If user quits
        if ((event.type == KEYDOWN and event.key == K_ESCAPE) or event.type == QUIT):
          self.end_game()
        
        # Increase / Decrease frame speed by 4 arrow keys
        if event.type == KEYDOWN:
          if event.key == pygame.K_UP:
            self.frame_speed *= 2
          if event.key == pygame.K_DOWN:
            self.frame_speed = self.frame_speed/2 + 1
            self.limitless = False
          if event.key == pygame.K_RIGHT:
            self.limitless = True
          if event.key == pygame.K_LEFT:
            self.limitless = False
            self.frame_speed = 120
            
        # Kill all birds, start breeding new gen
        if (event.type == KEYDOWN and event.key == pygame.K_k):
            self.dead_birds = self.population
      
      # When score reached 1000, end game and print successful genes without fitness
      if self.score == 1000:
        print('Successful genes: ')
        for bird in self.birds:
          if not bird.dead:
            print(self.gene_pool[bird.id, 0:-1])
        self.end_game()
      
      # if there's only 1 bird left 
      # and current score is already higher than high score by 20
      # kill it and start breeding new gen
      if (self.dead_birds == self.population - 1 and 
          self.score > self.high_score + 20):      
            self.dead_birds = self.population
            
      if (self.dead_birds >= self.population): 
        print('{:10}'.format(self.generation), '-', self.score)
        self.score_hist.append(self.score)
        self.new_generation()
        
      self.distance_traveled += self.pipe_speed    
      self.draw_objects()
      self.update_bird()
      self.update_pipes()
      self.collision_check()
     
      pygame.display.update()
  
  @staticmethod
  def start():
    game = FlappyBird()
    game.run_instance()

class GA:  
  """This class assumes every gene pool is a matrix of size (population x gene_size+1)
  By the end of each gene is the fitness of the gene. Newly created genes have 0 fitness
  The genes will be interpreted by Bird.form_network()"""
  def __init__(self, gene_size, 
               population, 
               selected_population,
               fund_select_rate = 0.25,
               mutation_rate = 0):
    self.gene_pool = 0    
    self.gene_size = gene_size
    self.pop = population
    self.selected_pop = selected_population
    self.mutation_rate = mutation_rate
    self.C = fund_select_rate  # fundamental selection rate
    
    # Create an array of probabilities, ranked from high to low
    # These are rank-based probabilities, independent of the fitness value
    self.P = np.zeros(population)
    for i in range(self.pop):
      self.P[i] = (1 - self.C)**i * self.C
    
  def generate_pool(self, high_lim = 10, low_lim = 0):
    """Generates a gene pool matrix of size (population x gene_size) with options:
       - high_lim: upper limit of the random
       - low_lim: lower limit of the random"""
    self.high_lim = high_lim
    self.low_lim = low_lim
    self.gene_range = high_lim - low_lim
    
    self.gene_pool = np.random.rand(self.pop, self.gene_size) * self.gene_range + self.low_lim
    self.gene_pool = np.concatenate((self.gene_pool, np.zeros((self.pop, 1))), axis=1)
    return self.gene_pool
  
  def evolve(self, fitness):
    # update fitness to self.gene_pool
    self.gene_pool[:, -1] = fitness
    self.gene_pool = self.selection()
    return self.gene_pool
  
  def selection(self):
    good_genes = []
    ranked_genes = np.flip(self.gene_pool[np.argsort(self.gene_pool[:, -1])], 0)
      
    # Iterate through the probabilities to pick genes until having picked enough
    # actually picking indices of good genes
    # real genes will be appended later to good_genes
    i = 0
    chosen = []  # stores the indices of chosen genes to breed
    while i < self.selected_pop:
      for j in range(self.pop):
        if np.random.rand() < self.P[j]:
          chosen.append(j)
          i += 1
          break    
    
    for i in chosen:
      good_genes.append(ranked_genes[i, :])
    good_genes = np.array(good_genes)
    self.gene_pool = self.breed(good_genes)
    
    return self.gene_pool
  
  def breed(self, good_genes):
    temp_gene_pool = np.zeros((self.pop + 1, self.gene_size + 1))
    n = 0    
    
    for i in range(self.selected_pop - 1):
      for j in range(i + 1, self.selected_pop):
        while n < self.pop:
          dad = np.copy(good_genes[i,:])
          mom = np.copy(good_genes[j,:])
          boy = np.copy(dad)
          girl = np.copy(mom)
          boy[-1] = 0  # reset fitness
          girl[-1] = 0  # reset fitness
          
          # Single point cross-over sequence
          k = np.random.randint(self.gene_size)
          temp = boy[0:k]
          boy[0:k] = girl[0:k]
          girl[0:k] = temp
          if self.mutation_rate/2 > np.random.rand():
            boy[np.random.randint(self.gene_size)] += 0.1 * self.gene_range * ((np.random.rand() < 0.5)*2 - 1)
            girl[np.random.randint(self.gene_size)] += 0.1 * self.gene_range * ((np.random.rand() < 0.5)*2 - 1) 
          
          # uniform cross-over sequence
#          for k in range(self.gene_size):
#            # swap gene by a chance of 50%
#            # if mutation occurs, add or subtract 10% of self.gene_range to that gene
#            sign = np.random.randint(2) * 2 - 1  # this returns either 1 or -1
#            if np.random.rand() >= 0.5:
#              boy[k] = mom[k]
#              girl[k] = dad[k]
#            boy[k] +=  0.05 * sign * (self.mutation_rate > np.random.rand())
#            girl[k] +=  0.05 * sign * (self.mutation_rate > np.random.rand())
                   
          temp_gene_pool[n,:] = np.copy(boy)
          temp_gene_pool[n+1,:] = np.copy(girl)
          n += 2

    return temp_gene_pool[0:self.pop, :]
   
class Bird:
  def __init__(self, birdID, xo, yo, genes, network_config = 0):
    self.id = birdID
    self.dead = False
    self.x = xo
    self.y = yo
    self.v = 0
    self.jump = False
    self.form_network(genes)
    self.fitness = 0     
    self.rect = pygame.Rect(xo, yo, 50, 50)
    
  def form_network(self, genes):
    self.A1 = np.reshape(genes[0:15], (5,3))
    self.b1 = np.reshape(genes[15:20], (5,1)) 
    self.A2 = np.reshape(genes[20:25], (1,5))
    self.b2 = genes[25]
      
  def decide(self, inputs):
    X1 = np.reshape(inputs, (3,1))
    X2 = np.dot(self.A1, X1) + self.b1
    X2 = np.tanh(X2) #, 0, X2)  # ReLU 
    Y = np.dot(self.A2, X2) + self.b2
    Y = sigmoid(Y)[0][0]
    
    self.jump = (Y >= 0.5)
  
  def update_fitness(x, h):
    return 0
       
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def combination(n, k):
  fact = np.math.factorial
  return int(fact(n) / fact(n-k) / fact(k))

def rand_offset(offset):
  return np.random.randint(-offset, offset)

if __name__ == "__main__":
  FlappyBird.start()       